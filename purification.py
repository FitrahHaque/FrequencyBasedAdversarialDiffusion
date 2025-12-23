import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from utils import diff2clf, clf2diff, normalize
import random

from typing import Optional
from frequency_masks import BaseFrequencyMask, RadialHardFrequencyMask, RadialSoftFrequencyMask
from wavelet_utils import multi_level_dwt, multi_level_idwt, WaveletBandMixer

def get_beta_schedule(beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(
        beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
    )
    assert betas.shape == (num_diffusion_timesteps,)
    return torch.from_numpy(betas).float()


class PurificationForward(torch.nn.Module):
    def __init__(
        self,
        clf,
        diffusion,
        max_timestep,
        attack_steps,
        sampling_method,
        is_imagenet,
        device,
        amplitude_cut_range,
        phase_cut_range,
        delta,
        forward_noise_steps,
        amplitude_mask: Optional[BaseFrequencyMask] = None,
        phase_mask: Optional[BaseFrequencyMask] = None,
        learnable_delta: bool = False,
        transform_type: str = 'dft',
        wavelet_levels: int = 2,
        wavelet_mixer: WaveletBandMixer | None = None,   # NEW
    ):
        super().__init__()
        self.clf = clf
        self.diffusion = diffusion
        self.device = device

        self.betas = get_beta_schedule(1e-4, 2e-2, 1000).to(device)
        self.max_timestep = max_timestep
        self.attack_steps = attack_steps

        self.sampling_method = sampling_method
        assert self.sampling_method in ["ddim", "ddpm"]
        self.eta = 0 if self.sampling_method == "ddim" else 1

        self.is_imagenet = is_imagenet
        self.forward_noise_steps = forward_noise_steps
        self.transform_type = transform_type
        self.wavelet_levels = wavelet_levels

        if amplitude_mask is None:
            self.amplitude_mask: BaseFrequencyMask = RadialHardFrequencyMask(
                cutoff=amplitude_cut_range,
                device=device,
                learnable=False,
            )
        else:
            self.amplitude_mask = amplitude_mask

        if phase_mask is None:
            self.phase_mask: BaseFrequencyMask = RadialHardFrequencyMask(
                cutoff=phase_cut_range,
                device=device,
                learnable=False,
            )
        else:
            self.phase_mask = phase_mask

        # Delta for phase projection (DFT & wavelet)
        delta_tensor = torch.as_tensor(float(delta), dtype=torch.float32, device=device)
        if learnable_delta:
            self.delta = torch.nn.Parameter(delta_tensor)
        else:
            self.register_buffer("delta", delta_tensor)

        # --- Wavelet mixer --------------------------------------
        if self.transform_type == 'wavelet':
            if wavelet_mixer is None:
                # Default: fixed weights; learning is enabled via train script.
                self.wavelet_mixer = WaveletBandMixer(learnable=False).to(device)
            else:
                self.wavelet_mixer = wavelet_mixer
        else:
            self.wavelet_mixer = None

    def compute_alpha(self, t):
        beta = torch.cat(
            [torch.zeros(1).to(self.betas.device), self.betas], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a
    
    def wavelet_exchange(self, x_ref, x_est):
        """
        Wavelet-based frequency exchange (Combo C).

        x_ref: Reference image (adversarial), (B, C, H, W) in [0, 1]
        x_est: Estimated image from diffusion, (B, C, H, W) in [0, 1]
        """
        # Forward multi-level DWT
        LL_ref, coeffs_ref, pads = multi_level_dwt(x_ref, levels=self.wavelet_levels)
        LL_est, coeffs_est, _ = multi_level_dwt(x_est, levels=self.wavelet_levels)

        # If we have a mixer (Combo C), use it; otherwise fall back to a simple baseline
        if self.wavelet_mixer is not None:
            LL_new, coeffs_new = self.wavelet_mixer(LL_ref, LL_est, coeffs_ref, coeffs_est)
        else:
            # Baseline: copy LL from ref, clamp details around ref within +/- delta
            LL_new = LL_ref
            coeffs_new = []
            for (LH_r, HL_r, HH_r), (LH_e, HL_e, HH_e) in zip(coeffs_ref, coeffs_est):
                LH_n = torch.clamp(LH_e, LH_r - self.delta, LH_r + self.delta)
                HL_n = torch.clamp(HL_e, HL_r - self.delta, HL_r + self.delta)
                HH_n = torch.clamp(HH_e, HH_r - self.delta, HH_r + self.delta)
                coeffs_new.append((LH_n, HL_n, HH_n))

        # Inverse DWT
        x_new = multi_level_idwt(LL_new, coeffs_new, pads)
        return torch.clamp(x_new, 0, 1)

    def get_noised_x(self, x, t):
        e = torch.randn_like(x)
        if type(t) == int:
            t = (torch.ones(x.shape[0]) * t).to(x.device).long()
        a = (1 - self.betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x = x * a.sqrt() + e * (1.0 - a).sqrt()
        return x

    def denoising_process(self,ori_x, x, seq):
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xt = x
        for i, j in zip(reversed(seq), reversed(seq_next)):

            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(t.long())
            at_next = self.compute_alpha(next_t.long())
            et = self.diffusion(xt, t)
            if self.is_imagenet:
                et, _ = torch.split(et, 3, dim=1)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_t = self.amplitude_phase_exchange_torch(ori_x,x0_t)
            c1 = (
                self.eta * ((1 - at / at_next) *
                            (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
        return xt

    def preprocess(self, x):
        # diffusion part
        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
        x_diff = clf2diff(x)
        for i in range(len(self.max_timestep)):
            noised_x = self.get_noised_x(x_diff, self.max_timestep[i])
            x_diff = self.denoising_process(noised_x, self.attack_steps[i])

        x_clf = diff2clf(x_diff)
        return x_clf

    def classify(self, x):
        logits = self.clf(x)
        return logits

    def forward(self, x):
        # diffusion part
        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
        x_diff = clf2diff(x)
        for i in range(len(self.max_timestep)):
            noised_x = self.get_noised_x(x_diff, self.max_timestep[i])
            x_diff = self.denoising_process(x_diff,noised_x, self.attack_steps[i])

        # classifier part
        if self.is_imagenet:
            x_clf = normalize(diff2clf(F.interpolate(x_diff, size=(
                224, 224), mode='bilinear', align_corners=False)))
        else:
            x_clf = diff2clf(x_diff)
        logits = self.clf(x_clf)
        return logits
    

    def compute_fft(self,image):  
        amplitude_channels = []  
        phase_channels = []  

        for channel in range(3):  
            f = torch.fft.fft2(image[channel, :, :])  
            fshift = torch.fft.fftshift(f)  
            amplitude = torch.abs(fshift)  
            amplitude_channels.append(amplitude)  
            phase = torch.angle(fshift)  
            phase_channels.append(phase + torch.pi)  

        return amplitude_channels, phase_channels

    def _frequency_radius_grid(self, rows: int, cols: int, device: torch.device) -> torch.Tensor:
        """
        Build a (rows, cols) tensor where each entry is the distance to the
        frequency origin in (u, v) space. Used by the masks.
        """
        u = torch.arange(-cols // 2, cols // 2, device=device)
        v = torch.arange(-rows // 2, rows // 2, device=device)
        # meshgrid returns (rows, cols) when indexing='ij'; default is fine here
        V, U = torch.meshgrid(v, u, indexing="ij") if hasattr(torch.meshgrid, "__call__") else torch.meshgrid(v, u)
        radius = torch.sqrt(U.float() ** 2 + V.float() ** 2)
        return radius

    def low_pass_exchange(self, amplitude_channels, amplitude_channels_0_t):
        """
        Amplitude Spectrum Exchange (ASE):

        For each color channel, we interpolate between:
            - amplitude from adversarial (amplitude_channels)
            - amplitude from current estimate (amplitude_channels_0_t)

        The interpolation weights come from self.amplitude_mask(radius).

        If the mask is boolean (RadialHardFrequencyMask), this reproduces the
        original behavior (hard swap inside low frequencies). If the mask is
        soft (RadialSoftFrequencyMask), this becomes a smooth blend.
        """
        filtered_amplitude_channels = []
        for i in range(3):
            rows, cols = amplitude_channels[i].shape
            radius = self._frequency_radius_grid(rows, cols, self.device)
            mask = self.amplitude_mask(radius)

            # Hard mask (original behavior)
            if mask.dtype == torch.bool:
                # same as torch.where(low_frequency, adv_amp, est_amp)
                filtered = torch.where(
                    mask,
                    amplitude_channels[i],
                    amplitude_channels_0_t[i],
                )
            else:
                # Soft mask in [0, 1]: linear interpolation
                mask = mask.to(amplitude_channels[i].dtype)
                filtered = mask * amplitude_channels[i] + (1.0 - mask) * amplitude_channels_0_t[i]

            filtered_amplitude_channels.append(filtered)
        return filtered_amplitude_channels


    def phase_low_pass_exchange(self, phase_channels, phase_channels_0_t):
        """
        Phase Spectrum Projection (PSP) in low frequencies.

        For each channel:
          1) Combine low-frequency phase from adversarial & estimate
             according to self.phase_mask.
          2) Clip the resulting phase to be within +/- self.delta around the
             adversarial phase (as in the original paper).
        """
        filtered_phase_channels = []
        for i in range(3):
            rows, cols = phase_channels[i].shape
            radius = self._frequency_radius_grid(rows, cols, self.device)
            mask = self.phase_mask(radius)

            adv_phase = phase_channels[i]
            est_phase = phase_channels_0_t[i]

            if mask.dtype == torch.bool:
                # Original behavior: hard replace low-frequency phase
                blended = torch.where(mask, adv_phase, est_phase)

                # Clip ONLY where we took from adversarial low frequencies
                lower = adv_phase - self.delta
                upper = adv_phase + self.delta
                clipped = blended.clone()
                clipped[mask] = torch.clamp(
                    blended[mask],
                    min=lower[mask],
                    max=upper[mask],
                )
            else:
                # Soft blend everywhere
                mask = mask.to(adv_phase.dtype)
                blended = mask * adv_phase + (1.0 - mask) * est_phase
                lower = adv_phase - self.delta
                upper = adv_phase + self.delta
                clipped = torch.clamp(blended, min=lower, max=upper)

            filtered_phase_channels.append(clipped)
        return filtered_phase_channels

    
    def phase_exchange(self,phase_channels,phase_channels_0_t):
        exchanged_phase_channels = []
        for i in range(3):
            rows, cols = phase_channels[i].shape
            exchange_matrix = self.generate_frequency_exchange_matrix(rows, cols)
            phase_channels_0_t[i][exchange_matrix] = phase_channels[i][exchange_matrix]
            exchanged_phase_channels.append(phase_channels_0_t[i])
        return exchanged_phase_channels
    
    def phase_clip(self,phase_channels,phase_channels_0_t,delta=0.6):
        phase_channels_clip=[]
        for i in range(3):
            phase_channels_clip.append(np.clip(phase_channels_0_t[i],phase_channels[i]-delta,phase_channels[i]+delta))
        return phase_channels_clip

    
    def reconstruct_image(self,filtered_amplitude_channels, phase_channels):
        reconstructed_image = []
        for channel in range(3):
            amplitude = filtered_amplitude_channels[channel]
            phase = phase_channels[channel]-torch.pi
            fshift_filtered = amplitude * torch.exp(1j * phase)
            f_ishift = torch.fft.ifftshift(fshift_filtered)
            img_reconstructed = torch.fft.ifft2(f_ishift)
            img_reconstructed = torch.abs(img_reconstructed)
            img_reconstructed = torch.clip(img_reconstructed,0,255)
            reconstructed_image.append(img_reconstructed/255)
        return torch.stack(reconstructed_image,dim=2)

        
    
    def amplitude_phase_exchange_torch(self,x,x_0_t):
        x_t = self.get_noised_x(x, self.forward_noise_steps)
        t = (torch.ones(x.size(0)) * self.forward_noise_steps).to(x.device)
        at = self.compute_alpha(t.long())
        et = self.diffusion(x_t, t)
        x =  (x_t - et * (1 - at).sqrt()) / at.sqrt()
        # save_image(diff2clf(x), 'new_x_0_t.png')
        
                
        # Wavelet-based exchange
        if self.transform_type == 'wavelet':
            x_ref = torch.clamp(diff2clf(x), 0, 1)
            x_est = torch.clamp(diff2clf(x_0_t), 0, 1)
            new_x_0_t_clf = self.wavelet_exchange(x_ref, x_est)
            return clf2diff(new_x_0_t_clf)
        
        # DFT-based exchange (original)
        x = torch.clip((diff2clf(x)* 255),0,255)
        x_0_t = torch.clip((diff2clf(x_0_t)* 255),0,255)

        batch,channel,height,width = x.shape
        new_x_0_t = torch.zeros(size=(batch,height,width,channel))
        for batch_idx in range(batch):

            amplitude_channels, phase_channels = self.compute_fft(x[batch_idx])

            amplitude_channels_0_t, phase_channels_0_t = self.compute_fft(x_0_t[batch_idx])

            amplitude_channels_0_t_exchange = self.low_pass_exchange(amplitude_channels,amplitude_channels_0_t)
            
            phase_channels_0_t_exchange = self.phase_low_pass_exchange(phase_channels,phase_channels_0_t)
            reconstructed_image = self.reconstruct_image(amplitude_channels_0_t_exchange,phase_channels_0_t_exchange)

            new_x_0_t[batch_idx] = reconstructed_image
        new_x_0_t = new_x_0_t.float().permute(0,3,1,2).to(self.device)
        new_x_0_t = clf2diff(new_x_0_t)
        return new_x_0_t

    def get_img_logits(self, x):

        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
        x_diff = clf2diff(x)
        for i in range(len(self.max_timestep)):
            noised_x = self.get_noised_x(x_diff, self.max_timestep[i])
            x_diff = self.denoising_process(x_diff,noised_x, self.attack_steps[i])

        # classifier part
        if self.is_imagenet:
            x_clf = normalize(diff2clf(F.interpolate(x_diff, size=(
                224, 224), mode='bilinear', align_corners=False)))
        x_clf = diff2clf(x_diff)
        logits = self.clf(x_clf)
        return x_clf,logits