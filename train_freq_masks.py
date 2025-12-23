import os
import math
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

from path import diffusion_model_path
from load_model import load_models
from purification import PurificationForward
from attacks.pgd_eot import PGD
from frequency_masks import RadialHardFrequencyMask, RadialSoftFrequencyMask
from wavelet_utils import WaveletBandMixer


# -------------------------------------------------------
# Utilities
# -------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_diffusion_params(max_timesteps: str, num_denoising_steps: str) -> Tuple[List[int], List[List[int]]]:
    """
    Re-implementation of the schedule helper used in ddp_test.py.

    Args:
        max_timesteps: string like "999" or "750,999"
        num_denoising_steps: string like "5" or "5,5"

    Returns:
        max_timestep_list: list of final timesteps (0-based)
        diffusion_steps:   list of lists of integer timesteps (0-based)
    """
    max_timestep_list = [int(s) for s in max_timesteps.split(",")]
    nd_steps_list = [int(s) for s in num_denoising_steps.split(",")]
    assert len(max_timestep_list) == len(nd_steps_list)

    diffusion_steps: List[List[int]] = []
    for idx, (mt, nd) in enumerate(zip(max_timestep_list, nd_steps_list)):
        step = mt // nd
        steps = list(range(step, mt + 1, step))
        # convert to 0-based indices as used in purification
        steps = [t - 1 for t in steps]
        diffusion_steps.append(steps)
        max_timestep_list[idx] = mt - 1  # also 0-based
    return max_timestep_list, diffusion_steps


def build_cifar10_val_loader(data_root: str, num_val: int, batch_size: int) -> DataLoader:
    """
    Simple CIFAR-10 loader (test split) with ToTensor only, matching the rest of the repo.
    """
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.CIFAR10(
        root=os.path.join(data_root, "cifar10"),
        train=False,
        download=True,
        transform=transform,
    )
    if num_val > 0 and num_val < len(dataset):
        indices = torch.randperm(len(dataset))[:num_val].tolist()
        dataset = Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return loader


def build_masks(args, device: torch.device):
    """
    Construct amplitude and phase masks based on CLI flags.
    """
    if args.freq_mask_type == "hard":
        amp_mask = RadialHardFrequencyMask(
            cutoff=args.amplitude_cut_range,
            device=device,
            learnable=args.learn_freq_masks,  # usually False for exact baseline
        )
        phase_mask = RadialHardFrequencyMask(
            cutoff=args.phase_cut_range,
            device=device,
            learnable=args.learn_freq_masks,
        )
    else:  # soft
        amp_mask = RadialSoftFrequencyMask(
            init_cutoff=args.amplitude_cut_range,
            init_sharpness=args.init_sharpness,
            device=device,
            learnable=args.learn_freq_masks,
        )
        phase_mask = RadialSoftFrequencyMask(
            init_cutoff=args.phase_cut_range,
            init_sharpness=args.init_sharpness,
            device=device,
            learnable=args.learn_freq_masks,
        )
    return amp_mask, phase_mask


def build_purifier(args, device: torch.device, clf: nn.Module, diffusion: nn.Module,
                   amp_mask: nn.Module, phase_mask: nn.Module,
                   wavelet_mixer: nn.Module | None = None) -> PurificationForward:
    """
    Create a PurificationForward instance with the given masks,
    freezing classifier and diffusion weights.
    """
    # Freeze backbone models
    for p in clf.parameters():
        p.requires_grad_(False)
    for p in diffusion.parameters():
        p.requires_grad_(False)

    max_timestep_list, diffusion_steps = get_diffusion_params(
        args.max_timesteps, args.num_denoising_steps
    )

    purifier = PurificationForward(
        clf=clf,
        diffusion=diffusion,
        max_timestep=max_timestep_list,
        attack_steps=diffusion_steps,
        sampling_method=args.sampling_method,
        is_imagenet=(args.dataset == "imagenet"),
        device=device,
        amplitude_cut_range=args.amplitude_cut_range,
        phase_cut_range=args.phase_cut_range,
        delta=args.delta,
        forward_noise_steps=args.forward_noise_steps,
        amplitude_mask=amp_mask,
        phase_mask=phase_mask,
        learnable_delta=args.learn_delta,
        transform_type=args.transform_type,       
        wavelet_levels=args.wavelet_levels,    
        wavelet_mixer=wavelet_mixer,
    ).to(device)

    return purifier


def build_attacker(purifier: PurificationForward, args) -> PGD:
    """
    PGD-EOT attacker that treats the purifier + classifier as the model.
    We do NOT differentiate through the attack when updating masks.
    """
    def get_logit(x):
        # x in [0,1], shape (B,3,H,W)
        return purifier(x)

    attacker = PGD(
        get_logit=get_logit,
        attack_steps=args.attack_steps,
        eps=args.eps,
        step_size=args.step_size,
        eot=args.eot,
    )
    return attacker


def collect_mask_parameters(purifier: PurificationForward, amp_mask: nn.Module,
                            phase_mask: nn.Module, learn_delta: bool) -> list:
    params = list(amp_mask.parameters()) + list(phase_mask.parameters())
    if learn_delta and isinstance(purifier.delta, nn.Parameter):
        params.append(purifier.delta)
    return params


# -------------------------------------------------------
# Train & evaluation loops
# -------------------------------------------------------

def evaluate(purifier: PurificationForward,
             attacker: PGD,
             val_loader: DataLoader,
             device: torch.device):
    purifier.eval()
    clean_correct = 0
    adv_correct = 0
    total = 0

    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)
        batch_size = x.size(0)
        total += batch_size

        # ----- Clean accuracy (no gradient needed) -----
        with torch.no_grad():
            logits_clean = purifier(x)
            pred_clean = logits_clean.argmax(dim=1)
            clean_correct += (pred_clean == y).sum().item()

        # ----- Adversarial examples (PGD needs gradients) -----
        with torch.enable_grad():       # <-- turn gradients ON for attacker
            x_adv = attacker(x, y)

        # We don't need gradients for evaluating the purified logits
        with torch.no_grad():
            logits_adv = purifier(x_adv)
            pred_adv = logits_adv.argmax(dim=1)
            adv_correct += (pred_adv == y).sum().item()

    clean_acc = clean_correct / total
    adv_acc = adv_correct / total
    return clean_acc, adv_acc


def train_masks(args):
    device = get_device()
    set_seed(args.seed)
    print(f"Using device: {device}")

    # ---------------------------
    # 1) Data
    # ---------------------------
    if args.dataset != "cifar10":
        raise NotImplementedError("This training script currently assumes CIFAR-10.")
    val_loader = build_cifar10_val_loader(
        data_root=args.data_root,
        num_val=args.num_val,
        batch_size=args.batch_size,
    )

    # ---------------------------
    # 2) Models
    # ---------------------------
    model_src = diffusion_model_path[args.dataset]
    clf, diffusion = load_models(args, model_src, device)

        # --- Build purifier + masks depending on transform type ---
    if args.transform_type == "dft":
        # Original: learn DFT radial masks
        amp_mask, phase_mask = build_masks(args, device)
        wavelet_mixer = None
        purifier = build_purifier(args, device, clf, diffusion, amp_mask, phase_mask, wavelet_mixer)
        attacker = build_attacker(purifier, args)

        mask_params = collect_mask_parameters(purifier, amp_mask, phase_mask, args.learn_delta)

    else:  # args.transform_type == "wavelet"
        # Wavelet Combo C: keep DFT masks fixed (unused) and learn tiny wavelet mixer
        amp_mask = RadialHardFrequencyMask(
            cutoff=args.amplitude_cut_range,
            device=device,
            learnable=False,
        )
        phase_mask = RadialHardFrequencyMask(
            cutoff=args.phase_cut_range,
            device=device,
            learnable=False,
        )
        wavelet_mixer = WaveletBandMixer(learnable=True).to(device)

        purifier = build_purifier(args, device, clf, diffusion, amp_mask, phase_mask, wavelet_mixer)
        attacker = build_attacker(purifier, args)

        mask_params = list(purifier.wavelet_mixer.parameters())
        if args.learn_delta and isinstance(purifier.delta, nn.Parameter):
            mask_params.append(purifier.delta)
    if not mask_params:
        raise RuntimeError("No learnable mask parameters found. Did you set --learn-freq-masks or --learn-delta?")

    optimizer = torch.optim.Adam(mask_params, lr=args.lr, weight_decay=args.weight_decay)

    print("Trainable parameters:")
    for name, p in purifier.named_parameters():
        if p.requires_grad:
            print(f"  {name}: {p.shape}")

    # Baseline cutoffs for regularization
    base_amp_cutoff = torch.tensor(float(args.amplitude_cut_range), device=device)
    base_phase_cutoff = torch.tensor(float(args.phase_cut_range), device=device)

    # ---------------------------
    # 4) Training loop
    # ---------------------------
    best_adv_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        purifier.train()  # masks are trainable; backbone is frozen
        running_loss = 0.0
        running_robust = 0.0
        running_clean = 0.0
        num_batches = 0

        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            # ---- Generate adversarial examples (no gradient into mask parameters) ----
            # PGD already uses autograd internally to get gradient w.r.t x_adv only.
            x_adv = attacker(x, y).detach()

            optimizer.zero_grad()

            # ---- Robust loss: CE on purified adversarial images ----
            logits_adv = purifier(x_adv)
            loss_robust = F.cross_entropy(logits_adv, y)

            # ---- Clean regularization: CE on purified clean images ----
            logits_clean = purifier(x)
            loss_clean = F.cross_entropy(logits_clean, y)

            loss = loss_robust + args.lambda_clean * loss_clean

            # ---- Optional cutoff regularization (keep cutoffs near initial values) ----
            if args.reg_cutoff > 0:
                reg_terms = []

                if isinstance(amp_mask, RadialSoftFrequencyMask):
                    reg_terms.append((amp_mask.cutoff - base_amp_cutoff) ** 2)

                if isinstance(phase_mask, RadialSoftFrequencyMask):
                    reg_terms.append((phase_mask.cutoff - base_phase_cutoff) ** 2)

                if reg_terms:
                    loss = loss + args.reg_cutoff * sum(reg_terms)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_robust += loss_robust.item()
            running_clean += loss_clean.item()
            num_batches += 1

        avg_loss = running_loss / max(1, num_batches)
        avg_robust = running_robust / max(1, num_batches)
        avg_clean = running_clean / max(1, num_batches)

        # Evaluation with current masks
        clean_acc, adv_acc = evaluate(purifier, attacker, val_loader, device)

        print(
            f"[Epoch {epoch:02d}] "
            f"loss={avg_loss:.4f} "
            f"(robust={avg_robust:.4f}, clean={avg_clean:.4f}) | "
            f"clean_acc={clean_acc:.4%}, adv_acc={adv_acc:.4%}"
        )

        # Save best
        if adv_acc > best_adv_acc:
            best_adv_acc = adv_acc
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            state = {
                "args": vars(args),
                "transform_type": args.transform_type,
            }

            if args.transform_type == "dft":
                state["amp_mask"] = amp_mask.state_dict()
                state["phase_mask"] = phase_mask.state_dict()

            if hasattr(purifier, "wavelet_mixer") and purifier.wavelet_mixer is not None:
                state["wavelet_mixer"] = purifier.wavelet_mixer.state_dict()

            # Save delta (whether or not it was learnable)
            state["delta"] = purifier.delta.detach().cpu()

            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(state, args.save_path)
            print(f"  >> New best adv acc {adv_acc:.4%}. Saved masks to {args.save_path}")


# -------------------------------------------------------
# CLI
# -------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train adaptive frequency masks for FreqPure")

    # Data / model
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"],
                        help="Dataset (currently only cifar10 supported).")
    parser.add_argument("--data-root", type=str, default="./datasets",
                        help="Root directory containing dataset subfolders (e.g., ./datasets/cifar10).")
    parser.add_argument("--num-val", type=int, default=1024,
                        help="Number of validation images to use for training masks (0 = all).")

    # Diffusion schedule / FreqPure hyperparameters
    parser.add_argument("--max-timesteps", type=str, default="999",
                        help="Max diffusion timestep(s), e.g. '999' or '750,999'.")
    parser.add_argument("--num-denoising-steps", type=str, default="5",
                        help="Number of denoising steps per schedule, e.g. '5' or '5,5'.")
    parser.add_argument("--sampling-method", type=str, default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--forward-noise-steps", type=int, default=50,
                        help="Forward noise steps used in amplitude_phase_exchange.")
    parser.add_argument("--amplitude-cut-range", type=float, default=18.0,
                        help="Initial low-frequency radius for amplitude mask.")
    parser.add_argument("--phase-cut-range", type=float, default=18.0,
                        help="Initial low-frequency radius for phase mask.")
    parser.add_argument("--delta", type=float, default=0.2,
                        help="Initial delta for phase projection.")

    # Frequency mask configuration
    parser.add_argument("--freq-mask-type", type=str, default="soft",
                        choices=["hard", "soft"],
                        help="Use 'hard' (binary) or 'soft' (sigmoid) radial masks.")
    parser.add_argument("--learn-freq-masks", action="store_true",
                        help="If set, make radial mask cutoffs learnable.")
    parser.add_argument("--learn-delta", action="store_true",
                        help="If set, make delta (for phase projection) learnable.")
    parser.add_argument("--init-sharpness", type=float, default=10.0,
                        help="Initial sharpness for soft masks.")

    # Adversarial training (PGD-EOT) hyperparameters
    parser.add_argument("--attack-steps", type=int, default=10,
                        help="Number of PGD steps used during training.")
    parser.add_argument("--eps", type=float, default=8.0 / 255.0,
                        help="L_inf perturbation budget.")
    parser.add_argument("--step-size", type=float, default=0.007,
                        help="PGD step size.")
    parser.add_argument("--eot", type=int, default=4,
                        help="Number of EOT samples for PGD gradient estimation.")

    # Optimization
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lambda-clean", type=float, default=0.5,
                        help="Weight for clean CE regularization term.")
    parser.add_argument("--reg-cutoff", type=float, default=1e-3,
                        help="L2 regularization weight to keep learned cutoffs near their initial values.")

    # Misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="./pretrained/freq_masks_cifar10.pth")
    
    # Transform type (DFT vs wavelet)
    parser.add_argument("--transform-type", type=str, default="dft",
                        choices=["dft", "wavelet"],
                        help="Use DFT-based or wavelet-based purification during mask training.")
    parser.add_argument("--wavelet-levels", type=int, default=2,
                        help="Number of wavelet decomposition levels when transform_type='wavelet'.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train_masks(args)