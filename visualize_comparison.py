"""
Visualization script for comparing DCT-FreqPure vs Wavelet-FreqPure.
Creates side-by-side comparisons of original, adversarial, and purified images.
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image, make_grid
from PIL import Image
import argparse

from load_data import load_dataset_by_name
from load_model import load_models
from purification import PurificationForward
from path import diffusion_model_path
from utils import diff2clf, clf2diff
from wavelet_utils import multi_level_dwt, multi_level_idwt
from attacks.pgd_eot import PGD


def visualize_comparison(num_samples=10, save_dir='./comparison_results'):
    """
    Generate comparison visualizations between DCT and Wavelet purification.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda:0')
    
    # Load dataset (small subset for visualization)
    dataset = load_dataset_by_name('cifar10', './dataset', num_samples)
    loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=False)
    
    # Load models
    class Args:
        dataset = 'cifar10'
    args = Args()
    model_src = diffusion_model_path['cifar10']
    clf, diffusion = load_models(args, model_src, device)
    
    # Common parameters
    def_max_timesteps = [999]
    def_diffusion_steps = [[i for i in range(9, 1000, 10)]]
    att_max_timesteps = [999]
    att_diffusion_steps = [[999]]
    
    # Create purification forwards
    dct_forward = PurificationForward(
        clf=clf, diffusion=diffusion, is_imagenet=False,
        max_timestep=def_max_timesteps, attack_steps=def_diffusion_steps,
        forward_noise_steps=50, amplitude_cut_range=10, phase_cut_range=10,
        delta=0.3, device=device, sampling_method='ddpm',
        transform_type='dct', wavelet_levels=2
    )
    
    wavelet_forward = PurificationForward(
        clf=clf, diffusion=diffusion, is_imagenet=False,
        max_timestep=def_max_timesteps, attack_steps=def_diffusion_steps,
        forward_noise_steps=50, amplitude_cut_range=10, phase_cut_range=10,
        delta=0.3, device=device, sampling_method='ddpm',
        transform_type='wavelet', wavelet_levels=2
    )
    
    # Create attack using DCT (standard attack)
    attack_forward = PurificationForward(
        clf=clf, diffusion=diffusion, is_imagenet=False,
        max_timestep=att_max_timesteps, attack_steps=att_diffusion_steps,
        forward_noise_steps=50, amplitude_cut_range=10, phase_cut_range=10,
        delta=0.3, device=device, sampling_method='ddpm',
        transform_type='dct', wavelet_levels=2
    )
    
    attack = PGD(
        attack_forward, attack_steps=50, eps=8/255, step_size=2/255, eot=5
    )
    
    # Process images
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        # Generate adversarial examples
        print("Generating adversarial examples...")
        with torch.enable_grad():
            x_adv = attack(x, y)
        
        # Purify with DCT
        print("Purifying with DCT...")
        with torch.no_grad():
            x_diff = clf2diff(x_adv)
            noised_x = dct_forward.get_noised_x(x_diff, dct_forward.max_timestep[0])
            x_purified_dct = dct_forward.denoising_process(x_diff, noised_x, dct_forward.attack_steps[0])
            x_purified_dct = diff2clf(x_purified_dct)
        
        # Purify with Wavelet
        print("Purifying with Wavelet...")
        with torch.no_grad():
            x_diff = clf2diff(x_adv)
            noised_x = wavelet_forward.get_noised_x(x_diff, wavelet_forward.max_timestep[0])
            x_purified_wavelet = wavelet_forward.denoising_process(x_diff, noised_x, wavelet_forward.attack_steps[0])
            x_purified_wavelet = diff2clf(x_purified_wavelet)
        
        # Compute perturbation magnitudes
        adv_pert = (x_adv - x).abs()
        dct_diff = (x_purified_dct - x).abs()
        wavelet_diff = (x_purified_wavelet - x).abs()
        
        # Get predictions
        with torch.no_grad():
            pred_orig = clf(x).argmax(dim=1)
            pred_adv = clf(x_adv).argmax(dim=1)
            pred_dct = clf(x_purified_dct).argmax(dim=1)
            pred_wavelet = clf(x_purified_wavelet).argmax(dim=1)
        
        # Class names for CIFAR-10
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Create comparison figure for each image
        for i in range(min(num_samples, 8)):  # Limit to 8 for readability
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            # Top row: Images
            axes[0, 0].imshow(x[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[0, 0].set_title(f'Original\n{classes[y[i]]}', fontsize=12)
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(x_adv[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[0, 1].set_title(f'Adversarial\nPred: {classes[pred_adv[i]]}', fontsize=12)
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(x_purified_dct[i].cpu().permute(1, 2, 0).clamp(0, 1))
            correct_dct = '✓' if pred_dct[i] == y[i] else '✗'
            axes[0, 2].set_title(f'DCT Purified {correct_dct}\nPred: {classes[pred_dct[i]]}', fontsize=12)
            axes[0, 2].axis('off')
            
            axes[0, 3].imshow(x_purified_wavelet[i].cpu().permute(1, 2, 0).clamp(0, 1))
            correct_wav = '✓' if pred_wavelet[i] == y[i] else '✗'
            axes[0, 3].set_title(f'Wavelet Purified {correct_wav}\nPred: {classes[pred_wavelet[i]]}', fontsize=12)
            axes[0, 3].axis('off')
            
            # Bottom row: Difference maps (enhanced for visibility)
            axes[1, 0].imshow((adv_pert[i] * 10).cpu().permute(1, 2, 0).clamp(0, 1))
            axes[1, 0].set_title(f'Adv Perturbation\n(10x enhanced)', fontsize=12)
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow((adv_pert[i].mean(dim=0) * 50).cpu().clamp(0, 1), cmap='hot')
            axes[1, 1].set_title('Perturbation Heatmap', fontsize=12)
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow((dct_diff[i].mean(dim=0) * 10).cpu().clamp(0, 1), cmap='viridis')
            mse_dct = dct_diff[i].pow(2).mean().item()
            axes[1, 2].set_title(f'DCT Error (MSE: {mse_dct:.4f})', fontsize=12)
            axes[1, 2].axis('off')
            
            axes[1, 3].imshow((wavelet_diff[i].mean(dim=0) * 10).cpu().clamp(0, 1), cmap='viridis')
            mse_wav = wavelet_diff[i].pow(2).mean().item()
            axes[1, 3].set_title(f'Wavelet Error (MSE: {mse_wav:.4f})', fontsize=12)
            axes[1, 3].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/comparison_sample_{i}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved comparison for sample {i}")
        
        # Create summary grid
        print("\nCreating summary grid...")
        fig, axes = plt.subplots(4, num_samples, figsize=(2*num_samples, 8))
        
        for i in range(num_samples):
            axes[0, i].imshow(x[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('Original', fontsize=14)
            
            axes[1, i].imshow(x_adv[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel('Adversarial', fontsize=14)
            
            axes[2, i].imshow(x_purified_dct[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[2, i].axis('off')
            if i == 0:
                axes[2, i].set_ylabel('DCT Purified', fontsize=14)
            
            axes[3, i].imshow(x_purified_wavelet[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[3, i].axis('off')
            if i == 0:
                axes[3, i].set_ylabel('Wavelet Purified', fontsize=14)
        
        plt.suptitle('DCT vs Wavelet FreqPure Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/summary_grid.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Compute and print statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        adv_acc = (pred_adv == y).float().mean().item() * 100
        dct_acc = (pred_dct == y).float().mean().item() * 100
        wav_acc = (pred_wavelet == y).float().mean().item() * 100
        
        print(f"Adversarial Accuracy (no defense): {adv_acc:.1f}%")
        print(f"DCT-FreqPure Accuracy: {dct_acc:.1f}%")
        print(f"Wavelet-FreqPure Accuracy: {wav_acc:.1f}%")
        print(f"\nImprovement (Wavelet vs DCT): {wav_acc - dct_acc:+.1f}%")
        
        # MSE comparison
        mse_dct_avg = dct_diff.pow(2).mean().item()
        mse_wav_avg = wavelet_diff.pow(2).mean().item()
        print(f"\nAvg MSE to Original (DCT): {mse_dct_avg:.6f}")
        print(f"Avg MSE to Original (Wavelet): {mse_wav_avg:.6f}")
        
        print(f"\nResults saved to: {save_dir}/")
        break


def visualize_wavelet_decomposition(save_dir='./comparison_results'):
    """
    Visualize the wavelet decomposition to understand what each subband captures.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda:0')
    
    # Load a sample image
    dataset = load_dataset_by_name('cifar10', './dataset', 1)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    for x, y in loader:
        x = x.to(device)
        
        # Perform DWT
        LL, coeffs, pads = multi_level_dwt(x, levels=2)
        
        # Create visualization
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        
        # Original image
        axes[0, 0].imshow(x[0].cpu().permute(1, 2, 0).clamp(0, 1))
        axes[0, 0].set_title('Original Image', fontsize=12)
        axes[0, 0].axis('off')
        
        # Level 1 subbands
        LH1, HL1, HH1 = coeffs[0]
        axes[0, 1].imshow(LL[0, 0].cpu().abs(), cmap='gray')
        axes[0, 1].set_title('LL (Approx)', fontsize=12)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(LH1[0, 0].cpu().abs(), cmap='gray')
        axes[0, 2].set_title('LH1 (Horiz Detail)', fontsize=12)
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(HL1[0, 0].cpu().abs(), cmap='gray')
        axes[0, 3].set_title('HL1 (Vert Detail)', fontsize=12)
        axes[0, 3].axis('off')
        
        axes[0, 4].imshow(HH1[0, 0].cpu().abs(), cmap='gray')
        axes[0, 4].set_title('HH1 (Diag Detail)', fontsize=12)
        axes[0, 4].axis('off')
        
        # Level 2 subbands
        LH2, HL2, HH2 = coeffs[1]
        axes[1, 0].text(0.5, 0.5, 'Level 2\nDecomposition', ha='center', va='center', fontsize=14)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(LL[0, 0].cpu().abs(), cmap='gray')
        axes[1, 1].set_title('LL2 (Coarsest)', fontsize=12)
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(LH2[0, 0].cpu().abs(), cmap='gray')
        axes[1, 2].set_title('LH2 (Horiz)', fontsize=12)
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(HL2[0, 0].cpu().abs(), cmap='gray')
        axes[1, 3].set_title('HL2 (Vert)', fontsize=12)
        axes[1, 3].axis('off')
        
        axes[1, 4].imshow(HH2[0, 0].cpu().abs(), cmap='gray')
        axes[1, 4].set_title('HH2 (Diag)', fontsize=12)
        axes[1, 4].axis('off')
        
        plt.suptitle('Haar Wavelet Decomposition (2 Levels)', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/wavelet_decomposition.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved wavelet decomposition visualization to {save_dir}/wavelet_decomposition.png")
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./comparison_results')
    args = parser.parse_args()
    
    print("="*60)
    print("DCT vs Wavelet FreqPure Visualization")
    print("="*60)
    
    # First visualize wavelet decomposition
    print("\n1. Visualizing wavelet decomposition...")
    visualize_wavelet_decomposition(args.save_dir)
    
    # Then create comparison
    print("\n2. Creating DCT vs Wavelet comparison...")
    visualize_comparison(args.num_samples, args.save_dir)
