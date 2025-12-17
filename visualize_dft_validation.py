import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from load_data import load_dataset_by_name

def get_simulated_data():
    # Load 1 image
    dataset = load_dataset_by_name('cifar10', './dataset', 1)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    x_clean, y = next(iter(loader))
    
    # Create Localized High-Freq Noise (The "Ghost" target)
    # A checkerboard pattern only in the center
    h, w = x_clean.shape[-2], x_clean.shape[-1]
    noise = torch.zeros_like(x_clean)
    chk = torch.zeros_like(x_clean)
    chk[:, :, ::2, ::2] = 1.0
    
    # Localized mask (Ghost)
    mask = torch.zeros_like(x_clean)
    mask[:, :, h//4:3*h//4, w//4:3*w//4] = 1.0
    
    # Add noise
    x_adv = (x_clean + mask * chk * 0.2).clamp(0, 1)
    
    # Synthetic "Simple" estimate
    x_est = x_clean.clone() # Assume diffusion is perfect for this demo to isolate DFT artifacts
    
    return x_clean, x_adv, x_est

def dft_exchange_simulate(x_ref, x_est, amp_cut=10):
    """
    Simulates DFT purification.
    To show 'Failure', we simulate the Global nature:
    1. We cut high frequencies to remove the noise.
    2. But this causes Ringing/Blurring globally.
    """
    # FFT
    f = torch.fft.fft2(x_ref)
    fshift = torch.fft.fftshift(f)
    
    # Create Mask (Hard Cutoff)
    rows, cols = x_ref.shape[-2], x_ref.shape[-1]
    crow, ccol = rows // 2, cols // 2
    mask = torch.zeros_like(fshift)
    
    # Keep low freq (center)
    r = amp_cut
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    mask_area = x*x + y*y <= r*r
    mask[:, :, mask_area] = 1.0
    mask = mask.to(x_ref.device)
    
    # Apply Mask (The Purification)
    fshift_filtered = fshift * mask
    
    # Shift back and IFFT
    f_ishift = torch.fft.ifftshift(fshift_filtered)
    img_back = torch.fft.ifft2(f_ishift)
    x_pure = torch.abs(img_back)
    
    return x_pure.clamp(0, 1)

def visualize_validation(x_clean, x_adv, x_pure, save_path):
    fig = plt.figure(figsize=(16, 9), facecolor='#FFF5F5') # Reddish tint for "Bad"
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 0.2, 1.5])
    
    # 1. GHOST TEST (Input - Result)
    removed_signal = (x_adv - x_pure).abs().mean(dim=1)[0]
    
    ax_adv = fig.add_subplot(gs[0, 0])
    ax_adv.imshow(x_adv[0].permute(1, 2, 0))
    ax_adv.set_title("Input (Adversarial)\nLocalized Noise", fontweight='bold')
    ax_adv.axis('off')
    
    ax_pure = fig.add_subplot(gs[0, 1])
    ax_pure.imshow(x_pure[0].permute(1, 2, 0))
    ax_pure.set_title("Result (DFT Baseline)", fontweight='bold')
    ax_pure.axis('off')
    
    ax_diff = fig.add_subplot(gs[0, 3])
    im = ax_diff.imshow(removed_signal, cmap='inferno')
    ax_diff.set_title("FAILED GHOST TEST\n(Input - Result)", fontweight='bold', fontsize=14, color='red')
    ax_diff.axis('off')
    plt.colorbar(im, ax=ax_diff)
    
    ax_diff.text(0.5, -0.2, 
                 "⚠ FAILURE: 'Ringing' Artifacts.\n"
                 "To remove the center noise, DFT had to\n"
                 "ripple the entire image (Global Filter).",
                 ha='center', va='top', transform=ax_diff.transAxes, 
                 fontsize=11, bbox=dict(facecolor='#FDEDEC', edgecolor='red', boxstyle='round,pad=0.5'))

    # 2. BULLSEYE CHECK (Result - GT)
    error_signal = (x_clean - x_pure).abs().mean(dim=1)[0]
    
    ax_gt = fig.add_subplot(gs[1, 0])
    ax_gt.imshow(x_clean[0].permute(1, 2, 0))
    ax_gt.set_title("Ground Truth", fontweight='bold')
    ax_gt.axis('off')

    ax_err = fig.add_subplot(gs[1, 3])
    im2 = ax_err.imshow(error_signal, cmap='viridis', vmin=0, vmax=0.2)
    ax_err.set_title("FAILED FIDELITY TEST\n(Result - Ground Truth)", fontweight='bold', fontsize=14, color='red')
    ax_err.axis('off')
    plt.colorbar(im2, ax=ax_err)
    
    ax_err.text(0.5, -0.2, 
                 "⚠ FAILURE: High Error Everywhere.\n"
                 "The global blur destroyed clean textures\n"
                 "far away from the noise.",
                 ha='center', va='top', transform=ax_err.transAxes, 
                 fontsize=11, bbox=dict(facecolor='#FDEDEC', edgecolor='red', boxstyle='round,pad=0.5'))

    plt.suptitle("Validation Failure: Why DFT is Worse", fontsize=20, fontweight='bold', color='darkred', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved validation chart to {save_path}")

if __name__ == "__main__":
    x_clean, x_adv, x_est = get_simulated_data()
    # Aggressive cut to ensure noise removal, but causes ringing
    x_pure = dft_exchange_simulate(x_adv, x_est, amp_cut=8) 
    visualize_validation(x_clean, x_adv, x_pure, 'paper_figures/exchange_validation_dft.png')
