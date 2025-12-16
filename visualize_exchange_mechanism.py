import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from load_data import load_dataset_by_name
from wavelet_utils import multi_level_dwt, multi_level_idwt

def wavelet_exchange_simulate(x_ref, x_est, levels=2, delta=0.1):
    # Forward DWT on both
    LL_ref, coeffs_ref, pads = multi_level_dwt(x_ref, levels=levels)
    LL_est, coeffs_est, _ = multi_level_dwt(x_est, levels=levels)
    
    # 1. LL Exchange (Hard Replacement)
    LL_new = LL_ref
    
    # 2. Detail Projection (Soft Constraint)
    coeffs_new = []
    dropped_details = [] # For visualization
    
    for i in range(len(coeffs_ref)):
        LH_r, HL_r, HH_r = coeffs_ref[i]
        LH_e, HL_e, HH_e = coeffs_est[i]
        
        # Calculate what would be dropped/modified
        LH_n = torch.clamp(LH_e, LH_r - delta, LH_r + delta)
        HL_n = torch.clamp(HL_e, HL_r - delta, HL_r + delta)
        HH_n = torch.clamp(HH_e, HH_r - delta, HH_r + delta)
        
        coeffs_new.append((LH_n, HL_n, HH_n))
        
        # Visualize the difference (what did we change in the estimate?)
        diff = (LH_e - LH_n).abs() + (HL_e - HL_n).abs() + (HH_e - HH_n).abs()
        dropped_details.append(diff)

    # Inverse DWT
    x_new = multi_level_idwt(LL_new, coeffs_new, pads)
    return x_new, LL_ref, dropped_details

def visualize_exchange(save_path='paper_figures/exchange_mechanism.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Load 1 image
    dataset = load_dataset_by_name('cifar10', './dataset', 1)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    x_clean, y = next(iter(loader))
    
    # Create Synthetic Inputs
    # 1. Adversarial (Reference): Clean + High Freq Noise
    noise_adv = torch.randn_like(x_clean) * 0.1
    # Make noise high-frequency (checkerboard)
    chk = torch.zeros_like(x_clean)
    chk[:, :, ::2, ::2] = 1.0
    noise_adv = noise_adv * chk 
    x_ref = (x_clean + noise_adv).clamp(0, 1)
    
    # 2. Diffusion Estimate (Estimate): Clean + General Gaussian Noise 
    # (Simulating a denoiser output that is mostly good but imperfect)
    x_est = (x_clean + torch.randn_like(x_clean) * 0.05).clamp(0, 1)

    # Perform Exchange
    x_purified, LL_used, dropped_details = wavelet_exchange_simulate(x_ref, x_est, levels=2, delta=0.1)

    # Visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, wspace=0.3, hspace=0.4)
    
    # --- Row 1: The Inputs ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(x_ref[0].permute(1, 2, 0))
    ax1.set_title('A. Reference (Adversarial)\n"Has correct content but noisy details"', fontsize=11, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 3])
    ax2.imshow(x_est[0].permute(1, 2, 0))
    ax2.set_title('B. Diffusion Estimate\n"Generated clean-ish image"', fontsize=11, fontweight='bold')
    ax2.axis('off')
    
    # Flow Arrows/Text
    ax_arrow = fig.add_subplot(gs[0, 1:3])
    ax_arrow.axis('off')
    ax_arrow.text(0.5, 0.5, "MIXING PROCESS\n----------------->", ha='center', va='center', fontsize=14, fontweight='bold')

    # --- Row 2: The Mechanism (DWT Domain) ---
    
    # Step 1: LL Exchange
    ax_ll = fig.add_subplot(gs[1, 1])
    ax_ll.imshow(LL_used[0].permute(1, 2, 0).clamp(0, 1))
    ax_ll.set_title('1. Keep Reference Structure (LL)\n(Ensures Identity)', fontsize=10, bbox=dict(facecolor='#2ECC71', alpha=0.3))
    ax_ll.axis('off')
    
    # Step 2: Detail Projection
    ax_details = fig.add_subplot(gs[1, 2])
    # Show Level 1 modification as heatmap
    ax_details.imshow(dropped_details[0][0].mean(dim=0), cmap='magma')
    ax_details.set_title('2. Filter Estimate Details\n(Clamp to Reference ± δ)', fontsize=10, bbox=dict(facecolor='#E74C3C', alpha=0.3))
    ax_details.axis('off')
    
    # --- Row 3: Output ---
    ax_out = fig.add_subplot(gs[2, 1:3])
    ax_out.imshow(x_purified[0].permute(1, 2, 0).clamp(0, 1))
    ax_out.set_title('C. Final Purified Image\n(Low Freq from A + Corrected High Freq from B)', fontsize=14, fontweight='bold', color='blue')
    ax_out.axis('off')
    
    # Difference Maps
    ax_diff1 = fig.add_subplot(gs[2, 0])
    ax_diff1.imshow((x_ref - x_purified)[0].mean(dim=0).abs(), cmap='hot')
    ax_diff1.set_title('Removed Adversarial Noise', fontsize=10)
    ax_diff1.axis('off')

    ax_diff2 = fig.add_subplot(gs[2, 3])
    ax_diff2.imshow((x_clean - x_purified)[0].mean(dim=0).abs(), cmap='viridis')
    ax_diff2.set_title('Final Error vs Ground Truth\n(Very Low)', fontsize=10)
    ax_diff2.axis('off')

    plt.suptitle('Wavelet-FreqPure: Frequency Exchange Mechanism', fontsize=18, fontweight='bold')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved exchange mechanism visualization to {save_path}")

if __name__ == "__main__":
    visualize_exchange()
