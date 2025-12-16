import torch
import matplotlib.pyplot as plt
import os
from load_data import load_dataset_by_name
from wavelet_utils import multi_level_dwt

def visualize_decomposition(save_path='paper_figures/wavelet_concept.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Load 1 image
    dataset = load_dataset_by_name('cifar10', './dataset', 1)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    x, y = next(iter(loader))
    
    # Analyze
    LL_final, coeffs, _ = multi_level_dwt(x, levels=2)
    
    # Create Figure with GridSpec for better control
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 4, wspace=0.3, hspace=0.6)
    
    # --- 1. Original Image (Left, large) ---
    ax_orig = fig.add_subplot(gs[:, 0])
    ax_orig.imshow(x[0].permute(1, 2, 0).clamp(0, 1))
    ax_orig.set_title('Original Input\n(32x32)', fontsize=14, fontweight='bold', pad=15)
    ax_orig.axis('off')
    
    # --- 2. Level 1 (High Frequencies) (Middle Column) ---
    LH1, HL1, HH1 = coeffs[0]
    
    # Title for the column
    ax_l1_title = fig.add_subplot(gs[0, 1:3])
    ax_l1_title.axis('off')
    ax_l1_title.text(0.5, 0.5, "Level 1: High-Frequency Details\n(Where Adv Noise Hides)", 
                     ha='center', va='center', fontsize=12, fontweight='bold', color='#C0392B')

    ax_lh1 = fig.add_subplot(gs[1, 1])
    ax_lh1.imshow(LH1[0].mean(dim=0).abs(), cmap='Reds')
    ax_lh1.set_title('Horizontal (LH1)', fontsize=10)
    ax_lh1.axis('off')
    
    ax_hl1 = fig.add_subplot(gs[1, 2])
    ax_hl1.imshow(HL1[0].mean(dim=0).abs(), cmap='Greens')
    ax_hl1.set_title('Vertical (HL1)', fontsize=10)
    ax_hl1.axis('off')
    
    ax_hh1 = fig.add_subplot(gs[2, 1:3]) # Centered below
    # Constrain aspect ratio
    ax_hh1.imshow(HH1[0].mean(dim=0).abs(), cmap='Blues')
    ax_hh1.set_title('Diagonal (HH1)', fontsize=10)
    ax_hh1.axis('off')
    ax_hh1.set_aspect('equal') # Fix stretching
    
    # --- 3. Level 2 (Coarse Structure) (Right Column) ---
    LL2, (LH2, HL2, HH2) = LL_final, coeffs[1]
    
    ax_l2_title = fig.add_subplot(gs[0, 3])
    ax_l2_title.axis('off')
    ax_l2_title.text(0.5, 0.5, "Level 2: Structure\n(Preserved)", 
                     ha='center', va='center', fontsize=12, fontweight='bold', color='#27AE60')

    # LL2 - The most important part
    ax_ll2 = fig.add_subplot(gs[1, 3])
    ax_ll2.imshow(LL2[0].permute(1, 2, 0).clamp(0, 1))
    ax_ll2.set_title('Approximation (LL2)\n(Hard Replacement)', fontsize=10, fontweight='bold', 
                     bbox=dict(facecolor='#2ECC71', alpha=0.2, edgecolor='none', boxstyle='round,pad=0.5'))
    ax_ll2.axis('off')
    
    # Just show one detail for L2 to save space
    ax_details2 = fig.add_subplot(gs[2, 3])
    ax_details2.imshow(LH2[0].mean(dim=0).abs(), cmap='Purples')
    ax_details2.set_title('L2 Details (Soft Filter)', fontsize=10)
    ax_details2.axis('off')
    
    # Arrows and Logic annotations
    # (Optional, but matplotlib arrows can be messy. Text is safer)
    
    plt.suptitle('Wavelet-FreqPure: Multi-Scale Frequency Decomposition', fontsize=18, fontweight='bold', y=0.98)
    
    # Explanation
    plt.figtext(0.5, 0.02, 
                "Key Idea: We strictly preserve the 'Approximation' (Green) to maintain identity,\n"
                "while aggressively filtering the 'High-Frequency Details' (Red/Blue) where adversarial noise is concentrated.", 
                ha="center", fontsize=11, style='italic', 
                bbox={"facecolor":"#ECF0F1", "alpha":1.0, "edgecolor":"gray", "boxstyle":"round,pad=1"})

    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved clean visualization to {save_path}")

if __name__ == "__main__":
    visualize_decomposition()
