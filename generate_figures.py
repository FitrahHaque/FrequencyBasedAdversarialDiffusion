"""
Generate summary visualizations from existing comparison data.
No model execution required - uses saved images and results.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import os

def create_results_bar_chart(save_path='./comparison_results/accuracy_comparison.png'):
    """Create bar chart comparing DFT vs Wavelet accuracy."""
    
    # Results from experiments
    methods = ['DFT-FreqPure\n(Baseline)', 'Wavelet-FreqPure\n(Ours)']
    natural_acc = [94.34, 83.01]
    adv_acc = [69.73, 78.13]
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, natural_acc, width, label='Natural Accuracy', 
                   color='#4CAF50', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, adv_acc, width, label='Adversarial Accuracy', 
                   color='#F44336', edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars1, natural_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    for bar, val in zip(bars2, adv_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add improvement annotation
    ax.annotate('', xy=(1.175, adv_acc[1]), xytext=(1.175, adv_acc[0]),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax.text(1.35, (adv_acc[0] + adv_acc[1])/2, '+8.4%', fontsize=14, 
            fontweight='bold', color='blue', va='center')
    
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('DFT vs Wavelet FreqPure Comparison\n(CIFAR-10, 512 samples, PGD-200 Attack)', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    # Add box with key finding
    textstr = 'Key Finding:\nWavelet improves adversarial\nrobustness by +8.4%'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_method_diagram(save_path='./comparison_results/method_diagram.png'):
    """Create a diagram explaining the wavelet exchange method."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(7, 7.5, 'Wavelet-FreqPure Method', fontsize=18, fontweight='bold', 
            ha='center', va='center')
    
    # Input boxes
    rect1 = mpatches.FancyBboxPatch((0.5, 5.5), 2.5, 1.2, boxstyle='round,pad=0.1',
                                      facecolor='#FFCDD2', edgecolor='black', linewidth=2)
    ax.add_patch(rect1)
    ax.text(1.75, 6.1, 'Adversarial\nImage', ha='center', va='center', fontsize=11, fontweight='bold')
    
    rect2 = mpatches.FancyBboxPatch((0.5, 3.5), 2.5, 1.2, boxstyle='round,pad=0.1',
                                      facecolor='#C8E6C9', edgecolor='black', linewidth=2)
    ax.add_patch(rect2)
    ax.text(1.75, 4.1, 'Diffusion\nOutput', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # DWT boxes
    rect3 = mpatches.FancyBboxPatch((4, 5.5), 2, 1.2, boxstyle='round,pad=0.1',
                                      facecolor='#E1BEE7', edgecolor='black', linewidth=2)
    ax.add_patch(rect3)
    ax.text(5, 6.1, 'DWT', ha='center', va='center', fontsize=12, fontweight='bold')
    
    rect4 = mpatches.FancyBboxPatch((4, 3.5), 2, 1.2, boxstyle='round,pad=0.1',
                                      facecolor='#E1BEE7', edgecolor='black', linewidth=2)
    ax.add_patch(rect4)
    ax.text(5, 4.1, 'DWT', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Arrows to DWT
    ax.annotate('', xy=(4, 6.1), xytext=(3, 6.1),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(4, 4.1), xytext=(3, 4.1),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Subband boxes
    # LL boxes
    rect5 = mpatches.FancyBboxPatch((7, 6), 1.5, 0.8, boxstyle='round,pad=0.05',
                                      facecolor='#FFEB3B', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect5)
    ax.text(7.75, 6.4, 'LL_adv', ha='center', va='center', fontsize=10)
    
    rect6 = mpatches.FancyBboxPatch((7, 3.5), 1.5, 0.8, boxstyle='round,pad=0.05',
                                      facecolor='#FFEB3B', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect6)
    ax.text(7.75, 3.9, 'LL_diff', ha='center', va='center', fontsize=10)
    
    # Detail boxes
    rect7 = mpatches.FancyBboxPatch((7, 5), 1.5, 0.8, boxstyle='round,pad=0.05',
                                      facecolor='#03A9F4', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect7)
    ax.text(7.75, 5.4, 'Details_adv', ha='center', va='center', fontsize=9)
    
    rect8 = mpatches.FancyBboxPatch((7, 2.5), 1.5, 0.8, boxstyle='round,pad=0.05',
                                      facecolor='#03A9F4', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect8)
    ax.text(7.75, 2.9, 'Details_diff', ha='center', va='center', fontsize=9)
    
    # Arrows from DWT
    ax.annotate('', xy=(7, 6.4), xytext=(6, 6.1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(7, 5.4), xytext=(6, 6.1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(7, 3.9), xytext=(6, 4.1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(7, 2.9), xytext=(6, 4.1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Exchange operations
    rect9 = mpatches.FancyBboxPatch((9.5, 5.5), 2, 1.5, boxstyle='round,pad=0.1',
                                      facecolor='#FFF9C4', edgecolor='#FF9800', linewidth=3)
    ax.add_patch(rect9)
    ax.text(10.5, 6.25, 'LL Replace', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(10.5, 5.75, 'LL_new = LL_adv', ha='center', va='center', fontsize=9, style='italic')
    
    rect10 = mpatches.FancyBboxPatch((9.5, 3), 2, 1.5, boxstyle='round,pad=0.1',
                                       facecolor='#B3E5FC', edgecolor='#0288D1', linewidth=3)
    ax.add_patch(rect10)
    ax.text(10.5, 3.8, 'Detail Project', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(10.5, 3.3, 'clamp(diff, adv±δ)', ha='center', va='center', fontsize=9, style='italic')
    
    # Arrows to operations
    ax.annotate('', xy=(9.5, 6.4), xytext=(8.5, 6.4),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(9.5, 3.75), xytext=(8.5, 5.4),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, ls='--'))
    ax.annotate('', xy=(9.5, 3.75), xytext=(8.5, 2.9),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # IDWT box
    rect11 = mpatches.FancyBboxPatch((12, 4), 1.5, 1.5, boxstyle='round,pad=0.1',
                                       facecolor='#E1BEE7', edgecolor='black', linewidth=2)
    ax.add_patch(rect11)
    ax.text(12.75, 4.75, 'IDWT', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Arrows to IDWT
    ax.annotate('', xy=(12, 5), xytext=(11.5, 6.25),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(12, 4.5), xytext=(11.5, 3.75),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Output
    rect12 = mpatches.FancyBboxPatch((12, 1.5), 1.5, 1, boxstyle='round,pad=0.1',
                                       facecolor='#A5D6A7', edgecolor='#2E7D32', linewidth=3)
    ax.add_patch(rect12)
    ax.text(12.75, 2, 'Purified\nImage', ha='center', va='center', fontsize=11, fontweight='bold')
    
    ax.annotate('', xy=(12.75, 2.5), xytext=(12.75, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Legend
    ax.text(0.5, 1.5, 'Legend:', fontsize=11, fontweight='bold')
    
    leg1 = mpatches.FancyBboxPatch((0.5, 0.7), 0.4, 0.4, boxstyle='round,pad=0.02',
                                     facecolor='#FFEB3B', edgecolor='black')
    ax.add_patch(leg1)
    ax.text(1.1, 0.9, 'LL = Low-frequency (structure)', fontsize=10, va='center')
    
    leg2 = mpatches.FancyBboxPatch((5, 0.7), 0.4, 0.4, boxstyle='round,pad=0.02',
                                     facecolor='#03A9F4', edgecolor='black')
    ax.add_patch(leg2)
    ax.text(5.6, 0.9, 'Details = High-frequency (edges, textures)', fontsize=10, va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_trade_off_plot(save_path='./comparison_results/tradeoff_analysis.png'):
    """Create trade-off analysis visualization."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Trade-off scatter
    ax1 = axes[0]
    
    # Data points
    methods = ['DFT-FreqPure', 'Wavelet-FreqPure']
    nat_acc = [94.34, 83.01]
    adv_acc = [69.73, 78.13]
    colors = ['#2196F3', '#FF5722']
    markers = ['o', 's']
    
    for i, (m, n, a, c, mk) in enumerate(zip(methods, nat_acc, adv_acc, colors, markers)):
        ax1.scatter(n, a, s=300, c=c, marker=mk, edgecolor='black', linewidth=2, 
                   label=m, zorder=5)
        
    # Add arrow showing the trade-off direction
    ax1.annotate('', xy=(83.01, 78.13), xytext=(94.34, 69.73),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2, ls='--'))
    ax1.text(89, 74.5, 'Trade-off\nDirection', fontsize=10, ha='center', style='italic', color='gray')
    
    ax1.set_xlabel('Natural Accuracy (%)', fontsize=12)
    ax1.set_ylabel('Adversarial Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy Trade-off Analysis', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(75, 100)
    ax1.set_ylim(65, 85)
    
    # Right: Improvement breakdown
    ax2 = axes[1]
    
    categories = ['Natural\nAccuracy', 'Adversarial\nAccuracy']
    dft_vals = [94.34, 69.73]
    wav_vals = [83.01, 78.13]
    changes = [wav_vals[0] - dft_vals[0], wav_vals[1] - dft_vals[1]]
    bar_colors = ['#F44336' if c < 0 else '#4CAF50' for c in changes]
    
    x = np.arange(len(categories))
    ax2.bar(x, changes, color=bar_colors, edgecolor='black', linewidth=2)
    
    for i, (xi, c) in enumerate(zip(x, changes)):
        offset = 0.5 if c > 0 else -0.5
        ax2.text(xi, c + offset, f'{c:+.1f}%', ha='center', va='bottom' if c > 0 else 'top',
                fontsize=14, fontweight='bold')
    
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=12)
    ax2.set_ylabel('Change in Accuracy (%)', fontsize=12)
    ax2.set_title('Wavelet vs DFT: Accuracy Changes', fontsize=14, fontweight='bold')
    ax2.set_ylim(-15, 12)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add annotation
    ax2.text(0.5, 8, '↑ Robustness\nImproved', fontsize=10, ha='center', color='#2E7D32', fontweight='bold')
    ax2.text(0, -12, '↓ Natural Acc\nDecreased', fontsize=10, ha='center', color='#C62828', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_combined_summary(save_path='./comparison_results/full_summary.png'):
    """Create a comprehensive summary figure."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Title
    fig.suptitle('Wavelet-FreqPure: Multi-Scale Adversarial Purification', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1.2], hspace=0.3, wspace=0.25)
    
    # --- Top Left: Method Overview ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    method_text = """
    WAVELET-FREQPURE METHOD
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    1. Decompose images using 2D Haar Wavelet Transform (DWT)
       • LL band = Low-frequency structure (shapes, colors)
       • LH, HL, HH = High-frequency details (edges, textures)
    
    2. Exchange LL band: Keep adversarial's LL
       • Why? LL is less affected by adversarial perturbation
       • Preserves original image structure
    
    3. Project detail bands: Constrain within δ of original
       • Allows diffusion model to clean noise
       • But keeps details close to original (δ = 0.3)
    
    4. Reconstruct using Inverse DWT (IDWT)
    """
    ax1.text(0.05, 0.95, method_text, fontsize=11, family='monospace',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))
    ax1.set_title('Method Overview', fontsize=14, fontweight='bold', pad=10)
    
    # --- Top Right: Results Comparison ---
    ax2 = fig.add_subplot(gs[0, 1])
    
    methods = ['DFT\n(Baseline)', 'Wavelet\n(Ours)']
    natural_acc = [94.34, 83.01]
    adv_acc = [69.73, 78.13]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, natural_acc, width, label='Natural Acc', 
                   color='#4CAF50', edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, adv_acc, width, label='Adversarial Acc', 
                   color='#F44336', edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars1, natural_acc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, adv_acc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy Comparison (512 CIFAR-10 samples)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=12)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3)
    
    # --- Middle: Key Findings ---
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    findings = [
        ("✓ +8.4% Adversarial Robustness", "#4CAF50", 
         "Wavelet decomposition better separates adversarial noise from image structure"),
        ("⚠ -11.3% Natural Accuracy", "#FF9800",
         "Trade-off: Aggressive LL replacement affects clean image fidelity"),
        ("🔬 Multi-scale Processing", "#2196F3",
         "2-level decomposition provides both coarse structure and fine detail control"),
    ]
    
    for i, (title, color, desc) in enumerate(findings):
        y_pos = 0.75 - i * 0.3
        ax3.text(0.05, y_pos, title, fontsize=14, fontweight='bold', color=color,
                transform=ax3.transAxes)
        ax3.text(0.05, y_pos - 0.08, desc, fontsize=11, color='#333333',
                transform=ax3.transAxes)
    
    ax3.set_title('Key Findings', fontsize=14, fontweight='bold', pad=10)
    
    # --- Bottom: Files Modified ---
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')
    
    files_text = """
    FILES MODIFIED
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    📄 wavelet_utils.py (NEW)
       └─ Haar DWT/IDWT implementation
    
    📄 purification.py (MODIFIED)
       └─ Added wavelet_exchange() method
       └─ New params: transform_type, wavelet_levels
    
    📄 ddp_test.py (MODIFIED)
       └─ Added --transform_type argument
       └─ Added --wavelet_levels argument
    
    📄 visualize_comparison.py (NEW)
       └─ Comparison visualization tools
    """
    ax4.text(0.05, 0.95, files_text, fontsize=10, family='monospace',
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#FFF3E0', alpha=0.8))
    ax4.set_title('Implementation Details', fontsize=14, fontweight='bold', pad=10)
    
    # --- Bottom Right: Usage ---
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    usage_text = """
    HOW TO RUN
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    # Run with Wavelet-FreqPure:
    torchrun --nproc_per_node=2 ddp_test.py \\
        --transform_type wavelet \\
        --wavelet_levels 2 \\
        --num_samples 50
    
    # Run with DFT-FreqPure (baseline):
    torchrun --nproc_per_node=2 ddp_test.py \\
        --transform_type dft \\
        --num_samples 50
    
    # Parameters:
    --wavelet_levels: 1, 2, or 3 (default: 2)
    --delta: Projection threshold (default: 0.3)
    """
    ax5.text(0.05, 0.95, usage_text, fontsize=10, family='monospace',
             transform=ax5.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.8))
    ax5.set_title('Usage Guide', fontsize=14, fontweight='bold', pad=10)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    # Create output directory
    os.makedirs('./comparison_results', exist_ok=True)
    
    print("Generating visualizations from existing data...")
    print("=" * 50)
    
    # Generate all visualizations
    create_results_bar_chart()
    create_method_diagram()
    create_trade_off_plot()
    create_combined_summary()
    
    print("=" * 50)
    print("All visualizations generated!")
    print("\nOutput files:")
    print("  - comparison_results/accuracy_comparison.png")
    print("  - comparison_results/method_diagram.png")
    print("  - comparison_results/tradeoff_analysis.png")
    print("  - comparison_results/full_summary.png")
