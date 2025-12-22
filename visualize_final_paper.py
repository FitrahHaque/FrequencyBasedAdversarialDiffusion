"""
Final Paper Visualization Script for Wavelet-FreqPure.
Generates high-quality plots for the technical report using full dataset results.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# Full Dataset Results (Based on User Table)
configs = [
    'DFT Baseline',
    'Wavelet (L=1, δ=0.3)', 
    'Wavelet (L=2, δ=0.1)',
    'Wavelet (L=3, δ=0.3)',
    'Wavelet + Learnable (L=2, δ=0.2)'
]

nat_acc = [94.34, 92.77, 88.87, 58.01, 92.77]
adv_acc = [69.73, 80.27, 85.55, 51.37, 82.23]

# Ensure directory
os.makedirs('paper_figures', exist_ok=True)

# Set up the plot style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12

# 1. Bar Chart Comparison
fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(configs))
width = 0.35

# Color scheme
colors_nat = ['#2ECC71'] * len(configs)
colors_adv = ['#E74C3C'] * len(configs)

# Highlight the last one
colors_nat[-1] = '#9B59B6' # Purple for Nat
colors_adv[-1] = '#F1C40F' # Gold for Adv

rects1 = ax.bar(x - width/2, nat_acc, width, label='Natural Acc', color=colors_nat, alpha=0.9, edgecolor='black')
rects2 = ax.bar(x + width/2, adv_acc, width, label='Adversarial Acc', color=colors_adv, alpha=0.9, edgecolor='black')

# Labels
ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
ax.set_title('Wavelet-FreqPure Performance Breakdown', fontweight='bold', fontsize=16, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(configs, rotation=15, ha='right', fontsize=11)

# Add legend manually because 'color' arg was a list
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ECC71', edgecolor='black', label='Natural Acc'),
    Patch(facecolor='#E74C3C', edgecolor='black', label='Adversarial Acc'),
    Patch(facecolor='#9B59B6', edgecolor='black', label='Nat (Learnable)'),
    Patch(facecolor='#F1C40F', edgecolor='black', label='Adv (Learnable)')
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

# Add values on top
def autolabel(rects, is_adv=False):
    for rect in rects:
        height = rect.get_height()
        # Bold the best adversarial
        weight = 'bold' if is_adv and height == max(adv_acc) else 'normal'
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight=weight)

autolabel(rects1)
autolabel(rects2, is_adv=True)

# Add line for baseline adv
baseline = adv_acc[0]
ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7, label='Baseline Adv Acc')

plt.ylim(0, 110) # Little more headroom
plt.tight_layout() # Should handle the legend if bbox_extra_artists is used, but bbox_to_anchor implies manual space needed. 
# tight_layout might truncate bottom legend.
# Instead of tight_layout, strictly set margins or use bbox_inches='tight' in savefig.

plt.savefig('paper_figures/main_results_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Scatter Plot (Trade-off)
fig, ax = plt.subplots(figsize=(10, 8))

# Plot points
markers = ['o', '^', '*', 'X', 'D']
sizes = [200, 200, 400, 200, 350]
colors = ['#95A5A6', '#3498DB', '#E67E22', '#C0392B', '#1ABC9C'] # Gray, Blue, Orange, Red, Teal

for i, conf in enumerate(configs):
    # Highlight Learnable and Best L2
    lw = 2.5 if i >= 2 else 1.5
    ax.scatter(nat_acc[i], adv_acc[i], s=sizes[i], c=colors[i], marker=markers[i], 
               label=conf, edgecolors='black', linewidth=lw, zorder=5)

# Annotate points
# Annotate points
for i, txt in enumerate(configs):
    xytext = (0, 10)
    ha = 'center'
    
    if i == 0: # DFT
        xytext = (0, -20)
        
    if i == 1: # L=1 (92, 80) -> BELOW
        xytext = (0, -35)
        
    if i == 2: # L=2 (88, 85) -> LEFT/TOP
        xytext = (-40, 5)
        
    if i == 3: # L=3 (58, 51)
        xytext = (20, 10)
        
    if i == 4: # Learnable (92, 82) -> ABOVE
        xytext = (0, 25)
    
    ax.annotate(txt, (nat_acc[i], adv_acc[i]), xytext=xytext, 
                textcoords='offset points', ha=ha, fontsize=11, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

# Ideal corner
ax.scatter(100, 100, s=100, c='black', marker='x')
ax.text(99, 99, 'Ideal', ha='right', va='top')

# Labels
ax.set_xlabel('Natural Accuracy (%)', fontweight='bold', fontsize=12)
ax.set_ylabel('Adversarial Accuracy (%)', fontweight='bold', fontsize=12)
ax.set_title('Robustness vs. Accuracy Trade-off', fontweight='bold', fontsize=14, pad=15)
ax.set_xlim(50, 102)
ax.set_ylim(50, 102)
plt.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('paper_figures/tradeoff_scatter.png', dpi=300)
plt.close()

print("Figures saved to paper_figures/")
