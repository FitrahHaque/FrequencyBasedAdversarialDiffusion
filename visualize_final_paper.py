"""
Final Paper Visualization Script for Wavelet-FreqPure.
Generates high-quality plots for the technical report using full dataset results.
"""
import matplotlib.pyplot as plt
import numpy as np

# Full Dataset Results (512 samples)
configs = [
    'DCT Baseline',
    'Wavelet (L=2, δ=0.3)',
    'Wavelet (L=1, δ=0.3)', 
    'Wavelet (L=2, δ=0.1)',
    'Wavelet (L=3, δ=0.3)'
]

nat_acc = [94.34, 83.01, 92.77, 88.87, 58.01]
adv_acc = [69.73, 78.13, 80.27, 85.55, 51.37]

# Set up the plot style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12

# 1. Bar Chart Comparison
fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(configs))
width = 0.35

# Color scheme
color_nat = '#2ECC71'  # Green
color_adv = '#E74C3C'  # Red
color_best = '#F1C40F' # Gold highlight

# Bars
rects1 = ax.bar(x - width/2, nat_acc, width, label='Natural Accuracy', color=color_nat, edgecolor='black', alpha=0.8)
rects2 = ax.bar(x + width/2, adv_acc, width, label='Adversarial Accuracy', color=color_adv, edgecolor='black', alpha=0.8)

# Highlight best robustness
rects2[3].set_linewidth(3)
rects2[3].set_edgecolor('gold')
rects2[3].set_hatch('//')

# Labels and Text
ax.set_ylabel('Accuracy (%)', fontweight='bold')
ax.set_title('Defense Performance Comparison (CIFAR-10, PGD-200, N=512)', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontweight='bold')
ax.set_ylim(0, 105)
ax.legend(loc='upper right')

# Add values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# Add improvement annotation
improvement = adv_acc[3] - adv_acc[0]
ax.annotate(f'+{improvement:.1f}% Improvement', 
            xy=(x[3] + width/2, adv_acc[3]), 
            xytext=(x[3] + width/2 + 0.5, adv_acc[3] + 10),
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=12, fontweight='bold', color='black', ha='center')

plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('paper_figures/main_results_comparison.png', dpi=300)
plt.close()

# 2. Scatter Plot Trade-off
fig, ax = plt.subplots(figsize=(10, 8))

# Plot points
# Plot points
markers = ['o', 's', '^', '*', 'X']
sizes = [200, 200, 200, 400, 200]
colors = ['#95A5A6', '#3498DB', '#9B59B6', '#E67E22', '#C0392B'] # Gray, Blue, Purple, Orange, Red

for i, conf in enumerate(configs):
    ax.scatter(nat_acc[i], adv_acc[i], s=sizes[i], c=colors[i], marker=markers[i], 
               label=conf, edgecolors='black', linewidth=1.5, zorder=5)

# Annotate points
for i, txt in enumerate(configs):
    offset_y = 1.5 if i != 2 else -2.5 # Adjust label position for L=1 to avoid overlap
    ax.annotate(txt, (nat_acc[i], adv_acc[i]), xytext=(0, 10 * offset_y), 
                textcoords='offset points', ha='center', fontsize=11, fontweight='bold')

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

# Draw arrows/lines? No, simple scatter is cleaner.

plt.tight_layout()
plt.savefig('paper_figures/tradeoff_scatter.png', dpi=300)
plt.close()

print("Figures saved to paper_figures/")
