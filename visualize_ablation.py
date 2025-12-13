"""
Ablation study visualization for Wavelet-FreqPure.
"""
import matplotlib.pyplot as plt
import numpy as np

# Ablation study results
results = {
    'DCT Baseline': {'levels': '-', 'delta': 0.3, 'nat': 94.34, 'adv': 69.73, 'samples': 512},
    'Level1_d0.3': {'levels': 1, 'delta': 0.3, 'nat': 98, 'adv': 82, 'samples': 50},
    'Level2_d0.1': {'levels': 2, 'delta': 0.1, 'nat': 88, 'adv': 90, 'samples': 50},
    'Level2_d0.2': {'levels': 2, 'delta': 0.2, 'nat': 84, 'adv': 72, 'samples': 50},
    'Level2_d0.3': {'levels': 2, 'delta': 0.3, 'nat': 80, 'adv': 72, 'samples': 50},
    'Level2_d0.5': {'levels': 2, 'delta': 0.5, 'nat': 84, 'adv': 72, 'samples': 50},
    'Level3_d0.3': {'levels': 3, 'delta': 0.3, 'nat': 60, 'adv': 52, 'samples': 50},
}

# Create figure
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Wavelet-FreqPure Ablation Study', fontsize=18, fontweight='bold')

# 1. Bar chart comparing all configurations
ax1 = fig.add_subplot(2, 2, 1)
names = list(results.keys())
nat_acc = [results[k]['nat'] for k in names]
adv_acc = [results[k]['adv'] for k in names]

x = np.arange(len(names))
width = 0.35

bars1 = ax1.bar(x - width/2, nat_acc, width, label='Natural Acc', color='#4CAF50', edgecolor='black')
bars2 = ax1.bar(x + width/2, adv_acc, width, label='Adversarial Acc', color='#F44336', edgecolor='black')

ax1.set_xlabel('Configuration')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('All Configurations Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=8)
ax1.legend()
ax1.set_ylim(0, 105)
ax1.grid(axis='y', alpha=0.3)

# Highlight best adversarial
best_adv_idx = adv_acc.index(max(adv_acc))
bars2[best_adv_idx].set_edgecolor('gold')
bars2[best_adv_idx].set_linewidth(3)

# 2. Delta effect (Level 2)
ax2 = fig.add_subplot(2, 2, 2)
deltas = [0.1, 0.2, 0.3, 0.5]
delta_nat = [88, 84, 80, 84]
delta_adv = [90, 72, 72, 72]

ax2.plot(deltas, delta_nat, 'o-', color='#4CAF50', linewidth=2, markersize=10, label='Natural Acc')
ax2.plot(deltas, delta_adv, 's-', color='#F44336', linewidth=2, markersize=10, label='Adversarial Acc')

# Mark best point
best_delta_idx = delta_adv.index(max(delta_adv))
ax2.scatter([deltas[best_delta_idx]], [delta_adv[best_delta_idx]], s=200, c='gold', 
           marker='*', zorder=5, edgecolor='black', linewidth=2)
ax2.annotate('Best', (deltas[best_delta_idx], delta_adv[best_delta_idx]+3), ha='center', fontsize=10)

ax2.set_xlabel('Delta (δ)')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Effect of Delta (at Level 2)')
ax2.legend()
ax2.set_xlim(0, 0.6)
ax2.set_ylim(50, 100)
ax2.grid(True, alpha=0.3)

# 3. Level effect (δ=0.3)
ax3 = fig.add_subplot(2, 2, 3)
levels = [1, 2, 3]
level_nat = [98, 80, 60]
level_adv = [82, 72, 52]

ax3.plot(levels, level_nat, 'o-', color='#4CAF50', linewidth=2, markersize=10, label='Natural Acc')
ax3.plot(levels, level_adv, 's-', color='#F44336', linewidth=2, markersize=10, label='Adversarial Acc')

ax3.set_xlabel('Wavelet Levels')
ax3.set_ylabel('Accuracy (%)')
ax3.set_title('Effect of Decomposition Levels (at δ=0.3)')
ax3.legend()
ax3.set_xticks([1, 2, 3])
ax3.set_ylim(40, 105)
ax3.grid(True, alpha=0.3)

# Add LL size annotations
for i, lv in enumerate(levels):
    ll_size = 32 // (2**lv)
    ax3.annotate(f'LL: {ll_size}×{ll_size}', (lv, level_nat[i]-5), ha='center', fontsize=9, color='gray')

# 4. Trade-off scatter plot
ax4 = fig.add_subplot(2, 2, 4)

for name, data in results.items():
    if name == 'DCT Baseline':
        marker = 'D'
        color = '#2196F3'
        size = 200
    elif name == 'Level2_d0.1':
        marker = '*'
        color = 'gold'
        size = 300
    else:
        marker = 'o'
        color = '#FF5722'
        size = 100
    
    ax4.scatter(data['nat'], data['adv'], s=size, c=color, marker=marker, 
               edgecolor='black', linewidth=1.5, label=name, zorder=5)

ax4.set_xlabel('Natural Accuracy (%)')
ax4.set_ylabel('Adversarial Accuracy (%)')
ax4.set_title('Accuracy Trade-off (Nat vs Adv)')
ax4.set_xlim(55, 100)
ax4.set_ylim(45, 95)
ax4.grid(True, alpha=0.3)

# Add legend
ax4.legend(loc='lower left', fontsize=7, ncol=2)

# Add annotation for best config
ax4.annotate('Best Config:\nLevel 2, δ=0.1', (88, 90), xytext=(75, 87),
            arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('./ablation_results/ablation_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ./ablation_results/ablation_visualization.png")

# Create summary table as image
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')

table_data = [
    ['Configuration', 'Levels', 'Delta', 'Natural Acc', 'Adv Acc', 'Notes'],
    ['DCT Baseline', '-', '0.3', '94.34%', '69.73%', 'Original (512 samples)'],
    ['Wavelet L1', '1', '0.3', '98%', '82%', 'Best natural'],
    ['Wavelet L2', '2', '0.1', '88%', '90%', '★ BEST ROBUST'],
    ['Wavelet L2', '2', '0.2', '84%', '72%', ''],
    ['Wavelet L2', '2', '0.3', '80%', '72%', 'Default'],
    ['Wavelet L2', '2', '0.5', '84%', '72%', ''],
    ['Wavelet L3', '3', '0.3', '60%', '52%', 'Too aggressive'],
]

colors = [['#E3F2FD']*6] + [['white']*6]*7
colors[3] = ['#C8E6C9']*6  # Highlight best

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.2, 0.1, 0.1, 0.15, 0.15, 0.25],
                cellColours=colors)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)

# Bold header
for j in range(6):
    table[(0, j)].set_text_props(fontweight='bold')

ax.set_title('Wavelet-FreqPure Ablation Study Results\n(CIFAR-10, PGD-200 Attack)', 
            fontsize=14, fontweight='bold', pad=20)

plt.savefig('./ablation_results/ablation_table.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: ./ablation_results/ablation_table.png")

print("\n" + "="*60)
print("ABLATION STUDY COMPLETE")
print("="*60)
print("\nResults saved to: ./ablation_results/")
print("\nKey Finding: Level 2, δ=0.1 achieves 90% adversarial accuracy!")
