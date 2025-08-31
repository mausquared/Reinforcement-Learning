"""Plot comparison of mean survival rate and mean nectar collected per flower-count scenario.
Data is taken from the user's parsed results (attachment).
Saves PNG and PDF to analysis/outputs/ at 300 DPI.

Usage:
    python analysis/scripts/plot_detailed_eval_comparison.py
"""
import os
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
OUT_DIR = os.path.join(ROOT, 'analysis', 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, 'detailed_evaluation_comparison.png')
OUT_PDF = os.path.join(OUT_DIR, 'detailed_evaluation_comparison.pdf')

# Data from attachment
flower_counts = [2, 4, 6, 8, 10]
mean_survival = [99.70, 33.50, 28.20, 43.50, 32.60]  # percent
mean_nectar = [45.32, 45.10, 53.08, 71.13, 65.92]

fig, ax1 = plt.subplots(figsize=(8, 5))

bar_width = 0.6
bars = ax1.bar([str(x) for x in flower_counts], mean_survival, color='#4c72b0', width=bar_width, label='Mean Survival Rate (%)')
ax1.set_ylabel('Mean Survival Rate (%)', color='#4c72b0')
ax1.set_ylim(0, 110)
ax1.set_xlabel('Number of Flowers')
ax1.tick_params(axis='y', labelcolor='#4c72b0')

# annotate survival bars
for b in bars:
    h = b.get_height()
    ax1.annotate(f'{h:.1f}%', xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 6), textcoords='offset points', ha='center', va='bottom', fontsize=9)

# secondary axis for mean nectar
ax2 = ax1.twinx()
ax2.plot([str(x) for x in flower_counts], mean_nectar, color='#dd8452', marker='o', linewidth=2, label='Mean Nectar Collected')
ax2.set_ylabel('Mean Nectar Collected', color='#dd8452')
ax2.tick_params(axis='y', labelcolor='#dd8452')

# annotate nectar points
for x, y in zip([str(x) for x in flower_counts], mean_nectar):
    ax2.annotate(f'{y:.2f}', xy=(x, y), xytext=(0, -12), textcoords='offset points', ha='center', va='top', fontsize=9, color='#444')

# combined legend
lines_labels = [ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels()]
lines = [lines_labels[0][0][0], lines_labels[1][0][0]]
labels = [lines_labels[0][1][0], lines_labels[1][1][0]]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2)

plt.title('Comparison: Mean Survival Rate and Mean Nectar Collected')
plt.tight_layout()

# save high-res PNG and PDF
plt.savefig(OUT_PNG, dpi=300)
plt.savefig(OUT_PDF, dpi=300)
print(f'Wrote plots: {OUT_PNG} and {OUT_PDF}')

# show when run interactively
plt.show()
