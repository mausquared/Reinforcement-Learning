#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from math import log2

ACTION_CSV = Path('action_hist_083435.csv')
EPISODE_CSV = Path('episode_log_083435_table.csv')
OUT_PDF = Path('action_hist_083435.pdf')
OUT_STATS = Path('action_stats_083435.txt')

# read action counts
act_df = pd.read_csv(ACTION_CSV)
# ensure sorted by action_index
act_df = act_df.sort_values('action_index')
labels = act_df['action_name'].tolist()
counts = act_df['count'].astype(int).tolist()

# plot with larger y-axis
fig, ax = plt.subplots(figsize=(9,5))
bars = ax.bar(labels, counts, color='C0')
ax.set_ylabel('Frequency')
ax.set_title('Action frequency (episode_log_083435)')
ax.set_xticklabels(labels, rotation=30, ha='right')
# increase y axis a bit
maxc = max(counts) if counts else 1
ax.set_ylim(0, maxc * 1.25)
# annotate
for bar in bars:
    h = bar.get_height()
    ax.annotate(str(h), xy=(bar.get_x()+bar.get_width()/2, h), xytext=(0,3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
fig.tight_layout()
fig.savefig(OUT_PDF, dpi=300)
plt.close(fig)

# compute stats from action counts and episode csv
total_steps = sum(counts)
hover_count = int(act_df.loc[act_df['action_index']==6, 'count'].values[0]) if 6 in act_df['action_index'].values else 0
movement_steps = total_steps - hover_count
proportions = [c/total_steps for c in counts]

# entropy
entropy = -sum((p*log2(p) if p>0 else 0) for p in proportions)

# load episode CSV for energy/nectar etc
if EPISODE_CSV.exists():
    ep = pd.read_csv(EPISODE_CSV)
    mean_energy = ep['agent_energy'].mean()
    min_energy = ep['agent_energy'].min()
    final_energy = ep['agent_energy'].iloc[-1]
    mean_nearest = ep['nearest_flower_distance'].mean() if 'nearest_flower_distance' in ep.columns else np.nan
    final_nectar = ep['total_nectar_collected'].iloc[-1] if 'total_nectar_collected' in ep.columns else np.nan
    # count feed events as positive increases in total_nectar_collected
    if 'total_nectar_collected' in ep.columns:
        nectar = ep['total_nectar_collected'].fillna(method='ffill').values
        diffs = np.diff(nectar)
        feed_events = int(np.count_nonzero(diffs>0))
        nectar_gained = float(max(0.0, nectar[-1] - nectar[0]))
    else:
        feed_events = 0
        nectar_gained = 0.0
else:
    mean_energy = min_energy = final_energy = mean_nearest = final_nectar = np.nan
    feed_events = 0
    nectar_gained = 0.0

# prepare text summary
lines = []
lines.append(f"Action frequency summary (total steps = {total_steps})")
lines.append('')
for idx, row in act_df.iterrows():
    ai = int(row['action_index'])
    name = row['action_name']
    cnt = int(row['count'])
    pct = 100.0 * cnt / total_steps if total_steps>0 else 0.0
    lines.append(f"{ai:>2d}  {name:25s} {cnt:6d}  ({pct:5.1f}%)")
lines.append('')
lines.append(f"Movement steps (0-5): {movement_steps} ({100.0*movement_steps/total_steps:5.1f}%)")
lines.append(f"Hover steps (6):       {hover_count} ({100.0*hover_count/total_steps:5.1f}%)")
lines.append('')
lines.append(f"Action entropy: {entropy:.3f} bits")
lines.append('')
lines.append('Energy & nectar statistics (from episode log):')
lines.append(f" mean_energy = {mean_energy:.2f}")
lines.append(f" min_energy  = {min_energy:.2f}")
lines.append(f" final_energy = {final_energy:.2f}")
lines.append(f" mean_nearest_flower_distance = {mean_nearest:.3f}")
lines.append(f" final_nectar_collected = {final_nectar}")
lines.append(f" feed_events (positive nectar increases) = {feed_events}")
lines.append(f" total_nectar_gained = {nectar_gained}")

# write stats
OUT_STATS.write_text('\n'.join(lines), encoding='utf-8')
print(OUT_STATS)
print('\n'.join(lines))
