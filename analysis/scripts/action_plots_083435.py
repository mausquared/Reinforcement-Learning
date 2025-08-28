#!/usr/bin/env python3
"""Create annotated proportion bar chart and action timeline PDF for episode_log_083435.
Saves:
 - action_hist_083435_props.pdf (300 dpi)
 - action_timeline_083435.pdf (300 dpi)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

LOG = Path('episode_log_083435.jsonl')
OUT_HIST = Path('action_hist_083435_props.pdf')
OUT_TIMELINE = Path('action_timeline_083435.pdf')

ACTION_MAP = {
    0: 'forward',
    1: 'backward',
    2: 'left',
    3: 'right',
    4: 'up',
    5: 'down',
    6: 'hover'
}

if not LOG.exists():
    raise SystemExit(f"Log not found: {LOG}")

# load per-step log
df = pd.read_json(LOG, lines=True)
# ensure action is scalar int
def extract_action(a):
    if isinstance(a, list) or isinstance(a, tuple):
        return int(a[0]) if len(a)>0 else None
    try:
        return int(a)
    except Exception:
        return None

df['action_idx'] = df['action'].apply(extract_action)
actions = df['action_idx'].dropna().astype(int).values
steps = len(actions)

# counts and proportions
unique, counts = np.unique(actions, return_counts=True)
# ensure include all action indices 0..6
all_idx = np.arange(0, max(ACTION_MAP.keys())+1)
counts_full = np.zeros_like(all_idx)
for i,c in zip(unique, counts):
    counts_full[i] = c
proportions = counts_full / counts_full.sum()
labels = [ACTION_MAP.get(i, str(i)) for i in all_idx]

# --- Bar chart with proportions ---
fig, ax = plt.subplots(figsize=(10,5))
bars = ax.bar(labels, counts_full, color=plt.cm.tab10(np.linspace(0,1,len(labels))))
ax.set_ylabel('Frequency')
ax.set_title('Action frequency with proportions (episode_log_083435)')
ax.set_ylim(0, counts_full.max()*1.35)
# annotate with count and proportion
for i, bar in enumerate(bars):
    h = bar.get_height()
    pct = proportions[i]*100 if counts_full.sum()>0 else 0
    ax.annotate(f"{int(h)}\n{pct:.1f}%", xy=(bar.get_x()+bar.get_width()/2, h), xytext=(0,6), textcoords='offset points', ha='center', va='bottom', fontsize=9)
fig.tight_layout()
fig.savefig(OUT_HIST, dpi=300)
plt.close(fig)

# --- Timeline raster ---
# Create a 2D array for imshow: shape (num_actions, steps) with 1 where action==idx
num_actions = len(all_idx)
timeline = np.zeros((num_actions, steps), dtype=int)
for t, a in enumerate(actions):
    if 0 <= a < num_actions:
        timeline[a, t] = 1

# For display, create an integer map of action per step (action index array)
action_per_step = actions
# create a colormap with distinct colors for each action
cmap = plt.get_cmap('tab10', num_actions)
norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, num_actions+0.5, 1), ncolors=num_actions)

fig, ax = plt.subplots(figsize=(12,3))
# show as a colored row where color is action index over time
img = ax.imshow(action_per_step[np.newaxis, :], aspect='auto', cmap=cmap, norm=norm)
ax.set_yticks([])
ax.set_xlabel('Step')
ax.set_title('Action timeline (color = action index)')
# create custom legend mapping colors
from matplotlib.patches import Patch
legend_handles = [Patch(color=cmap(i), label=f"{i}: {ACTION_MAP.get(i)}") for i in range(num_actions)]
ax.legend(handles=legend_handles, bbox_to_anchor=(1.01, 1), loc='upper left')
fig.tight_layout()
fig.savefig(OUT_TIMELINE, dpi=300)
plt.close(fig)

print(f"Wrote: {OUT_HIST} and {OUT_TIMELINE}")
