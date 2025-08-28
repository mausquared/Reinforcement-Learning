#!/usr/bin/env python3
"""Overlay action colors on the XY trajectory for episode_log_083435.jsonl
Saves: trajectory_actions_083435.pdf (300 dpi) and PNG.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LOG = Path('episode_log_083435.jsonl')
OUT_PDF = Path('trajectory_actions_083435.pdf')
OUT_PNG = Path('trajectory_actions_083435.png')

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

xs = []
ys = []
actions = []
steps = []
with LOG.open('r', encoding='utf-8') as fh:
    for line in fh:
        if not line.strip():
            continue
        obj = json.loads(line)
        a = obj.get('action')
        if isinstance(a, (list,tuple)):
            a = a[0] if len(a)>0 else None
        try:
            a = int(a)
        except Exception:
            a = None
        obs = obj.get('obs', {}) or {}
        agent = None
        if isinstance(obs, dict):
            agent = obs.get('agent')
        if agent and len(agent)>=2:
            x = float(agent[0]); y = float(agent[1])
        else:
            # fallback: use info altitude? skip
            continue
        xs.append(x); ys.append(y); actions.append(a); steps.append(obj.get('step'))

if not xs:
    raise SystemExit('No trajectory points extracted from log')

# map actions to integer indices and colors
unique_actions = sorted(set([a for a in actions if a is not None]))
num_actions = max(ACTION_MAP.keys())+1
cmap = plt.get_cmap('tab10', num_actions)
colors = [cmap(a) if (a is not None and 0<=a<num_actions) else (0.5,0.5,0.5) for a in actions]

# plot
fig, ax = plt.subplots(figsize=(8,8))
# plot faint trajectory line
ax.plot(xs, ys, color='0.8', linewidth=0.8, alpha=0.6, label='trajectory')
# scatter colored by action
sc = ax.scatter(xs, ys, c=actions, cmap=cmap, s=80, edgecolors='k', linewidths=0.4, vmin=0, vmax=num_actions-1)
# legend with action names and colors
legend_handles = []
for ai in range(num_actions):
    if ai in ACTION_MAP:
        legend_handles.append(Line2D([0],[0], marker='o', color='w', markerfacecolor=cmap(ai), markeredgecolor='k', markersize=8, label=f"{ai}: {ACTION_MAP[ai]}"))
ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.01,1))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Trajectory colored by action (episode 083435)')
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
fig.savefig(OUT_PDF, dpi=300)
fig.savefig(OUT_PNG, dpi=200)
plt.close(fig)
print(f"Saved: {OUT_PDF} and {OUT_PNG}")
