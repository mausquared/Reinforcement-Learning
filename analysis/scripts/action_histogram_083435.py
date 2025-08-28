#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG = Path('episode_log_083435.jsonl')
OUT_PNG = Path('action_hist_083435.png')
OUT_CSV = Path('action_hist_083435.csv')

# mapping from hummingbird_env (Discrete(7))
ACTION_MAP = {
    0: 'forward (north)',
    1: 'backward (south)',
    2: 'left (west)',
    3: 'right (east)',
    4: 'up',
    5: 'down',
    6: 'hover'
}

if not LOG.exists():
    print(f'Log not found: {LOG}')
    raise SystemExit(1)

counts = Counter()
with LOG.open('r', encoding='utf-8') as fh:
    for line in fh:
        line=line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        a = obj.get('action')
        # sometimes action may be list/array; handle
        if isinstance(a, list) or isinstance(a, tuple):
            if len(a)>0:
                a = a[0]
        try:
            a_int = int(a)
        except Exception:
            a_int = a
        counts[a_int] += 1

# build table with mapping
rows = []
for k in sorted(counts.keys()):
    name = ACTION_MAP.get(k, str(k))
    rows.append((k, name, counts[k]))

# print table
print('action_index, action_name, count')
for k,name,c in rows:
    print(f'{k}, {name}, {c}')

# save CSV
with OUT_CSV.open('w', encoding='utf-8') as fh:
    fh.write('action_index,action_name,count\n')
    for k,name,c in rows:
        fh.write(f'{k},{name},{c}\n')

# plot
if rows:
    labels = [r[1] for r in rows]
    counts_vals = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(8,4))
    bars = ax.bar(labels, counts_vals, color='C0')
    ax.set_ylabel('Frequency')
    ax.set_title('Action frequency (episode_log_083435)')
    ax.set_xticklabels(labels, rotation=30, ha='right')
    # annotate
    for bar in bars:
        h = bar.get_height()
        ax.annotate(str(h), xy=(bar.get_x()+bar.get_width()/2, h), xytext=(0,3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)
    print(f'Saved histogram PNG: {OUT_PNG}')
else:
    print('No actions found in log.')
