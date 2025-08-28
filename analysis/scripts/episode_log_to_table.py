#!/usr/bin/env python3
"""Convert episode JSONL log to a flattened pandas table and save CSV.
Usage:
    python episode_log_to_table.py [in.jsonl] [out.csv]
"""
import sys
import json
from pathlib import Path

in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('episode_log_083435.jsonl')
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(in_path.stem + '_table.csv')

try:
    import pandas as pd
except Exception:
    print('pandas is required to run this script. Install with: pip install pandas')
    raise

records = []
with in_path.open('r', encoding='utf-8') as fh:
    for line in fh:
        if not line.strip():
            continue
        j = json.loads(line)
        rec = {}
        rec['step'] = j.get('step')
        rec['action'] = j.get('action')
        rec['reward'] = j.get('reward')
        rec['done'] = j.get('done')
        # obs flatten
        obs = j.get('obs', {})
        agent = obs.get('agent') if isinstance(obs, dict) else None
        if agent and len(agent) >= 4:
            rec['agent_x'] = agent[0]
            rec['agent_y'] = agent[1]
            rec['agent_z'] = agent[2]
            rec['agent_energy'] = agent[3]
        else:
            rec['agent_x'] = rec['agent_y'] = rec['agent_z'] = rec['agent_energy'] = None
        # optionally include nearest flower position info if present in obs.flowers
        # info flatten
        info = j.get('info', {}) or {}
        rec['total_nectar_collected'] = info.get('total_nectar_collected')
        rec['energy'] = info.get('energy')
        rec['steps_so_far'] = info.get('steps')
        rec['last_flower_visited'] = info.get('last_flower_visited')
        rec['flowers_found_this_episode'] = info.get('flowers_found_this_episode')
        rec['nearest_flower_distance'] = info.get('nearest_flower_distance')
        rec['altitude'] = info.get('altitude')
        records.append(rec)

if not records:
    print('No records parsed from', in_path)
    sys.exit(2)

df = pd.DataFrame.from_records(records)
# order columns
cols = ['step','action','reward','done','agent_x','agent_y','agent_z','agent_energy','altitude','steps_so_far','energy','total_nectar_collected','flowers_found_this_episode','last_flower_visited','nearest_flower_distance']
cols = [c for c in cols if c in df.columns]

df = df[cols]
# save CSV
df.to_csv(out_path, index=False)
print(f'Saved CSV: {out_path}  (rows={len(df)})\n')
# print head
print(df.head(20).to_string(index=False))
