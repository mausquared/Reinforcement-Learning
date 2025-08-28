import json
import numpy as np
from pathlib import Path
p=Path('models/trajectory_autonomous_training_1_25000k_2025-08-27_083435.json')
meta=json.loads(p.read_text(encoding='utf-8'))
visits=meta.get('flower_visits',{})
traj=np.asarray(meta.get('trajectory',[]))
flowers=np.asarray(meta.get('flower_positions',[]))
collision=float(meta.get('collision_radius',1.2))

COOLDOWN=3  # if gap between frames <= COOLDOWN, treat as same feed event

print(f"File: {p.name}  collision_radius={collision}  cooldown={COOLDOWN}\n")
for i in range(len(flowers)):
    steps=sorted(int(s) for s in visits.get(str(i),[]))
    raw=len(steps)
    events=[]
    if steps:
        start=steps[0]
        group=[start]
        for s in steps[1:]:
            if s - group[-1] <= COOLDOWN:
                group.append(s)
            else:
                events.append(group)
                group=[s]
        events.append(group)
    print(f"Flower {i}: pos={flowers[i].tolist()}  raw_frames={raw}  collapsed_events={len(events)}")
    if events:
        for e in events:
            # representative step = first step
            rep=e[0]
            pos=traj[rep].tolist() if rep < len(traj) else None
            dist=float(np.linalg.norm(np.asarray(pos)-flowers[i])) if pos is not None else None
            print(f"  event steps={e} rep={rep} rep_pos={pos} dist_to_flower={round(dist,4) if dist is not None else None} size={len(e)}")
    print()
