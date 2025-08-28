import json
import numpy as np
from pathlib import Path
p=Path('models/trajectory_autonomous_training_1_25000k_2025-08-27_083435.json')
meta=json.loads(p.read_text(encoding='utf-8'))
traj=np.asarray(meta.get('trajectory',[]))
flowers=np.asarray(meta.get('flower_positions',[]))
visits=meta.get('flower_visits',{})
collision=float(meta.get('collision_radius',1.2))
print(f"file: {p.name}")
print(f"model: {meta.get('model')} saved_at: {meta.get('saved_at')} steps: {meta.get('steps')} collision_radius: {collision}\n")
for i in range(len(flowers)):
    steps=visits.get(str(i),[])
    print(f"Flower {i} position: {flowers[i].tolist()}\n  raw_frames_count: {len(steps)}")
    if len(steps)==0:
        print('  (no visits)\n')
        continue
    dists=[]
    details=[]
    for s in steps:
        sidx=int(s)
        if sidx < len(traj):
            pos=traj[sidx]
            dist=float(np.linalg.norm(pos-flowers[i]))
            # nearest flower
            all_dists=np.linalg.norm(traj[sidx]-flowers,axis=1)
            nearest=int(np.argmin(all_dists))
            details.append((sidx, pos.tolist(), round(dist,4), nearest, round(float(all_dists[nearest]),4), dist<=collision))
            dists.append(dist)
        else:
            details.append((int(s), None, None, None, None, False))
    import statistics
    print(f"  distances -> min: {min(dists):.4f}, mean: {statistics.mean(dists):.4f}, max: {max(dists):.4f}")
    within=sum(1 for x in dists if x<=collision)
    print(f"  within_collision_count: {within} / {len(dists)} ({within/len(dists):.2%})")
    print('  per-step details:')
    for det in details:
        sidx,pos,dist,nearest,ndist,ok=det
        if pos is None:
            print(f"    step {sidx}: index out of range\n")
        else:
            print(f"    step {sidx}: pos={pos} dist_to_assigned={dist} nearest_flower={nearest} dist_to_nearest={ndist} within={ok}")
    print()
