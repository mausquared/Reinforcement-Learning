import json
from pathlib import Path
import numpy as np

meta_path = Path('models/trajectory_autonomous_training_1_25000k_2025-08-27_083435.json')
meta = json.loads(meta_path.read_text(encoding='utf-8'))
traj = np.asarray(meta.get('trajectory', []))
flowers = np.asarray(meta.get('flower_positions', []))
visits = meta.get('flower_visits', {})
collision_radius = float(meta.get('collision_radius', 1.2))

print('trajectory length =', len(traj))
print('num flowers =', len(flowers))
print('collision_radius =', collision_radius)

for fk, steps in visits.items():
    fi = int(fk)
    print(f"\nFlower {fi}: position={flowers[fi] if fi < len(flowers) else 'MISSING'} visits={steps}")
    for s in steps:
        try:
            sidx = int(s)
        except Exception:
            print('  step parse failed:', s); continue
        out_of_range = sidx >= len(traj)
        if out_of_range:
            print(f"  step {sidx}: OUT OF RANGE (traj length {len(traj)})")
            continue
        pos = traj[sidx]
        fpos = flowers[fi]
        dist = np.linalg.norm(pos[:3] - fpos[:3])
        # nearest flower
        dists_all = np.linalg.norm(traj[sidx,:3] - flowers[:,:3], axis=1)
        nearest_idx = int(np.argmin(dists_all))
        nearest_dist = float(dists_all[nearest_idx])
        print(f"  step {sidx}: traj_pos={pos}, dist_to_recorded_flower={dist:.3f}, nearest_flower={nearest_idx}, dist_to_nearest={nearest_dist:.3f}, within_radius={dist <= collision_radius}")

print('\nSummary:')
for fk, steps in visits.items():
    fi = int(fk)
    good = 0; bad = 0; oob = 0
    for s in steps:
        sidx = int(s)
        if sidx >= len(traj):
            oob += 1
            continue
        pos = traj[sidx]
        dist = np.linalg.norm(pos[:3] - flowers[fi][:3])
        if dist <= collision_radius:
            good += 1
        else:
            bad += 1
    print(f"Flower {fi}: good_within_radius={good}, outside={bad}, out_of_range={oob}")
