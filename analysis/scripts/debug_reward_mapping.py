import json
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from hummingbird_env import ComplexHummingbird3DMatplotlibEnv

META = Path('models/trajectory_autonomous_training_1_25000k_2025-08-27_083516.json')
if not META.exists():
    print('metadata not found:', META)
    raise SystemExit(1)
meta = json.loads(META.read_text(encoding='utf-8'))
model_path = meta.get('model_path') or meta.get('model')
if not model_path or not Path(model_path).exists():
    print('model not found:', model_path)
    raise SystemExit(2)

print('Using model:', model_path)
print('Metadata steps:', meta.get('steps'))

model = PPO.load(model_path)
num_flowers = int(meta.get('num_flowers', 5))
print('num_flowers:', num_flowers)

env = ComplexHummingbird3DMatplotlibEnv(render_mode=None, num_flowers=num_flowers)
try:
    reset_ret = env.reset()
    if isinstance(reset_ret, tuple):
        obs = reset_ret[0]
    else:
        obs = reset_ret
except Exception as e:
    print('reset error:', e)
    obs = env.reset()

flowers = np.asarray(getattr(env, 'flowers', []))
print('flower positions (env):')
for i, f in enumerate(flowers):
    print(i, f[:3])

max_steps = int(meta.get('steps', 300))
reward_visits = {i: [] for i in range(len(flowers))}
traj = []
for step in range(max_steps):
    try:
        action, _ = model.predict(obs, deterministic=True)
    except Exception:
        if isinstance(obs, dict) and 'agent' in obs:
            action, _ = model.predict(obs['agent'], deterministic=True)
        else:
            raise
    obs, reward, terminated, truncated, info = env.step(action)
    pos = getattr(env, 'agent_pos', None)
    if pos is None:
        if isinstance(obs, dict) and 'agent' in obs:
            pos = np.asarray(obs['agent'])[:3]
        else:
            pos = np.array([0.0,0.0,0.0])
    traj.append(pos)
    if reward and reward > 0:
        # map to nearest flower by XY
        if len(flowers) > 0:
            dists = np.linalg.norm(flowers[:, :2] - np.asarray(pos)[:2], axis=1)
            fidx = int(np.argmin(dists))
            reward_visits[fidx].append(step)
            print(f"step {step}: reward={reward:.3f} -> flower {fidx} dist={dists[fidx]:.3f}")
        else:
            print(f"step {step}: reward={reward:.3f} -> no flowers known")
    else:
        print(f"step {step}: reward={reward}")
    if terminated or truncated:
        print('terminated at step', step)
        break

print('\nSummary of reward-mapped feeds:')
for k, v in reward_visits.items():
    print(k, len(v), v[:10])

# also show visits from metadata for comparison
visits_meta = meta.get('flower_visits', {})
print('\nMetadata visits (frames) per flower:')
for k in sorted(visits_meta.keys(), key=lambda x: int(x)):
    print(k, len(visits_meta[k]), visits_meta[k][:10])

# show trajectory length
print('\nReplayed trajectory length:', len(traj))

try:
    env.close()
except Exception:
    pass
