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

# safe model load with fallback custom_objects
custom_objects = {'lr_schedule': (lambda _: 0.0), 'clip_range': (lambda _: 0.2)}
try:
    model = PPO.load(model_path)
    print('Loaded model without custom_objects')
except Exception as e:
    print('Load without custom_objects failed:', e)
    print('Retrying with custom_objects...')
    model = PPO.load(model_path, custom_objects=custom_objects)
    print('Loaded model with custom_objects')

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
    except Exception as e:
        print('model.predict error at step', step, e)
        if isinstance(obs, dict) and 'agent' in obs:
            try:
                action, _ = model.predict(obs['agent'], deterministic=True)
            except Exception as e2:
                print('predict fallback failed:', e2)
                break
        else:
            break

    try:
        out = env.step(action)
    except Exception as e:
        print('env.step error at step', step, e)
        break

    # handle different return signatures
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
    elif len(out) == 4:
        obs, reward, terminated, info = out
        truncated = False
    else:
        # unexpected
        print('env.step returned unexpected len', len(out))
        break

    pos = getattr(env, 'agent_pos', None)
    if pos is None:
        if isinstance(obs, dict) and 'agent' in obs:
            pos = np.asarray(obs['agent'])[:3]
        else:
            pos = np.array([0.0,0.0,0.0])
    traj.append(pos)

    # print reward type and info keys
    try:
        rtype = type(reward)
        rrepr = repr(reward)
    except Exception:
        rtype = str(type(reward))
        rrepr = '<unreprable>'
    print(f'step {step}: reward type={rtype} repr={rrepr} info_keys={list(info.keys()) if isinstance(info, dict) else type(info)}')

    # check for positive reward
    is_reward = False
    try:
        arr = np.asarray(reward)
        if arr.size == 1:
            is_reward = float(arr) > 0
        else:
            is_reward = np.any(arr > 0)
    except Exception as e:
        print('reward check error:', e)

    if is_reward:
        # prefer explicit info field if present
        fidx = None
        if isinstance(info, dict):
            # look for common keys
            for key in ('flower_idx', 'flower_id', 'fed_flower', 'flower'):
                if key in info:
                    try:
                        fidx = int(info[key])
                        break
                    except Exception:
                        pass
        if fidx is None and len(flowers) > 0:
            dists = np.linalg.norm(flowers[:, :2] - np.asarray(pos)[:2], axis=1)
            fidx = int(np.argmin(dists))
        reward_visits[fidx].append(step)
        print(f'  -> reward mapped to flower {fidx} dist={dists[fidx]:.3f}' if len(flowers)>0 else f'  -> reward mapped to None')

    if terminated or truncated:
        print('terminated at step', step)
        break

print('\nSummary of reward-mapped feeds:')
for k, v in reward_visits.items():
    print(k, len(v), v[:20])

# also show visits from metadata for comparison
visits_meta = meta.get('flower_visits', {})
print('\nMetadata visits (frames) per flower:')
for k in sorted(visits_meta.keys(), key=lambda x: int(x)):
    print(k, len(visits_meta[k]), visits_meta[k][:20])

print('\nReplayed trajectory length:', len(traj))

try:
    env.close()
except Exception:
    pass
