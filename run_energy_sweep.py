import os
import csv
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from detailed_evaluation import evaluate_model_stats, SURVIVAL_THRESHOLD
from hummingbird_env import ComplexHummingbird3DMatplotlibEnv

# Configuration (quick run)
runs_per_setting = 1
episodes_per_setting = 50
energy_factors = [0.5, 0.75, 1.0, 1.25, 1.5]

# Locate default model if none provided
MODELS_DIR = os.path.join('.', 'models')
DEFAULT_MODEL = None
if len(sys.argv) >= 2:
    DEFAULT_MODEL = sys.argv[1]
else:
    # prefer 2500k, fallback to 25000k, else best_model.zip
    for fname in os.listdir(MODELS_DIR):
        if '2500k' in fname and fname.endswith('.zip'):
            DEFAULT_MODEL = os.path.join(MODELS_DIR, fname)
            break
    if DEFAULT_MODEL is None:
        for fname in os.listdir(MODELS_DIR):
            if '25000k' in fname and fname.endswith('.zip'):
                DEFAULT_MODEL = os.path.join(MODELS_DIR, fname)
                print(f"Using fallback model: {fname}")
                break
    if DEFAULT_MODEL is None:
        candidate = os.path.join(MODELS_DIR, 'best_model.zip')
        if os.path.exists(candidate):
            DEFAULT_MODEL = candidate

if DEFAULT_MODEL is None:
    print('No model found in models/. Provide path as first arg.')
    sys.exit(1)

print('Model used:', DEFAULT_MODEL)
model = PPO.load(DEFAULT_MODEL)

# Infer num_flowers from model obs if possible
num_flowers = 5
try:
    obs_space = model.observation_space
    if 'flowers' in obs_space.spaces:
        num_flowers = obs_space.spaces['flowers'].shape[0]
except Exception:
    pass

# Determine base max_energy by instantiating a temp env
tmp = ComplexHummingbird3DMatplotlibEnv(num_flowers=num_flowers)
base_max_energy = getattr(tmp, 'max_energy', 100)
try:
    tmp.close()
except Exception:
    pass

results = []

for factor in energy_factors:
    max_energy = int(base_max_energy * factor)
    print(f"\nEvaluating max_energy={max_energy} (factor {factor}) -> {episodes_per_setting} episodes")
    # create env with overridden max_energy
    env = make_vec_env(ComplexHummingbird3DMatplotlibEnv, n_envs=1, env_kwargs=dict(num_flowers=num_flowers, max_energy=max_energy))

    lengths, nectar = evaluate_model_stats(model, env, num_episodes=episodes_per_setting)
    # compute survival percent
    survival_pct = 100.0 * sum(1 for L in lengths if L >= SURVIVAL_THRESHOLD) / max(1, len(lengths))
    mean_nectar = float(np.mean(nectar)) if nectar else float('nan')
    print(f"Result: survival@{SURVIVAL_THRESHOLD} = {survival_pct:.1f}%, mean_nectar = {mean_nectar:.2f}")

    results.append({'factor': factor, 'max_energy': max_energy, 'survival_pct': survival_pct, 'mean_nectar': mean_nectar})
    try:
        env.close()
    except Exception:
        pass

# Save CSV
base = os.path.basename(DEFAULT_MODEL)
base = base.replace('.zip', '').replace('.', '_')
out_csv = os.path.join('models', f'energy_sweep_{base}.csv')
with open(out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['factor', 'max_energy', 'survival_pct', 'mean_nectar'])
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print('\nSaved energy sweep results to', out_csv)
