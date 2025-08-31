import os
import sys
import csv
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from hummingbird_env import ComplexHummingbird3DMatplotlibEnv

# Config
EPISODES_PER_SETTING = 50
ENERGY_FACTORS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
SURVIVAL_THRESHOLD = 200
MODELS_DIR = os.path.join('.', 'models')


def find_default_model():
    # prefer 2500k, fallback 25000k, then best_model
    if not os.path.exists(MODELS_DIR):
        return None
    files = os.listdir(MODELS_DIR)
    for f in files:
        if '2500k' in f and f.endswith('.zip'):
            return os.path.join(MODELS_DIR, f)
    for f in files:
        if '25000k' in f and f.endswith('.zip'):
            return os.path.join(MODELS_DIR, f)
    candidate = os.path.join(MODELS_DIR, 'best_model.zip')
    if os.path.exists(candidate):
        return candidate
    return None


def run_for_model(model_path, energy_factors=ENERGY_FACTORS, episodes=EPISODES_PER_SETTING):
    print('Loading model:', model_path)
    model = PPO.load(model_path)

    # infer num_flowers
    num_flowers = 5
    try:
        obs_space = model.observation_space
        if 'flowers' in obs_space.spaces:
            num_flowers = obs_space.spaces['flowers'].shape[0]
    except Exception:
        pass

    # base max_energy
    tmp = ComplexHummingbird3DMatplotlibEnv(num_flowers=num_flowers)
    base_max_energy = getattr(tmp, 'max_energy', 100)
    try:
        tmp.close()
    except Exception:
        pass

    out_rows = []
    summary = []

    for factor in energy_factors:
        max_energy = int(base_max_energy * factor)
        print(f"\n=== Evaluating max_energy={max_energy} (factor {factor}) ===")

        # create env with overridden max_energy
        env = make_vec_env(ComplexHummingbird3DMatplotlibEnv, n_envs=1, env_kwargs=dict(num_flowers=num_flowers, max_energy=max_energy))

        # Run episodes and collect per-episode info
        per_steps = []
        per_nectar = []
        per_energy_end = []
        died_count = 0
        ways_to_end = {'energy_depletion': 0, 'time_limit': 0, 'other': 0}

        for ep in range(episodes):
            obs = env.reset()
            terminated = [False]
            truncated = [False]
            steps = 0
            info = None
            while not (terminated[0] or truncated[0]):
                action, _ = model.predict(obs, deterministic=True)
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, _, terminated, truncated, info = step_result
                else:
                    obs, _, done, info = step_result
                    terminated = done
                    truncated = [False]
                steps += 1
                # safety: prevent infinite loops
                if steps > 5000:
                    print('Warning: long episode, breaking')
                    break

            info_dict = info[0] if isinstance(info, (list, tuple)) else info
            steps_reported = info_dict.get('steps', steps) if isinstance(info_dict, dict) else steps
            nectar = info_dict.get('total_nectar_collected', float('nan')) if isinstance(info_dict, dict) else float('nan')
            energy_end = info_dict.get('energy', float('nan')) if isinstance(info_dict, dict) else float('nan')

            died = False
            if isinstance(info_dict, dict) and 'energy' in info_dict and info_dict['energy'] <= 0:
                died = True
                died_count += 1
                ways_to_end['energy_depletion'] += 1
            elif steps_reported >= 300:
                ways_to_end['time_limit'] += 1
            else:
                ways_to_end['other'] += 1

            per_steps.append(steps_reported)
            per_nectar.append(nectar)
            per_energy_end.append(energy_end)

            out_rows.append({
                'model': os.path.basename(model_path),
                'factor': factor,
                'max_energy': max_energy,
                'episode': ep,
                'steps': steps_reported,
                'nectar': nectar,
                'energy_end': energy_end,
                'died': int(died)
            })

        env.close()

        # summary stats
        survival_pct = 100.0 * sum(1 for s in per_steps if s >= SURVIVAL_THRESHOLD) / max(1, len(per_steps))
        mean_nectar = float(np.nanmean(per_nectar)) if per_nectar else float('nan')
        median_steps = float(np.nanmedian(per_steps))
        q1 = float(np.nanpercentile(per_steps, 25))
        q3 = float(np.nanpercentile(per_steps, 75))
        min_steps = int(np.nanmin(per_steps))
        max_steps = int(np.nanmax(per_steps))

        summary.append({
            'factor': factor,
            'max_energy': max_energy,
            'survival_pct': survival_pct,
            'mean_nectar': mean_nectar,
            'median_steps': median_steps,
            'q1_steps': q1,
            'q3_steps': q3,
            'min_steps': min_steps,
            'max_steps': max_steps,
            'died_count': died_count,
            'ways_to_end': ways_to_end
        })

        print(f"Result: survival@{SURVIVAL_THRESHOLD} = {survival_pct:.1f}%, mean_nectar = {mean_nectar:.2f}")
        print(f" steps: median={median_steps:.1f}, q1={q1:.1f}, q3={q3:.1f}, min={min_steps}, max={max_steps}")
        print(f" died_count={died_count}, ways_to_end={ways_to_end}")

    # save detailed per-episode CSV and summary CSV
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join('models', f'energy_sweep_detailed_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)

    per_file = os.path.join(out_dir, 'per_episode.csv')
    with open(per_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model','factor','max_energy','episode','steps','nectar','energy_end','died'])
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    summary_file = os.path.join(out_dir, 'summary.csv')
    with open(summary_file, 'w', newline='') as f:
        fieldnames = ['factor','max_energy','survival_pct','mean_nectar','median_steps','q1_steps','q3_steps','min_steps','max_steps','died_count','ways_to_end']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summary:
            # convert ways_to_end to string
            s2 = s.copy()
            s2['ways_to_end'] = str(s2['ways_to_end'])
            writer.writerow(s2)

    print('\nSaved detailed results to', out_dir)
    return out_dir


if __name__ == '__main__':
    model_path = None
    if len(sys.argv) >= 2:
        model_path = sys.argv[1]
    else:
        model_path = find_default_model()
    if model_path is None:
        print('No model found in models/. Provide model path as first arg.')
        sys.exit(1)
    run_energy_sweep_detailed = run_for_model(model_path)
    print('Done.')
