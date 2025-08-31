"""Collect detailed evaluation statistics for multiple models and write a CSV summary.
This re-uses functions from `detailed_evaluation.py` so results match previous runs.

Usage:
    python analysis/scripts/collect_detailed_eval.py
"""
import os
import csv
import numpy as np
from scipy import stats
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import detailed_evaluation as de
from hummingbird_env import ComplexHummingbird3DMatplotlibEnv

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODELS = [
    os.path.join(ROOT, 'models', 'hummingbird_model_2_flowers.zip'),
    os.path.join(ROOT, 'models', 'hummingbird_model_4_flowers.zip'),
    os.path.join(ROOT, 'models', 'hummingbird_model_6_flowers.zip'),
    os.path.join(ROOT, 'models', 'hummingbird_model_8_flowers.zip'),
    os.path.join(ROOT, 'models', 'hummingbird_model_10_flowers.zip'),
]

OUT_DIR = os.path.join(ROOT, 'analysis', 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, 'detailed_evaluation_summary.csv')

FIELDNAMES = [
    'model',
    'mean_survival_rate',
    'std_dev_survival',
    'ci_lower',
    'ci_upper',
    'mean_nectar',
    'std_dev_nectar',
    'correlation',
    'total_episodes'
]


def compute_stats_for_model(model_path: str):
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None

    print(f"Evaluating: {model_path}")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Failed to load {model_path}: {e}")
        return None

    # detect number of flowers from observation space if available
    obs_space = model.observation_space
    if hasattr(obs_space, 'spaces') and 'flowers' in obs_space.spaces:
        num_flowers = obs_space.spaces['flowers'].shape[0]
        eval_env = make_vec_env(
            ComplexHummingbird3DMatplotlibEnv,
            n_envs=1,
            env_kwargs=dict(num_flowers=num_flowers)
        )
    else:
        eval_env = make_vec_env(ComplexHummingbird3DMatplotlibEnv, n_envs=1)

    all_survival_rates = []
    all_episode_lengths = []
    all_nectar_collected = []

    for i in range(de.NUM_RUNS):
        lengths, nectar = de.evaluate_model_stats(model, eval_env, num_episodes=de.EPISODES_PER_RUN)
        survival_count = sum(1 for length in lengths if length >= de.SURVIVAL_THRESHOLD)
        survival_rate = (survival_count / de.EPISODES_PER_RUN) * 100
        all_survival_rates.append(survival_rate)
        all_episode_lengths.extend(lengths)
        all_nectar_collected.extend(nectar)

    eval_env.close()

    mean_survival_rate = float(np.mean(all_survival_rates))
    std_dev_survival = float(np.std(all_survival_rates))
    # handle small-sample edge cases
    try:
        ci = stats.t.interval(0.95, len(all_survival_rates)-1, loc=mean_survival_rate, scale=stats.sem(all_survival_rates))
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
    except Exception:
        ci_lower, ci_upper = (np.nan, np.nan)

    mean_nectar = float(np.mean(all_nectar_collected)) if len(all_nectar_collected) > 0 else float('nan')
    std_dev_nectar = float(np.std(all_nectar_collected)) if len(all_nectar_collected) > 0 else float('nan')

    if np.std(all_episode_lengths) > 0 and np.std(all_nectar_collected) > 0:
        correlation = float(np.corrcoef(all_episode_lengths, all_nectar_collected)[0, 1])
    else:
        correlation = float('nan')

    return {
        'model': os.path.basename(model_path),
        'mean_survival_rate': mean_survival_rate,
        'std_dev_survival': std_dev_survival,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'mean_nectar': mean_nectar,
        'std_dev_nectar': std_dev_nectar,
        'correlation': correlation,
        'total_episodes': de.NUM_RUNS * de.EPISODES_PER_RUN
    }


def main():
    rows = []
    for m in MODELS:
        stats_row = compute_stats_for_model(m)
        if stats_row:
            rows.append(stats_row)

    if not rows:
        print("No evaluation results to write.")
        return

    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote summary CSV: {OUT_CSV}")

if __name__ == '__main__':
    main()
