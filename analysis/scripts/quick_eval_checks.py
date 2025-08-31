import os
import sys
import numpy as np
from stable_baselines3 import PPO
from hummingbird_env import ComplexHummingbird3DMatplotlibEnv


def run_eval(model_path, deterministic, num_episodes=200):
    print(f"\nRunning eval: model={os.path.basename(model_path)}, deterministic={deterministic}, episodes={num_episodes}")
    model = PPO.load(model_path)

    # try to detect num_flowers from model obs_space
    obs_space = model.observation_space
    if hasattr(obs_space, 'spaces') and 'flowers' in obs_space.spaces:
        num_flowers = obs_space.spaces['flowers'].shape[0]
    else:
        num_flowers = 5

    env = ComplexHummingbird3DMatplotlibEnv(num_flowers=num_flowers, render_mode=None)

    episode_lengths = []
    episode_nectar = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        total_nectar = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            step_res = env.step(action)
            # handle both gym API variants
            if len(step_res) == 5:
                obs, reward, terminated, truncated, info = step_res
                done = terminated[0] if isinstance(terminated, (list, np.ndarray)) else terminated
                # extract info dict
                info_dict = info[0] if isinstance(info, (list, tuple)) else info
            else:
                obs, reward, done, info = step_res
                info_dict = info[0] if isinstance(info, (list, tuple)) else info

            steps += 1
            # collect nectar if present
            if isinstance(info_dict, dict) and 'total_nectar_collected' in info_dict:
                total_nectar = info_dict.get('total_nectar_collected', total_nectar)

            # safety: cap steps
            if steps > 5000:
                break

        episode_lengths.append(steps)
        episode_nectar.append(total_nectar)

    env.close()
    return np.array(episode_lengths), np.array(episode_nectar)


if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/best_model.zip'
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'quick_eval_results.txt')

    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model not found: {model_path}')

        # settings to test
        thresholds = [200, 300]
        out_lines = []
        for det in [True, False]:
            lengths, nectar = run_eval(model_path, deterministic=det, num_episodes=200)
            out_lines.append(f"Results deterministic={det}:")
            for t in thresholds:
                successes = np.sum(lengths >= t)
                rate = successes / len(lengths) * 100
                out_lines.append(f"  Threshold {t}: {successes}/{len(lengths)} = {rate:.1f}%")
            out_lines.append(f"  Mean length: {np.mean(lengths):.1f}, Mean nectar: {np.mean(nectar):.1f}\n")

        with open(out_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(out_lines))

        print(f"Wrote quick eval results to: {out_file}")
    except Exception as e:
        # write exception info
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write('ERROR during quick eval run:\n')
            import traceback
            f.write(traceback.format_exc())
        print(f"Error occurred; wrote traceback to {out_file}")
