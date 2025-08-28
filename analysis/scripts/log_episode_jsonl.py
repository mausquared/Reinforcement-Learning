#!/usr/bin/env python3
"""Run one episode with a saved model and write per-step JSONL logs.

Usage:
  python log_episode_jsonl.py [metadata.json] [out.jsonl]

Defaults to the 083435 metadata in the repo.
"""
import json
import argparse
from pathlib import Path
import numpy as np

def sanitize(obj):
    """Convert numpy/scalars to plain Python types for JSON serialization."""
    if obj is None:
        return None
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [sanitize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    try:
        import numpy as _np
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    # Fallback: try to stringify
    try:
        return json.loads(json.dumps(obj))
    except Exception:
        return str(obj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('meta', nargs='?', default='models/trajectory_autonomous_training_1_25000k_2025-08-27_083435.json')
    parser.add_argument('--out', '-o', default=None)
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic actions')
    args = parser.parse_args()

    meta_path = Path(args.meta)
    if not meta_path.exists():
        print(f"Metadata not found: {meta_path}")
        return 2

    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    model_path = meta.get('model_path') or meta.get('model')
    if not model_path:
        print("No model_path found in metadata JSON. Provide model with --model or set model_path in JSON.")
        return 2

    model_path = Path(model_path)
    if not model_path.exists():
        # try relative to repo
        candidate = Path('models') / model_path.name
        if candidate.exists():
            model_path = candidate
        else:
            print(f"Model archive not found: {model_path}")
            return 2

    out_path = Path(args.out) if args.out else Path(f"episode_log_{meta_path.stem}.jsonl")
    print(f"Logging episode to: {out_path}")

    # Lazy imports that may be heavy
    try:
        from train import create_environment_for_model
    except Exception as e:
        print(f"Failed to import environment helper from train.py: {e}")
        return 3

    try:
        # stable-baselines3 import
        from stable_baselines3 import PPO
    except Exception as e:
        print(f"Failed to import stable_baselines3.PPO: {e}")
        return 4

    # create env
    env = create_environment_for_model(str(model_path), render_mode=None)

    # load model
    try:
        model = PPO.load(str(model_path))
    except Exception as e:
        print(f"Failed to load model: {e}")
        env.close()
        return 5

    # run one episode
    rec_count = 0
    with out_path.open('w', encoding='utf-8') as fh:
        obs_info = env.reset()
        # env.reset may return (obs, info) or only obs depending on env API
        if isinstance(obs_info, tuple) and len(obs_info) == 2:
            obs, info = obs_info
        else:
            obs = obs_info
            info = {}

        done = False
        step_idx = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=args.deterministic)
            step_ret = env.step(action)
            # support both new (obs, reward, terminated, truncated, info) and old (obs, reward, done, info)
            if len(step_ret) == 5:
                obs, reward, terminated, truncated, info = step_ret
                done_flag = terminated or truncated
            else:
                obs, reward, done_flag, info = step_ret
                terminated = done_flag
                truncated = False

            record = {
                'step': step_idx,
                'action': sanitize(action),
                'obs': sanitize(obs),
                'reward': float(reward) if reward is not None else None,
                'done': bool(done_flag),
                'info': sanitize(info)
            }
            fh.write(json.dumps(record) + '\n')
            rec_count += 1
            step_idx += 1
            if done_flag:
                break

    env.close()
    print(f"Wrote {rec_count} records to {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
