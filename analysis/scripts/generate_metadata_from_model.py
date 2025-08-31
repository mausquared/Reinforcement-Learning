#!/usr/bin/env python3
"""Run one episode from a model and save metadata JSON usable by save_topdown.py

Usage:
  python generate_metadata_from_model.py <model.zip> [--num-flowers N] [--out metadata.json]

Saves: metadata JSON with keys: trajectory, flower_positions, flower_visits, model_path, steps, num_flowers
"""
import argparse
import json
from pathlib import Path
import numpy as np

def main():
    print('generate_metadata_from_model.py starting...')
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Path to model .zip')
    parser.add_argument('--num-flowers', type=int, default=None)
    parser.add_argument('--out', default=None)
    parser.add_argument('--stochastic', action='store_true')
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print('Model not found:', model_path)
        return 2

    # Lazy imports
    try:
        from stable_baselines3 import PPO
        from hummingbird_env import ComplexHummingbird3DMatplotlibEnv
    except Exception as e:
        print('Failed to import dependencies:', e)
        return 3

    # Create env
    env_kwargs = {'render_mode': None}
    if args.num_flowers is not None:
        env_kwargs['num_flowers'] = int(args.num_flowers)
    env = ComplexHummingbird3DMatplotlibEnv(**env_kwargs)

    # Load model
    try:
        model = PPO.load(str(model_path))
    except Exception:
        # try safe load
        try:
            model = PPO.load(str(model_path), custom_objects={'lr_schedule': (lambda _: 0.0)})
        except Exception as e:
            print('Failed to load model:', e)
            env.close()
            return 4

    # Run one episode
    try:
        obs = env.reset()
    except Exception:
        obs, _ = env.reset()
    traj = []
    flower_visits = {}
    max_steps = getattr(env, 'MAX_STEPS', 300)
    for step in range(int(max_steps)):
        try:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
        except Exception:
            if isinstance(obs, dict) and 'agent' in obs:
                action, _ = model.predict(obs['agent'], deterministic=not args.stochastic)
            else:
                raise
        step_ret = env.step(action)
        if len(step_ret) == 5:
            obs, reward, terminated, truncated, info = step_ret
            done = terminated or truncated
        else:
            obs, reward, done, info = step_ret
        # get agent pos
        pos = getattr(env, 'agent_pos', None)
        if pos is None:
            if isinstance(obs, dict) and 'agent' in obs:
                pos = np.asarray(obs['agent'])[:3]
            else:
                pos = np.array([0.0, 0.0, 0.0])
        traj.append(np.asarray(pos).tolist())
        # record any info about last flower visited
        if isinstance(info, dict) and 'last_flower_visited' in info and info['last_flower_visited'] is not None:
            try:
                fi = int(info['last_flower_visited'])
                flower_visits.setdefault(str(fi), []).append(step)
            except Exception:
                pass
        if done:
            break

    # flower positions from env
    flowers = None
    try:
        if hasattr(env, 'flowers') and env.flowers is not None:
            flowers = np.asarray(env.flowers)[:, :3].tolist()
    except Exception:
        flowers = None

    collision_radius = getattr(env, 'FLOWER_COLLISION_RADIUS', 1.2)

    # fallback: if no explicit flower_visits recorded, compute proximity-based visits
    if not flower_visits and flowers is not None:
        flower_visits = {str(i): [] for i in range(len(flowers))}
        traj_arr = np.asarray(traj)
        for fi, f in enumerate(flowers):
            dists = np.linalg.norm(traj_arr - np.asarray(f), axis=1)
            hit_idxs = np.where(dists <= float(collision_radius))[0]
            if hit_idxs.size > 0:
                flower_visits[str(fi)] = hit_idxs.tolist()

    metadata = {
        'model': model_path.stem,
        'model_path': str(model_path),
        'saved_at': None,
        'steps': len(traj),
        'seed': None,
        'stochastic': bool(args.stochastic),
        'num_flowers': args.num_flowers if args.num_flowers is not None else (len(flowers) if flowers is not None else None),
        'trajectory': traj,
        'flower_positions': flowers or [],
        'collision_radius': float(collision_radius),
        'format': 'png',
        'dpi': 300,
        'matplotlib_backend': None
    }

    out_path = Path(args.out) if args.out else Path(f"models/trajectory_{model_path.stem}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Wrote metadata to:', out_path)
    env.close()
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
