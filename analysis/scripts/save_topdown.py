#!/usr/bin/env python3
"""Save a standalone top-down XY projection from a trajectory metadata JSON.

Usage:
    python save_topdown.py path/to/metadata.json [out_pdf] [out_png]

If out paths are omitted, defaults are `report_topdown.pdf` and `report_topdown.png`.
"""
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) < 1:
        print('Usage: save_topdown.py path/to/metadata.json [out_pdf] [out_png]')
        return 1
    meta_path = Path(argv[0])
    out_pdf = Path(argv[1]) if len(argv) >= 2 and not str(argv[1]).startswith('--') else Path('report_topdown.pdf')
    out_png = Path(argv[2]) if len(argv) >= 3 and not str(argv[2]).startswith('--') else Path('report_topdown.png')

    # optional flags: --dpi N to set PDF DPI, --png-dpi N to set PNG DPI, --cooldown N to set feed cooldown (in steps)
    dpi_pdf = 300
    dpi_png = 150
    cooldown = 30
    use_reward = False
    try:
        # simple parse for flags in argv
        for i, a in enumerate(argv):
            if a == '--dpi' and i + 1 < len(argv):
                dpi_pdf = int(argv[i + 1])
            if a == '--png-dpi' and i + 1 < len(argv):
                dpi_png = int(argv[i + 1])
            if a == '--cooldown' and i + 1 < len(argv):
                cooldown = int(argv[i + 1])
            if a == '--use-reward':
                use_reward = True
    except Exception:
        pass

    if not meta_path.exists():
        print('Metadata not found:', meta_path)
        return 2

    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    traj = np.asarray(meta.get('trajectory', []))
    flowers = np.asarray(meta.get('flower_positions', []))
    visits = meta.get('flower_visits', {})
    collision_radius = float(meta.get('collision_radius', 1.2))

    if traj.size == 0 or flowers.size == 0:
        print('Missing trajectory or flower_positions in metadata.'); return 3

    fig, ax = plt.subplots(figsize=(8, 8))

    # trajectory path
    try:
        ax.plot(traj[:, 0], traj[:, 1], color='0.5', linewidth=1.2, alpha=0.9, label='trajectory')
    except Exception:
        pass

    # flowers
    try:
        ax.scatter(flowers[:, 0], flowers[:, 1], s=140, c='tab:green', marker='X', edgecolor='k', label='flowers')
    except Exception:
        pass

    # (Option A) Minimal view: plot only feed events as color-graded circles (avoid crowding)
    feed_steps = []
    feed_coords = []
    for fi, steps in visits.items():
        for s in steps:
            sidx = int(s)
            if sidx < len(traj):
                feed_steps.append(sidx)
                feed_coords.append(traj[sidx, :2])
    if feed_coords:
        feed_coords = np.vstack(feed_coords)
        sc = ax.scatter(feed_coords[:, 0], feed_coords[:, 1], c=feed_steps, cmap='viridis', s=120, edgecolor='k', marker='o', zorder=5)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('step index')

    # collision circles
    try:
        for fi, fpos in enumerate(flowers):
            circ = mpatches.Circle((fpos[0], fpos[1]), collision_radius, fill=False, edgecolor='C3', linestyle='--', alpha=0.6)
            ax.add_patch(circ)
    except Exception:
        pass

    # (Removed) dashed lines from visits to flowers to reduce clutter

    # numbered labels and clearer feed counts placed to the right
    try:
        frames_counts = {int(k): len(v) for k, v in visits.items()}
        feeds_counts = None

        if use_reward:
            # attempt to replay model and use reward spikes to detect feeds
            try:
                from stable_baselines3 import PPO
                from hummingbird_env import ComplexHummingbird3DMatplotlibEnv

                model_path = meta.get('model_path') or meta.get('model')
                if model_path and Path(model_path).exists():
                    # load model
                    try:
                        model = PPO.load(model_path)
                    except Exception:
                        model = PPO.load(model_path, custom_objects={'lr_schedule': (lambda _: 0.0)})

                    env = ComplexHummingbird3DMatplotlibEnv(render_mode=None, num_flowers=int(meta.get('num_flowers', 5)))
                    try:
                        obs = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
                    except Exception:
                        obs = env.reset()

                    reward_visits = {i: [] for i in range(len(flowers))}
                    traj_replay = []
                    max_steps = int(meta.get('steps', len(traj)))
                    stochastic = bool(meta.get('stochastic', True))
                    prev_flowers_found = None
                    for step in range(max_steps):
                        try:
                            action, _ = model.predict(obs, deterministic=(not stochastic))
                        except Exception:
                            if isinstance(obs, dict) and 'agent' in obs:
                                action, _ = model.predict(obs['agent'], deterministic=(not stochastic))
                            else:
                                raise
                        obs, reward, terminated, truncated, info = env.step(action)
                        pos = getattr(env, 'agent_pos', None)
                        if pos is None:
                            if isinstance(obs, dict) and 'agent' in obs:
                                pos = np.asarray(obs['agent'])[:3]
                            else:
                                pos = np.array([0.0, 0.0, 0.0])
                        traj_replay.append(np.asarray(pos))
                        # Prefer explicit env info fields to detect feeding
                        fidx = None
                        try:
                            if isinstance(info, dict):
                                # prefer last_flower_visited if present
                                if 'last_flower_visited' in info and info['last_flower_visited'] is not None:
                                    try:
                                        fidx = int(info['last_flower_visited'])
                                    except Exception:
                                        fidx = None
                                # else detect increments in flowers_found_this_episode
                                elif 'flowers_found_this_episode' in info:
                                    if prev_flowers_found is None:
                                        prev_flowers_found = int(info.get('flowers_found_this_episode', 0))
                                    else:
                                        curr = int(info.get('flowers_found_this_episode', prev_flowers_found))
                                        if curr > prev_flowers_found:
                                            # map to nearest flower when count increases
                                            dists = np.linalg.norm(flowers[:, :2] - np.asarray(pos)[:2], axis=1)
                                            fidx = int(np.argmin(dists))
                                            prev_flowers_found = curr
                        except Exception:
                            pass

                        # fallback: use reward>0 if no info field detected
                        if fidx is None:
                            try:
                                if reward and (isinstance(reward, (int, float)) and reward > 0):
                                    dists = np.linalg.norm(flowers[:, :2] - np.asarray(pos)[:2], axis=1)
                                    fidx = int(np.argmin(dists))
                            except Exception:
                                pass

                        if fidx is not None:
                            reward_visits[fidx].append(step)
                        if terminated or truncated:
                            break

                    feeds_counts = {k: len(v) for k, v in reward_visits.items()}
                    visits = {k: [str(x) for x in v] for k, v in reward_visits.items()}
                    print('Used reward-based feed detection (replayed model)')
                    try:
                        env.close()
                    except Exception:
                        pass
                else:
                    print('Model path not found in metadata; falling back to cooldown-based feeds')
            except Exception as e:
                print('Reward-based detection failed, falling back to cooldown-based feeds:', e)

        if feeds_counts is None:
            # compute feeds using first-contact-after-cooldown per flower (fallback)
            feeds_counts = {}
            for k, steps in visits.items():
                ssorted = sorted(int(x) for x in steps)
                last_feed = None
                feeds = 0
                for s in ssorted:
                    if last_feed is None or (s - last_feed) >= cooldown:
                        feeds += 1
                        last_feed = s
                feeds_counts[int(k)] = feeds

        for i, pos in enumerate(flowers):
            ax.annotate(str(i), xy=(pos[0], pos[1]), xytext=(0, 8), textcoords='offset points',
                        ha='center', va='bottom', fontsize=10, weight='bold', color='black',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=0.9))
            # clearer feed count label to the right of the flower marker (minimal: feeds only)
            fc = feeds_counts.get(i, 0)
            ax.annotate(f"feeds: {fc}", xy=(pos[0], pos[1]), xytext=(8, 0), textcoords='offset points',
                        ha='left', va='center', fontsize=9, color='black',
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.85))
    except Exception:
        pass

    ax.set_aspect('equal', adjustable='box')
    # Title: keep generic on the plot; place model name into PDF metadata instead
    model_name = meta.get('model', meta.get('model_path', 'unknown'))
    ax.set_title(f"Top-down XY projection (collision radius={collision_radius})")

    # Build a legend with explicit handles so it's clear what each symbol means
    try:
        legend_handles = []
        # trajectory line
        legend_handles.append(Line2D([0], [0], color='0.5', lw=1.2, label='trajectory'))
        # flower marker
        legend_handles.append(Line2D([0], [0], marker='X', color='w', markerfacecolor='tab:green', markeredgecolor='k', markersize=10, label='flower (index)'))
        # feed marker (circle) if present
        if len(feed_steps) > 0:
            legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='C0', markeredgecolor='k', markersize=8, label='feed (step-colored)'))
        # collision circle
        legend_handles.append(mpatches.Patch(facecolor='none', edgecolor='C3', label=f'collision radius = {collision_radius}'))
        ax.legend(handles=legend_handles, loc='upper left', framealpha=0.9)
    except Exception:
        pass
    ax.set_xlabel('X'); ax.set_ylabel('Y')

    fig.tight_layout()
    try:
        # Save PDF with model name as Title metadata
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(out_pdf) as pdf:
            try:
                pdf.infodict()['Title'] = str(model_name)
            except Exception:
                pass
            pdf.savefig(fig, dpi=dpi_pdf, bbox_inches='tight')
        print('Saved PDF:', out_pdf, 'dpi=', dpi_pdf)
    except Exception as e:
        print('Failed to save PDF:', e)
    try:
        fig.savefig(out_png, dpi=dpi_png, bbox_inches='tight')
        print('Saved PNG:', out_png, 'dpi=', dpi_png)
    except Exception as e:
        print('Failed to save PNG:', e)
    plt.close(fig)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
