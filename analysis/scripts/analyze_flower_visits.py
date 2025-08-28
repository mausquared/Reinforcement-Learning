#!/usr/bin/env python3
"""
analyze_flower_visits.py

Simple analyzer for the JSON metadata produced by `visualize_trajectory.py`.
Produces per-flower statistics and plots to help quantify foraging behavior.

Outputs:
 - Printed table: flower index, total visits, first visit step, mean inter-visit interval
 - Plots: bar chart of visit counts, timeline scatter of visits, histogram/boxplot of inter-visit intervals

Usage:
    python analyze_flower_visits.py path/to/trajectory_metadata.json --show
    python analyze_flower_visits.py path/to/trajectory_metadata.json --save-plots out_dir

"""
import os
import sys
import json
import argparse
from collections import defaultdict
import numpy as np
import matplotlib
# prefer GUI backend when available
try:
    matplotlib.use('TkAgg')
except Exception:
    pass
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from hummingbird_env import ComplexHummingbird3DMatplotlibEnv


def load_metadata(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def summarize_flower_visits(flower_visits):
    """Return summary dict per flower and global lists."""
    summaries = {}
    total_visits = 0
    all_intervals = []
    visited_flowers = 0

    for k, lst in flower_visits.items():
        # keys might be strings
        fi = int(k)
        steps = sorted(int(x) for x in lst)
        count = len(steps)
        if count == 0:
            first = None
            mean_interval = None
            intervals = []
        else:
            first = steps[0]
            if count >= 2:
                intervals = np.diff(steps).tolist()
                mean_interval = float(np.mean(intervals))
                all_intervals.extend(intervals)
            else:
                intervals = []
                mean_interval = None

        summaries[fi] = {
            'count': count,
            'first_visit': first,
            'mean_interval': mean_interval,
            'intervals': intervals,
            'steps': steps,
        }
        total_visits += count
        if count > 0:
            visited_flowers += 1

    return summaries, total_visits, visited_flowers, all_intervals


def print_summary_table(summaries, total_visits, visited_flowers):
    print('\nFlower visit summary:')
    print('Flower  Visits  FirstStep  MeanInterval')
    for fi in sorted(summaries.keys()):
        s = summaries[fi]
        mv = f"{s['mean_interval']:.2f}" if s['mean_interval'] is not None else '-' 
        fs = str(s['first_visit']) if s['first_visit'] is not None else '-'
        print(f"{fi:6d} {s['count']:7d} {fs:10s} {mv:12s}")

    print(f"\nTotal visits: {total_visits}")
    print(f"Unique flowers visited: {visited_flowers} / {len(summaries)}")


def plot_stats(summaries, all_intervals, metadata, out_dir=None, show=True):
    # Prepare arrays
    flower_indices = sorted(summaries.keys())
    counts = np.array([summaries[i]['count'] for i in flower_indices])

    # Figure 1: bar chart counts
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(flower_indices, counts, color='tab:orange')
    ax1.set_xlabel('Flower index')
    ax1.set_ylabel('Total visits')
    ax1.set_title('Visits per flower')
    for i, v in zip(flower_indices, counts):
        ax1.text(i, v + 0.1, str(int(v)), ha='center', va='bottom', fontsize=8)

    # Figure 2: timeline scatter of visits (step vs flower)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for fi in flower_indices:
        steps = summaries[fi]['steps']
        if steps:
            ax2.scatter(steps, [fi] * len(steps), label=f'F{fi}', s=30)
    ax2.set_xlabel('Step index')
    ax2.set_ylabel('Flower index')
    ax2.set_title('Visit timeline: step index vs flower')
    ax2.grid(axis='x', linestyle='--', alpha=0.4)

    # Figure 3: inter-visit interval distribution
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    if all_intervals:
        ax3.hist(all_intervals, bins=30, color='tab:blue', alpha=0.8)
        ax3.set_xlabel('Inter-visit interval (steps)')
        ax3.set_title('Distribution of inter-visit intervals (all flowers)')
    else:
        ax3.text(0.5, 0.5, 'No inter-visit intervals (too few visits)', ha='center')

    # Optional per-flower detailed plots (small multiples) - only if small number of flowers
    figs = [fig1, fig2, fig3]
    if len(flower_indices) <= 12:
        fig4, axs = plt.subplots(nrows=min(4, len(flower_indices)), ncols=1, figsize=(8, 2 * min(4, len(flower_indices))))
        if not isinstance(axs, (list, np.ndarray)):
            axs = [axs]
        for ax, fi in zip(axs, flower_indices[:len(axs)]):
            s = summaries[fi]
            ax.plot(s['steps'], np.arange(len(s['steps'])), marker='o')
            ax.set_title(f'Flower {fi}: visits over time (step indices)')
            ax.set_xlabel('Step index')
            ax.set_ylabel('Visit order')
        figs.append(fig4)

    # Save plots if requested
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(metadata.get('model_path', 'trajectory')))[0]
        for idx, fig in enumerate(figs, start=1):
            p_png = os.path.join(out_dir, f'{base}_analysis_{idx}.png')
            p_pdf = os.path.join(out_dir, f'{base}_analysis_{idx}.pdf')
            try:
                fig.savefig(p_png, bbox_inches='tight', dpi=150)
                print(f"Saved plot: {p_png}")
            except Exception:
                print(f"Failed to save PNG: {p_png}")
            try:
                fig.savefig(p_pdf, bbox_inches='tight', dpi=150)
                print(f"Saved plot: {p_pdf}")
            except Exception:
                print(f"Failed to save PDF: {p_pdf}")

    if show:
        plt.show()
    else:
        plt.close('all')


def plot_spatial_from_metadata(metadata, visits, out_path=None, show=True):
    """Create a 3D spatial plot when trajectory and flower_positions are present in metadata."""
    traj = metadata.get('trajectory', None)
    flowers = metadata.get('flower_positions', None)
    if not traj or not flowers:
        print('No trajectory or flower_positions found in metadata; skipping spatial plot.')
        return

    traj = np.asarray(traj)
    flowers = np.asarray(flowers)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # plot trajectory (thin gray)
    try:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='0.6', linewidth=1.0, alpha=0.8, label='trajectory')
    except Exception:
        pass

    # plot flowers
    try:
        ax.scatter(flowers[:, 0], flowers[:, 1], flowers[:, 2], s=150, c='tab:green', marker='X', label='flowers', edgecolor='k')
    except Exception:
        pass

    # plot visit points colored by time
    all_steps = []
    all_coords = []
    for fi, steps in visits.items():
        for s in steps:
            sidx = int(s)
            if sidx < len(traj):
                all_steps.append(sidx)
                all_coords.append(traj[sidx])

    if all_coords:
        coords = np.vstack(all_coords)
        sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=all_steps, cmap='viridis', s=40, marker='o')
        cbar = fig.colorbar(sc, ax=ax, pad=0.1)
        cbar.set_label('step index (time)')

    # annotate counts next to flowers
    try:
        counts = {int(k): len(v) for k, v in visits.items()}
        for i, pos in enumerate(flowers):
            # draw a clear numbered label above each flower (3D)
            ax.text(pos[0], pos[1], pos[2] + 0.15, str(i), fontsize=9, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=0.9))
            # also show visit count slightly above the number
            ax.text(pos[0], pos[1], pos[2] + 0.05, f"({counts.get(i,0)})", fontsize=8, ha='center', va='bottom')
    except Exception:
        pass

    ax.set_title(metadata.get('model', 'trajectory'))
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved spatial figure: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_report_pdf(metadata, summaries, all_intervals, visits, out_path):
    """Create a publication-ready multi-panel PDF combining spatial and quantitative plots."""
    traj = metadata.get('trajectory', None)
    flowers = metadata.get('flower_positions', None)
    if traj is None or flowers is None:
        raise ValueError('metadata must contain "trajectory" and "flower_positions" for report generation')

    traj = np.asarray(traj)
    flowers = np.asarray(flowers)

    # Create figure with GridSpec
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.gridspec as gridspec

    import matplotlib.patches as mpatches
    with PdfPages(out_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))  # landscape letter-like
        gs = gridspec.GridSpec(3, 4, figure=fig, width_ratios=[1, 1, 1, 1], height_ratios=[1, 0.8, 0.9])

        # 3D spatial panel (left-top, spans 2 rows)
        ax3d = fig.add_subplot(gs[0:2, 0:2], projection='3d')
        ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='0.6', linewidth=1.0, alpha=0.9)
        ax3d.scatter(flowers[:, 0], flowers[:, 1], flowers[:, 2], s=120, c='tab:green', marker='X', edgecolor='k')
        ax3d.set_title('3D Trajectory and Flowers')
        ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')

    # Top-down XY projection (right-top, spans 2 columns)
        ax_xy = fig.add_subplot(gs[0:2, 2:4])
        ax_xy.plot(traj[:, 0], traj[:, 1], color='0.5', linewidth=1.2, alpha=0.9)
        ax_xy.scatter(flowers[:, 0], flowers[:, 1], s=140, c='tab:green', marker='X', edgecolor='k')
        # Plot visits as colored by time
        all_steps = []
        all_coords_xy = []
        for fi, steps in visits.items():
            for s in steps:
                sidx = int(s)
                if sidx < len(traj):
                    all_steps.append(sidx)
                    all_coords_xy.append(traj[sidx, :2])
        if all_coords_xy:
            coords_xy = np.vstack(all_coords_xy)
            sc = ax_xy.scatter(coords_xy[:, 0], coords_xy[:, 1], c=all_steps, cmap='viridis', s=40)
            cbar = fig.colorbar(sc, ax=ax_xy, fraction=0.046, pad=0.04)
            cbar.set_label('step index')

        # Draw collision circles around flowers (top-down) and dashed visit lines for clarity
        collision_radius = float(metadata.get('collision_radius', 1.2))
        try:
            for fi, fpos in enumerate(flowers):
                circ = mpatches.Circle((fpos[0], fpos[1]), collision_radius, fill=False, edgecolor='C3', linestyle='--', alpha=0.6)
                ax_xy.add_patch(circ)
        except Exception:
            pass

        # Draw dashed lines from each visit point to its flower (in XY)
        try:
            for fi, steps in visits.items():
                fidx = int(fi)
                fxy = flowers[fidx, :2]
                for s in steps:
                    sidx = int(s)
                    if sidx < len(traj):
                        vxy = traj[sidx, :2]
                        ax_xy.plot([fxy[0], vxy[0]], [fxy[1], vxy[1]], color='C3', linestyle='--', linewidth=0.6, alpha=0.5)
        except Exception:
            pass

        # Annotate each flower with visit count
        try:
            counts = {int(k): len(v) for k, v in visits.items()}
            for i, pos in enumerate(flowers):
                # numbered label (bright bbox) for easy identification on the top-down map
                ax_xy.annotate(str(i), xy=(pos[0], pos[1]), xytext=(0, 6), textcoords='offset points',
                               ha='center', va='bottom', fontsize=9, weight='bold', color='black',
                               bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=0.9))
                # small visit-count label below the number
                ax_xy.annotate(f"({counts.get(i,0)})", xy=(pos[0], pos[1]), xytext=(0, -10), textcoords='offset points',
                               ha='center', va='top', fontsize=8, color='black')
        except Exception:
            pass

        # Ensure equal aspect so circles look round
        try:
            ax_xy.set_aspect('equal', adjustable='box')
        except Exception:
            pass
        ax_xy.set_title('Top-down XY projection')
        ax_xy.set_xlabel('X'); ax_xy.set_ylabel('Y')

        # Bar chart: visits per flower (bottom-left)
        ax_bar = fig.add_subplot(gs[2, 0:2])
        flower_idxs = sorted(summaries.keys())
        counts = [summaries[i]['count'] for i in flower_idxs]
        ax_bar.bar(flower_idxs, counts, color='C0')
        ax_bar.set_xlabel('Flower index'); ax_bar.set_ylabel('Visits'); ax_bar.set_title('Visits per flower')

        # Timeline scatter (bottom middle)
        ax_tl = fig.add_subplot(gs[2, 2])
        for fi in flower_idxs:
            steps = summaries[fi]['steps']
            if steps:
                ax_tl.scatter(steps, [fi] * len(steps), s=20)
        ax_tl.set_xlabel('Step index'); ax_tl.set_ylabel('Flower index'); ax_tl.set_title('Visit timeline')

        # Inter-visit histogram (bottom right)
        ax_hist = fig.add_subplot(gs[2, 3])
        if all_intervals:
            ax_hist.hist(all_intervals, bins=20, color='C2')
            ax_hist.set_xlabel('Inter-visit interval (steps)'); ax_hist.set_title('Inter-visit intervals')
        else:
            ax_hist.text(0.5, 0.5, 'No inter-visit intervals', ha='center')

    # Add a caption describing what is shown and key parameters
    caption = (f"Model: {metadata.get('model', metadata.get('model_path', 'unknown'))} — "
           f"Total visits: {sum([summaries[i]['count'] for i in flower_idxs])}, "
           f"Unique flowers visited: {len([i for i in flower_idxs if summaries[i]['count']>0])}/{len(flower_idxs)}. "
           f"Collision radius (3D): {collision_radius} units. Visits detected by 3D proximity <= radius.")
    fig.suptitle(metadata.get('model', 'Trajectory Analysis'), fontsize=14)
    fig.text(0.5, 0.01, caption, ha='center', fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)
    print(f"Saved report PDF: {out_path}")


def replay_trajectory_from_model(model_path, num_flowers, stochastic=True, max_steps=300, seed=None):
    """Load model and step the environment to reconstruct trajectory and flower positions.

    Runs without GUI (render_mode=None) and returns (trajectory_array, flower_positions_array).
    """
    # Load model
    custom_objects = {'lr_schedule': (lambda _: 0.0), 'clip_range': (lambda _: 0.2)}
    try:
        model = PPO.load(model_path, custom_objects=custom_objects)
    except Exception:
        model = PPO.load(model_path)

    env = ComplexHummingbird3DMatplotlibEnv(render_mode=None, num_flowers=int(num_flowers))
    try:
        if seed is not None:
            obs, info = env.reset(seed=int(seed))
        else:
            obs, info = env.reset()

        traj = []
        for step in range(int(max_steps)):
            try:
                action, _ = model.predict(obs, deterministic=(not stochastic))
            except Exception:
                # fallback if obs is dict
                if isinstance(obs, dict) and 'agent' in obs:
                    action, _ = model.predict(obs['agent'], deterministic=(not stochastic))
                else:
                    raise
            obs, reward, terminated, truncated, info = env.step(action)
            # try to get agent_pos from env
            pos = getattr(env, 'agent_pos', None)
            if pos is None:
                # fallback to obs
                if isinstance(obs, dict) and 'agent' in obs:
                    pos = np.asarray(obs['agent'])[:3]
                else:
                    pos = np.array([0.0, 0.0, 0.0])
            traj.append(np.asarray(pos).tolist())
            if terminated or truncated:
                break

        flower_positions = None
        try:
            if hasattr(env, 'flowers') and env.flowers is not None:
                flower_positions = np.asarray(env.flowers)[:, :3]
        except Exception:
            flower_positions = None

        return np.asarray(traj), (np.asarray(flower_positions) if flower_positions is not None else None)
    finally:
        try:
            env.close()
        except Exception:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description='Analyze flower visits from trajectory metadata JSON')
    parser.add_argument('metadata', help='Path to metadata JSON (saved by visualize_trajectory.py)')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    parser.add_argument('--save-plots', default=None, help='Directory to save analysis plots')
    parser.add_argument('--save-spatial', default=None, help='Path to save spatial 3D figure (png/pdf). If omitted, spatial figure will be shown when --show is used and data exists in metadata')
    parser.add_argument('--save-report', default=None, help='Path to save a multi-panel publication-ready PDF report (overrides --save-spatial)')
    parser.add_argument('--augment', action='store_true', help='If metadata lacks trajectory/flower_positions, replay the model to reconstruct them and optionally save back to JSON')
    args = parser.parse_args(argv)

    if not os.path.exists(args.metadata):
        print(f"Metadata file not found: {args.metadata}")
        sys.exit(1)

    meta = load_metadata(args.metadata)
    flower_visits = meta.get('flower_visits', {})
    if not flower_visits:
        print('No flower_visits found in metadata. Did you run visualize_trajectory with visit logging?')
        sys.exit(1)

    # If trajectory/flower_positions missing, optionally reconstruct by replaying the model
    if ('trajectory' not in meta or not meta.get('flower_positions')):
        print('trajectory or flower_positions missing from metadata.')
        # try to reconstruct if model_path present
        model_path = meta.get('model_path')
        if model_path and os.path.exists(model_path):
            print('Attempting to reconstruct trajectory by replaying the model (no GUI)...')
            try:
                recon_traj, recon_flowers = replay_trajectory_from_model(model_path, int(meta.get('num_flowers', 5)),
                                                                          stochastic=bool(meta.get('stochastic', True)),
                                                                          max_steps=int(meta.get('steps', 300)),
                                                                          seed=meta.get('seed'))
                meta['trajectory'] = recon_traj.tolist()
                meta['flower_positions'] = recon_flowers.tolist() if recon_flowers is not None else []
                print(f"Reconstructed trajectory ({len(recon_traj)} steps) and {0 if recon_flowers is None else len(recon_flowers)} flowers.")
                if args.augment:
                    # write back to JSON
                    with open(args.metadata, 'w', encoding='utf-8') as f:
                        json.dump(meta, f, indent=2)
                    print(f"Augmented metadata saved to: {args.metadata}")
            except Exception as e:
                print(f"Failed to reconstruct trajectory: {e}")
        else:
            print('No valid model_path found in metadata to reconstruct trajectory.')

    summaries, total_visits, visited_flowers, all_intervals = summarize_flower_visits(flower_visits)
    print(f"Loaded metadata for model: {meta.get('model', meta.get('model_path', 'unknown'))}")
    print_summary_table(summaries, total_visits, visited_flowers)

    plot_stats(summaries, all_intervals, meta, out_dir=args.save_plots, show=args.show)

    # Spatial plot if requested or if metadata contains trajectory and flower positions
    if args.save_spatial:
        plot_spatial_from_metadata(meta, flower_visits, out_path=args.save_spatial, show=False)
    elif args.show:
        # show spatial if data present
        plot_spatial_from_metadata(meta, flower_visits, out_path=None, show=True)

    # Generate a multi-panel publication-ready PDF if requested
    if args.save_report:
        try:
            plot_report_pdf(meta, summaries, all_intervals, flower_visits, out_path=args.save_report)
        except Exception as e:
            print(f"Failed to generate report: {e}")


if __name__ == '__main__':
    main()
