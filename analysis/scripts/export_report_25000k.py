#!/usr/bin/env python3
"""Generate two report PDFs for a trajectory metadata JSON:
1) 3D trajectory and flowers
2) Top-down XY projection

Usage:
    python export_report_25000k.py <metadata.json> <out_3d.pdf> <out_topdown.pdf>

This script is headless (uses Agg) and prints a small per-flower summary.
"""
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


def load_meta(p):
    p = Path(p)
    if not p.exists():
        raise SystemExit(f"Metadata not found: {p}")
    return json.loads(p.read_text(encoding='utf-8'))


def save_3d(metadata, out_pdf):
    traj = np.asarray(metadata.get('trajectory', []))
    flowers = np.asarray(metadata.get('flower_positions', []))
    visits = metadata.get('flower_visits', {})
    collision_radius = float(metadata.get('collision_radius', 1.2))

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    if traj.size:
        ax.plot(traj[:,0], traj[:,1], traj[:,2], color='tab:blue', linewidth=1.5, alpha=0.9, label='trajectory')
    if flowers.size:
        ax.scatter(flowers[:,0], flowers[:,1], flowers[:,2], s=120, c='tab:green', marker='X', edgecolor='k', label='flowers')

    # (Intentionally omit feed/visit markers here) -- show only the agent trajectory and flower markers
    # This keeps the 3D plot focused on exploration without per-visit overlays.

    ax.set_title(metadata.get('model', metadata.get('model_path', 'trajectory')))
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved 3D PDF: {out_pdf}")


def save_topdown(metadata, out_pdf):
    traj = np.asarray(metadata.get('trajectory', []))
    flowers = np.asarray(metadata.get('flower_positions', []))
    visits = metadata.get('flower_visits', {})
    collision_radius = float(metadata.get('collision_radius', 1.2))

    fig, ax = plt.subplots(figsize=(8,8))
    if traj.size:
        ax.plot(traj[:,0], traj[:,1], color='0.5', linewidth=1.2, alpha=0.9, label='trajectory')
    if flowers.size:
        ax.scatter(flowers[:,0], flowers[:,1], s=140, c='tab:green', marker='X', edgecolor='k', label='flowers')

    # visits as colored circles
    feed_steps = []
    feed_coords = []
    for fi, steps in visits.items():
        for s in steps:
            sidx = int(s)
            if sidx < len(traj):
                feed_steps.append(sidx)
                feed_coords.append(traj[sidx,:2])
    if feed_coords:
        feed_coords = np.vstack(feed_coords)
        sc = ax.scatter(feed_coords[:,0], feed_coords[:,1], c=feed_steps, cmap='plasma', s=120, edgecolor='k', marker='o', zorder=5)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('step index')

    # collision circles
    try:
        for fpos in flowers:
            circ = mpatches.Circle((fpos[0], fpos[1]), collision_radius, fill=False, edgecolor='C3', linestyle='--', alpha=0.6)
            ax.add_patch(circ)
    except Exception:
        pass

    # annotate
    try:
        counts = {int(k): len(v) for k, v in visits.items()}
        for i, pos in enumerate(flowers):
            ax.annotate(str(i), xy=(pos[0], pos[1]), xytext=(0,8), textcoords='offset points', ha='center', va='bottom', fontsize=10, weight='bold', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=0.9))
            ax.annotate(f"feeds: {counts.get(i,0)}", xy=(pos[0], pos[1]), xytext=(8,0), textcoords='offset points', ha='left', va='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.85))
    except Exception:
        pass

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_title(f"Top-down XY projection (collision radius={collision_radius})")
    # build a clear legend describing plotted symbols
    try:
        legend_handles = []
        legend_handles.append(Line2D([0], [0], color='0.5', lw=1.2, label='trajectory'))
        legend_handles.append(Line2D([0], [0], marker='X', color='w', markerfacecolor='tab:green', markeredgecolor='k', markersize=10, label='flower (index)'))
        if len(feed_steps) > 0:
            legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='C0', markeredgecolor='k', markersize=8, label='feed (step-colored)'))
        legend_handles.append(mpatches.Patch(facecolor='none', edgecolor='C3', label=f'collision radius = {collision_radius}'))
        ax.legend(handles=legend_handles, loc='upper left', framealpha=0.9)
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved top-down PDF: {out_pdf}")


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) < 3:
        print('Usage: export_report_25000k.py metadata.json out_3d.pdf out_topdown.pdf')
        return 2
    meta_path = argv[0]
    out_3d = argv[1]
    out_top = argv[2]

    meta = load_meta(meta_path)
    visits = meta.get('flower_visits', {})
    counts = {int(k): len(v) for k, v in visits.items()}
    print('Per-flower frames-in-radius counts (raw):', counts)

    save_3d(meta, out_3d)
    save_topdown(meta, out_top)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
