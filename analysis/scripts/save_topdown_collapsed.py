import json
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

def collapse_events(steps, cooldown=3):
    if not steps:
        return []
    steps = sorted(int(s) for s in steps)
    events = []
    current = [steps[0]]
    for s in steps[1:]:
        if s - current[-1] <= cooldown:
            current.append(s)
        else:
            events.append(current)
            current = [s]
    events.append(current)
    return events

def make_topdown(meta_path, out_pdf, out_png, cooldown=3, dpi_pdf=600, dpi_png=300, make_heatmap=False, heatmap_res=200):
    meta = json.loads(Path(meta_path).read_text(encoding='utf-8'))
    traj = np.asarray(meta.get('trajectory', []))
    flowers = np.asarray(meta.get('flower_positions', []))
    visits = meta.get('flower_visits', {})
    collision = float(meta.get('collision_radius', 1.2))
    model_name = meta.get('model', Path(meta_path).stem)

    # collect collapsed events
    events = []
    for i in range(len(flowers)):
        steps = visits.get(str(i), [])
        groups = collapse_events(steps, cooldown=cooldown)
        for g in groups:
            rep = int(g[0])
            rep_pos = traj[rep] if rep < len(traj) else None
            events.append({'flower': i, 'steps': g, 'rep': rep, 'rep_pos': rep_pos, 'size': len(g)})

    rep_steps = [e['rep'] for e in events if e['rep_pos'] is not None]
    rep_positions = np.array([e['rep_pos'] for e in events if e['rep_pos'] is not None]) if events else np.zeros((0,3))

    # Top-down figure (collapsed events)
    fig, ax = plt.subplots(figsize=(8,8))
    if len(traj) > 0:
        ax.plot(traj[:,0], traj[:,1], color='0.7', linewidth=0.8, alpha=0.6, label='trajectory')

    for i,(x,y,z) in enumerate(flowers):
        circ = mpatches.Circle((x,y), collision, fill=False, edgecolor='C3', linestyle='--', alpha=0.6)
        ax.add_patch(circ)
        ax.scatter([x],[y], marker='x', c='C0', s=60, linewidths=2)
        collapsed = sum(1 for ev in events if ev['flower']==i)
        ax.text(x + 0.18, y, f'feeds: {collapsed}', fontsize=9, va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    if len(rep_positions) > 0:
        cmap = plt.get_cmap('viridis')
        sc = ax.scatter(rep_positions[:,0], rep_positions[:,1], c=rep_steps, cmap=cmap, s=140, edgecolors='k', linewidth=0.6, zorder=5)
        cbar = fig.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label('rep step (time)')
        legend_handles = [
            plt.Line2D([0],[0], color='0.7', lw=1, label='trajectory'),
            plt.Line2D([0],[0], marker='x', color='w', markerfacecolor='C0', markersize=8, label='flower (X)'),
            plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='C2', markersize=8, markeredgecolor='k', label='collapsed feed event')
        ]
    else:
        legend_handles = [
            plt.Line2D([0],[0], color='0.7', lw=1, label='trajectory'),
            plt.Line2D([0],[0], marker='x', color='w', markerfacecolor='C0', markersize=8, label='flower (X)'),
        ]

    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Top-down XY projection')
    ax.legend(handles=legend_handles, loc='upper left')
    plt.tight_layout()

    with PdfPages(out_pdf) as pdf:
        pdf_inf = pdf.infodict()
        pdf_inf['Title'] = model_name
        pdf.savefig(fig, dpi=dpi_pdf)
    fig.savefig(out_png, dpi=dpi_png)
    plt.close(fig)

    print(f"Saved: {out_pdf}, {out_png}")

    # Optional heatmap
    if make_heatmap:
        if len(traj) == 0:
            print("No trajectory for heatmap.")
        else:
            xs = traj[:,0]
            ys = traj[:,1]
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            H, xedges, yedges = np.histogram2d(xs, ys, bins=heatmap_res, range=[[xmin, xmax],[ymin, ymax]])
            # try smoothing if scipy available
            try:
                from scipy.ndimage import gaussian_filter
                Hs = gaussian_filter(H, sigma=heatmap_res//100 if heatmap_res>=100 else 1)
            except Exception:
                Hs = H
            fig2, ax2 = plt.subplots(figsize=(8,8))
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax2.imshow(Hs.T, origin='lower', extent=extent, cmap='hot', aspect='equal')
            # overlay flowers and collision circles
            for i,(x,y,z) in enumerate(flowers):
                circ = mpatches.Circle((x,y), collision, fill=False, edgecolor='white', linestyle='--', alpha=0.8)
                ax2.add_patch(circ)
                ax2.scatter([x],[y], marker='x', c='white', s=60, linewidths=2)
            cbar2 = fig2.colorbar(im, ax=ax2, pad=0.01)
            cbar2.set_label('occupancy (counts)')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title(f'Occupancy heatmap ({model_name})')
            plt.tight_layout()
            heat_pdf = Path(str(out_pdf).replace('.pdf','_heatmap.pdf'))
            heat_png = Path(str(out_png).replace('.png','_heatmap.png'))
            with PdfPages(heat_pdf) as pdf:
                pdf_inf = pdf.infodict()
                pdf_inf['Title'] = model_name + ' occupancy'
                pdf.savefig(fig2, dpi=dpi_pdf)
            fig2.savefig(heat_png, dpi=dpi_png)
            plt.close(fig2)
            print(f"Saved heatmap: {heat_pdf}, {heat_png}")

    # print summary counts
    total_events = len(events)
    per_flower = {i: sum(1 for e in events if e['flower']==i) for i in range(len(flowers))}
    print(f"collision_radius={collision} cooldown={cooldown}")
    for i in range(len(flowers)):
        print(f"Flower {i}: pos={flowers[i].tolist()}  raw_frames={len(visits.get(str(i),[]))}  collapsed_events={per_flower[i]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('meta', nargs='?', default='models/trajectory_autonomous_training_1_25000k_2025-08-27_083435.json')
    parser.add_argument('--out-prefix', default=None)
    parser.add_argument('--cooldown', type=int, default=3)
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--png-dpi', type=int, default=300)
    parser.add_argument('--heatmap', action='store_true')
    parser.add_argument('--heatmap-res', type=int, default=200)
    args = parser.parse_args()

    md = Path(args.meta)
    stem = md.stem
    out_prefix = args.out_prefix or f"topdown_{stem}_collapsed"
    out_pdf = Path(f"{out_prefix}.pdf")
    out_png = Path(f"{out_prefix}.png")
    make_topdown(md, out_pdf, out_png, cooldown=args.cooldown, dpi_pdf=args.dpi, dpi_png=args.png_dpi, make_heatmap=args.heatmap, heatmap_res=args.heatmap_res)