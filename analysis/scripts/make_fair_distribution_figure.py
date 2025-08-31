import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hummingbird_env import ComplexHummingbird3DMatplotlibEnv


def main(outfile='./models/fair_flower_distribution_figure.png'):
    env = ComplexHummingbird3DMatplotlibEnv()
    obs, _ = env.reset()

    flowers = env.flowers.copy()  # Nx4: x,y,z,nectar
    agent = env.agent_pos.copy()

    # params
    x_mid = env.grid_size / 2
    y_mid = env.grid_size / 2
    z_mid = env.max_height / 2
    MIN_FLOWER_DISTANCE = 3.0
    MIN_AGENT_DISTANCE = 2.0

    # octant index
    octants = ((flowers[:,0] >= x_mid).astype(int) * 4 +
               (flowers[:,1] >= y_mid).astype(int) * 2 +
               (flowers[:,2] >= z_mid).astype(int))

    cmap = plt.get_cmap('tab10')
    colors = cmap(octants % 10)

    # Sample candidate points to show rejected-by-energy examples
    rng = np.random.default_rng(12345)
    samples = rng.uniform([1,1,1], [env.grid_size-1, env.grid_size-1, env.max_height-1], size=(800,3))
    rejected_energy = []
    rejected_spacing = []
    accepted_samples = []
    for s in samples:
        # agent distance
        if np.linalg.norm(s - agent) < MIN_AGENT_DISTANCE:
            continue
        # spacing
        dists = np.linalg.norm(flowers[:,:3] - s, axis=1)
        if np.any(dists < MIN_FLOWER_DISTANCE):
            rejected_spacing.append(s)
            continue
        # energy accessibility
        # replicate cost calc from env._is_energy_accessible
        manhattan = np.sum(np.abs(s - agent))
        horiz = np.sum(np.abs(s[:2] - agent[:2]))
        vert = abs(s[2] - agent[2])
        if s[2] > agent[2]:
            vert_cost = vert * env.MOVE_UP_ENERGY_COST
        else:
            vert_cost = vert * env.MOVE_DOWN_ENERGY_COST
        est_cost = horiz * env.MOVE_HORIZONTAL_COST + vert_cost + manhattan * env.METABOLIC_COST
        # accessibility threshold
        reachable = est_cost <= (env.max_energy * 0.8)
        if not reachable:
            rejected_energy.append(s)
        else:
            accepted_samples.append(s)

    rejected_energy = np.array(rejected_energy)
    rejected_spacing = np.array(rejected_spacing)
    accepted_samples = np.array(accepted_samples)

    # compute nearest neighbor distances for flowers
    N = len(flowers)
    nn = []
    for i in range(N):
        d = np.linalg.norm(flowers[:,:3] - flowers[i,:3], axis=1)
        d[i] = np.inf
        nn.append(np.min(d))
    nn = np.array(nn)

    # energy costs to flowers
    energy_costs = []
    for f in flowers:
        s = f[:3]
        manhattan = np.sum(np.abs(s - agent))
        horiz = np.sum(np.abs(s[:2] - agent[:2]))
        vert = abs(s[2] - agent[2])
        if s[2] > agent[2]:
            vert_cost = vert * env.MOVE_UP_ENERGY_COST
        else:
            vert_cost = vert * env.MOVE_DOWN_ENERGY_COST
        est_cost = horiz * env.MOVE_HORIZONTAL_COST + vert_cost + manhattan * env.METABOLIC_COST
        energy_costs.append(est_cost)
    energy_costs = np.array(energy_costs)
    reach_thresh = env.max_energy * 0.8

    # Begin plotting
    fig = plt.figure(figsize=(12,6))

    # Left: 3D spatial view
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.scatter(flowers[:,0], flowers[:,1], flowers[:,2], c=colors, s=80, edgecolors='k')
    # agent
    ax.scatter(agent[0], agent[1], agent[2], marker='*', s=200, c='k', label='Agent start')

    # draw mid planes
    xx = np.linspace(0, env.grid_size, 2)
    yy = np.linspace(0, env.grid_size, 2)
    X, Y = np.meshgrid(xx, yy)
    # x = x_mid plane
    Z = np.full_like(X, z_mid)
    # draw three planes: x_mid (vertical y-z), y_mid (vertical x-z), z_mid (horizontal x-y)
    # x_mid plane (as vertical rectangle)
    x_plane = np.array([[x_mid, x_mid],[x_mid, x_mid]])
    y_plane = np.array([[0, env.grid_size],[0, env.grid_size]])
    z_plane = np.array([[0,0],[env.max_height, env.max_height]])
    ax.plot_surface(x_plane, y_plane, z_plane, color='gray', alpha=0.08)
    # y_mid plane
    ax.plot_surface(y_plane, x_plane, z_plane, color='gray', alpha=0.08)
    # z_mid (horizontal)
    ax.plot_surface(X, Y, np.full_like(X, z_mid), color='gray', alpha=0.06)

    # projected spacing circles on XY plane at each flower (shows min spacing)
    for f in flowers:
        circ = plt.Circle((f[0], f[1]), MIN_FLOWER_DISTANCE, fill=False, color='gray', alpha=0.25)
        ax_proj = fig.add_axes([0,0,0,0])  # dummy to avoid warnings
        # use 2D projection trick: plot circle in 3D by sampling points on circle and putting z at flower z
        theta = np.linspace(0, 2*np.pi, 50)
        xs = f[0] + MIN_FLOWER_DISTANCE * np.cos(theta)
        ys = f[1] + MIN_FLOWER_DISTANCE * np.sin(theta)
        zs = np.full_like(xs, f[2])
        ax.plot(xs, ys, zs, color='gray', alpha=0.15)

    # mark rejected energy candidates (project them as red x)
    if rejected_energy.size:
        ax.scatter(rejected_energy[:,0], rejected_energy[:,1], rejected_energy[:,2], c='red', marker='x', s=30, label='Rejected (energy)')
    if rejected_spacing.size:
        ax.scatter(rejected_spacing[:,0], rejected_spacing[:,1], rejected_spacing[:,2], c='orange', marker='o', s=10, alpha=0.4, label='Rejected (spacing)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Spatial layout: octant-colored flowers, min-spacing (rings), rejected candidates')
    ax.view_init(elev=20, azim=-60)
    ax.legend(loc='upper left')

    # Right: diagnostics (stacked)
    gs = fig.add_gridspec(2,2, right=1.0, left=0.52, wspace=0.3, hspace=0.4)
    ax_cost = fig.add_subplot(gs[0,0])
    ax_nn = fig.add_subplot(gs[1,0])

    # Energy cost scatter on XY colored by estimated cost
    # build a grid for contour
    xx = np.linspace(0.5, env.grid_size-0.5, 80)
    yy = np.linspace(0.5, env.grid_size-0.5, 80)
    XX, YY = np.meshgrid(xx, yy)
    COST = np.zeros_like(XX)
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            p = np.array([XX[i,j], YY[i,j], env.agent_pos[2]])
            manhattan = np.sum(np.abs(p - agent))
            horiz = np.sum(np.abs(p[:2] - agent[:2]))
            vert = abs(p[2] - agent[2])
            vert_cost = vert * env.MOVE_DOWN_ENERGY_COST
            est = horiz * env.MOVE_HORIZONTAL_COST + vert_cost + manhattan * env.METABOLIC_COST
            COST[i,j] = est
    cs = ax_cost.contourf(XX, YY, COST, levels=30, cmap='viridis')
    fig.colorbar(cs, ax=ax_cost, label='Estimated energy cost (proxy)')
    ax_cost.scatter(flowers[:,0], flowers[:,1], c='white', edgecolors='k', s=50)
    ax_cost.scatter(agent[0], agent[1], marker='*', c='k', s=120)
    # accessible vs not
    accessible = energy_costs <= reach_thresh
    ax_cost.scatter(flowers[accessible,0], flowers[accessible,1], facecolors='none', edgecolors='g', s=150, linewidths=1.8, label='Accessible')
    ax_cost.scatter(flowers[~accessible,0], flowers[~accessible,1], marker='x', c='r', s=80, label='Inaccessible')
    ax_cost.set_title('Energy accessibility (XY projection)')
    ax_cost.set_xlabel('X')
    ax_cost.set_ylabel('Y')
    ax_cost.legend()

    # NN distance histogram and energy histogram inset
    ax_nn.hist(nn, bins=10, alpha=0.7, color='C0', label='Nearest-neighbor dist')
    ax_nn.axvline(MIN_FLOWER_DISTANCE, color='r', linestyle='--', label=f'Min spacing = {MIN_FLOWER_DISTANCE}')
    ax_nn.set_xlabel('Nearest neighbor distance')
    ax_nn.set_ylabel('Count')
    ax_nn.set_title('Spacing diagnostics')
    ax_nn.legend()

    # energy histogram to the right of nn
    ax_e = fig.add_subplot(gs[:,1])
    ax_e.hist(energy_costs, bins=10, alpha=0.7, color='C1')
    ax_e.axvline(reach_thresh, color='r', linestyle='--', label=f'Reach thresh = {reach_thresh:.1f}')
    ax_e.set_xlabel('Estimated energy cost')
    ax_e.set_ylabel('Count')
    frac = np.sum(energy_costs <= reach_thresh) / len(energy_costs)
    ax_e.set_title(f'Energy costs (accessible: {frac*100:.0f}%)')
    ax_e.legend()

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print('Saved figure to', outfile)


if __name__ == '__main__':
    main()
