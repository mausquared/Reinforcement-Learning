import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# If script is executed from the scripts/ folder, the project root may not be on sys.path.
# Insert the project root (two levels up from this file) so imports like `hummingbird_env`
# resolve regardless of current working directory.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from hummingbird_env import ComplexHummingbird3DMatplotlibEnv
except ModuleNotFoundError as e:
    # Provide a clearer error message with the attempted import location.
    raise ModuleNotFoundError(
        f"No module named 'hummingbird_env'. Tried to import from project root: {project_root}. "
        "Make sure you're running the script from the project root or that the package is available in your PYTHONPATH and virtualenv is activated."
    ) from e

os.makedirs("docs", exist_ok=True)

# Create env with bird in center and 5 flowers
env = ComplexHummingbird3DMatplotlibEnv(
    grid_size=10,
    num_flowers=5,
    max_energy=100,
    max_height=8,
    render_mode="matplotlib"  # env supports matplotlib rendering in this repo
)

# Reset environment to get initial positions/infos
obs, info = env.reset()

# Helper to try multiple possible keys for agent and flower positions
def try_get(keys, container):
    for k in keys:
        try:
            val = container.get(k) if isinstance(container, dict) else None
            if val is not None:
                return np.asarray(val)
        except Exception:
            continue
    return None

# Common fallbacks for agent position
agent_pos = None
# try info first
agent_pos = try_get(['agent_position', 'position', 'pos', 'agent'], info)
# try obs if still None
if agent_pos is None and isinstance(obs, dict):
    agent_pos = try_get(['agent', 'agent_position', 'position', 'pos'], obs)
# if still None, default to center
if agent_pos is None:
    agent_pos = np.array([env.grid_size / 2, env.grid_size / 2, env.max_height / 2]) if hasattr(env, 'grid_size') and hasattr(env, 'max_height') else np.array([5.0, 5.0, 4.0])
agent_pos = agent_pos.ravel()[:3]

# Common fallbacks for flower coordinates
flowers = None
flowers = try_get(['flower_positions', 'flowers', 'flower_coords', 'flower_locations'], info)
if flowers is None and isinstance(obs, dict):
    flowers = try_get(['flowers', 'flower_positions', 'flower_coords'], obs)
if flowers is None:
    # fallback: arrange 5 flowers around agent in a small circle
    theta = np.linspace(0, 2 * np.pi, 6)[:-1]
    radius = max(1.5, min(3.0, env.grid_size * 0.2 if hasattr(env, 'grid_size') else 2.0))
    flowers = np.stack([agent_pos[0] + radius * np.cos(theta),
                        agent_pos[1] + radius * np.sin(theta),
                        np.clip(agent_pos[2] + np.zeros_like(theta), 0, env.max_height if hasattr(env, 'max_height') else 8)], axis=1)
else:
    flowers = np.asarray(flowers)
    if flowers.ndim == 1:
        flowers = flowers.reshape(-1, 3)
    # If per-flower dict structure, try to extract coords
    if flowers.dtype == object:
        try:
            flowers = np.asarray([f.get('position') if isinstance(f, dict) else f for f in flowers])
        except Exception:
            pass

# Try to use env.render() if it returns a matplotlib Figure
saved = False
try:
    fig_or_none = env.render()
    if hasattr(fig_or_none, "savefig"):  # often plt.gcf() returned
        out_path = os.path.join("docs", "environment_snapshot.pdf")
        fig_or_none.savefig(out_path, bbox_inches="tight")
        print(f"Saved environment PDF via env.render(): {out_path}")
        saved = True
except Exception:
    pass

if not saved:
    # Manual plotting fallback
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(agent_pos[0], agent_pos[1], agent_pos[2], c="red", s=160, label="Agent (hummingbird)", depthshade=True)
    ax.scatter(flowers[:, 0], flowers[:, 1], flowers[:, 2], c="magenta", s=120, marker="^", label="Flowers")

    # axis limits
    try:
        gs = getattr(env, "grid_size", 10)
        mh = getattr(env, "max_height", 8)
        ax.set_xlim(0, gs)
        ax.set_ylim(0, gs)
        ax.set_zlim(0, mh)
    except Exception:
        ax.auto_scale_xyz([np.min(flowers[:,0].min(), agent_pos[0])-1, np.max(flowers[:,0].max(), agent_pos[0])+1],
                          [np.min(flowers[:,1].min(), agent_pos[1])-1, np.max(flowers[:,1].max(), agent_pos[1])+1],
                          [0, max(flowers[:,2].max(), agent_pos[2]) + 1])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Hummingbird Environment — bird (red) and 5 flowers (magenta)")
    ax.legend(loc="upper right")

    out_path = os.path.join("docs", "environment_snapshot.pdf")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved environment PDF: {out_path}")

env.close()