import os
import sys
import numpy as np
import matplotlib
import torch

# Try to select a GUI backend so plt.show() will open a window when possible.
def _ensure_gui_backend():
    # Preferred backends on Windows
    preferred = ['TkAgg', 'Qt5Agg', 'Qt4Agg', 'WXAgg']
    for bk in preferred:
        try:
            matplotlib.use(bk, force=True)
            return bk
        except Exception:
            continue
    # Fall back to whatever matplotlib chose
    return matplotlib.get_backend()

backend_in_use = _ensure_gui_backend()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from stable_baselines3 import PPO
from hummingbird_env import ComplexHummingbird3DMatplotlibEnv
from datetime import datetime
import argparse
import json
import threading
import time


def visualize_and_save_trajectory(model_path, show=False, out_format='png', stochastic=False,
                                  num_flowers=None, seed=None, render_steps=False,
                                  dpi=200, timeout=None, annotate_flowers=False, open_after_save=False,
                                  plot_all_visits=False, color_by='flower', draw_visit_lines=False):
    """
    Finds and visualizes the trajectory of the first successful episode and saves the plot.

    A successful episode is defined as one where the agent survives for the environment's
    maximum number of steps. If the environment doesn't expose `MAX_STEPS`, this function
    falls back to 300 steps which matches the `ComplexHummingbird3DMatplotlibEnv` success
    threshold in `hummingbird_env.py`.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from: {model_path}")
    # Try to load with safe replacements for objects that sometimes fail to deserialize
    custom_objects = {
        'lr_schedule': (lambda _: 0.0),
        'clip_range': (lambda _: 0.2)
    }
    try:
        model = PPO.load(model_path, custom_objects=custom_objects)
    except Exception as e:
        # Fall back to a naked load to surface the original error if necessary
        try:
            model = PPO.load(model_path)
        except Exception as e2:
            print(f"Failed to load model. First attempt error: {e}; second attempt error: {e2}")
            return

    # Initialize environment with matplotlib rendering for consistent internal state
    env_kwargs = {'render_mode': 'matplotlib'}
    if num_flowers is not None:
        env_kwargs['num_flowers'] = int(num_flowers)
    env = ComplexHummingbird3DMatplotlibEnv(**env_kwargs)

    # Determine required steps for success. Use env.MAX_STEPS if present, otherwise fallback.
    target_steps = getattr(env, "MAX_STEPS", None)
    if target_steps is None:
        # The environment implements success at 300 steps in its step() method.
        target_steps = 300

    print(f"Searching for a successful episode (target steps = {target_steps})...")
    # Inform about the matplotlib backend - useful for debugging why show() may not open
    try:
        print(f"Matplotlib backend in use: {matplotlib.get_backend()}")
        if matplotlib.get_backend().lower() in ('agg', 'pdf', 'svg', 'ps'):
            print("Warning: matplotlib is using a non-GUI backend; --show will not open a window.")
    except Exception:
        pass

    try:
        while True:
            # Pass seed to reset when provided to reproduce exact episode
            if seed is not None:
                obs, info = env.reset(seed=int(seed))
            else:
                obs, info = env.reset()
            done = False
            trajectory_points = []

            # Record initial position
            try:
                trajectory_points.append(env.agent_pos.copy())
            except Exception:
                # Fallback: if agent_pos is not available, try to extract from observation
                if isinstance(obs, dict) and 'agent' in obs:
                    ap = np.asarray(obs['agent'])[:3]
                    trajectory_points.append(ap.copy())

            while True:
                # Predict action. SB3 policies may expect the raw observation that was used at training time.
                try:
                    action, _states = model.predict(obs, deterministic=(not stochastic))
                except Exception:
                    # Try using only the 'agent' portion if the model was trained on that
                    if isinstance(obs, dict) and 'agent' in obs:
                        action, _states = model.predict(obs['agent'], deterministic=(not stochastic))
                    else:
                        raise

                obs, reward, terminated, truncated, info = env.step(action)
                trajectory_points.append(env.agent_pos.copy())
                # Optionally render each step to reproduce the interactive view
                if render_steps:
                    try:
                        env.render()
                    except Exception:
                        pass

                if terminated or truncated:
                    break

            steps = getattr(env, 'steps_taken', info.get('steps', None))
            # Some envs provide 'steps' in info; ensure integer
            try:
                steps = int(steps)
            except Exception:
                steps = len(trajectory_points)

            # Successful if steps >= target_steps
            if steps >= int(target_steps) or truncated:
                print(f"Found successful episode: steps = {steps}")
                trajectory_points = np.array(trajectory_points)

                # Plotting
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')

                # Agent trajectory
                ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2],
                        label='Agent Trajectory', color='blue', linewidth=2)

                # Flowers - attempt to get positions from env
                flower_positions = None
                try:
                    # env.flowers is an array with columns [x, y, z, nectar]
                    if hasattr(env, 'flowers') and env.flowers is not None:
                        flower_positions = np.asarray(env.flowers)[:, :3]
                except Exception:
                    flower_positions = None

                if flower_positions is not None and len(flower_positions) > 0:
                    ax.scatter(flower_positions[:, 0], flower_positions[:, 1], flower_positions[:, 2],
                               label='Flowers', color='red', s=80, marker='o')

                # Mark intersections between trajectory and flowers
                try:
                    collision_radius = getattr(env, 'FLOWER_COLLISION_RADIUS', None)
                    if collision_radius is None:
                        collision_radius = 1.2

                    # Prepare storage
                    flower_visits = {int(i): [] for i in range(len(flower_positions))}
                    visit_points = []  # list of (flower_index, step_idx, point)

                    for fi, fpos in enumerate(flower_positions):
                        dists = np.linalg.norm(trajectory_points - np.asarray(fpos), axis=1)
                        hit_idxs = np.where(dists <= float(collision_radius))[0]
                        if hit_idxs.size > 0:
                            flower_visits[int(fi)] = hit_idxs.tolist()
                            if plot_all_visits:
                                for step_idx in hit_idxs.tolist():
                                    visit_points.append((fi, int(step_idx), trajectory_points[int(step_idx)]))
                            else:
                                # keep only first visit for compact view
                                visit_points.append((fi, int(hit_idxs[0]), trajectory_points[int(hit_idxs[0])]))

                    if visit_points:
                        # Decide coloring and markers
                        num_flowers = len(flower_positions)
                        cmap = plt.get_cmap('tab20')
                        # marker list for up to many flowers
                        markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'D', 'X']

                        # If coloring by time, compute normalization across trajectory steps
                        if color_by == 'time':
                            all_steps = np.array([vp[1] for vp in visit_points])
                            min_step, max_step = all_steps.min(), all_steps.max()
                            step_range = max(1, max_step - min_step)

                        # Plot each visit point individually so we can color/marker per flower or by time
                        for fi, step_idx, point in visit_points:
                            if color_by == 'flower':
                                col = cmap(fi % 20)
                            elif color_by == 'time':
                                tnorm = (step_idx - min_step) / step_range
                                col = plt.get_cmap('viridis')(tnorm)
                            else:
                                col = 'green'

                            marker = markers[fi % len(markers)]
                            ax.scatter(point[0], point[1], point[2], color=col, s=80, marker=marker,
                                       edgecolors='k', linewidths=0.6, alpha=0.9)

                            # Optionally draw a faint line from visit point to the flower for clarity
                            if draw_visit_lines:
                                try:
                                    f = flower_positions[fi]
                                    ax.plot([f[0], point[0]], [f[1], point[1]], [f[2], point[2]],
                                            color=col, linestyle='--', linewidth=0.8, alpha=0.6)
                                except Exception:
                                    pass

                        # Optionally annotate flower visit counts next to the flower marker
                        if annotate_flowers:
                            try:
                                for fi in range(len(flower_positions)):
                                    count = len(flower_visits.get(fi, []))
                                    f = flower_positions[fi]
                                    ax.text(f[0], f[1], f[2] + 0.3, f'F{fi}: {count}', fontsize=9, ha='center')
                            except Exception:
                                pass
                except Exception:
                    flower_visits = {}

                ax.set_title('Learned Hummingbird Trajectory: Successful Episode')
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
                ax.set_zlabel('Z Position')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.6)

                # Save with timestamped unique filename
                timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
                model_name = os.path.splitext(os.path.basename(model_path))[0]
                ext = out_format.lower().lstrip('.')
                save_name = f"trajectory_{model_name}_{timestamp}.{ext}"
                save_path = os.path.join(os.path.dirname(model_path) or '.', save_name)

                # Optionally show the plot interactively before saving.
                # Ensure the display is blocking so saving happens after the window is closed.
                if show:
                    try:
                        was_interactive = plt.isinteractive()
                        if was_interactive:
                            plt.ioff()

                        # Try to draw and raise the figure so it appears in front on common backends
                        try:
                            plt.draw()
                            plt.pause(0.1)
                            mgr = plt.get_current_fig_manager()
                            try:
                                # Qt
                                mgr.window.activateWindow()
                                mgr.window.raise_()
                            except Exception:
                                try:
                                    # TkAgg
                                    mgr.window.attributes("-topmost", 1)
                                    mgr.window.attributes("-topmost", 0)
                                except Exception:
                                    try:
                                        # WX
                                        mgr.window.Raise()
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                        # If a timeout is requested, start a timer that will close the figure
                        timer = None
                        if timeout is not None and timeout > 0:
                            def _close_after_timeout():
                                try:
                                    plt.close(fig)
                                except Exception:
                                    pass
                            timer = threading.Timer(timeout, _close_after_timeout)
                            timer.start()

                        # Force a blocking show so user can view/close the window (or timeout)
                        plt.show(block=True)

                        if timer is not None:
                            timer.cancel()

                        if was_interactive:
                            plt.ion()
                    except Exception:
                        pass

                # Save the figure after the optional interactive display
                try:
                    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
                    print(f"Plot saved to: {save_path}")
                except Exception as e:
                    print(f"Failed to save plot: {e}")

                # Save metadata JSON next to the image
                try:
                    meta = {
                        "model": model_name,
                        "model_path": model_path,
                        "saved_at": timestamp,
                        "steps": int(steps),
                        "seed": seed,
                        "stochastic": bool(stochastic),
                        "num_flowers": num_flowers,
                        # record detected visit indices per flower
                        "flower_visits": flower_visits if 'flower_visits' in locals() else {},
                        # save full trajectory points and flower positions so spatial plots can be recreated without replay
                        "trajectory": trajectory_points.tolist(),
                        "flower_positions": (flower_positions.tolist() if (flower_positions is not None) else []),
                        "format": ext,
                        "dpi": dpi,
                        "matplotlib_backend": matplotlib.get_backend()
                    }
                    meta_path = os.path.splitext(save_path)[0] + ".json"
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)
                    print(f"Metadata saved to: {meta_path}")
                except Exception as e:
                    print(f"Failed to save metadata: {e}")

                # Optionally open the saved file (Windows)
                if open_after_save:
                    try:
                        if os.name == 'nt':
                            os.startfile(save_path)
                        else:
                            # macOS / Linux fallback
                            import subprocess
                            if sys.platform == "darwin":
                                subprocess.run(["open", save_path])
                            else:
                                subprocess.run(["xdg-open", save_path])
                    except Exception:
                        pass

                # Close the figure and turn off interactive mode to ensure the process can exit
                try:
                    plt.close(fig)
                    plt.ioff()
                except Exception:
                    pass

                # Close environment and return immediately to stop further episodes
                try:
                    env.close()
                except Exception:
                    pass

                return
            else:
                print(f"Episode ended after {steps} steps (target {target_steps}). Retrying...")
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize trajectory of a trained PPO hummingbird model')
    parser.add_argument('model', nargs='?', help='Path to model .zip (optional). If omitted or --choose used, you can pick interactively from ./models')
    parser.add_argument('--show', action='store_true', help='Show the plot window before closing')
    parser.add_argument('--choose', action='store_true', help='Interactively choose a model from the models directory')
    parser.add_argument('--models-dir', default='models', help='Directory to search for models when choosing or resolving a model name')
    parser.add_argument('--format', choices=['png', 'pdf'], default='png', help='Output file format')
    parser.add_argument('--stochastic', action='store_true', help='Use stochastic actions (like launcher option 6)')
    parser.add_argument('--num-flowers', type=int, default=None, help='Override number of flowers in the environment')
    parser.add_argument('--seed', type=int, default=None, help='Seed the environment reset to reproduce an episode')
    parser.add_argument('--render-steps', action='store_true', help='Call env.render() on every step (reproduce launcher interactive rendering)')
    parser.add_argument('--dpi', type=int, default=200, help='DPI for saved images (or resolution for raster output)')
    parser.add_argument('--timeout', type=float, default=None, help='If --show used, auto-close the window after N seconds')
    parser.add_argument('--annotate-flowers', action='store_true', help='Annotate flower indices on the plot')
    parser.add_argument('--plot-all-visits', action='store_true', help='Plot every visit point to flowers (not only first)')
    parser.add_argument('--color-by', choices=['flower', 'time', 'none'], default='flower', help='Color visit markers by flower index or by time')
    parser.add_argument('--draw-visit-lines', action='store_true', help='Draw dashed lines from each visit point to its flower')
    parser.add_argument('--open-after-save', action='store_true', help='Open saved file after saving (uses system opener)')
    args = parser.parse_args()
    def choose_model(models_dir):
        dir_path = os.path.abspath(models_dir)
        if not os.path.isdir(dir_path):
            print(f"Models directory not found: {dir_path}")
            return None

        # Find .zip files in the directory
        candidates = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith('.zip')]
        if not candidates:
            print(f"No .zip model files found in {dir_path}")
            return None

        # Sort by modification time (newest first)
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)

        print('\nAvailable models:')
        for i, p in enumerate(candidates):
            mtime = datetime.fromtimestamp(os.path.getmtime(p)).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  [{i}] {os.path.basename(p)}    ({mtime})")

        while True:
            choice = input('\nEnter model index to use (or q to quit): ').strip()
            if choice.lower() in ('q', 'quit', 'exit'):
                return None
            try:
                idx = int(choice)
                if 0 <= idx < len(candidates):
                    return candidates[idx]
            except ValueError:
                pass
            print('Invalid selection, try again.')

    # Resolve model path
    model_path = args.model
    def choose_model_option6(models_dir):
        """Interactive chooser matching launcher option 6 behaviour.

        Lists models (alphabetically), offers a "use default (best_model.zip)" entry,
        then prompts for number of flowers (required) and returns (model_path, num_flowers).
        """
        dir_path = os.path.abspath(models_dir)
        if not os.path.isdir(dir_path):
            print(f"Models directory not found: {dir_path}")
            return None, None

        model_files = [f for f in os.listdir(dir_path) if f.endswith('.zip')]
        model_files.sort()  # alphabetical like launcher
        if not model_files:
            print(f"No trained models found in {dir_path}.")
            return None, None

        print(f"\nAvailable models ({len(model_files)} found):")
        for i, model in enumerate(model_files, 1):
            print(f"  {i}. {model}")
        print(f"  {len(model_files) + 1}. Use default (best_model.zip)")

        selected_model_path = None
        while True:
            model_choice = input(f"\nChoose model (1-{len(model_files) + 1}): ").strip()
            try:
                choice_num = int(model_choice)
                if 1 <= choice_num <= len(model_files):
                    selected_model_path = os.path.join(models_dir, model_files[choice_num - 1])
                    break
                elif choice_num == len(model_files) + 1:
                    selected_model_path = os.path.join(models_dir, 'best_model.zip')
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        # Prompt for number of flowers (required)
        num_flowers_val = None
        while True:
            num_flowers_input = input("Enter number of flowers this model was trained with (e.g., 2, 4, 5, 8, 10): ").strip()
            try:
                num_flowers_val = int(num_flowers_input)
                if num_flowers_val <= 0:
                    print("Error: Number of flowers must be a positive integer.")
                    continue
                break
            except ValueError:
                print("Error: Invalid number of flowers. Please enter a valid integer.")

        return selected_model_path, num_flowers_val

    if args.choose or not model_path:
        # Use the option-6 style chooser so behaviour matches the launcher: interactive selection + required num_flowers
        selected, nf = choose_model_option6(args.models_dir)
        if not selected:
            print('No model selected. Exiting.')
            sys.exit(1)
        model_path = selected
        # Force option-6 defaults: show interactive plot and use stochastic actions
        args.show = True
        args.stochastic = True
        args.num_flowers = nf
    else:
        # If a model argument was provided but doesn't exist, try resolving in models dir
        if not os.path.exists(model_path):
            # Try common fixes: look in models dir, append .zip, or fuzzy match
            mdir = os.path.abspath(args.models_dir)
            candidate = None
            # direct join
            try_path = os.path.join(mdir, model_path)
            if os.path.exists(try_path):
                candidate = try_path
            else:
                # try with .zip
                if not model_path.lower().endswith('.zip'):
                    try_zip = os.path.join(mdir, model_path + '.zip')
                    if os.path.exists(try_zip):
                        candidate = try_zip
                # fuzzy match by substring
                if candidate is None and os.path.isdir(mdir):
                    for f in os.listdir(mdir):
                        if f.lower().endswith('.zip') and model_path.lower() in f.lower():
                            candidate = os.path.join(mdir, f)
                            break

            if candidate:
                print(f"Resolved model to: {candidate}")
                model_path = candidate
            else:
                print(f"Model not found: {model_path}")
                print(f"Use --choose to pick interactively from {args.models_dir}")
                sys.exit(1)

    visualize_and_save_trajectory(
        model_path,
        show=args.show,
        out_format=args.format,
        stochastic=args.stochastic,
    num_flowers=args.num_flowers,
        seed=args.seed,
        render_steps=args.render_steps,
        dpi=args.dpi,
        timeout=args.timeout,
    annotate_flowers=args.annotate_flowers,
    plot_all_visits=args.plot_all_visits,
    color_by=args.color_by,
    draw_visit_lines=args.draw_visit_lines,
        open_after_save=args.open_after_save
    )
