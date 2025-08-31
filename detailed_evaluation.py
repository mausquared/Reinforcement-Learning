import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from hummingbird_env import ComplexHummingbird3DMatplotlibEnv

# --- Configuration ---
NUM_RUNS = 10
EPISODES_PER_RUN = 100
SURVIVAL_THRESHOLD = 200  # Steps required to be considered a "survival"

def evaluate_model_stats(model, env, num_episodes=100):
    """
    Evaluates the model for a given number of episodes and returns detailed statistics.
    """
    all_episode_lengths = []
    all_nectar_collected = []
    
    # Reset the vectorized environment
    obs = env.reset()
    
    for i in range(num_episodes):
        terminated = [False]
        truncated = [False]
        steps = 0
        while not (terminated[0] or truncated[0]):
            action, _ = model.predict(obs, deterministic=True)
            
            # Corrected line to handle both old (4-value) and new (5-value) step returns
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, _, terminated, truncated, info = step_result
            else: # Handle older gym versions
                obs, _, done, info = step_result
                terminated = done
                truncated = [False] # Assume no truncation for older versions
                
            steps += 1
        
        info_dict = info[0]
        
        all_episode_lengths.append(info_dict.get('steps', steps))
        all_nectar_collected.append(info_dict.get('total_nectar_collected', 0))

    return all_episode_lengths, all_nectar_collected

def main(model_path):
    """Main statistical evaluation function kept for backward compatibility.
    Internally calls evaluate_one which returns a stats dict.
    """
    stats = evaluate_one(model_path, num_runs=NUM_RUNS, episodes_per_run=EPISODES_PER_RUN)
    if stats is None:
        sys.exit(1)
    return stats


def evaluate_one(model_path, num_runs=10, episodes_per_run=100):
    """Evaluate a single model and save reports/plots. Returns a stats dict or None on error."""
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return None

    print("🔬 Starting Detailed Statistical Evaluation...")
    print(f" Model: {os.path.basename(model_path)}")
    print(f" Total Episodes: {num_runs * episodes_per_run}")

    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

    obs_space = model.observation_space
    if 'flowers' in obs_space.spaces:
        num_flowers_from_model = obs_space.spaces['flowers'].shape[0]
        print(f"📊 Using environment configuration with {num_flowers_from_model} flowers from the trained model.")

        eval_env = make_vec_env(
            ComplexHummingbird3DMatplotlibEnv,
            n_envs=1,
            env_kwargs=dict(num_flowers=num_flowers_from_model)
        )
    else:
        print("⚠️ Warning: Could not detect the number of flowers from the model's observation space.")
        print("Falling back to default environment settings (5 flowers).")
        eval_env = make_vec_env(ComplexHummingbird3DMatplotlibEnv, n_envs=1)

    all_survival_rates = []
    all_episode_lengths = []
    all_nectar_collected = []

    print("\nRunning evaluation episodes...")
    for i in range(num_runs):
        print(f"🔄 Run {i+1}/{num_runs}...")
        lengths, nectar = evaluate_model_stats(model, eval_env, num_episodes=episodes_per_run)

        survival_count = sum(1 for length in lengths if length >= SURVIVAL_THRESHOLD)
        survival_rate = (survival_count / episodes_per_run) * 100

        all_survival_rates.append(survival_rate)
        all_episode_lengths.extend(lengths)
        all_nectar_collected.extend(nectar)

    eval_env.close()

    # --- Statistical Analysis ---
    mean_survival_rate = np.mean(all_survival_rates)
    std_dev_survival = np.std(all_survival_rates)
    if len(all_survival_rates) > 1:
        confidence_interval = stats.t.interval(0.95, len(all_survival_rates)-1, loc=mean_survival_rate, scale=stats.sem(all_survival_rates))
    else:
        confidence_interval = (float('nan'), float('nan'))

    mean_nectar = np.mean(all_nectar_collected) if all_nectar_collected else float('nan')
    std_dev_nectar = np.std(all_nectar_collected) if all_nectar_collected else float('nan')

    if np.std(all_episode_lengths) > 0 and np.std(all_nectar_collected) > 0:
        correlation = np.corrcoef(all_episode_lengths, all_nectar_collected)[0, 1]
    else:
        correlation = "Not applicable (no variance in data)"

    # --- Output Results ---
    print("\n" + "="*40)
    print("📊 Comprehensive Statistical Report")
    print("="*40)

    print("🎯 Survival Rate Analysis:")
    print(f"  Mean Survival Rate (@{SURVIVAL_THRESHOLD} steps): {mean_survival_rate:.2f}%")
    print(f"  Standard Deviation: {std_dev_survival:.2f}%")
    print(f"  95% Confidence Interval: ({confidence_interval[0]:.2f}%, {confidence_interval[1]:.2f}%)")

    print("\n🍯 Nectar Collection Analysis:")
    print(f"  Mean Nectar Collected: {mean_nectar:.2f}")
    print(f"  Standard Deviation: {std_dev_nectar:.2f}")

    print("\n🔗 Behavioral Correlation:")
    print(f"  Correlation (Length vs. Nectar): {correlation}")
    if isinstance(correlation, (float, int)):
        if abs(correlation) > 0.5:
            print("  This suggests a strong relationship between survival time and nectar collection.")
        elif abs(correlation) > 0.2:
            print("  This suggests a moderate relationship.")
        else:
            print("  This suggests a weak relationship.")

    base_save_path = os.path.join(os.path.dirname(model_path), f"evaluation_plots_{os.path.basename(model_path).replace('.zip', '')}")
    os.makedirs(base_save_path, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(all_episode_lengths, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Episode Lengths')
    plt.xlabel('Steps Survived')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.5)
    plt.savefig(os.path.join(base_save_path, 'episode_length_histogram.png'))
    plt.close()
    print("\n✅ Saved plot: episode_length_histogram.png")

    plt.figure(figsize=(10, 6))
    plt.hist(all_nectar_collected, bins=20, color='coral', edgecolor='black')
    plt.title('Distribution of Nectar Collected per Episode')
    plt.xlabel('Total Nectar Collected')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.5)
    plt.savefig(os.path.join(base_save_path, 'nectar_collection_histogram.png'))
    plt.close()
    print("✅ Saved plot: nectar_collection_histogram.png")

    print("\nComplete! The full statistical report and plots are saved.")

    return {
        'model': model_path,
        'mean_survival_rate': mean_survival_rate,
        'std_survival_rate': std_dev_survival,
        'ci': confidence_interval,
        'mean_nectar': mean_nectar,
        'std_nectar': std_dev_nectar,
        'correlation': correlation,
        'plots_dir': base_save_path
    }


def batch_evaluate():
    """Find requested models and run evaluate_one for each sequentially.
    Targets:
      - autonomous model matching 2500k (fallback to 25000k if 2500k not found)
      - hummingbird_model_{2,4,6,8,10}_flowers.zip
    """
    models_dir = os.path.join('.', 'models')
    if not os.path.exists(models_dir):
        print("Models directory not found.")
        return

    files = os.listdir(models_dir)

    # Find the '2500k' autonomous model; fallback to '25000k' if needed
    target_autonomous = None
    for fname in files:
        if '2500k' in fname and fname.endswith('.zip'):
            target_autonomous = os.path.join(models_dir, fname)
            break
    if target_autonomous is None:
        for fname in files:
            if '25000k' in fname and fname.endswith('.zip'):
                target_autonomous = os.path.join(models_dir, fname)
                print(f"Note: '2500k' model not found, using fallback: {fname}")
                break

    # Find hummingbird models for 2,4,6,8,10 flowers
    flower_models = []
    for n in [2, 4, 6, 8, 10]:
        pattern = f'hummingbird_model_{n}_flowers.zip'
        if pattern in files:
            flower_models.append(os.path.join(models_dir, pattern))
        else:
            print(f"Warning: expected model not found: {pattern}")

    to_run = []
    if target_autonomous:
        to_run.append(target_autonomous)
    to_run.extend(flower_models)

    if not to_run:
        print("No models found to evaluate.")
        return

    results = []
    for m in to_run:
        print(f"\n=== Evaluating: {m} ===")
        res = evaluate_one(m, num_runs=NUM_RUNS, episodes_per_run=EPISODES_PER_RUN)
        if res is not None:
            results.append(res)

    # Summary
    print("\nBatch evaluation complete. Summary:")
    for r in results:
        print(f"Model: {os.path.basename(r['model'])} | Mean survival @{SURVIVAL_THRESHOLD}: {r['mean_survival_rate']:.2f}% | Mean nectar: {r['mean_nectar']:.2f} | Plots: {r['plots_dir']}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python detailed_evaluation.py [path_to_model.zip]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    main(model_path)