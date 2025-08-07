"""
🔬 Detailed Model Evaluation Script
------------------------------------
This script performs a rigorous, statistical evaluation of a single trained model.
It runs a model for multiple sets of episodes (e.g., 10 runs of 100 episodes)
to gather robust performance metrics and assess the consistency of the agent's strategy.

Usage:
    python detailed_evaluation.py [path_to_model.zip]

The script will output a formatted table with the following statistics:
- Mean survival rate (%)
- Standard deviation of the survival rate
- Min/Max survival rates across runs
- 95% Confidence Interval for the mean
- A qualitative measure of consistency
"""

import os
import sys
import numpy as np
import torch
from stable_baselines3 import PPO
from train import create_environment_for_model
from scipy import stats

# --- Configuration ---
NUM_RUNS = 10
EPISODES_PER_RUN = 100
SURVIVAL_THRESHOLD = 200  # Steps required to be considered a "survival"

def evaluate_survival_rate(model, env, num_episodes=100):
    """
    Evaluates the model for a given number of episodes and returns the survival rate.
    """
    episode_lengths = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
        # Use the 'steps' from the info dict if available, otherwise use the counter
        episode_lengths.append(info.get('steps', steps))
        
    survival_count = sum(1 for length in episode_lengths if length >= SURVIVAL_THRESHOLD)
    survival_rate = (survival_count / num_episodes) * 100
    return survival_rate

def main(model_path):
    """Main evaluation function."""
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        sys.exit(1)

    print("🔬 Starting Detailed Evaluation...")
    print(f"   Model: {os.path.basename(model_path)}")
    print(f"   Runs: {NUM_RUNS} x {EPISODES_PER_RUN} episodes ({NUM_RUNS * EPISODES_PER_RUN} total)")
    print("-" * 60)

    # --- Environment Creation ---
    # The create_environment_for_model function will automatically detect
    # if the model is 'autonomous' or 'legacy' based on its filename and
    # print the appropriate information.
    env = create_environment_for_model(model_path)
    
    # Load the model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PPO.load(model_path, env=env, device=device)
        print(f"   Model loaded successfully on '{device}' device.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        env.close()
        sys.exit(1)

    survival_rates = []
    print("\nRunning evaluations...")
    for i in range(NUM_RUNS):
        print(f"   Run {i + 1}/{NUM_RUNS}... ", end="", flush=True)
        rate = evaluate_survival_rate(model, env, EPISODES_PER_RUN)
        survival_rates.append(rate)
        print(f"Survival: {rate:.1f}%")

    env.close()

    # --- Statistical Analysis ---
    mean_survival = np.mean(survival_rates)
    std_survival = np.std(survival_rates)
    min_survival = np.min(survival_rates)
    max_survival = np.max(survival_rates)
    
    # 95% Confidence Interval
    if std_survival > 0 and len(survival_rates) > 1:
        ci = stats.t.interval(0.95, len(survival_rates)-1, loc=mean_survival, scale=stats.sem(survival_rates))
        ci_str = f"{ci[0]:.1f}-{ci[1]:.1f}%"
    else:
        ci_str = "N/A"

    # Consistency Score
    if std_survival < 3.0:
        consistency = "High"
    elif std_survival < 6.0:
        consistency = "Variable"
    else:
        consistency = "Low"

    # --- Display Results ---
    model_name = os.path.basename(model_path).replace('.zip', '')
    
    print("\n" + "=" * 120)
    print("🏆 DETAILED EVALUATION RESULTS")
    print("=" * 120)
    header = f"{'Rank':<5}{'Model':<45}{'Mean%':<8}{'±Std':<8}{'Min%':<8}{'Max%':<8}{'95% CI':<13}{'Consist':<9}{'Episodes':<10}"
    print(header)
    print("-" * 120)
    
    row = (f"{'1':<5}"
           f"{model_name:<45}"
           f"{mean_survival:<8.1f}"
           f"{'±' + str(round(std_survival, 1)):<8}"
           f"{min_survival:<8.1f}"
           f"{max_survival:<8.1f}"
           f"{ci_str:<13}"
           f"{consistency:<9}"
           f"{NUM_RUNS * EPISODES_PER_RUN:<10}")
    print(row)
    print("-" * 120)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detailed_evaluation.py [path_to_model.zip]")
    else:
        main(sys.argv[1])
