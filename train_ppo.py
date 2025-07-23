"""
Train the HummingbirdEnv using PPO (Proximal Policy Optimization) algorithm.
PPO is a modern, stable RL algorithm that works well for most environments.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import os
import pandas as pd
from hummingbird_gymnasium import HummingbirdEnv


class TrainingProgressCallback(BaseCallback):
    """Custom callback to track training progress and statistics."""
    
    def __init__(self, eval_freq=1000, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.eval_episodes = []
        
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        # Get training statistics from the monitor logs
        if len(self.model.ep_info_buffer) > 0:
            recent_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
            recent_lengths = [ep_info['l'] for ep_info in self.model.ep_info_buffer]
            
            self.episode_rewards.extend(recent_rewards)
            self.episode_lengths.extend(recent_lengths)
            
            # Calculate success rate (episodes where reward >= 95)
            successful_episodes = sum(1 for r in recent_rewards if r >= 95)
            success_rate = successful_episodes / len(recent_rewards) * 100 if recent_rewards else 0
            self.success_rates.append(success_rate)
            
        return True


def create_env():
    """Create and wrap the environment for training."""
    return HummingbirdEnv(grid_size=10, render_mode=True)  # No rendering during training


def train_ppo():
    """Train PPO agent on the HummingbirdEnv."""
    print("Starting PPO Training...")
    
    # Create directories for saving models and logs
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create vectorized environment (helps with training stability)
    env = make_vec_env(create_env, n_envs=4)  # 4 parallel environments
    
    # Create evaluation environment
    eval_env = Monitor(HummingbirdEnv(grid_size=10, render_mode=None), "logs/eval_env")
    
    # Create custom callback to track progress
    progress_callback = TrainingProgressCallback(eval_freq=1000)
    
    # Create PPO model
    model = PPO(
        "MultiInputPolicy",  # Policy for Dict observation spaces
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./logs/"
    )
    
    # Callback to stop training when reward threshold is reached
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=95, verbose=1)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False,
        callback_on_new_best=callback_on_best
    )
    
    # Combine callbacks
    from stable_baselines3.common.callbacks import CallbackList
    callback = CallbackList([progress_callback, eval_callback])
    
    # Train the model
    print("Training PPO agent...")
    model.learn(
        total_timesteps=200000,  # Adjust based on your needs
        callback=callback,
        progress_bar=True
    )
    
    # Save the final model
    model.save("models/ppo_hummingbird_final")
    print("Training completed! Model saved to 'models/ppo_hummingbird_final.zip'")
    
    # Save training statistics
    training_stats = {
        'episode_rewards': progress_callback.episode_rewards,
        'episode_lengths': progress_callback.episode_lengths,
        'success_rates': progress_callback.success_rates
    }
    
    import pickle
    with open("models/ppo_training_stats.pkl", 'wb') as f:
        pickle.dump(training_stats, f)
    print("Training statistics saved to 'models/ppo_training_stats.pkl'")
    
    return model, training_stats


def test_trained_model(model_path="models/ppo_hummingbird_final"):
    """Test the trained PPO model with visual rendering."""
    print(f"Testing trained model: {model_path}")
    print("🎮 Opening visual window - watch the trained hummingbird!")
    
    # Load the trained model
    model = PPO.load(model_path)
    
    # Create environment for testing (with rendering)
    env = HummingbirdEnv(grid_size=10, render_mode="human")
    
    # Test for several episodes
    for episode in range(5):
        observation, info = env.reset(seed=episode)
        print(f"\n--- Test Episode {episode + 1} ---")
        print(f"🐦 Agent starts at: {observation['agent']}")
        print(f"🌺 Flower is at: {observation['flower']}")
        print("Watch the blue circle (hummingbird) find the red circle (flower)!")
        
        terminated = False
        truncated = False
        step_count = 0
        total_reward = 0
        
        while not (terminated or truncated) and step_count < 100:
            # Use trained model to predict action
            action, _states = model.predict(observation, deterministic=True)
            
            # Convert numpy array to integer (fix for PPO output)
            action = int(action)
            
            # Execute action
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Render environment (this shows the visual)
            env.render()
            
            # Add action name for better understanding
            action_names = ["⬆️ Up", "⬇️ Down", "⬅️ Left", "➡️ Right"]
            if step_count % 5 == 0:  # Print every 5 steps to avoid spam
                print(f"Step {step_count}: {action_names[action]} -> Agent at {observation['agent']}")
            
            if terminated:
                print(f"🎉 SUCCESS! Found flower in {step_count} steps with reward {total_reward}")
                print("⏳ Starting next episode in 2 seconds...")
                import time
                time.sleep(2)  # Pause between episodes
                break
        
        if step_count >= 100:
            print(f"⏰ Episode timeout after {step_count} steps with reward {total_reward}")
    
    print("\n🏁 Visual testing completed! Close the window when ready.")
    env.close()


def evaluate_model(model_path="models/ppo_hummingbird_final", n_episodes=100):
    """Evaluate the trained model over multiple episodes."""
    print(f"Evaluating model over {n_episodes} episodes...")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment (no rendering for evaluation)
    env = HummingbirdEnv(grid_size=10, render_mode=None)
    
    success_count = 0
    total_steps = 0
    total_rewards = 0
    
    for episode in range(n_episodes):
        observation, info = env.reset()
        terminated = False
        truncated = False
        step_count = 0
        episode_reward = 0
        
        while not (terminated or truncated) and step_count < 200:
            action, _states = model.predict(observation, deterministic=True)
            
            # Convert numpy array to integer (fix for PPO output)
            action = int(action)
            
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            if terminated:
                success_count += 1
                break
        
        total_steps += step_count
        total_rewards += episode_reward
        
        if (episode + 1) % 20 == 0:
            print(f"Episodes {episode + 1}/{n_episodes} - Success rate: {success_count/(episode+1)*100:.1f}%")
    
    env.close()
    
    # Print final statistics
    success_rate = success_count / n_episodes * 100
    avg_steps = total_steps / n_episodes
    avg_reward = total_rewards / n_episodes
    
    print(f"\n--- Evaluation Results ---")
    print(f"Success Rate: {success_rate:.1f}% ({success_count}/{n_episodes})")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Average Reward: {avg_reward:.1f}")


def plot_training_progress(stats_path="models/ppo_training_stats.pkl", logs_path="logs/eval_env"):
    """Plot PPO training progress."""
    print("Generating PPO training plots...")
    
    # Load training statistics
    try:
        import pickle
        with open(stats_path, 'rb') as f:
            training_stats = pickle.load(f)
        
        episode_rewards = training_stats['episode_rewards']
        episode_lengths = training_stats['episode_lengths']
        success_rates = training_stats.get('success_rates', [])
        
    except FileNotFoundError:
        print(f"Training statistics not found at {stats_path}")
        print("Trying to load from monitor logs...")
        
        # Try to load from monitor logs as fallback
        try:
            monitor_path = f"{logs_path}/monitor.csv"
            if os.path.exists(monitor_path):
                df = pd.read_csv(monitor_path, skiprows=1)
                episode_rewards = df['r'].tolist()
                episode_lengths = df['l'].tolist()
                success_rates = [(sum(episode_rewards[max(0, i-99):i+1]) / min(i+1, 100)) >= 95 * 100 
                               for i in range(len(episode_rewards))]
            else:
                print("No training data found. Please train the model first.")
                return
        except Exception as e:
            print(f"Error loading monitor logs: {e}")
            return
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    episodes = range(1, len(episode_rewards) + 1)
    
    # Plot 1: Episode Rewards
    ax1.plot(episodes, episode_rewards, alpha=0.7, color='blue')
    if len(episode_rewards) >= 100:
        # Add moving average
        window_size = 100
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size, len(episode_rewards) + 1), moving_avg, 
                color='red', linewidth=2, label=f'Moving Avg ({window_size})')
        ax1.legend()
    ax1.set_title('PPO Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Episode Lengths (Steps)
    ax2.plot(episodes, episode_lengths, alpha=0.7, color='green')
    if len(episode_lengths) >= 100:
        window_size = 100
        moving_avg = np.convolve(episode_lengths, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(range(window_size, len(episode_lengths) + 1), moving_avg, 
                color='red', linewidth=2, label=f'Moving Avg ({window_size})')
        ax2.legend()
    ax2.set_title('PPO Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps to Complete')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Success Rate
    if success_rates:
        ax3.plot(range(1, len(success_rates) + 1), success_rates, color='orange', linewidth=2)
        ax3.set_title('PPO Success Rate (Rolling Average)')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Success Rate (%)')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
    else:
        # Calculate success rate manually
        success_rate_manual = []
        window_size = 100
        for i in range(len(episode_rewards)):
            start_idx = max(0, i - window_size + 1)
            recent_rewards = episode_rewards[start_idx:i+1]
            success_count = sum(1 for r in recent_rewards if r >= 95)
            success_rate = success_count / len(recent_rewards) * 100
            success_rate_manual.append(success_rate)
        
        ax3.plot(episodes, success_rate_manual, color='orange', linewidth=2)
        ax3.set_title('PPO Success Rate (Rolling 100 episodes)')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Success Rate (%)')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
    
    # Plot 4: Reward Distribution
    ax4.hist(episode_rewards, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax4.axvline(np.mean(episode_rewards), color='red', linestyle='--', 
               label=f'Mean: {np.mean(episode_rewards):.1f}')
    ax4.axvline(np.median(episode_rewards), color='green', linestyle='--', 
               label=f'Median: {np.median(episode_rewards):.1f}')
    ax4.set_title('PPO Reward Distribution')
    ax4.set_xlabel('Episode Reward')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('PPO Training Progress Analysis', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('models/ppo_training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\n--- PPO Training Summary ---")
    print(f"Total Episodes: {len(episode_rewards)}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Best Reward: {np.max(episode_rewards):.2f}")
    print(f"Average Steps: {np.mean(episode_lengths):.2f}")
    print(f"Best (Minimum) Steps: {np.min(episode_lengths):.2f}")
    
    # Final success rate
    final_100_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
    final_success_rate = sum(1 for r in final_100_rewards if r >= 95) / len(final_100_rewards) * 100
    print(f"Final Success Rate (last {len(final_100_rewards)} episodes): {final_success_rate:.1f}%")
    
    print("Training progress plot saved to 'models/ppo_training_progress.png'")


def compare_with_qlearning():
    """Compare PPO and Q-Learning performance."""
    print("Comparing PPO vs Q-Learning performance...")
    
    # Load PPO stats
    try:
        import pickle
        with open("models/ppo_training_stats.pkl", 'rb') as f:
            ppo_stats = pickle.load(f)
        ppo_rewards = ppo_stats['episode_rewards']
        ppo_lengths = ppo_stats['episode_lengths']
    except FileNotFoundError:
        print("PPO training statistics not found.")
        return
    
    # Load Q-Learning stats
    try:
        from train_qlearning import QLearningAgent
        agent = QLearningAgent(grid_size=8)
        agent.load_model("models/q_learning_hummingbird.pkl")
        q_rewards = agent.episode_rewards
        q_lengths = agent.episode_steps
    except (FileNotFoundError, ImportError):
        print("Q-Learning model not found.")
        return
    
    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Reward comparison
    if len(ppo_rewards) >= 100:
        ppo_moving_avg = np.convolve(ppo_rewards, np.ones(100)/100, mode='valid')
        ax1.plot(range(100, len(ppo_rewards) + 1), ppo_moving_avg, 
                label='PPO', color='blue', linewidth=2)
    
    if len(q_rewards) >= 100:
        q_moving_avg = np.convolve(q_rewards, np.ones(100)/100, mode='valid')
        ax1.plot(range(100, len(q_rewards) + 1), q_moving_avg, 
                label='Q-Learning', color='red', linewidth=2)
    
    ax1.set_title('Reward Comparison (Moving Average)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Steps comparison
    if len(ppo_lengths) >= 100:
        ppo_steps_avg = np.convolve(ppo_lengths, np.ones(100)/100, mode='valid')
        ax2.plot(range(100, len(ppo_lengths) + 1), ppo_steps_avg, 
                label='PPO', color='blue', linewidth=2)
    
    if len(q_lengths) >= 100:
        q_steps_avg = np.convolve(q_lengths, np.ones(100)/100, mode='valid')
        ax2.plot(range(100, len(q_lengths) + 1), q_steps_avg, 
                label='Q-Learning', color='red', linewidth=2)
    
    ax2.set_title('Steps Comparison (Moving Average)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Steps')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final performance comparison
    algorithms = ['PPO', 'Q-Learning']
    final_rewards = [np.mean(ppo_rewards[-100:]), np.mean(q_rewards[-100:])]
    final_steps = [np.mean(ppo_lengths[-100:]), np.mean(q_lengths[-100:])]
    
    x_pos = np.arange(len(algorithms))
    ax3.bar(x_pos, final_rewards, color=['blue', 'red'], alpha=0.7)
    ax3.set_title('Final Performance - Average Reward')
    ax3.set_xlabel('Algorithm')
    ax3.set_ylabel('Average Reward (last 100 episodes)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(algorithms)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(final_rewards):
        ax3.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom')
    
    # Plot 4: Steps comparison
    ax4.bar(x_pos, final_steps, color=['blue', 'red'], alpha=0.7)
    ax4.set_title('Final Performance - Average Steps')
    ax4.set_xlabel('Algorithm')
    ax4.set_ylabel('Average Steps (last 100 episodes)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(algorithms)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(final_steps):
        ax4.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom')
    
    plt.suptitle('PPO vs Q-Learning Performance Comparison', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('models/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Algorithm comparison plot saved to 'models/algorithm_comparison.png'")


def main():
    """Main function to train and test PPO agent."""
    print("PPO Training for HummingbirdEnv")
    print("=" * 40)
    
    choice = input("Choose option:\n1. Train new model\n2. Test existing model\n3. Evaluate model\n4. Plot training progress\n5. Compare with Q-Learning\n6. Train and test\nEnter choice (1-6): ")
    
    if choice == "1":
        train_ppo()
    elif choice == "2":
        if os.path.exists("models/ppo_hummingbird_final.zip"):
            test_trained_model()
        else:
            print("No trained model found. Please train first (option 1).")
    elif choice == "3":
        if os.path.exists("models/ppo_hummingbird_final.zip"):
            evaluate_model()
        else:
            print("No trained model found. Please train first (option 1).")
    elif choice == "4":
        if os.path.exists("models/ppo_training_stats.pkl") or os.path.exists("logs/eval_env"):
            plot_training_progress()
        else:
            print("No training data found. Please train first (option 1).")
    elif choice == "5":
        compare_with_qlearning()
    elif choice == "6":
        model, stats = train_ppo()
        print("\nTesting the trained model...")
        test_trained_model()
        evaluate_model()
        plot_training_progress()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
