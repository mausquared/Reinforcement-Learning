"""
Train the Complex Hummingbird Environment with energy and multiple flowers using PPO.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import os
import pandas as pd
from complex_hummingbird_env import ComplexHummingbirdEnv


class ComplexTrainingCallback(BaseCallback):
    """Custom callback to track complex environment statistics."""
    
    def __init__(self, eval_freq=1000, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.nectar_collected = []
        self.survival_rate = []
        self.energy_efficiency = []
        
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        # Get training statistics from episode info
        if len(self.model.ep_info_buffer) > 0:
            recent_episodes = list(self.model.ep_info_buffer)
            
            for ep_info in recent_episodes:
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
                
                # Extract custom info if available
                if 'total_nectar_collected' in ep_info:
                    self.nectar_collected.append(ep_info['total_nectar_collected'])
                    
                if 'energy' in ep_info:
                    # Calculate survival (if energy > 0 at end)
                    survived = ep_info['energy'] > 0
                    self.survival_rate.append(1.0 if survived else 0.0)
                    
                    # Energy efficiency: nectar collected per step
                    efficiency = ep_info.get('total_nectar_collected', 0) / max(ep_info['l'], 1)
                    self.energy_efficiency.append(efficiency)
        
        return True


def create_complex_env():
    """Create the complex hummingbird environment."""
    return ComplexHummingbirdEnv(
        grid_size=12,
        num_flowers=5,
        max_energy=100,
        render_mode=None
    )


def train_complex_ppo():
    """Train PPO on the complex hummingbird environment."""
    print("🐦 Training PPO on Complex Hummingbird Environment...")
    print("Features: Energy system, Multiple flowers, Hovering, Nectar collection")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create vectorized environment
    env = make_vec_env(create_complex_env, n_envs=4)
    
    # Create evaluation environment
    eval_env = Monitor(ComplexHummingbirdEnv(grid_size=12, num_flowers=5, render_mode=None))
    
    # Custom callback
    progress_callback = ComplexTrainingCallback()
    
    # Create PPO model with adjusted hyperparameters for complex environment
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=0.0003,
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
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=20000,
        deterministic=True,
        render=False
    )
    
    # Combine callbacks
    from stable_baselines3.common.callbacks import CallbackList
    callback = CallbackList([progress_callback, eval_callback])
    
    # Train the model
    print("Starting training...")
    model.learn(
        total_timesteps=500000,  # More timesteps for complex environment
        callback=callback,
        progress_bar=True
    )
    
    # Save model
    model.save("models/complex_hummingbird_ppo")
    print("Training completed! Model saved.")
    
    # Save statistics
    import pickle
    training_stats = {
        'episode_rewards': progress_callback.episode_rewards,
        'episode_lengths': progress_callback.episode_lengths,
        'nectar_collected': progress_callback.nectar_collected,
        'survival_rate': progress_callback.survival_rate,
        'energy_efficiency': progress_callback.energy_efficiency
    }
    
    with open("models/complex_training_stats.pkl", 'wb') as f:
        pickle.dump(training_stats, f)
    
    return model, training_stats


def test_complex_model(model_path="models/complex_hummingbird_ppo"):
    """Test the trained complex model."""
    print("🎮 Testing Complex Hummingbird Model...")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment with rendering
    env = ComplexHummingbirdEnv(grid_size=12, num_flowers=5, render_mode="human")
    
    for episode in range(3):
        observation, info = env.reset(seed=episode)
        print(f"\n--- Episode {episode + 1} ---")
        print(f"🐦 Agent: Energy={observation['agent'][2]:.0f}")
        print(f"🌸 Flowers: {len(observation['flowers'])} flowers available")
        
        terminated = False
        truncated = False
        step_count = 0
        
        while not (terminated or truncated) and step_count < 300:
            # Predict action
            action, _states = model.predict(observation, deterministic=True)
            action = int(action)
            
            # Execute action
            observation, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Render
            env.render()
            
            # Print status updates
            action_names = ["⬆️ Up", "⬇️ Down", "⬅️ Left", "➡️ Right", "🚁 Hover"]
            if step_count % 15 == 0:
                print(f"Step {step_count}: {action_names[action]} | "
                      f"Energy: {info['energy']:.0f} | "
                      f"Nectar: {info['total_nectar_collected']:.0f} | "
                      f"Active Flowers: {info['flowers_available']}")
            
            if terminated:
                if info['energy'] <= 0:
                    print(f"💀 Agent died from energy depletion!")
                else:
                    print(f"🎉 Episode completed successfully!")
                break
        
        print(f"Final Stats: Energy={info['energy']:.0f}, "
              f"Nectar={info['total_nectar_collected']:.0f}, "
              f"Steps={step_count}")
        
        # Pause between episodes
        import time
        time.sleep(2)
    
    env.close()


def evaluate_complex_model(model_path="models/complex_hummingbird_ppo", n_episodes=50):
    """Evaluate the complex model performance."""
    print(f"📊 Evaluating Complex Model over {n_episodes} episodes...")
    
    model = PPO.load(model_path)
    env = ComplexHummingbirdEnv(grid_size=12, num_flowers=5, render_mode=None)
    
    survival_count = 0
    total_nectar = 0
    total_steps = 0
    total_rewards = 0
    episode_nectars = []
    episode_survivals = []
    
    for episode in range(n_episodes):
        observation, info = env.reset()
        terminated = False
        truncated = False
        step_count = 0
        
        while not (terminated or truncated) and step_count < 400:
            action, _states = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(int(action))
            step_count += 1
        
        # Record statistics
        survived = info['energy'] > 0
        if survived:
            survival_count += 1
        
        episode_nectar = info['total_nectar_collected']
        total_nectar += episode_nectar
        total_steps += step_count
        
        episode_nectars.append(episode_nectar)
        episode_survivals.append(1 if survived else 0)
        
        if (episode + 1) % 10 == 0:
            current_survival_rate = survival_count / (episode + 1) * 100
            avg_nectar = total_nectar / (episode + 1)
            print(f"Episodes {episode + 1}/{n_episodes} - "
                  f"Survival: {current_survival_rate:.1f}% - "
                  f"Avg Nectar: {avg_nectar:.1f}")
    
    env.close()
    
    # Final statistics
    survival_rate = survival_count / n_episodes * 100
    avg_nectar = total_nectar / n_episodes
    avg_steps = total_steps / n_episodes
    
    print(f"\n--- Complex Environment Evaluation ---")
    print(f"Survival Rate: {survival_rate:.1f}% ({survival_count}/{n_episodes})")
    print(f"Average Nectar Collected: {avg_nectar:.1f}")
    print(f"Average Steps per Episode: {avg_steps:.1f}")
    print(f"Best Nectar Collection: {max(episode_nectars):.1f}")
    print(f"Efficiency (Nectar/Step): {avg_nectar/avg_steps:.3f}")


def plot_complex_training(stats_path="models/complex_training_stats.pkl"):
    """Plot training progress for complex environment."""
    try:
        import pickle
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
    except FileNotFoundError:
        print("Training statistics not found. Train the model first.")
        return
    
    # Create comprehensive plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    episodes = range(1, len(stats['episode_rewards']) + 1)
    
    # Plot 1: Episode Rewards
    ax1.plot(episodes, stats['episode_rewards'], alpha=0.6, color='blue')
    if len(stats['episode_rewards']) >= 100:
        moving_avg = np.convolve(stats['episode_rewards'], np.ones(100)/100, mode='valid')
        ax1.plot(range(100, len(stats['episode_rewards']) + 1), moving_avg, 
                color='red', linewidth=2, label='Moving Average (100)')
        ax1.legend()
    ax1.set_title('Complex Environment: Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Nectar Collection
    if stats['nectar_collected']:
        ax2.plot(episodes[:len(stats['nectar_collected'])], stats['nectar_collected'], 
                alpha=0.6, color='green')
        if len(stats['nectar_collected']) >= 100:
            nectar_avg = np.convolve(stats['nectar_collected'], np.ones(100)/100, mode='valid')
            ax2.plot(range(100, len(stats['nectar_collected']) + 1), nectar_avg, 
                    color='darkgreen', linewidth=2, label='Moving Average (100)')
            ax2.legend()
        ax2.set_title('Nectar Collection Over Time')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Total Nectar Collected')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Survival Rate
    if stats['survival_rate']:
        # Calculate rolling survival rate
        survival_window = 100
        survival_rolling = []
        for i in range(len(stats['survival_rate'])):
            start_idx = max(0, i - survival_window + 1)
            window_data = stats['survival_rate'][start_idx:i+1]
            survival_rolling.append(np.mean(window_data) * 100)
        
        ax3.plot(episodes[:len(survival_rolling)], survival_rolling, color='orange', linewidth=2)
        ax3.set_title('Survival Rate (Rolling 100 episodes)')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Survival Rate (%)')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Energy Efficiency
    if stats['energy_efficiency']:
        ax4.plot(episodes[:len(stats['energy_efficiency'])], stats['energy_efficiency'], 
                alpha=0.6, color='purple')
        if len(stats['energy_efficiency']) >= 100:
            eff_avg = np.convolve(stats['energy_efficiency'], np.ones(100)/100, mode='valid')
            ax4.plot(range(100, len(stats['energy_efficiency']) + 1), eff_avg, 
                    color='darkmagenta', linewidth=2, label='Moving Average (100)')
            ax4.legend()
        ax4.set_title('Energy Efficiency (Nectar per Step)')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Efficiency')
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Complex Hummingbird Environment Training Analysis', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('models/complex_training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Complex training analysis saved to 'models/complex_training_analysis.png'")


def main():
    """Main function for complex hummingbird training."""
    print("🐦 Complex Hummingbird Environment - PPO Training")
    print("=" * 60)
    print("Features:")
    print("  🔋 Energy system (hover costs more than movement)")
    print("  🌸 Multiple flowers with regenerating nectar")
    print("  ⚡ Energy management strategy required")
    print("  🎯 Survival and efficiency optimization")
    
    choice = input("\nChoose option:\n"
                  "1. Train new model\n"
                  "2. Test existing model\n"
                  "3. Evaluate model\n"
                  "4. Plot training progress\n"
                  "5. Train and test\n"
                  "6. Test environment manually\n"
                  "Enter choice (1-6): ")
    
    if choice == "1":
        train_complex_ppo()
    elif choice == "2":
        if os.path.exists("models/complex_hummingbird_ppo.zip"):
            test_complex_model()
        else:
            print("No trained model found. Please train first (option 1).")
    elif choice == "3":
        if os.path.exists("models/complex_hummingbird_ppo.zip"):
            evaluate_complex_model()
        else:
            print("No trained model found. Please train first (option 1).")
    elif choice == "4":
        plot_complex_training()
    elif choice == "5":
        model, stats = train_complex_ppo()
        print("\nTesting the trained model...")
        test_complex_model()
        evaluate_complex_model()
        plot_complex_training()
    elif choice == "6":
        from complex_hummingbird_env import test_complex_environment
        test_complex_environment()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
