#!/usr/bin/env python3
"""
3D Complex Hummingbird PPO Training Script with Matplotlib Visualization
"""

import os
import sys
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Import our 3D matplotlib environment
from hummingbird_env import ComplexHummingbird3DMatplotlibEnv


class Complex3DMatplotlibTrainingCallback(BaseCallback):
    """Enhanced callback for 3D matplotlib environment training with comprehensive logging."""
    
    def __init__(self, log_freq=1000, training_num=1, total_timesteps=500000, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.training_num = training_num  # Store training session number
        self.total_timesteps = total_timesteps  # Store total timesteps for progress
        self.training_stats = {
            'episodes': 0,
            'total_rewards': [],
            'episode_lengths': [],
            'energy_at_death': [],
            'nectar_collected': [],
            'survival_rates': [],
            'altitude_stats': [],
            'crash_reasons': []
        }
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.episode_nectar = 0
        self.episode_altitudes = []
        
    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals.get('rewards', [0])[0]
        self.current_episode_length += 1
        
        # Track episode statistics - get the most recent nectar count
        if 'infos' in self.locals and len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            if 'total_nectar_collected' in info:
                self.episode_nectar = info['total_nectar_collected']  # Always update to latest
            if 'altitude' in info:
                self.episode_altitudes.append(info['altitude'])
        
        # Episode ended
        if self.locals.get('dones', [False])[0]:
            self.training_stats['episodes'] += 1
            self.training_stats['total_rewards'].append(self.current_episode_reward)
            self.training_stats['episode_lengths'].append(self.current_episode_length)
            self.training_stats['nectar_collected'].append(self.episode_nectar)
            
            # Survival rate (episodes that reach 300 steps = success!)
            survival = 1 if self.current_episode_length >= 300 else 0
            self.training_stats['survival_rates'].append(survival)
            
            # Altitude statistics
            if self.episode_altitudes:
                avg_altitude = np.mean(self.episode_altitudes)
                self.training_stats['altitude_stats'].append(avg_altitude)
            
            # Determine crash reason from info
            if 'infos' in self.locals and len(self.locals['infos']) > 0:
                info = self.locals['infos'][0]
                if 'energy' in info:
                    self.training_stats['energy_at_death'].append(info['energy'])
                    if info['energy'] <= 0:
                        self.training_stats['crash_reasons'].append('energy_depletion')
                    elif info.get('altitude', 1) <= 0:
                        self.training_stats['crash_reasons'].append('ground_crash')
                    else:
                        self.training_stats['crash_reasons'].append('time_limit')
            
            # Reset episode tracking
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.episode_nectar = 0
            self.episode_altitudes = []
        
        # Periodic logging
        if self.num_timesteps % self.log_freq == 0 and self.training_stats['episodes'] > 0:
            recent_episodes = min(100, len(self.training_stats['total_rewards']))
            if recent_episodes > 0:
                recent_rewards = self.training_stats['total_rewards'][-recent_episodes:]
                recent_lengths = self.training_stats['episode_lengths'][-recent_episodes:]
                recent_survival = self.training_stats['survival_rates'][-recent_episodes:]
                recent_nectar = self.training_stats['nectar_collected'][-recent_episodes:]
                
                avg_reward = np.mean(recent_rewards)
                avg_length = np.mean(recent_lengths)
                survival_rate = np.mean(recent_survival) * 100
                avg_nectar = np.mean(recent_nectar)
                
                # Simple progress indicator with training session number and progress
                progress_percent = (self.num_timesteps / self.total_timesteps) * 100
                progress_bar = "█" * int(progress_percent // 5) + "░" * (20 - int(progress_percent // 5))
                print(f"🐦 Training #{self.training_num} | Step {self.num_timesteps:,} [{progress_bar}] {progress_percent:.0f}%")
                print(f"   📊 Reward {avg_reward:.1f} | Nectar {avg_nectar:.1f} | Survival {survival_rate:.0f}%")
        
        return True


def create_3d_matplotlib_env():
    """Create a 3D matplotlib hummingbird environment for training."""
    return ComplexHummingbird3DMatplotlibEnv(
        grid_size=10,
        num_flowers=5,
        max_energy=100,
        max_height=8,
        render_mode=None  # No rendering during training for speed
    )


def get_next_training_number():
    """Get the next training session number by checking existing models."""
    models_dir = "models"
    if not os.path.exists(models_dir):
        return 1
    
    # Look for existing training sessions
    existing_models = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
    training_numbers = []
    
    for model in existing_models:
        # Extract training numbers from filenames like "training_5_500k.zip"
        if 'training_' in model:
            try:
                parts = model.split('_')
                if len(parts) >= 2:
                    num = int(parts[1])
                    training_numbers.append(num)
            except ValueError:
                continue
    
    return max(training_numbers) + 1 if training_numbers else 1


def train_complex_3d_matplotlib_ppo(timesteps=500000, model_name=None):
    """Train PPO on the 3D matplotlib complex hummingbird environment."""
    
    # Generate training session info
    training_num = get_next_training_number()
    timesteps_label = f"{timesteps//1000}k" if timesteps >= 1000 else str(timesteps)
    
    if model_name is None:
        model_name = f"training_{training_num}_{timesteps_label}"
    
    print("🐦 3D HUMMINGBIRD TRAINING SESSION")
    print("=" * 40)
    print(f"📋 Training Session: #{training_num}")
    print(f"🎯 Target Timesteps: {timesteps:,}")
    print(f"💾 Model Name: {model_name}")
    print("=" * 40)
    
    # Create vectorized training environment (multiple parallel envs for faster training)
    n_envs = 16  # Increased from 4 to 8 for faster training
    env = make_vec_env(create_3d_matplotlib_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)  # SubprocVecEnv for true parallelism
    
    # Create evaluation environment (single env for consistent evaluation)
    eval_env = Monitor(create_3d_matplotlib_env())
    
    # PPO hyperparameters optimized for 3D complex environment with memory learning
    model = PPO(
        "MultiInputPolicy",  # Required for Dict observation spaces
        env,
        learning_rate=5e-4,  # Increased from 3e-4 for faster memory learning
        n_steps=2048 // n_envs,  # Adjust for multiple environments
        batch_size=256,  # Increased from 128 for better gradient estimates
        n_epochs=15,  # Increased from 10 for more thorough learning
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.02,  # Increased exploration in 3D space with memory
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log="./logs/",
        policy_kwargs=dict(
            net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),  # Larger networks for memory processing
            activation_fn=nn.Tanh  # Use PyTorch activation function
        ),
        verbose=0  # Reduce verbosity - no detailed logs
    )
    
    print(f"Model created with MultiInputPolicy for 3D Dict observation space")
    print(f"Using {n_envs} parallel environments for training")
    
    # Set up callbacks
    training_callback = Complex3DMatplotlibTrainingCallback(
        log_freq=25000, 
        training_num=training_num, 
        total_timesteps=timesteps
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    callbacks = [training_callback, eval_callback]
    
    # Train the model
    print("\n🚀 Starting training...")
    start_time = datetime.now()
    
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            tb_log_name="PPO_3D_matplotlib"
        )
        
        training_time = datetime.now() - start_time
        print(f"\n✅ Training completed in {training_time}")
        
        # Save the final model
        model.save(f"./models/{model_name}")
        print(f"Model saved as {model_name}")
        
        # Save training statistics
        with open(f"./models/{model_name}_training_stats.pkl", 'wb') as f:
            pickle.dump(training_callback.training_stats, f)
        print("Training statistics saved")
        
        # Generate training analysis plots
        create_training_analysis_plots(training_callback.training_stats, model_name)
        
        return model, training_callback.training_stats
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        model.save(f"./models/{model_name}_interrupted")
        return model, training_callback.training_stats


def create_training_analysis_plots(stats, model_name):
    """Create comprehensive training analysis plots for 3D environment."""
    
    if not stats['episodes'] or len(stats['total_rewards']) == 0:
        print("No training data available for plotting")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'3D Matplotlib Complex Hummingbird Training Analysis - {model_name}', fontsize=16)
    
    episodes = range(1, len(stats['total_rewards']) + 1)
    
    # 1. Rewards over time
    axes[0, 0].plot(episodes, stats['total_rewards'], alpha=0.6, color='blue')
    if len(stats['total_rewards']) > 50:
        # Add moving average
        window = min(50, len(stats['total_rewards']) // 10)
        moving_avg = np.convolve(stats['total_rewards'], np.ones(window)/window, mode='valid')
        axes[0, 0].plot(episodes[window-1:], moving_avg, color='red', linewidth=2, label=f'MA({window})')
        axes[0, 0].legend()
    
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Episode lengths (survival time)
    axes[0, 1].plot(episodes, stats['episode_lengths'], alpha=0.6, color='green')
    if len(stats['episode_lengths']) > 50:
        window = min(50, len(stats['episode_lengths']) // 10)
        moving_avg = np.convolve(stats['episode_lengths'], np.ones(window)/window, mode='valid')
        axes[0, 1].plot(episodes[window-1:], moving_avg, color='darkgreen', linewidth=2, label=f'MA({window})')
        axes[0, 1].legend()
    
    axes[0, 1].set_title('Episode Lengths (Survival Time)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps Survived')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Survival rate over time
    if stats['survival_rates']:
        window_size = min(100, len(stats['survival_rates']) // 5)
        if window_size > 0:
            survival_windows = []
            for i in range(window_size, len(stats['survival_rates']) + 1):
                window_survival = np.mean(stats['survival_rates'][i-window_size:i]) * 100
                survival_windows.append(window_survival)
            
            axes[0, 2].plot(range(window_size, len(stats['survival_rates']) + 1), 
                          survival_windows, color='purple', linewidth=2)
    
    axes[0, 2].set_title(f'Survival Rate (Moving Average, Window={window_size})')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Survival Rate (%)')
    axes[0, 2].set_ylim(0, 100)
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Nectar collection over time
    axes[1, 0].plot(episodes, stats['nectar_collected'], alpha=0.6, color='orange')
    if len(stats['nectar_collected']) > 50:
        window = min(50, len(stats['nectar_collected']) // 10)
        moving_avg = np.convolve(stats['nectar_collected'], np.ones(window)/window, mode='valid')
        axes[1, 0].plot(episodes[window-1:], moving_avg, color='darkorange', linewidth=2, label=f'MA({window})')
        axes[1, 0].legend()
    
    axes[1, 0].set_title('Nectar Collection Progress')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Total Nectar Collected')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Altitude statistics (3D specific)
    if stats['altitude_stats']:
        axes[1, 1].plot(range(len(stats['altitude_stats'])), stats['altitude_stats'], 
                       alpha=0.6, color='skyblue')
        if len(stats['altitude_stats']) > 50:
            window = min(50, len(stats['altitude_stats']) // 10)
            moving_avg = np.convolve(stats['altitude_stats'], np.ones(window)/window, mode='valid')
            axes[1, 1].plot(range(window-1, len(stats['altitude_stats'])), moving_avg, 
                           color='navy', linewidth=2, label=f'MA({window})')
            axes[1, 1].legend()
    
    axes[1, 1].set_title('Average Flight Altitude')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Average Altitude')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Crash reasons analysis
    if stats['crash_reasons']:
        crash_types = ['energy_depletion', 'ground_crash', 'time_limit']
        crash_counts = [stats['crash_reasons'].count(crash_type) for crash_type in crash_types]
        colors = ['red', 'brown', 'gray']
        
        axes[1, 2].pie(crash_counts, labels=crash_types, colors=colors, autopct='%1.1f%%')
        axes[1, 2].set_title('Crash Reasons Distribution')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'./models/{model_name}_3d_matplotlib_training_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Training analysis plot saved as {model_name}_3d_matplotlib_training_analysis.png")
    
    plt.show()


def test_trained_model_3d_matplotlib(model_path, num_episodes=10, render=True):
    """Test a trained model in the 3D matplotlib environment."""
    
    print(f"🐦 Testing trained 3D matplotlib model: {model_path}")
    
    # Load the model
    model = PPO.load(model_path)
    
    # Create test environment with rendering
    env = ComplexHummingbird3DMatplotlibEnv(
        grid_size=10,
        num_flowers=5,
        max_energy=100,
        max_height=8,
        render_mode="matplotlib" if render else None
    )
    
    episode_rewards = []
    episode_lengths = []
    nectar_totals = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        print(f"\n--- 3D Matplotlib Test Episode {episode + 1} ---")
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Get action from trained model (use stochastic like in training)
            action, _states = model.predict(obs, deterministic=False)  # Changed to False
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            if render:
                env.render()
            
            if step_count % 25 == 0:
                print(f"  Step {step_count}: Energy {info['energy']:.1f}, "
                      f"Altitude {info['altitude']:.1f}, "
                      f"Nectar {info['total_nectar_collected']:.1f}")
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        nectar_totals.append(info['total_nectar_collected'])
        
        print(f"Episode {episode + 1} completed:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {step_count} steps")
        print(f"  Final Energy: {info['energy']:.1f}")
        print(f"  Final Altitude: {info['altitude']:.1f}")
        print(f"  Nectar Collected: {info['total_nectar_collected']:.1f}")
        
        if terminated:
            if info['energy'] <= 0:
                print("  💀 Died from energy depletion")
            elif info['altitude'] <= 0:
                print("  💥 Crashed to ground")
        else:
            print("  ✅ Survived to time limit")
    
    env.close()
    
    # Summary statistics
    print(f"\n📊 3D Matplotlib Test Summary ({num_episodes} episodes):")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average Nectar: {np.mean(nectar_totals):.1f} ± {np.std(nectar_totals):.1f}")
    
    return episode_rewards, episode_lengths, nectar_totals


def evaluate_model_comprehensive(model_path, num_episodes=20, render=False):
    """Comprehensive evaluation of a model with detailed statistics and no rendering."""
    
    print(f"🔍 Comprehensive Evaluation: {model_path}")
    print(f"Running {num_episodes} episodes for detailed statistics...")
    
    # Load the model
    model = PPO.load(model_path)
    
    # Create test environment without rendering for faster evaluation
    env = ComplexHummingbird3DMatplotlibEnv(
        grid_size=10,
        num_flowers=5,
        max_energy=100,
        max_height=8,
        render_mode=None  # No rendering for comprehensive eval
    )
    
    episode_rewards = []
    episode_lengths = []
    nectar_totals = []
    final_energies = []
    survival_count = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Get action from trained model (stochastic like training)
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        nectar_totals.append(info['total_nectar_collected'])
        final_energies.append(info['energy'])
        
        if not terminated:  # Survived to time limit
            survival_count += 1
        
        # Brief progress indicator
        if (episode + 1) % 5 == 0:
            print(f"  Episodes {episode + 1}/{num_episodes} completed...")
    
    env.close()
    
    # Comprehensive statistics
    print(f"\n📊 COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print("-" * 50)
    print(f"🏆 Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"🌸 Average Nectar: {np.mean(nectar_totals):.1f} ± {np.std(nectar_totals):.1f}")
    print(f"⏱️  Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"🔋 Average Final Energy: {np.mean(final_energies):.1f} ± {np.std(final_energies):.1f}")
    print(f"💪 Survival Rate: {(survival_count / num_episodes) * 100:.1f}% ({survival_count}/{num_episodes})")
    print("-" * 50)
    print(f"📈 Best Episode Reward: {np.max(episode_rewards):.2f}")
    print(f"📉 Worst Episode Reward: {np.min(episode_rewards):.2f}")
    print(f"🌸 Max Nectar Collected: {np.max(nectar_totals):.1f}")
    print(f"⏱️  Longest Survival: {np.max(episode_lengths)} steps")
    print("=" * 50)
    
    return episode_rewards, episode_lengths, nectar_totals, survival_count


def evaluate_all_models():
    """Evaluate all available models and compare their performance."""
    
    print("🔍 EVALUATING ALL MODELS")
    print("=" * 50)
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("No models directory found.")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
    if not model_files:
        print("No trained models found.")
        return
    
    results = {}
    
    for model_file in model_files:
        model_path = f"./models/{model_file}"
        print(f"\n🤖 Evaluating: {model_file}")
        
        try:
            rewards, lengths, nectars, survival_count = evaluate_model_comprehensive(model_path, num_episodes=10, render=False)
            
            results[model_file] = {
                'avg_reward': np.mean(rewards),
                'avg_nectar': np.mean(nectars),
                'avg_length': np.mean(lengths),
                'survival_rate': (survival_count / len(lengths)) * 100
            }
        except Exception as e:
            print(f"❌ Failed to evaluate {model_file}: {e}")
            continue
    
    # Summary comparison
    print(f"\n🏆 MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Model':<35} {'Reward':<10} {'Nectar':<10} {'Length':<10} {'Survival':<10}")
    print("-" * 80)
    
    # Sort by average reward
    for model, stats in sorted(results.items(), key=lambda x: x[1]['avg_reward'], reverse=True):
        print(f"{model[:34]:<35} {stats['avg_reward']:<10.1f} {stats['avg_nectar']:<10.1f} {stats['avg_length']:<10.1f} {stats['survival_rate']:<10.1f}%")
    
    print("=" * 80)
    
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['avg_reward'])
        print(f"🥇 Best performing model: {best_model[0]} (Reward: {best_model[1]['avg_reward']:.1f})")


def main():
    """Main training script for 3D matplotlib environment."""
    if len(sys.argv) < 2:
        print("Usage: python train_complex_3d_matplotlib_ppo.py <action>")
        print("Actions:")
        print("  1 - Train new model (500K timesteps)")
        print("  2 - Train new model (1M timesteps)")
        print("  custom <timesteps> - Train new model (custom timesteps)")
        print("  3 - Test best model (3 episodes, with visualization)")
        print("  4 - Test specific model (provide path)")
        print("  5 - Comprehensive evaluation (20 episodes, no visualization)")
        print("  6 - Evaluate all models and compare")
        return
    
    action = sys.argv[1]
    
    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    if action == "1":
        train_complex_3d_matplotlib_ppo(timesteps=500000)
    elif action == "2":
        train_complex_3d_matplotlib_ppo(timesteps=1000000)
    elif action == "custom":
        if len(sys.argv) < 3:
            print("Please provide custom timesteps number")
            return
        try:
            custom_timesteps = int(sys.argv[2])
            if custom_timesteps <= 0:
                print("Error: Timesteps must be a positive number")
                return
            print(f"🎛️ Starting custom training with {custom_timesteps:,} timesteps...")
            train_complex_3d_matplotlib_ppo(timesteps=custom_timesteps)
        except ValueError:
            print("Error: Invalid timesteps number. Please provide a valid integer.")
            return
    elif action == "3":
        test_trained_model_3d_matplotlib("./models/best_model", render=True)
    elif action == "4":
        if len(sys.argv) < 3:
            print("Please provide model path")
            return
        model_path = sys.argv[2]
        test_trained_model_3d_matplotlib(model_path, render=True)
    elif action == "5":
        if len(sys.argv) < 3:
            print("Please provide model path")
            return
        model_path = sys.argv[2]
        evaluate_model_comprehensive(model_path, num_episodes=20, render=False)
    elif action == "6":
        evaluate_all_models()
    else:
        print("Invalid action. Use 1, 2, custom, 3, 4, 5, or 6.")


if __name__ == "__main__":
    main()
