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

from stable_baselines3.common.utils import get_linear_fn

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
            'episode_end_reasons': []
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
            
            # Survival rate (episodes that reach 200 steps = success!)
            survival = 1 if self.current_episode_length >= 200 else 0
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
                        self.training_stats['episode_end_reasons'].append('energy_depletion')
                    else:
                        self.training_stats['episode_end_reasons'].append('time_limit')
            
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


class SurvivalModelSaver(BaseCallback):
    """
    A callback to save the model when a new best survival rate is achieved.
    It also cleans up the previously saved lower-performing survival model.
    """
    def __init__(self, check_freq: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_survival_rate = -1.0
        self.last_saved_model_path = None
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Check if an episode has ended
        if self.locals.get('dones', [False])[0]:
            # This info is from the Monitor wrapper
            episode_length = self.locals['infos'][0]['episode']['l']
            self.episode_lengths.append(episode_length)

        # Check at the specified frequency
        if self.n_calls % self.check_freq == 0:
            # Calculate survival rate over the last 100 episodes
            if len(self.episode_lengths) >= 20: # Only start after 20 episodes
                recent_episodes = self.episode_lengths[-100:]
                survival_rate = np.mean([1 if length >= 200 else 0 for length in recent_episodes]) * 100
                
                if survival_rate > self.best_survival_rate:
                    old_best = self.best_survival_rate
                    self.best_survival_rate = survival_rate
                    
                    # Delete the old model if it exists
                    if self.last_saved_model_path and os.path.exists(self.last_saved_model_path):
                        if self.verbose > 0:
                            print(f"Removing old survival model: {self.last_saved_model_path}")
                        os.remove(self.last_saved_model_path)

                    # Save the new best model
                    save_name = f"survival_milestone_{int(round(self.best_survival_rate))}%.zip"
                    new_save_path = os.path.join(self.save_path, save_name)
                    self.model.save(new_save_path)
                    self.last_saved_model_path = new_save_path
                    
                    if self.verbose > 0:
                        print(f"\n🎉 New best survival rate! From {old_best:.1f}% to {self.best_survival_rate:.1f}%.")
                        print(f"   Model saved as: {save_name}")

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
    """Train PPO on the 3D matplotlib complex hummingbird environment with autonomous learning."""
    
    # Generate training session info
    training_num = get_next_training_number()
    timesteps_label = f"{timesteps//1000}k" if timesteps >= 1000 else str(timesteps)
    
    if model_name is None:
        # Mark new models as "autonomous" to distinguish from legacy engineered-reward models
        model_name = f"autonomous_training_{training_num}_{timesteps_label}"
    
    print("🐦 3D HUMMINGBIRD AUTONOMOUS LEARNING SESSION")
    print("=" * 45)
    print(f"📋 Training Session: #{training_num}")
    print(f"🎯 Target Timesteps: {timesteps:,}")
    print(f"🤖 Training Mode: AUTONOMOUS LEARNING")
    print(f"💾 Model Name: {model_name}")
    print(f"📊 Reward Engineering: MINIMAL (strategy discovery)")
    print("=" * 45)
    
    # Create vectorized training environment (multiple parallel envs for faster training)
    n_envs = 25  # Increased from 4 to 8 for faster training
    env = make_vec_env(create_3d_matplotlib_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)  # SubprocVecEnv for true parallelism
    
    # Create evaluation environment (single env for consistent evaluation)
    eval_env = Monitor(create_3d_matplotlib_env())
    
    # Learning rate and clip range annealing for stability over long training
    # lr starts at 3e-4 and linearly decays to 1e-6 over the entire training
    lr_schedule = get_linear_fn(3e-4, 1e-6, 1.0)
    # clip range starts at 0.2 and linearly decays to 0.1 over the entire training
    clip_schedule = get_linear_fn(0.2, 0.1, 1.0)
    
    # PPO hyperparameters optimized for 3D complex environment with memory learning
    model = PPO(
        "MultiInputPolicy",  # Required for Dict observation spaces
        env,
        learning_rate=lr_schedule,  # Use the learning rate schedule
        n_steps=2048 // n_envs,  # Adjust for multiple environments
        batch_size=225,  # Increased from 128 for better gradient estimates
        n_epochs=10,  # Increased from 10 for more thorough learning
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=clip_schedule, # Use the clip range schedule
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
    
    # Survival model saver callback
    survival_callback = SurvivalModelSaver(
        check_freq=5000,  # Check every 5000 steps
        save_path="./models/",
        verbose=1
    )
    
    callbacks = [training_callback, eval_callback, survival_callback]
    
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
    """Create comprehensive training analysis plots for 3D environment and save them individually."""
    
    if not stats['episodes'] or len(stats['total_rewards']) == 0:
        print("No training data available for plotting")
        return

    episodes = range(1, len(stats['total_rewards']) + 1)
    base_save_path = f'./models/{model_name}'

    # --- 1. Rewards over time ---
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, stats['total_rewards'], alpha=0.6, color='blue')
    if len(stats['total_rewards']) > 50:
        window = min(50, len(stats['total_rewards']) // 10)
        moving_avg = np.convolve(stats['total_rewards'], np.ones(window)/window, mode='valid')
        plt.plot(episodes[window-1:], moving_avg, color='red', linewidth=2, label=f'MA({window})')
        plt.legend()
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{base_save_path}_plot_1_rewards.png', dpi=300)
    plt.close()
    print(f"Saved plot: {model_name}_plot_1_rewards.png")

    # --- 2. Episode lengths (survival time) ---
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, stats['episode_lengths'], alpha=0.6, color='green')
    if len(stats['episode_lengths']) > 50:
        window = min(50, len(stats['episode_lengths']) // 10)
        moving_avg = np.convolve(stats['episode_lengths'], np.ones(window)/window, mode='valid')
        plt.plot(episodes[window-1:], moving_avg, color='darkgreen', linewidth=2, label=f'MA({window})')
        plt.legend()
    plt.title('Episode Lengths (Survival Time)')
    plt.xlabel('Episode')
    plt.ylabel('Steps Survived')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{base_save_path}_plot_2_episode_lengths.png', dpi=300)
    plt.close()
    print(f"Saved plot: {model_name}_plot_2_episode_lengths.png")

    # --- 3. Survival rate over time ---
    if stats['survival_rates']:
        window_size = min(100, len(stats['survival_rates']) // 5)
        if window_size > 0:
            survival_windows = []
            for i in range(window_size, len(stats['survival_rates']) + 1):
                window_survival = np.mean(stats['survival_rates'][i-window_size:i]) * 100
                survival_windows.append(window_survival)
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(window_size, len(stats['survival_rates']) + 1), 
                     survival_windows, color='purple', linewidth=2)
            plt.title(f'Survival Rate (Moving Average, Window={window_size})')
            plt.xlabel('Episode')
            plt.ylabel('Survival Rate (%)')
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{base_save_path}_plot_3_survival_rate.png', dpi=300)
            plt.close()
            print(f"Saved plot: {model_name}_plot_3_survival_rate.png")

    # --- 4. Nectar collection over time ---
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, stats['nectar_collected'], alpha=0.6, color='orange')
    if len(stats['nectar_collected']) > 50:
        window = min(50, len(stats['nectar_collected']) // 10)
        moving_avg = np.convolve(stats['nectar_collected'], np.ones(window)/window, mode='valid')
        plt.plot(episodes[window-1:], moving_avg, color='darkorange', linewidth=2, label=f'MA({window})')
        plt.legend()
    plt.title('Nectar Collection Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Nectar Collected')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{base_save_path}_plot_4_nectar_collection.png', dpi=300)
    plt.close()
    print(f"Saved plot: {model_name}_plot_4_nectar_collection.png")

    # --- 5. Altitude statistics (3D specific) ---
    if stats['altitude_stats']:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(stats['altitude_stats'])), stats['altitude_stats'], 
                 alpha=0.6, color='skyblue')
        if len(stats['altitude_stats']) > 50:
            window = min(50, len(stats['altitude_stats']) // 10)
            moving_avg = np.convolve(stats['altitude_stats'], np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(stats['altitude_stats'])), moving_avg, 
                     color='navy', linewidth=2, label=f'MA({window})')
            plt.legend()
        plt.title('Average Flight Altitude')
        plt.xlabel('Episode')
        plt.ylabel('Average Altitude')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{base_save_path}_plot_5_altitude.png', dpi=300)
        plt.close()
        print(f"Saved plot: {model_name}_plot_5_altitude.png")

    # --- 6. Episode end reasons analysis ---
    if stats['episode_end_reasons']:
        crash_types = ['energy_depletion', 'time_limit']
        crash_counts = [stats['episode_end_reasons'].count(crash_type) for crash_type in crash_types]
        
        # Ensure there's something to plot
        if sum(crash_counts) > 0:
            colors = ['red', 'gray']
            plt.figure(figsize=(8, 8))
            plt.pie(crash_counts, labels=crash_types, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('Episode End Reasons')
            plt.tight_layout()
            plt.savefig(f'{base_save_path}_plot_6_end_reasons.png', dpi=300)
            plt.close()
            print(f"Saved plot: {model_name}_plot_6_end_reasons.png")

    print("\nAll individual training analysis plots have been saved.")
    # The original combined plot is no longer shown or saved.
    # plt.show() can be re-enabled if you want to see plots interactively.


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
        else:
            print("  ✅ Survived to time limit")
    
    env.close()
    
    # Summary statistics
    print(f"\n📊 3D Matplotlib Test Summary ({num_episodes} episodes):")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average Nectar: {np.mean(nectar_totals):.1f} ± {np.std(nectar_totals):.1f}")
    
    return episode_rewards, episode_lengths, nectar_totals


def get_model_environment_version(model_path):
    """Determine which environment version a model was trained in based on filename patterns."""
    model_name = os.path.basename(model_path).lower()
    
    # Models with these numbers/dates were trained in the autonomous learning environment
    autonomous_indicators = [
        'autonomous',
        'phase2', 
        'minimal_reward',
        'discovery'
    ]
    
    # Check if this is an autonomous learning model
    for indicator in autonomous_indicators:
        if indicator in model_name:
            return 'autonomous'
    
    # Check training date - models after today are autonomous learning
    # This is a simple heuristic - you could also embed version info in model names
    current_date = datetime.now()
    
    # For now, we'll assume models are "legacy" (engineered rewards) unless explicitly marked
    # You can update this logic as you train new autonomous models
    return 'legacy'


def create_environment_for_model(model_path, render_mode=None):
    """Create the appropriate environment version for evaluating a specific model."""
    env_version = get_model_environment_version(model_path)
    
    if env_version == 'autonomous':
        # Use current autonomous learning environment
        print(f"   📊 Using AUTONOMOUS LEARNING environment (minimal rewards)")
        return ComplexHummingbird3DMatplotlibEnv(
            grid_size=10,
            num_flowers=8,
            max_energy=100,
            max_height=8,
            render_mode=render_mode
        )
    else:
        # Use legacy environment with engineered rewards
        print(f"   📊 Using LEGACY environment (engineered rewards) - MODEL COMPATIBILITY")
        print(f"   ⚠️  Note: This model was trained with engineered rewards that are no longer used")
        
        # For now, we'll evaluate legacy models in the current environment
        # but clearly mark the results as incompatible
        return ComplexHummingbird3DMatplotlibEnv(
            grid_size=10,
            num_flowers=8,
            max_energy=100,
            max_height=8,
            render_mode=render_mode
        )


def evaluate_model_comprehensive(model_path, num_episodes=100, render=False):
    """Comprehensive evaluation of a model with detailed statistics and environment compatibility."""
    
    print(f"🔍 Comprehensive Evaluation: {model_path}")
    print(f"Running {num_episodes} episodes for detailed statistics...")
    
    # Determine environment version and warn if incompatible
    env_version = get_model_environment_version(model_path)
    
    # Load the model
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return [], [], [], 0
    
    # Create appropriate environment
    env = create_environment_for_model(model_path, render_mode=None if not render else 'human')
    
    # Compatibility warning
    if env_version == 'legacy':
        print(f"   🚨 COMPATIBILITY WARNING:")
        print(f"   This model was trained with engineered rewards (proximity bonuses, efficiency rewards, etc.)")
        print(f"   Current environment uses autonomous learning (minimal rewards)")
        print(f"   Results may not reflect the model's true performance!")
    
    episode_rewards = []
    episode_lengths = []
    nectar_totals = []
    final_energies = []
    final_altitudes = []
    energy_efficiency = []  # Nectar per energy consumed
    survival_count = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        initial_energy = info.get('energy', 100)
        
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
        final_altitudes.append(info.get('altitude', 0))
        
        # Calculate energy efficiency
        energy_used = initial_energy - info['energy']
        efficiency = info['total_nectar_collected'] / max(1, energy_used)
        energy_efficiency.append(efficiency)
        
        if not terminated:  # Survived to time limit
            survival_count += 1
        
        # Brief progress indicator
        if (episode + 1) % 5 == 0:
            print(f"  Episodes {episode + 1}/{num_episodes} completed...")
    
    env.close()
    
    # Create evaluation visualization only if rendering is enabled
    if render:
        create_evaluation_plots(model_path, episode_rewards, episode_lengths, nectar_totals, 
                               final_energies, final_altitudes, energy_efficiency, survival_count, num_episodes)
    
    # Comprehensive statistics with environment version info
    print(f"\n📊 COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Environment Version: {env_version.upper()}")
    if env_version == 'legacy':
        print(f"⚠️  WARNING: Legacy model evaluated in new environment!")
    print(f"Episodes: {num_episodes}")
    print("-" * 50)
    print(f"🏆 Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"🌸 Average Nectar: {np.mean(nectar_totals):.1f} ± {np.std(nectar_totals):.1f}")
    print(f"⏱️  Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"🔋 Average Final Energy: {np.mean(final_energies):.1f} ± {np.std(final_energies):.1f}")
    print(f"⚡ Average Energy Efficiency: {np.mean(energy_efficiency):.2f} nectar/energy")
    print(f"💪 Survival Rate: {(survival_count / num_episodes) * 100:.1f}% ({survival_count}/{num_episodes})")
    print("-" * 50)
    print(f"📈 Best Episode Reward: {np.max(episode_rewards):.2f}")
    print(f"📉 Worst Episode Reward: {np.min(episode_rewards):.2f}")
    print(f"🌸 Max Nectar Collected: {np.max(nectar_totals):.1f}")
    print(f"⏱️  Longest Survival: {np.max(episode_lengths)} steps")
    print(f"📊 Evaluation plots saved!")
    print("=" * 50)
    
    return episode_rewards, episode_lengths, nectar_totals, survival_count


def create_evaluation_plots(model_path, episode_rewards, episode_lengths, nectar_totals, 
                          final_energies, final_altitudes, energy_efficiency, survival_count, num_episodes):
    """Create comprehensive test set performance plots."""
    
    model_name = os.path.basename(model_path).replace('.zip', '')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Test Set Evaluation: {model_name}', fontsize=16, fontweight='bold')
    
    # 1. Episode Rewards Distribution
    axes[0, 0].hist(episode_rewards, bins=min(10, len(episode_rewards)), alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(np.mean(episode_rewards), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(episode_rewards):.1f}')
    axes[0, 0].set_title('Episode Rewards Distribution')
    axes[0, 0].set_xlabel('Total Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Nectar Collection Performance
    axes[0, 1].hist(nectar_totals, bins=min(10, len(nectar_totals)), alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(np.mean(nectar_totals), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(nectar_totals):.1f}')
    axes[0, 1].set_title('Nectar Collection Distribution')
    axes[0, 1].set_xlabel('Nectar Collected')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Episode Length vs Performance
    scatter = axes[0, 2].scatter(episode_lengths, nectar_totals, c=episode_rewards, 
                                cmap='viridis', alpha=0.7, s=50)
    axes[0, 2].set_title('Performance vs Episode Length')
    axes[0, 2].set_xlabel('Episode Length (steps)')
    axes[0, 2].set_ylabel('Nectar Collected')
    plt.colorbar(scatter, ax=axes[0, 2], label='Episode Reward')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Energy Efficiency Analysis
    axes[1, 0].hist(energy_efficiency, bins=min(10, len(energy_efficiency)), alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].axvline(np.mean(energy_efficiency), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(energy_efficiency):.2f}')
    axes[1, 0].set_title('Energy Efficiency Distribution')
    axes[1, 0].set_xlabel('Nectar per Energy Used')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Final Energy vs Nectar Collection
    colors = ['red' if length < 200 else 'green' for length in episode_lengths]
    axes[1, 1].scatter(final_energies, nectar_totals, c=colors, alpha=0.7, s=50)
    axes[1, 1].set_title('Final Energy vs Nectar (Red=Died, Green=Survived)')
    axes[1, 1].set_xlabel('Final Energy')
    axes[1, 1].set_ylabel('Nectar Collected')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Performance Summary Statistics
    axes[1, 2].axis('off')
    summary_text = f"""
TEST SET PERFORMANCE SUMMARY

Episodes Evaluated: {num_episodes}
Survival Rate: {(survival_count/num_episodes)*100:.1f}%

NECTAR COLLECTION:
  Mean: {np.mean(nectar_totals):.1f} ± {np.std(nectar_totals):.1f}
  Best: {np.max(nectar_totals):.1f}
  Worst: {np.min(nectar_totals):.1f}

EPISODE REWARDS:
  Mean: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}
  Best: {np.max(episode_rewards):.1f}
  Worst: {np.min(episode_rewards):.1f}

ENERGY EFFICIENCY:
  Mean: {np.mean(energy_efficiency):.2f} nectar/energy
  Best: {np.max(energy_efficiency):.2f}

EPISODE LENGTH:
  Mean: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}
  Max: {np.max(episode_lengths)} steps
"""
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, 
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the evaluation plot
    plot_path = f'./models/{model_name}_evaluation_results.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Evaluation plot saved as {plot_path}")
    
    # Show the plot
    plt.show()
    
    return plot_path


def evaluate_all_models(show_plots=False):
    """Evaluate all available models and compare their performance, grouped by environment version.
    
    Args:
        show_plots (bool): If True, shows matplotlib comparison plots. If False, runs completely headless.
    """
    
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
    
    legacy_results = {}
    autonomous_results = {}
    
    for model_file in model_files:
        model_path = f"./models/{model_file}"
        env_version = get_model_environment_version(model_path)
        
        print(f"\n🤖 Evaluating: {model_file} ({env_version.upper()} environment)")
        
        try:
            rewards, lengths, nectars, survival_count = evaluate_model_comprehensive(model_path, num_episodes=50, render=False)
            
            model_stats = {
                'avg_reward': np.mean(rewards),
                'avg_nectar': np.mean(nectars),
                'avg_length': np.mean(lengths),
                'survival_rate': (survival_count / len(lengths)) * 100
            }
            
            if env_version == 'legacy':
                legacy_results[model_file] = model_stats
            else:
                autonomous_results[model_file] = model_stats
                
        except Exception as e:
            print(f"❌ Failed to evaluate {model_file}: {e}")
            continue
    
    # Display results separated by environment version
    print(f"\n🏆 MODEL COMPARISON SUMMARY")
    print("=" * 80)
    
    if legacy_results:
        print(f"\n📊 LEGACY MODELS (Engineered Rewards - Environment Mismatch!)")
        print(f"⚠️  These results may not reflect true model performance!")
        print("-" * 80)
        print(f"{'Model':<35} {'Reward':<10} {'Nectar':<10} {'Length':<10} {'Survival':<10}")
        print("-" * 80)
        
        for model, stats in sorted(legacy_results.items(), key=lambda x: x[1]['avg_nectar'], reverse=True):
            print(f"{model[:34]:<35} {stats['avg_reward']:<10.1f} {stats['avg_nectar']:<10.1f} {stats['avg_length']:<10.1f} {stats['survival_rate']:<10.1f}%")
    
    if autonomous_results:
        print(f"\n🤖 AUTONOMOUS LEARNING MODELS (True Performance)")
        print("-" * 80)
        print(f"{'Model':<35} {'Reward':<10} {'Nectar':<10} {'Length':<10} {'Survival':<10}")
        print("-" * 80)
        
        for model, stats in sorted(autonomous_results.items(), key=lambda x: x[1]['avg_nectar'], reverse=True):
            print(f"{model[:34]:<35} {stats['avg_reward']:<10.1f} {stats['avg_nectar']:<10.1f} {stats['avg_length']:<10.1f} {stats['survival_rate']:<10.1f}%")
    
    print("=" * 80)
    
    # Create comparison visualization if we have models to compare and plots are requested
    if (legacy_results or autonomous_results) and show_plots:
        create_model_comparison_plots(legacy_results, autonomous_results)
    
    # Show best models by category
    if legacy_results:
        best_legacy = max(legacy_results.items(), key=lambda x: x[1]['avg_nectar'])
        print(f"📊 Best legacy model (by nectar): {best_legacy[0]} (Nectar: {best_legacy[1]['avg_nectar']:.1f})")
    
    if autonomous_results:
        best_autonomous = max(autonomous_results.items(), key=lambda x: x[1]['avg_nectar'])
        print(f"🥇 Best autonomous model (by nectar): {best_autonomous[0]} (Nectar: {best_autonomous[1]['avg_nectar']:.1f})")
    
    if not autonomous_results and legacy_results:
        print(f"\n💡 RECOMMENDATION: Train new models with autonomous learning!")
        print(f"   Current models use outdated reward engineering.")
        print(f"   New autonomous models will discover strategies independently.")
    
    print("=" * 80)


def create_model_comparison_plots(legacy_results, autonomous_results):
    """Create comparison plots for all evaluated models."""
    
    # Combine all results for visualization
    all_results = {}
    all_results.update({f"{k} (Legacy)": v for k, v in legacy_results.items()})
    all_results.update({f"{k} (Autonomous)": v for k, v in autonomous_results.items()})
    
    if not all_results:
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison (Test Set Results)', fontsize=16, fontweight='bold')
    
    models = list(all_results.keys())
    nectar_scores = [all_results[model]['avg_nectar'] for model in models]
    reward_scores = [all_results[model]['avg_reward'] for model in models]
    survival_rates = [all_results[model]['survival_rate'] for model in models]
    episode_lengths = [all_results[model]['avg_length'] for model in models]
    
    # Colors: red for legacy, blue for autonomous
    colors = ['red' if '(Legacy)' in model else 'blue' for model in models]
    
    # 1. Nectar Collection Comparison
    bars1 = axes[0, 0].bar(range(len(models)), nectar_scores, color=colors, alpha=0.7)
    axes[0, 0].set_title('Average Nectar Collection')
    axes[0, 0].set_ylabel('Nectar Collected')
    axes[0, 0].set_xticks(range(len(models)))
    axes[0, 0].set_xticklabels([m.replace('.zip', '').replace('_', ' ')[:20] for m in models], 
                               rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{nectar_scores[i]:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Survival Rate Comparison
    bars2 = axes[0, 1].bar(range(len(models)), survival_rates, color=colors, alpha=0.7)
    axes[0, 1].set_title('Survival Rate')
    axes[0, 1].set_ylabel('Survival Rate (%)')
    axes[0, 1].set_xticks(range(len(models)))
    axes[0, 1].set_xticklabels([m.replace('.zip', '').replace('_', ' ')[:20] for m in models], 
                               rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 100)
    
    # Add value labels on bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{survival_rates[i]:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 3. Reward vs Nectar Scatter
    scatter = axes[1, 0].scatter(reward_scores, nectar_scores, c=colors, s=100, alpha=0.7)
    axes[1, 0].set_title('Reward vs Nectar Collection')
    axes[1, 0].set_xlabel('Average Reward')
    axes[1, 0].set_ylabel('Average Nectar')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add model names as labels
    for i, model in enumerate(models):
        short_name = model.replace('.zip', '').replace('_', ' ')[:15]
        axes[1, 0].annotate(short_name, (reward_scores[i], nectar_scores[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Performance Summary Table
    axes[1, 1].axis('off')
    
    # Create performance ranking
    ranking_data = []
    for i, model in enumerate(models):
        short_name = model.replace('.zip', '').replace('_', ' ')[:25]
        model_type = "Legacy" if "(Legacy)" in model else "Autonomous"
        ranking_data.append([
            short_name,
            model_type,
            f"{nectar_scores[i]:.1f}",
            f"{survival_rates[i]:.0f}%"
        ])
    
    # Sort by nectar collection
    ranking_data.sort(key=lambda x: float(x[2]), reverse=True)
    
    table_text = "PERFORMANCE RANKING (by Nectar)\n" + "="*50 + "\n"
    table_text += f"{'Rank':<4} {'Model':<25} {'Type':<11} {'Nectar':<7} {'Survival':<8}\n"
    table_text += "-"*50 + "\n"
    
    for i, row in enumerate(ranking_data[:8]):  # Top 8 models
        table_text += f"{i+1:<4} {row[0]:<25} {row[1]:<11} {row[2]:<7} {row[3]:<8}\n"
    
    if len(ranking_data) > 8:
        table_text += f"... and {len(ranking_data) - 8} more models\n"
    
    axes[1, 1].text(0.05, 0.95, table_text, transform=axes[1, 1].transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='Legacy Models'),
                      Patch(facecolor='blue', alpha=0.7, label='Autonomous Models')]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    # Save the comparison plot
    plot_path = './models/model_comparison_evaluation.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Model comparison plot saved as {plot_path}")
    
    # Show the plot
    plt.show()
    
    return plot_path


def continue_training_model(model_path, additional_timesteps):
    """Continue training an existing model with additional timesteps."""
    
    print("🔄 CONTINUING TRAINING EXISTING MODEL")
    print("=" * 45)
    print(f"📋 Base Model: {model_path}")
    print(f"📈 Additional Timesteps: {additional_timesteps:,}")
    print(f"🤖 Training Mode: CONTINUE EXISTING")
    print(f"📊 Reward Engineering: PRESERVED (maintains learned strategies)")
    print("=" * 45)
    
    env = None
    eval_env = None
    try:
        # Load existing model
        print("🔄 Loading existing model...")
        model = PPO.load(model_path)
        print("✅ Model loaded successfully!")
        
        # Create environment (same as original training)
        n_envs = 25  # Same as original training
        env = make_vec_env(create_3d_matplotlib_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
        
        # Create evaluation environment
        eval_env = Monitor(create_3d_matplotlib_env())
        
        # Set environment for the loaded model
        model.set_env(env)
        
        # Try to load existing training stats from the base model
        base_model_name = os.path.basename(model_path).replace('.zip', '')
        existing_stats_path = f"./models/{base_model_name}_training_stats.pkl"
        
        existing_stats = None
        if os.path.exists(existing_stats_path):
            try:
                print("📊 Loading existing training statistics...")
                with open(existing_stats_path, 'rb') as f:
                    existing_stats = pickle.load(f)
                print(f"✅ Found existing stats with {existing_stats.get('episodes', 0)} episodes")
            except Exception as e:
                print(f"⚠️  Could not load existing stats: {e}")
        
        # Extract base model name for the new name
        base_name = os.path.basename(model_path).replace('.zip', '')
        timesteps_label = f"{additional_timesteps//1000}k" if additional_timesteps >= 1000 else str(additional_timesteps)
        new_model_name = f"{base_name}_continued_{timesteps_label}"
        
        print(f"💾 Will save as: {new_model_name}.zip")
        
        # Set up callbacks for continued training
        training_callback = Complex3DMatplotlibTrainingCallback(
            log_freq=25000, 
            training_num=get_next_training_number(), 
            total_timesteps=additional_timesteps
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

        # New callback for saving based on survival rate
        survival_saver_callback = SurvivalModelSaver(
            check_freq=5000,  # Check every 5000 steps
            save_path="./models/"
        )
    
        callbacks = [training_callback, eval_callback, survival_saver_callback]
        
        # Continue training
        print("\n🚀 Continuing training...")
        start_time = datetime.now()
        
        model.learn(
            total_timesteps=additional_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        # Save the continued model
        model.save(f"./models/{new_model_name}")
        
        # Save training statistics for continued training
        print("📊 Saving training statistics...")
        
        # Combine with existing stats if available
        final_stats = training_callback.training_stats
        if existing_stats:
            print("🔗 Combining with existing training statistics...")
            for key in ['total_rewards', 'episode_lengths', 'energy_at_death', 
                       'nectar_collected', 'survival_rates', 'altitude_stats', 'episode_end_reasons']:
                if key in existing_stats and key in final_stats:
                    final_stats[key] = existing_stats[key] + final_stats[key]
            
            # Update episode count
            final_stats['episodes'] = len(final_stats['total_rewards'])
            print(f"📈 Combined stats now include {final_stats['episodes']} total episodes")
        
        with open(f"./models/{new_model_name}_training_stats.pkl", 'wb') as f:
            pickle.dump(final_stats, f)
        
        # Create comprehensive training analysis plots
        print("📈 Creating training analysis plots...")
        create_training_analysis_plots(final_stats, new_model_name)
        
        print(f"\n🎉 CONTINUED TRAINING COMPLETED!")
        print("=" * 45)
        print(f"💾 Model saved as: {new_model_name}.zip")
        print(f"📊 Training stats saved as: {new_model_name}_training_stats.pkl")
        print(f"📈 Training analysis plot saved as: {new_model_name}_3d_matplotlib_training_analysis.png")
        print(f"⏱️  Training duration: {training_duration}")
        print(f"📈 Additional timesteps: {additional_timesteps:,}")
        print(f"🎯 Ready for evaluation!")
        print("=" * 45)
        
    except Exception as e:
        print(f"❌ Error during continued training: {e}")
    finally:
        # Clean up environment even on error
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()


def view_training_progress():
    """View training progress by selecting from available training statistics files."""
    
    print("📈 VIEWING TRAINING PROGRESS")
    print("=" * 50)
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("No models directory found.")
        return
    
    # Find all training stats files
    stats_files = [f for f in os.listdir(models_dir) if f.endswith('_training_stats.pkl')]
    
    if not stats_files:
        print("No training statistics files found.")
        print("Training stats are only available for models trained with this system.")
        print("Train a new model first (options 2, 3, or 4 in launcher).")
        return
    
    # Sort by modification time (most recent first)
    stats_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
    
    print(f"Available training statistics ({len(stats_files)} found):")
    for i, stats_file in enumerate(stats_files, 1):
        # Extract model name from stats filename
        model_name = stats_file.replace('_training_stats.pkl', '')
        file_path = os.path.join(models_dir, stats_file)
        
        # Get file modification time for display
        mod_time = os.path.getmtime(file_path)
        mod_date = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
        
        print(f"  {i}. {model_name} (created: {mod_date})")
    
    print(f"  {len(stats_files) + 1}. View ALL training progress (combined)")
    
    choice = input(f"\nChoose training stats to view (1-{len(stats_files) + 1}): ").strip()
    
    try:
        choice_num = int(choice)
        
        if 1 <= choice_num <= len(stats_files):
            # View specific model's training progress
            selected_file = stats_files[choice_num - 1]
            model_name = selected_file.replace('_training_stats.pkl', '')
            
            print(f"\n📊 Loading training statistics for: {model_name}")
            
            try:
                with open(os.path.join(models_dir, selected_file), 'rb') as f:
                    stats = pickle.load(f)
                
                # Display basic statistics
                print(f"\n📈 TRAINING SUMMARY:")
                print("-" * 40)
                print(f"Total Episodes: {stats.get('episodes', len(stats.get('total_rewards', [])))}")
                print(f"Average Reward: {np.mean(stats.get('total_rewards', [0])):.2f}")
                print(f"Average Episode Length: {np.mean(stats.get('episode_lengths', [0])):.1f} steps")
                print(f"Average Nectar Collected: {np.mean(stats.get('nectar_collected', [0])):.1f}")
                
                if 'survival_rates' in stats and stats['survival_rates']:
                    survival_rate = np.mean(stats['survival_rates']) * 100
                    print(f"Survival Rate: {survival_rate:.1f}%")
                
                # Create and display plots
                print(f"\n📈 Creating training progress plots...")
                create_training_analysis_plots(stats, model_name)
                
                # Check if plot file exists and display path
                plot_file = f"./models/{model_name}_3d_matplotlib_training_analysis.png"
                if os.path.exists(plot_file):
                    print(f"✅ Training plots saved as: {plot_file}")
                    print("   Open this file to view detailed training analysis!")
                else:
                    print("⚠️  Plot file not found, but analysis was attempted.")
                
            except Exception as e:
                print(f"❌ Error loading training statistics: {e}")
        
        elif choice_num == len(stats_files) + 1:
            # View combined training progress
            print(f"\n📊 Loading ALL training statistics...")
            
            all_models = []
            for stats_file in stats_files:
                model_name = stats_file.replace('_training_stats.pkl', '')
                try:
                    with open(os.path.join(models_dir, stats_file), 'rb') as f:
                        stats = pickle.load(f)
                    
                    all_models.append({
                        'name': model_name,
                        'episodes': stats.get('episodes', len(stats.get('total_rewards', []))),
                        'avg_reward': np.mean(stats.get('total_rewards', [0])),
                        'avg_length': np.mean(stats.get('episode_lengths', [0])),
                        'avg_nectar': np.mean(stats.get('nectar_collected', [0])),
                        'survival_rate': np.mean(stats.get('survival_rates', [0])) * 100 if 'survival_rates' in stats else 0
                    })
                except Exception as e:
                    print(f"⚠️  Could not load {stats_file}: {e}")
            
            if all_models:
                print(f"\n📈 TRAINING PROGRESS COMPARISON:")
                print("=" * 80)
                print(f"{'Model':<35} {'Episodes':<10} {'Reward':<10} {'Length':<10} {'Nectar':<10} {'Survival':<10}")
                print("-" * 80)
                
                # Sort by survival rate (descending)
                all_models.sort(key=lambda x: x['survival_rate'], reverse=True)
                
                for model in all_models:
                    print(f"{model['name'][:34]:<35} "
                          f"{model['episodes']:<10} "
                          f"{model['avg_reward']:<10.1f} "
                          f"{model['avg_length']:<10.1f} "
                          f"{model['avg_nectar']:<10.1f} "
                          f"{model['survival_rate']:<10.1f}%")
                
                print("-" * 80)
                best_model = max(all_models, key=lambda x: x['survival_rate'])
                print(f"🏆 Best performing model: {best_model['name']} "
                      f"(Survival: {best_model['survival_rate']:.1f}%)")
                print("=" * 80)
        
        else:
            print("Invalid selection.")
    
    except ValueError:
        print("Invalid input. Please enter a number.")


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
        print("  5 - Comprehensive evaluation (100 episodes, no visualization)")
        print("  6 - Evaluate all models and compare")
        print("  progress - View training progress from saved statistics")
        print("  continue <model_path> <timesteps> - Continue training existing model")
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
        evaluate_model_comprehensive(model_path, render=False)  # Use default 100 episodes
    elif action == "6":
        evaluate_all_models(show_plots=False)  # Headless evaluation for bulk processing
    elif action == "progress":
        view_training_progress()
    elif action == "continue":
        if len(sys.argv) < 4:
            print("Please provide model path and additional timesteps")
            print("Usage: python train.py continue <model_path> <additional_timesteps>")
            return
        model_path = sys.argv[2]
        try:
            additional_timesteps = int(sys.argv[3])
            if additional_timesteps <= 0:
                print("Error: Additional timesteps must be a positive number")
                return
            continue_training_model(model_path, additional_timesteps)
        except ValueError:
            print("Error: Invalid timesteps number. Please provide a valid integer.")
            return
    else:
        print("Invalid action. Use 1, 2, custom, 3, 4, 5, 6, progress, or continue.")


if __name__ == "__main__":
    main()
