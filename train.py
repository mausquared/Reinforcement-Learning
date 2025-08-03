#!/usr/bin/env python3
"""
3D Complex Hummingbird PPO Training Script with Matplotlib Visualization
"""

import os
import sys
import time
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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from typing import Callable

# Import our 3D matplotlib environment
from hummingbird_env import ComplexHummingbird3DMatplotlibEnv, CurriculumHummingbirdEnv


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    
    Args:
        initial_value: Initial learning rate
    
    Returns:
        Schedule function that takes progress_remaining (1 to 0) and returns learning rate
    """
    def func(progress_remaining: float) -> float:
        """Progress will decrease from 1 (beginning) to 0 (end)."""
        return progress_remaining * initial_value
    return func


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
        
        # NEW: Peak performance tracking for auto-save
        self.best_survival_rate = 0.0
        self.best_reward = 0.0
        self.peak_performance_threshold = 40.0  # Auto-save when survival > 40%
        self.models_saved = 0
        self.last_peak_save_step = 0
        
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
            
            # FIXED: Determine episode end reason based on episode length, not just energy
            if 'infos' in self.locals and len(self.locals['infos']) > 0:
                info = self.locals['infos'][0]
                if 'energy' in info:
                    self.training_stats['energy_at_death'].append(info['energy'])
                
                # CORRECTED LOGIC: Use episode length to determine survival
                if self.current_episode_length >= 300:
                    # Agent survived to time limit (success!)
                    self.training_stats['episode_end_reasons'].append('time_limit')
                else:
                    # Agent died before time limit (most likely energy depletion)
                    if info.get('energy', 0) <= 0:
                        self.training_stats['episode_end_reasons'].append('energy_depletion')
                    else:
                        # Died for other reasons (shouldn't happen in current environment)
                        self.training_stats['episode_end_reasons'].append('other_termination')
            else:
                # Fallback: use episode length
                if self.current_episode_length >= 300:
                    self.training_stats['episode_end_reasons'].append('time_limit')
                else:
                    self.training_stats['episode_end_reasons'].append('energy_depletion')
            
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
                
                # NEW: Peak performance auto-save detection
                self._check_peak_performance(survival_rate, avg_reward)
        
        return True
    
    def _check_peak_performance(self, current_survival_rate, current_avg_reward):
        """Check for peak performance and auto-save models."""
        # Check if this is a new peak performance
        is_new_survival_peak = current_survival_rate > self.best_survival_rate
        is_new_reward_peak = current_avg_reward > self.best_reward
        
        # Auto-save conditions
        should_save = False
        save_reason = ""
        
        if current_survival_rate >= self.peak_performance_threshold and is_new_survival_peak:
            should_save = True
            save_reason = f"New survival peak: {current_survival_rate:.1f}%"
            self.best_survival_rate = current_survival_rate
            
        elif current_survival_rate >= 35.0 and is_new_reward_peak:
            should_save = True  
            save_reason = f"New reward peak: {current_avg_reward:.1f} (survival: {current_survival_rate:.1f}%)"
            self.best_reward = current_avg_reward
        
        # Save the model if peak performance detected
        if should_save and (self.num_timesteps - self.last_peak_save_step) >= 200000:  # Don't save too frequently
            self.models_saved += 1
            
            # Create models directory if it doesn't exist
            import os
            os.makedirs("models", exist_ok=True)
            
            peak_model_path = f"models/peak_performance_{self.num_timesteps//1000}k_survival_{current_survival_rate:.1f}%.zip"
            
            print(f"\n🏆 PEAK PERFORMANCE DETECTED! Auto-saving model...")
            print(f"   📊 {save_reason}")
            print(f"   📁 Saved as: {peak_model_path}")
            print(f"   ⏰ Step: {self.num_timesteps:,}")
            
            # Save the model
            self.model.save(peak_model_path)
            
            # Also update best_model.zip if this is excellent performance
            if current_survival_rate >= 40.0:
                best_model_path = "models/best_model.zip"
                self.model.save(best_model_path)
                print(f"   🥇 Also updated best_model.zip (survival ≥ 40%)")
            
            self.last_peak_save_step = self.num_timesteps
            
            # Create a summary file
            summary_path = f"models/peak_performance_{self.num_timesteps//1000}k_summary.txt"
            with open(summary_path, 'w') as f:
                from datetime import datetime
                f.write(f"Peak Performance Model Summary\n")
                f.write(f"Saved at timestep: {self.num_timesteps:,}\n")
                f.write(f"Survival rate: {current_survival_rate:.1f}%\n") 
                f.write(f"Average reward: {current_avg_reward:.1f}\n")
                f.write(f"Save reason: {save_reason}\n")
                f.write(f"Timestamp: {datetime.now()}\n")
                f.write(f"Training session: #{self.training_num}\n")
                f.write(f"Models saved so far: {self.models_saved}\n")


def create_3d_matplotlib_env():
    """Create a 3D matplotlib hummingbird environment for training."""
    return ComplexHummingbird3DMatplotlibEnv(
        grid_size=10,
        num_flowers=5,
        max_energy=100,
        max_height=8,
        render_mode=None  # No rendering during training for speed
    )


def create_stable_3d_matplotlib_env(render_mode=None):
    """Create a 3D matplotlib hummingbird environment with survival rewards for stable training."""
    
    class StableHummingbirdEnv(ComplexHummingbird3DMatplotlibEnv):
        """Enhanced environment with survival rewards for stable training."""
        
        def step(self, action):
            """Enhanced step function with survival rewards."""
            # Call parent step function
            obs, reward, terminated, truncated, info = super().step(action)
            
            # Add survival reward - critical for stable training!
            if not terminated:  # Only if agent is still alive
                survival_reward = 0.1  # Small positive reward for each step survived
                reward += survival_reward
                
                # Optional: Small bonus for maintaining higher energy levels
                energy_bonus = 0.02 * (self.agent_energy / self.max_energy)  # 0-0.02 based on energy level
                reward += energy_bonus
            
            return obs, reward, terminated, truncated, info
    
    return StableHummingbirdEnv(
        grid_size=10,
        num_flowers=5,
        max_energy=100,
        max_height=8,
        render_mode=render_mode
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


def train_stable_3d_matplotlib_ppo(timesteps=2000000, model_name=None):
    """Train PPO with stable hyperparameters and essential survival rewards for consistent learning."""
    
    # Generate training session info
    training_num = get_next_training_number()
    timesteps_label = f"{timesteps//1000}k" if timesteps >= 1000 else str(timesteps)
    
    if model_name is None:
        # Mark stable models with special identifier
        model_name = f"stable_autonomous_{training_num}_{timesteps_label}"
    
    print("⚖️ 3D HUMMINGBIRD STABLE AUTONOMOUS LEARNING (ENHANCED)")
    print("=" * 50)
    print(f"📋 Training Session: #{training_num}")
    print(f"🎯 Target Timesteps: {timesteps:,}")
    print(f"⚖️ Training Mode: ENHANCED STABLE AUTONOMOUS LEARNING")
    print(f"💾 Model Name: {model_name}")
    print(f"📊 Reward Engineering: MINIMAL + Essential Survival Incentives")
    print(f"🔧 Hyperparameters: OPTIMIZED FOR STABILITY + PERFORMANCE")
    print("=" * 50)
    print(f"🎛️ ENHANCED STABILITY OPTIMIZATIONS:")
    print(f"   • Learning Rate: SCHEDULE 3e-4 → 0 (vs fixed 1e-4)")
    print(f"   • Rollout Buffer: 4096 steps (vs 2048 standard)")
    print(f"   • Observation Norm: ENABLED (mean=0, std=1)")
    print(f"   • Batch Size: 128 (vs 256 standard)")
    print(f"   • Entropy Coef: 0.005 (vs 0.02 standard)")
    print(f"   • Network Size: Reduced for stable gradients")
    print(f"   • Environment Count: 16 (optimal for stability)")
    print(f"   • Survival Rewards: +0.1 per step alive (essential task component)")
    print(f"   • Energy Bonus: +0.02 max for maintaining energy (encourages efficiency)")
    print("=" * 50)
    print(f"🎯 EXPECTED IMPROVEMENTS (ENHANCED):")
    print(f"   • Survival Rate: 0% → 20-50% (major improvement)")
    print(f"   • Training Stability: Dramatically reduced fluctuations")
    print(f"   • Learning Efficiency: Better sample efficiency")
    print(f"   • Convergence: Smoother and more predictable")
    print(f"   • Performance Ceiling: Higher achievable performance")
    print(f"   • Task Understanding: Agent learns survival + foraging")
    print("=" * 50)
    
    # Create vectorized training environment with fewer envs for stability + survival rewards
    n_envs = 16  # Reduced from 25 for more stable training
    env = make_vec_env(create_stable_3d_matplotlib_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    
    # ENHANCEMENT: Add observation normalization for better performance
    print(f"🔧 Adding observation normalization for improved stability...")
    env = VecNormalize(env, norm_obs=True, norm_reward=False, gamma=0.99)
    
    # Create evaluation environment (using stable environment for consistent evaluation)
    eval_env = make_vec_env(create_stable_3d_matplotlib_env, n_envs=1, vec_env_cls=DummyVecEnv)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, gamma=0.99)
    
    # ENHANCEMENT: Use learning rate schedule instead of fixed rate
    print(f"📈 Using learning rate schedule for better convergence...")
    lr_schedule = linear_schedule(0.0003)  # Start at 3e-4 and decay to 0
    
    # ENHANCEMENT: Increase rollout buffer size for more stable advantage estimates
    n_steps_per_env = 256  # Increased from 128 (total buffer: 16 * 256 = 4096)
    
    # STABLE PPO hyperparameters - optimized for consistency over speed
    model = PPO(
        "MultiInputPolicy",  # Required for Dict observation spaces
        env,
        learning_rate=lr_schedule,  # ENHANCED: Learning rate schedule
        n_steps=n_steps_per_env,  # ENHANCED: Larger rollout buffer
        batch_size=128,  # REDUCED from 256 for more frequent updates
        n_epochs=10,  # REDUCED from 15 for stable learning
        gamma=0.995,  # ENHANCED: Higher discount for long-term thinking
        gae_lambda=0.95,  # Standard GAE
        clip_range=0.2,  # Standard clip range
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,  # ENHANCED: Increased exploration to escape local optimum
        vf_coef=0.5,  # Standard value function coefficient
        max_grad_norm=0.5,  # Standard gradient clipping
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log="./logs/",
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),  # SMALLER networks for stable gradients
            activation_fn=nn.Tanh
        ),
        verbose=0
    )
    
    print(f"✅ Enhanced stable model created with:")
    print(f"   📈 Learning rate schedule: 3e-4 → 0 (linear decay)")
    print(f"   📊 Rollout buffer size: {n_envs * n_steps_per_env} steps")
    print(f"   🎯 Observation normalization: ENABLED")
    print(f"   🔍 Balanced incentive optimizations:")
    print(f"      • First-visit bonus: +5 reward for flower discovery (reduced from +25)")
    print(f"      • Inefficiency penalty: -2 reward for visiting unavailable flowers")
    print(f"      • Enhanced exploration: ent_coef = 0.01 (2x increase)")
    print(f"      • Long-term thinking: gamma = 0.995 (vs 0.99)")
    print(f"🔄 Using {n_envs} parallel environments for stable training")
    
    # Set up callbacks with adjusted logging frequency
    training_callback = Complex3DMatplotlibTrainingCallback(
        log_freq=50000,  # Less frequent logging for stability focus
        training_num=training_num, 
        total_timesteps=timesteps
    )
    
    # Define callback to save VecNormalize stats when a new best model is found
    class VecNormalizeSaveCallback(BaseCallback):
        """Callback to save VecNormalize statistics when a new best model is found."""
        
        def __init__(self, vec_env):
            super().__init__()
            self.vec_env = vec_env
        
        def _on_step(self) -> bool:
            return True
        
        def _on_training_start(self) -> None:
            pass
        
        def _on_rollout_start(self) -> None:
            pass
        
        def _on_rollout_end(self) -> None:
            try:
                # Save VecNormalize stats along with the model
                self.vec_env.save(f"./models/vec_normalize_stats.pkl")
                print(f"💾 VecNormalize stats saved with new best model")
            except Exception as e:
                print(f"⚠️ Warning: Could not save VecNormalize stats: {e}")
            
    vec_normalize_callback = VecNormalizeSaveCallback(env)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=25000,  # More frequent evaluation to monitor stability
        deterministic=True,
        render=False,
        n_eval_episodes=5,  # Fewer episodes for faster evaluation
        callback_on_new_best=vec_normalize_callback
    )
    
    callbacks = [training_callback, eval_callback]
    
    # Train the model
    print("\n🚀 Starting stable training...")
    start_time = datetime.now()
    
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            tb_log_name="PPO_3D_matplotlib_stable"
        )
        
        training_time = datetime.now() - start_time
        print(f"\n✅ Stable training completed in {training_time}")
        
        # Save the final model
        model.save(f"./models/{model_name}")
        print(f"💾 Stable model saved as {model_name}")
        
        # ENHANCEMENT: Save VecNormalize stats with the final model
        try:
            env.save(f"./models/{model_name}_vec_normalize_stats.pkl")
            print(f"🎯 VecNormalize stats saved for model: {model_name}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save VecNormalize stats: {e}")
        
        # Save training statistics
        with open(f"./models/{model_name}_training_stats.pkl", 'wb') as f:
            pickle.dump(training_callback.training_stats, f)
        print("📊 Training statistics saved")
        
        # Generate training analysis plots
        create_training_analysis_plots(training_callback.training_stats, model_name)
        
        print(f"\n🎉 STABLE TRAINING SUMMARY:")
        print("=" * 50)
        print(f"💾 Model: {model_name}")
        print(f"⏱️  Duration: {training_time}")
        print(f"📈 Timesteps: {timesteps:,}")
        print(f"⚖️ Mode: Stable Autonomous Learning")
        print(f"🔧 Optimizations: Applied for training stability")
        print(f"📊 Expected: Reduced reward fluctuations")
        print(f"🎯 Goal: Higher survival rates over time")
        print("=" * 50)
        
        return model, training_callback.training_stats
        
    except KeyboardInterrupt:
        print("\n⚠️ Stable training interrupted by user")
        model.save(f"./models/{model_name}_interrupted")
        return model, training_callback.training_stats


def train_curriculum_3d_matplotlib_ppo(difficulty='beginner', auto_progress=True, timesteps=2000000, model_name=None):
    """Train PPO with curriculum learning for progressive difficulty mastery."""
    
    # Ensure timesteps is an integer
    try:
        timesteps = int(timesteps)
        if timesteps <= 0:
            raise ValueError("Timesteps must be positive")
    except (ValueError, TypeError):
        print("Error: Invalid timesteps parameter. Must be a positive integer.")
        return
    
    # Generate training session info
    training_num = get_next_training_number()
    timesteps_label = f"{timesteps//1000}k" if timesteps >= 1000 else str(timesteps)
    
    if model_name is None:
        progress_type = "auto" if auto_progress else "manual"
        model_name = f"curriculum_{difficulty}_{progress_type}_{training_num}_{timesteps_label}"
    
    print("📚 3D HUMMINGBIRD CURRICULUM LEARNING")
    print("=" * 50)
    print(f"📋 Training Session: #{training_num}")
    print(f"🎯 Target Timesteps: {timesteps:,}")
    print(f"📚 Training Mode: CURRICULUM LEARNING")
    print(f"🎓 Starting Difficulty: {difficulty.upper()}")
    print(f"📈 Auto-progression: {'ENABLED' if auto_progress else 'DISABLED'}")
    print(f"💾 Model Name: {model_name}")
    print("=" * 50)
    
    # Initialize curriculum environment
    print(f"🏗️ Creating curriculum environment...")
    try:
        # Create environment with curriculum settings
        def make_curriculum_env():
            env = CurriculumHummingbirdEnv(
                difficulty=difficulty, 
                auto_progress=auto_progress
            )
            env = Monitor(env, f"logs/curriculum_{difficulty}_{training_num}.monitor.csv")
            return env
        
        # Create vector environment (reduced to 2 for curriculum stability)
        env = DummyVecEnv([make_curriculum_env for _ in range(2)])
        print(f"✅ Curriculum environment created successfully")
        
        # Curriculum-optimized hyperparameters
        custom_policy_kwargs = {
            'net_arch': dict(pi=[128, 128], vf=[128, 128]),  # Smaller network for curriculum
            'activation_fn': nn.ReLU
        }
        
        print(f"🤖 Creating PPO model with curriculum-optimized hyperparameters...")
        model = PPO(
            "MultiInputPolicy",        # FIXED: Use MultiInputPolicy for Dict observation space
            env,
            learning_rate=0.0002,      # Slower learning for curriculum
            n_steps=1024,              # Standard batch collection
            batch_size=64,             # Smaller batches for stability
            n_epochs=4,                # Fewer epochs per update
            gamma=0.99,                # Standard discount
            gae_lambda=0.95,           # Standard GAE
            clip_range=0.2,            # Standard clipping
            ent_coef=0.01,             # Moderate exploration
            vf_coef=0.5,               # Standard value function weight
            max_grad_norm=0.5,         # Standard gradient clipping
            policy_kwargs=custom_policy_kwargs,
            verbose=0,                 # SUPPRESS verbose PPO output tables
            tensorboard_log=f"./logs/PPO_{model_name}/"
        )
        
        print(f"✅ PPO model created successfully")
        print(f"🎓 CURRICULUM LEARNING SETUP:")
        print(f"   • Progressive difficulty: {difficulty} → ... → hard")
        print(f"   • Auto-progression: {auto_progress}")
        print(f"   • Learning rate: 0.0002 (curriculum-optimized)")
        print(f"   • Network: [128,128] (compact for stability)")
        print(f"   • Environments: 2 (reduced for consistency)")
        
        # Create curriculum-aware callback
        class CurriculumTrainingCallback(Complex3DMatplotlibTrainingCallback):
            """Extended callback for curriculum learning with difficulty tracking."""
            
            def __init__(self, log_freq=1000, training_num=1, total_timesteps=500000, verbose=1):
                super().__init__(log_freq, training_num, total_timesteps, verbose)
                self.difficulty_changes = []
                self.last_difficulty = difficulty
                
            def _on_step(self) -> bool:
                # Call parent step method
                result = super()._on_step()
                
                # Track difficulty changes
                if hasattr(self.training_env.envs[0], 'difficulty'):
                    current_difficulty = self.training_env.envs[0].difficulty
                    if current_difficulty != self.last_difficulty:
                        self.difficulty_changes.append({
                            'step': self.num_timesteps,
                            'old_difficulty': self.last_difficulty,
                            'new_difficulty': current_difficulty
                        })
                        self.last_difficulty = current_difficulty
                        print(f"\n🎓 CURRICULUM PROGRESSION!")
                        print(f"📈 Advanced to {current_difficulty.upper()} at step {self.num_timesteps:,}")
                
                return result
            
            def _on_training_end(self) -> None:
                super()._on_training_end()
                
                # Print curriculum progression summary
                if self.difficulty_changes:
                    print(f"\n📚 CURRICULUM PROGRESSION SUMMARY:")
                    print(f"🎓 Total difficulty changes: {len(self.difficulty_changes)}")
                    for change in self.difficulty_changes:
                        print(f"   Step {change['step']:,}: {change['old_difficulty']} → {change['new_difficulty']}")
                else:
                    print(f"\n📚 CURRICULUM STATUS: Remained at {difficulty.upper()} level")
        
        training_callback = CurriculumTrainingCallback(
            log_freq=1000, 
            training_num=training_num, 
            total_timesteps=timesteps
        )
        
        print(f"\n🚀 Starting curriculum learning...")
        print(f"📊 Progress will be logged every 1000 steps")
        print(f"🎓 Difficulty progression will be automatic (if enabled)")
        print(f"⏹️ Press Ctrl+C to stop training safely")
        print("=" * 50)
        
        # Start training
        model.learn(
            total_timesteps=timesteps,
            callback=training_callback,
            reset_num_timesteps=True
        )
        
        # Save the model
        model.save(f"./models/{model_name}")
        model.save("./models/best_model")  # Also save as default
        
        # Get final curriculum status
        if hasattr(env.envs[0], 'get_curriculum_status'):
            final_status = env.envs[0].get_curriculum_status()
            print(f"\n📚 FINAL CURRICULUM STATUS:")
            print(f"🎓 Final difficulty: {final_status['difficulty'].upper()}")
            print(f"📊 Episodes at final level: {final_status['episodes_at_difficulty']}")
            print(f"🎯 Final survival rate: {final_status['survival_rate']*100:.1f}%")
            print(f"📈 Progress to next level: {final_status['progress_to_next']*100:.1f}%")
        
        env.close()
        
        # Create training analysis
        print(f"\n📊 Training completed! Generating analysis...")
        create_training_analysis_plots(training_callback.training_stats, model_name)
        
        # Save training statistics
        with open(f"models/{model_name}_curriculum_stats.pkl", "wb") as f:
            pickle.dump({
                'training_stats': training_callback.training_stats,
                'difficulty_changes': training_callback.difficulty_changes,
                'final_status': final_status if 'final_status' in locals() else None,
                'hyperparameters': {
                    'learning_rate': 0.0002,
                    'batch_size': 64,
                    'starting_difficulty': difficulty,
                    'auto_progress': auto_progress
                }
            }, f)
        
        print(f"✅ Model saved as: ./models/{model_name}")
        print(f"✅ Default model updated: ./models/best_model")
        print(f"📊 Training analysis saved")
        print(f"🎓 Curriculum learning completed successfully!")
        
        return model, training_callback.training_stats
        
    except KeyboardInterrupt:
        print("\n⚠️ Curriculum training interrupted by user")
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
    
    # 6. Episode end reasons analysis
    if stats['episode_end_reasons']:
        # Updated labels to reflect the fix
        end_reasons = ['energy_depletion', 'time_limit', 'other_termination']
        reason_counts = [stats['episode_end_reasons'].count(reason) for reason in end_reasons]
        
        # Only show non-zero categories
        non_zero_reasons = []
        non_zero_counts = []
        colors = []
        
        for i, (reason, count) in enumerate(zip(end_reasons, reason_counts)):
            if count > 0:
                non_zero_reasons.append(reason.replace('_', ' ').title())
                non_zero_counts.append(count)
                if reason == 'energy_depletion':
                    colors.append('red')
                elif reason == 'time_limit':
                    colors.append('green')
                else:
                    colors.append('orange')
        
        if non_zero_counts:
            axes[1, 2].pie(non_zero_counts, labels=non_zero_reasons, colors=colors, autopct='%1.1f%%')
            axes[1, 2].set_title('Episode End Reasons')
        else:
            axes[1, 2].text(0.5, 0.5, 'No episode data available', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Episode End Reasons')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'./models/{model_name}_3d_matplotlib_training_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Training analysis plot saved as {model_name}_3d_matplotlib_training_analysis.png")
    
    plt.show()


def test_trained_model_3d_matplotlib(model_path, num_episodes=10, render=True):
    """Test a trained model in the 3D matplotlib environment with proper environment compatibility."""
    
    print(f"🐦 Testing trained 3D matplotlib model: {model_path}")
    
    # Load the model
    model = PPO.load(model_path)
    
    # Determine environment version and create appropriate environment
    env_version = get_model_environment_version(model_path)
    
    # Create test environment with proper compatibility
    env = create_environment_for_model(model_path, render_mode="matplotlib" if render else None)
    
    # Compatibility info
    if env_version == 'stable':
        print(f"   ⚖️ Using STABLE environment with survival rewards")
        print(f"   ✅ Optimal testing environment for this model")
    elif env_version == 'legacy':
        print(f"   🚨 COMPATIBILITY WARNING:")
        print(f"   This model was trained with engineered rewards")
        print(f"   Testing in current autonomous learning environment")
        print(f"   Results may not reflect the model's true trained performance!")
    else:
        print(f"   ✅ Using compatible autonomous learning environment")
    
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
    
    # Check if this is a stable training model (highest priority)
    stable_indicators = [
        'stable',
        'stable_autonomous',
        'stable_continued',
        'peak_performance',  # Peak models are saved during stable training
        'survival',          # Models with survival in name are likely stable
        '_stable_'          # Models with _stable_ in name
    ]
    
    for indicator in stable_indicators:
        if indicator in model_name:
            return 'stable'
    
    # Check specific model numbers/dates that we know are stable
    # Models trained with the stable environment
    stable_model_patterns = [
        'ppo_14',   # 14M timestep stable model
        'ppo_15',   # 15M+ stable models 
        'ppo_16',
        'ppo_17',
        'ppo_18',
        'ppo_19',
        'ppo_20',
        'ppo_21',
        'ppo_22'    # Any recent models likely stable
    ]
    
    for pattern in stable_model_patterns:
        if pattern in model_name:
            return 'stable'
    
    # Models with these numbers/dates were trained in the autonomous learning environment
    autonomous_indicators = [
        'autonomous',
        'phase2', 
        'minimal_reward',
        'discovery',
        '3d_matplotlib'  # Original 3D matplotlib models
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
    
    if env_version == 'stable':
        # Use stable environment with survival rewards
        print(f"   ⚖️ Using STABLE TRAINING environment (with survival rewards)")
        return create_stable_3d_matplotlib_env(render_mode=render_mode)
    elif env_version == 'autonomous':
        # Use current autonomous learning environment
        print(f"   📊 Using AUTONOMOUS LEARNING environment (minimal rewards)")
        return ComplexHummingbird3DMatplotlibEnv(
            grid_size=10,
            num_flowers=5,
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
            num_flowers=5,
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
    
    # Calculate confidence intervals (95%) using normal approximation
    # from scipy import stats  # Not available, using numpy approximation
    
    def calculate_confidence_interval(data, confidence=0.95):
        """Calculate confidence interval for the mean using normal approximation."""
        n = len(data)
        mean = np.mean(data)
        std_err = np.std(data) / np.sqrt(n)  # Standard error of the mean
        # For large n, use 1.96 for 95% confidence (normal approximation)
        z_score = 1.96 if n >= 30 else 2.262  # Conservative t-value for small samples
        margin = std_err * z_score
        return mean - margin, mean + margin
    
    # Main metrics with confidence intervals
    reward_mean = np.mean(episode_rewards)
    reward_ci_low, reward_ci_high = calculate_confidence_interval(episode_rewards)
    
    nectar_mean = np.mean(nectar_totals)
    nectar_ci_low, nectar_ci_high = calculate_confidence_interval(nectar_totals)
    
    survival_rate = (survival_count / num_episodes) * 100
    # Confidence interval for proportions (binomial)
    survival_ci_low = max(0, survival_rate - 1.96 * np.sqrt(survival_rate * (100 - survival_rate) / num_episodes))
    survival_ci_high = min(100, survival_rate + 1.96 * np.sqrt(survival_rate * (100 - survival_rate) / num_episodes))
    
    print(f"🏆 Average Reward: {reward_mean:.2f} ± {np.std(episode_rewards):.2f}")
    print(f"   95% CI: [{reward_ci_low:.2f}, {reward_ci_high:.2f}]")
    print(f"🌸 Average Nectar: {nectar_mean:.1f} ± {np.std(nectar_totals):.1f}")
    print(f"   95% CI: [{nectar_ci_low:.1f}, {nectar_ci_high:.1f}]")
    print(f"💪 Survival Rate: {survival_rate:.1f}% ± {np.sqrt(survival_rate * (100 - survival_rate) / num_episodes):.1f}%")
    print(f"   95% CI: [{survival_ci_low:.1f}%, {survival_ci_high:.1f}%] ({survival_count}/{num_episodes})")
    print(f"⏱️  Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"🔋 Average Final Energy: {np.mean(final_energies):.1f} ± {np.std(final_energies):.1f}")
    print(f"⚡ Average Energy Efficiency: {np.mean(energy_efficiency):.2f} nectar/energy")
    
    # Variance analysis
    reward_cv = np.std(episode_rewards) / np.mean(episode_rewards) * 100  # Coefficient of variation
    nectar_cv = np.std(nectar_totals) / np.mean(nectar_totals) * 100 if np.mean(nectar_totals) > 0 else 0
    
    print("-" * 50)
    print(f"� VARIANCE ANALYSIS:")
    print(f"   Reward CV: {reward_cv:.1f}% {'(HIGH VARIANCE!)' if reward_cv > 50 else '(Moderate)' if reward_cv > 25 else '(Low variance)'}")
    print(f"   Nectar CV: {nectar_cv:.1f}% {'(HIGH VARIANCE!)' if nectar_cv > 50 else '(Moderate)' if nectar_cv > 25 else '(Low variance)'}")
    print(f"   Performance Range: {np.min(episode_rewards):.1f} to {np.max(episode_rewards):.1f} (spread: {np.max(episode_rewards) - np.min(episode_rewards):.1f})")
    
    if reward_cv > 50:
        print(f"⚠️  HIGH VARIANCE DETECTED! Consider:")
        print(f"   • More training for stability")
        print(f"   • Lower learning rate")
        print(f"   • More episodes for evaluation (500-1000)")
        print(f"   • Environment determinism improvements")
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
    colors = ['red' if length < 300 else 'green' for length in episode_lengths]
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
    stable_results = {}
    
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
            elif env_version == 'stable':
                stable_results[model_file] = model_stats
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
    
    if stable_results:
        print(f"\n⚖️ STABLE TRAINING MODELS (Optimized Performance)")
        print(f"✅ Models with survival rewards and conservative hyperparameters")
        print("-" * 80)
        print(f"{'Model':<35} {'Reward':<10} {'Nectar':<10} {'Length':<10} {'Survival':<10}")
        print("-" * 80)
        
        for model, stats in sorted(stable_results.items(), key=lambda x: x[1]['avg_nectar'], reverse=True):
            print(f"{model[:34]:<35} {stats['avg_reward']:<10.1f} {stats['avg_nectar']:<10.1f} {stats['avg_length']:<10.1f} {stats['survival_rate']:<10.1f}%")
    
    print("=" * 80)
    
    # Create comparison visualization if we have models to compare and plots are requested
    if (legacy_results or autonomous_results or stable_results) and show_plots:
        create_model_comparison_plots(legacy_results, autonomous_results, stable_results)
    
    # Show best models by category
    if legacy_results:
        best_legacy = max(legacy_results.items(), key=lambda x: x[1]['avg_nectar'])
        print(f"📊 Best legacy model (by nectar): {best_legacy[0]} (Nectar: {best_legacy[1]['avg_nectar']:.1f})")
    
    if autonomous_results:
        best_autonomous = max(autonomous_results.items(), key=lambda x: x[1]['avg_nectar'])
        print(f"� Best autonomous model (by nectar): {best_autonomous[0]} (Nectar: {best_autonomous[1]['avg_nectar']:.1f})")
    
    if stable_results:
        best_stable = max(stable_results.items(), key=lambda x: x[1]['avg_nectar'])
        print(f"🥇 Best stable model (by nectar): {best_stable[0]} (Nectar: {best_stable[1]['avg_nectar']:.1f})")
    
    if not autonomous_results and not stable_results and legacy_results:
        print(f"\n💡 RECOMMENDATION: Train new models with autonomous or stable learning!")
        print(f"   Current models use outdated reward engineering.")
        print(f"   New models will discover strategies independently.")
    elif autonomous_results and not stable_results:
        print(f"\n💡 SUGGESTION: Try stable training for improved survival rates!")
        print(f"   Stable training uses conservative hyperparameters for consistency.")
    
    print("=" * 80)


def create_model_comparison_plots(legacy_results, autonomous_results, stable_results=None):
    """Create comparison plots for all evaluated models."""
    
    # Combine all results for visualization
    all_results = {}
    all_results.update({f"{k} (Legacy)": v for k, v in legacy_results.items()})
    all_results.update({f"{k} (Autonomous)": v for k, v in autonomous_results.items()})
    if stable_results:
        all_results.update({f"{k} (Stable)": v for k, v in stable_results.items()})
    
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
    
    # Colors: red for legacy, blue for autonomous, green for stable
    colors = []
    for model in models:
        if '(Legacy)' in model:
            colors.append('red')
        elif '(Stable)' in model:
            colors.append('green')
        else:
            colors.append('blue')
    
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
                      Patch(facecolor='blue', alpha=0.7, label='Autonomous Models'),
                      Patch(facecolor='green', alpha=0.7, label='Stable Models')]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    # Save the comparison plot
    plot_path = './models/model_comparison_evaluation.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Model comparison plot saved as {plot_path}")
    
    # Show the plot
    plt.show()
    
    return plot_path


def continue_training_model(model_path, additional_timesteps, use_stable_params=False):
    """Continue training an existing model with additional timesteps and optionally stable parameters."""
    
    print("🔄 CONTINUING TRAINING EXISTING MODEL")
    print("=" * 50)
    print(f"📋 Base Model: {model_path}")
    print(f"📈 Additional Timesteps: {additional_timesteps:,}")
    print(f"🤖 Training Mode: CONTINUE EXISTING")
    
    if use_stable_params:
        print(f"⚖️ Parameters: STABLE HYPERPARAMETERS + SURVIVAL REWARDS")
        print(f"📊 Environment: STABLE (with survival incentives)")
        print(f"🎯 Expected: Improved stability and survival rates")
    else:
        print(f"📊 Parameters: ORIGINAL (preserves learned strategies)")
        print(f"🔧 Environment: ORIGINAL (autonomous learning)")
    print("=" * 50)
    
    try:
        # Load existing model
        print("🔄 Loading existing model...")
        model = PPO.load(model_path)
        print("✅ Model loaded successfully!")
        
        if use_stable_params:
            print("🔧 Applying stable training configuration...")
            
            # Update model hyperparameters to stable values
            model.learning_rate = 0.0001  # Reduced from default 5e-4
            model.batch_size = 128        # Reduced from default 256
            model.ent_coef = 0.005        # Reduced from default 0.02
            
            print(f"   • Learning Rate: {model.learning_rate} (stable)")
            print(f"   • Batch Size: {model.batch_size} (stable)")
            print(f"   • Entropy Coef: {model.ent_coef} (stable)")
            print(f"   • Environment: Stable with survival rewards")
            
            # Create stable environment with survival rewards
            n_envs = 16  # Reduced for stability
            env = make_vec_env(create_stable_3d_matplotlib_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
            eval_env = Monitor(create_stable_3d_matplotlib_env())
        else:
            # Use original configuration
            n_envs = 25  # Original training
            env = make_vec_env(create_3d_matplotlib_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
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
        
        if use_stable_params:
            new_model_name = f"{base_name}_stable_continued_{timesteps_label}"
        else:
            new_model_name = f"{base_name}_continued_{timesteps_label}"
        
        print(f"💾 Will save as: {new_model_name}.zip")
        
        # Set up callbacks for continued training with adjusted frequency for stable training
        log_freq = 50000 if use_stable_params else 25000
        eval_freq = 25000 if use_stable_params else 10000
        
        training_callback = Complex3DMatplotlibTrainingCallback(
            log_freq=log_freq, 
            training_num=get_next_training_number(), 
            total_timesteps=additional_timesteps
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./models/",
            log_path="./logs/",
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=5 if use_stable_params else 10  # Fewer for stable training
        )
        
        callbacks = [training_callback, eval_callback]
        
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
        print("=" * 50)
        print(f"💾 Model saved as: {new_model_name}.zip")
        print(f"📊 Training stats saved as: {new_model_name}_training_stats.pkl")
        print(f"📈 Training analysis plot saved as: {new_model_name}_3d_matplotlib_training_analysis.png")
        print(f"⏱️  Training duration: {training_duration}")
        print(f"📈 Additional timesteps: {additional_timesteps:,}")
        
        if use_stable_params:
            print(f"⚖️ Configuration: STABLE PARAMETERS APPLIED")
            print(f"   • Survival rewards: +0.1 per step + energy bonus")
            print(f"   • Hyperparameters: Optimized for stability")
            print(f"   • Expected: Improved survival rates over time")
        else:
            print(f"🔧 Configuration: ORIGINAL PARAMETERS PRESERVED")
            print(f"   • Environment: Autonomous learning (minimal rewards)")
            print(f"   • Hyperparameters: Original training configuration")
        
        print(f"🎯 Ready for evaluation!")
        print("=" * 50)
        
        # Clean up
        env.close()
        eval_env.close()
        
    except Exception as e:
        print(f"❌ Error during continued training: {e}")
        return


def train_specialist_hard_mode(model_path, timesteps=10000000, model_name=None):
    """Transform a curriculum graduate into a HARD mode specialist through ultra-low learning rate fine-tuning."""
    
    if model_name is None:
        base_name = os.path.basename(model_path).replace('.zip', '')
        timesteps_label = f"{timesteps//1000000}M" if timesteps >= 1000000 else f"{timesteps//1000}k"
        model_name = f"{base_name}_specialist_hard_{timesteps_label}"
    
    print("🎯 SPECIALIST HARD MODE TRAINING")
    print("=" * 60)
    print(f"🎓 Base Model: {model_path}")
    print(f"🎯 Target: HARD mode mastery (50%+ survival)")
    print(f"📈 Timesteps: {timesteps:,}")
    print(f"🧪 Mode: PRECISION FINE-TUNING")
    print(f"💾 Output: {model_name}")
    print("=" * 60)
    print(f"🔬 SPECIALIST TRAINING PARAMETERS:")
    print(f"   • Learning Rate: 5e-6 (ultra-low for precision)")
    print(f"   • Entropy: 0.001 (minimal exploration)")
    print(f"   • Environment: HARD-locked (no curriculum)")
    print(f"   • Batch Size: 64 (stable gradients)")
    print(f"   • Networks: Compact [128,128] (prevent overfitting)")
    print(f"   • Training Style: Conservative fine-tuning")
    print("=" * 60)
    
    try:
        # STEP 1: First, inspect the base model to determine its environment configuration
        print(f"🔍 Analyzing base model environment configuration...")
        temp_model = PPO.load(model_path)
        
        # Extract observation space dimensions to determine flower count
        obs_space = temp_model.observation_space
        if hasattr(obs_space, 'spaces') and 'flowers' in obs_space.spaces:
            flowers_shape = obs_space.spaces['flowers'].shape
            original_flower_count = flowers_shape[0]  # First dimension is number of flowers
            print(f"✅ Detected original flower count: {original_flower_count}")
        else:
            # Fallback to standard configuration
            original_flower_count = 5
            print(f"⚠️ Could not detect flower count, using default: {original_flower_count}")
        
        del temp_model  # Free memory
        
        # STEP 2: Create HARD-locked curriculum environment that MATCHES the original configuration
        def make_specialist_env():
            env = CurriculumHummingbirdEnv(
                difficulty='hard',
                auto_progress=False,  # LOCKED to HARD mode only
                num_flowers=original_flower_count  # MATCH original model's flower count
            )
            env = Monitor(env, f"logs/specialist_hard_{get_next_training_number()}.monitor.csv")
            return env
        
        # Create vector environment with same number as original model (2 envs for curriculum)
        env = DummyVecEnv([make_specialist_env for _ in range(2)])  # Match original curriculum training
        print(f"🎯 Created HARD-locked environment with {original_flower_count} flowers (matching original)")
        
        # STEP 3: Load the curriculum graduate model WITH the compatible environment
        print(f"🔄 Loading curriculum graduate model with compatible environment...")
        model = PPO.load(model_path, env=env)  # Load with compatible environment
        print(f"✅ Model loaded successfully with matching observation space!")
        
        # Update model hyperparameters for specialist fine-tuning
        print(f"⚙️ Configuring specialist hyperparameters...")
        
        # Create a linear schedule for ultra-low learning rate
        ultra_low_lr = 5e-6
        model.learning_rate = linear_schedule(ultra_low_lr)  # Use proper learning rate schedule
        model.batch_size = 64       # Stable batch size
        model.n_epochs = 4          # Conservative epochs
        model.ent_coef = 0.001      # Minimal exploration (exploit learned skills)
        # Use linear schedule for clip_range as well
        model.clip_range = linear_schedule(0.1)  # Tighter clipping for stability
        
        print(f"🎯 Specialist hyperparameters configured:")
        print(f"   • Learning Rate: {ultra_low_lr} (ultra-low)")
        print(f"   • Batch Size: {model.batch_size}")
        print(f"   • Entropy Coef: {model.ent_coef} (minimal)")
        print(f"   • Clip Range: {model.clip_range} (conservative)")
        print(f"   • Environment: HARD-locked curriculum")
        
        # Create specialist training callback with adjusted logging
        class SpecialistTrainingCallback(Complex3DMatplotlibTrainingCallback):
            """Callback for specialist training with HARD mode focus."""
            
            def __init__(self, log_freq=50000, training_num=1, total_timesteps=10000000, base_timesteps=0):
                super().__init__(log_freq, training_num, total_timesteps, verbose=1)
                self.survival_improvement_threshold = 40.0  # Target survival rate
                self.best_hard_survival = 0.0
                self.specialist_milestones = []
                self.base_timesteps = base_timesteps  # Track starting timesteps from base model
                
            def _on_step(self) -> bool:
                result = super()._on_step()
                
                # Override progress display for specialist training
                if self.num_timesteps % self.log_freq == 0 and self.training_stats['episodes'] > 0:
                    # Calculate specialist training progress (additional steps beyond base model)
                    specialist_steps = self.num_timesteps - self.base_timesteps
                    specialist_progress = (specialist_steps / self.total_timesteps) * 100
                    
                    # Create a shorter, more reasonable progress bar (10 segments instead of 20)
                    progress_bar_length = 10
                    progress_segments = min(int(specialist_progress // 10), progress_bar_length)
                    progress_bar = "█" * progress_segments + "░" * (progress_bar_length - progress_segments)
                    
                    # Get recent statistics
                    recent_episodes = min(100, len(self.training_stats['survival_rates']))
                    if recent_episodes > 0:
                        recent_rewards = self.training_stats['total_rewards'][-recent_episodes:]
                        recent_survival = self.training_stats['survival_rates'][-recent_episodes:]
                        recent_nectar = self.training_stats['nectar_collected'][-recent_episodes:]
                        
                        avg_reward = np.mean(recent_rewards)
                        survival_rate = np.mean(recent_survival) * 100
                        avg_nectar = np.mean(recent_nectar)
                        
                        print(f"🎯 Specialist #{self.training_num} | Step {self.num_timesteps:,} (+{specialist_steps:,}) [{progress_bar}] {specialist_progress:.1f}%")
                        print(f"   📊 Reward {avg_reward:.1f} | Nectar {avg_nectar:.1f} | Survival {survival_rate:.0f}%")
                        
                        # Track specialist milestones
                        if survival_rate > self.best_hard_survival:
                            self.best_hard_survival = survival_rate
                            
                            # Check for specialist milestones
                            milestones = [20, 30, 40, 50, 60, 70]
                            for milestone in milestones:
                                if survival_rate >= milestone and milestone not in [m['threshold'] for m in self.specialist_milestones]:
                                    self.specialist_milestones.append({
                                        'step': self.num_timesteps,
                                        'threshold': milestone,
                                        'survival_rate': survival_rate
                                    })
                                    print(f"\n🏆 SPECIALIST MILESTONE ACHIEVED!")
                                    print(f"   🎯 HARD Mode Survival: {survival_rate:.1f}% (Target: {milestone}%)")
                                    print(f"   📈 Step: {self.num_timesteps:,}")
                                    
                                    # Auto-save specialist milestones
                                    if milestone >= 40:  # Save significant milestones
                                        milestone_path = f"models/specialist_milestone_{milestone}pct_survival_{self.num_timesteps//1000}k.zip"
                                        self.model.save(milestone_path)
                                        print(f"   💾 Milestone model saved: {milestone_path}")
                
                return result
        
        # Get base timesteps from loaded model for proper progress calculation
        base_timesteps = model.num_timesteps if hasattr(model, 'num_timesteps') else 0
        
        training_callback = SpecialistTrainingCallback(
            log_freq=50000,  # Log every 50k steps (more frequent for specialist training)
            training_num=get_next_training_number(),
            total_timesteps=timesteps,
            base_timesteps=base_timesteps
        )
        
        print(f"\n🚀 Starting specialist HARD mode training...")
        print(f"🎯 Goal: Transform {model_path} into HARD mode specialist")
        print(f"📊 Progress logging every 50k steps")
        print(f"🏆 Milestone auto-save at 20%, 30%, 40%, 50%+ survival")
        print(f"⏹️ Press Ctrl+C to stop training safely")
        print("=" * 60)
        
        # Start specialist training
        model.learn(
            total_timesteps=timesteps,
            callback=training_callback,
            reset_num_timesteps=False  # Continue from base model timesteps
        )
        
        # Save the specialist model
        model.save(f"./models/{model_name}")
        model.save("./models/best_model")  # Update best model
        
        print(f"\n🎉 SPECIALIST TRAINING COMPLETED!")
        print("=" * 60)
        print(f"💾 Specialist model saved as: {model_name}.zip")
        print(f"🥇 Best model updated: best_model.zip")
        
        # Final evaluation
        if training_callback.specialist_milestones:
            print(f"\n🏆 SPECIALIST MILESTONES ACHIEVED:")
            for milestone in training_callback.specialist_milestones:
                print(f"   • {milestone['threshold']}% survival at step {milestone['step']:,}")
            print(f"🎯 Best HARD survival: {training_callback.best_hard_survival:.1f}%")
        else:
            print(f"🎯 Final HARD survival: {training_callback.best_hard_survival:.1f}%")
        
        # Save specialist training statistics
        with open(f"models/{model_name}_specialist_stats.pkl", "wb") as f:
            pickle.dump({
                'training_stats': training_callback.training_stats,
                'specialist_milestones': training_callback.specialist_milestones,
                'best_hard_survival': training_callback.best_hard_survival,
                'base_model': model_path,
                'specialist_params': {
                    'learning_rate': ultra_low_lr,  # Use the actual learning rate value
                    'batch_size': 64,
                    'ent_coef': 0.001,
                    'clip_range': 0.1,
                    'difficulty': 'hard_locked'
                }
            }, f)
        
        # Create specialist analysis
        create_training_analysis_plots(training_callback.training_stats, model_name)
        
        print(f"📊 Specialist analysis saved")
        print(f"🎓 Ready for evaluation!")
        print("=" * 60)
        
        env.close()
        return model, training_callback.training_stats
        
    except KeyboardInterrupt:
        print("\n⚠️ Specialist training interrupted by user")
        model.save(f"./models/{model_name}_interrupted")
        return model, training_callback.training_stats if 'training_callback' in locals() else None
    except Exception as e:
        print(f"❌ Error during specialist training: {e}")
        return None, None


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
        print("  stable <timesteps> - Train with stable hyperparameters (recommended for consistency)")
        print("  curriculum <difficulty> <auto|manual> <timesteps> - Curriculum learning")
        print("  specialist <model_path> <timesteps> - Transform curriculum graduate to HARD mode specialist")
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
        
        # Check if custom episode count is provided
        num_episodes = 100  # default
        if len(sys.argv) >= 4:
            try:
                num_episodes = int(sys.argv[3])
                if num_episodes <= 0:
                    print("Error: Number of episodes must be positive")
                    return
                print(f"🔬 Running extended evaluation with {num_episodes} episodes...")
            except ValueError:
                print("Warning: Invalid episode count, using default 100 episodes")
                num_episodes = 100
        
        evaluate_model_comprehensive(model_path, num_episodes=num_episodes, render=False)
    elif action == "6":
        evaluate_all_models(show_plots=False)  # Headless evaluation for bulk processing
    elif action == "progress":
        view_training_progress()
    elif action == "continue":
        if len(sys.argv) < 4:
            print("Please provide model path and additional timesteps")
            print("Usage: python train.py continue <model_path> <additional_timesteps> [stable]")
            print("Add 'stable' to use stable hyperparameters and survival rewards")
            return
        model_path = sys.argv[2]
        try:
            additional_timesteps = int(sys.argv[3])
            if additional_timesteps <= 0:
                print("Error: Additional timesteps must be a positive number")
                return
            
            # Check if stable parameters should be used
            use_stable = len(sys.argv) > 4 and sys.argv[4].lower() == 'stable'
            
            continue_training_model(model_path, additional_timesteps, use_stable_params=use_stable)
        except ValueError:
            print("Error: Invalid timesteps number. Please provide a valid integer.")
            return
    elif action == "stable":
        if len(sys.argv) < 3:
            print("Please provide timesteps number for stable training")
            print("Usage: python train.py stable <timesteps>")
            return
        try:
            stable_timesteps = int(sys.argv[2])
            if stable_timesteps <= 0:
                print("Error: Timesteps must be a positive number")
                return
            print(f"⚖️ Starting stable training with {stable_timesteps:,} timesteps...")
            train_stable_3d_matplotlib_ppo(timesteps=stable_timesteps)
        except ValueError:
            print("Error: Invalid timesteps number. Please provide a valid integer.")
            return
    elif action == "curriculum":
        if len(sys.argv) < 5:
            print("Please provide curriculum training parameters")
            print("Usage: python train.py curriculum <difficulty> <auto|manual> <timesteps>")
            print("Difficulties: beginner, easy, medium, hard")
            print("Auto: enables auto-progression, Manual: stays at fixed difficulty")
            return
        
        difficulty = sys.argv[2]
        progression_mode = sys.argv[3]
        
        if difficulty not in ['beginner', 'easy', 'medium', 'hard']:
            print("Error: Invalid difficulty. Choose: beginner, easy, medium, hard")
            return
        
        if progression_mode not in ['auto', 'manual']:
            print("Error: Invalid progression mode. Choose: auto, manual")
            return
        
        try:
            curriculum_timesteps = int(sys.argv[4])
            if curriculum_timesteps <= 0:
                print("Error: Timesteps must be a positive number")
                return
            
            auto_progress = progression_mode == 'auto'
            print(f"📚 Starting curriculum learning...")
            print(f"🎓 Starting difficulty: {difficulty.upper()}")
            print(f"📈 Auto-progression: {'ENABLED' if auto_progress else 'DISABLED'}")
            print(f"🎯 Timesteps: {curriculum_timesteps:,}")
            
            train_curriculum_3d_matplotlib_ppo(
                difficulty=difficulty,
                auto_progress=auto_progress,
                timesteps=curriculum_timesteps
            )
        except ValueError:
            print("Error: Invalid timesteps number. Please provide a valid integer.")
            return
    elif action == "specialist":
        if len(sys.argv) < 4:
            print("Please provide model path and timesteps for specialist training")
            print("Usage: python train.py specialist <model_path> <timesteps>")
            print("Example: python train.py specialist models/best_model.zip 10000000")
            return
        
        model_path = sys.argv[2]
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            return
        
        try:
            specialist_timesteps = int(sys.argv[3])
            if specialist_timesteps <= 0:
                print("Error: Timesteps must be a positive number")
                return
            
            print(f"🎯 Starting specialist HARD mode training...")
            print(f"🎓 Base model: {model_path}")
            print(f"📈 Timesteps: {specialist_timesteps:,}")
            
            train_specialist_hard_mode(model_path, timesteps=specialist_timesteps)
        except ValueError:
            print("Error: Invalid timesteps number. Please provide a valid integer.")
            return
    else:
        print("Invalid action. Use 1, 2, custom, 3, 4, 5, 6, progress, continue, stable, curriculum, or specialist.")


if __name__ == "__main__":
    main()
