#!/usr/bin/env python3
"""
Test the exact same environment creation as the launcher evaluation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from hummingbird_env import ComplexHummingbird3DMatplotlibEnv

def test_launcher_evaluation():
    """Test the exact same evaluation process as the launcher."""
    print("🔧 LAUNCHER EVALUATION TEST")
    print("=" * 50)
    
    model_path = "./models/training_24_500k.zip"
    
    # Load the model (same as evaluate_model_comprehensive)
    print("Loading model...")
    model = PPO.load(model_path)
    print(f"✅ Model loaded: {model_path}")
    
    # Create test environment (exact same as evaluate_model_comprehensive)
    print("Creating environment...")
    env = ComplexHummingbird3DMatplotlibEnv(
        grid_size=10,
        num_flowers=5,
        max_energy=100,
        max_height=8,
        render_mode=None  # No rendering for comprehensive eval
    )
    print(f"✅ Environment created")
    print(f"📊 Observation space: {env.observation_space}")
    
    # Test one complete episode (same loop as evaluate_model_comprehensive)
    print("Starting test episode...")
    obs, info = env.reset()
    print(f"✅ Environment reset")
    print(f"🤖 Initial obs shape: agent={obs['agent'].shape}, flowers={obs['flowers'].shape}")
    
    episode_reward = 0
    step_count = 0
    terminated = False
    truncated = False
    
    try:
        while not (terminated or truncated) and step_count < 5:  # Just test 5 steps
            print(f"  Step {step_count + 1}:")
            print(f"    Agent obs: {obs['agent']}")
            
            # Get action from trained model (same as evaluation)
            action, _states = model.predict(obs, deterministic=False)
            print(f"    Predicted action: {action}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"    Reward: {reward:.2f}")
            print(f"    New obs shape: agent={obs['agent'].shape}")
            
            episode_reward += reward
            step_count += 1
        
        print(f"✅ Test completed successfully!")
        print(f"📊 Total reward: {episode_reward:.2f}")
        print(f"📊 Steps taken: {step_count}")
        
    except Exception as e:
        print(f"❌ Error during evaluation:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print(f"   Step: {step_count}")
        return False
    
    return True

if __name__ == "__main__":
    test_launcher_evaluation()
