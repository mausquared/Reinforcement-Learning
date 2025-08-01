#!/usr/bin/env python3
"""
Debug script to test the newest model with detailed error information
"""

import numpy as np
from stable_baselines3 import PPO
from hummingbird_env import ComplexHummingbird3DMatplotlibEnv

def debug_model_compatibility():
    """Debug the model compatibility issue step by step."""
    print("🔧 DEBUG: Model Compatibility Test")
    print("=" * 50)
    
    # Step 1: Create environment and check observation space
    print("Step 1: Creating environment...")
    env = ComplexHummingbird3DMatplotlibEnv(
        grid_size=10,
        num_flowers=5,
        max_energy=100,
        max_height=8,
        render_mode=None
    )
    
    print(f"✅ Environment created successfully")
    print(f"📊 Observation space: {env.observation_space}")
    print()
    
    # Step 2: Reset environment and check observation shape
    print("Step 2: Resetting environment...")
    obs, info = env.reset()
    print(f"✅ Environment reset successfully")
    print(f"🤖 Agent obs shape: {obs['agent'].shape}")
    print(f"🌸 Flower obs shape: {obs['flowers'].shape}")
    print(f"🤖 Agent obs: {obs['agent']}")
    print()
    
    # Step 3: Try loading the newest model
    print("Step 3: Loading newest model...")
    model_path = "./models/training_24_500k.zip"
    
    try:
        model = PPO.load(model_path)
        print(f"✅ Model loaded successfully: {model_path}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # Step 4: Try prediction
    print("Step 4: Testing model prediction...")
    try:
        action, _states = model.predict(obs, deterministic=True)
        print(f"✅ Prediction successful!")
        print(f"🎯 Predicted action: {action}")
        print()
        
        # Step 5: Try environment step
        print("Step 5: Testing environment step...")
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Environment step successful!")
        print(f"🏆 Reward: {reward}")
        print(f"📍 New position: {obs['agent'][:3]}")
        print(f"🔋 Energy: {obs['agent'][3]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction failed with error:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print()
        
        # Additional debugging
        print("🔍 DEBUGGING INFO:")
        print(f"   Model policy: {model.policy}")
        if hasattr(model.policy, 'observation_space'):
            print(f"   Model expects obs space: {model.policy.observation_space}")
        else:
            print(f"   Model observation space: Unknown")
        print(f"   Environment provides: {env.observation_space}")
        
        return False

if __name__ == "__main__":
    debug_model_compatibility()
