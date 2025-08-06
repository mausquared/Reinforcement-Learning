#!/usr/bin/env python3
"""
Test stable model observation space
"""

import os
from stable_baselines3 import PPO

def main():
    model_path = './models/stable_autonomous_28_14000k_stable_continued_5000k.zip'
    print(f'Testing stable model observation space: {model_path}')
    
    if os.path.exists(model_path):
        try:
            # Load the model to see what observation space it expects
            model = PPO.load(model_path)
            print(f'✅ Model loaded successfully!')
            print(f'📊 Expected observation space: {model.observation_space}')
            print(f'🔍 Observation space type: {type(model.observation_space)}')
            if hasattr(model.observation_space, 'spaces'):
                print(f'📄 Observation space keys: {list(model.observation_space.spaces.keys())}')
                for key, space in model.observation_space.spaces.items():
                    print(f'   {key}: {space}')
            
        except Exception as e:
            print(f'❌ ERROR loading model: {e}')
    else:
        print(f'❌ Model not found: {model_path}')

if __name__ == "__main__":
    main()
