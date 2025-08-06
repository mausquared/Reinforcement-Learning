#!/usr/bin/env python3
"""
Quick test to understand legacy model observation space requirements
"""

import os
import numpy as np
from stable_baselines3 import PPO

def main():
    model_path = './models/peak_performance_3006k_survival_50.0%.zip'
    print(f'Testing legacy model observation space: {model_path}')
    
    if os.path.exists(model_path):
        try:
            # Load the model to see what observation space it expects
            model = PPO.load(model_path)
            print(f'✅ Model loaded successfully!')
            print(f'📊 Expected observation space: {model.observation_space}')
            print(f'🔍 Observation space type: {type(model.observation_space)}')
            print(f'📏 Observation space shape: {model.observation_space.shape}')
            
        except Exception as e:
            print(f'❌ ERROR loading model: {e}')
    else:
        print(f'❌ Model not found: {model_path}')

if __name__ == "__main__":
    main()
