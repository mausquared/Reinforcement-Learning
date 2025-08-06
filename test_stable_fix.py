#!/usr/bin/env python3
"""
Test stable model testing functionality
"""

import os
import sys
from train import test_trained_model_3d_matplotlib

def main():
    model_path = './models/stable_autonomous_28_14000k_stable_continued_5000k.zip'
    print(f'Testing stable model with fixed environment: {model_path}')
    
    if os.path.exists(model_path):
        try:
            # Test with 1 episode, no rendering for quick test
            test_trained_model_3d_matplotlib(model_path, num_episodes=1, render=False)
            print('✅ SUCCESS! Stable model testing works!')
        except Exception as e:
            print(f'❌ ERROR: {e}')
    else:
        print(f'❌ Model not found: {model_path}')

if __name__ == "__main__":
    main()
