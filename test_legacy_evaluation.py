#!/usr/bin/env python3
"""
Test legacy model evaluation functionality
"""

import os
import sys
from train import evaluate_model_comprehensive

def main():
    model_path = './models/peak_performance_3006k_survival_50.0%.zip'
    print(f'Testing legacy model evaluation: {model_path}')
    
    if os.path.exists(model_path):
        try:
            # Test with 3 episodes for quick test
            rewards, lengths, nectar, survival_rate = evaluate_model_comprehensive(model_path, num_episodes=3, render=False)
            print(f'✅ SUCCESS! Legacy model evaluation works!')
            print(f'📊 Results: {len(rewards)} episodes, Survival rate: {survival_rate:.1%}')
            print(f'🎯 Average reward: {sum(rewards)/len(rewards):.2f}')
        except Exception as e:
            print(f'❌ ERROR: {e}')
            import traceback
            traceback.print_exc()
    else:
        print(f'❌ Model not found: {model_path}')

if __name__ == "__main__":
    main()
