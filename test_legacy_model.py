#!/usr/bin/env python3
"""
Quick test script for legacy model evaluation
"""

import os
import sys
from train import evaluate_model_comprehensive

def main():
    model_path = './models/peak_performance_3006k_survival_50.0%.zip'
    print(f'Testing legacy model: {model_path}')
    
    if os.path.exists(model_path):
        try:
            rewards, _, _, survival_rate = evaluate_model_comprehensive(model_path, num_episodes=5, render=False)
            print(f'✅ SUCCESS! Results: {len(rewards)} episodes, Survival rate: {survival_rate:.1%}')
        except Exception as e:
            print(f'❌ ERROR: {e}')
    else:
        print(f'❌ Model not found: {model_path}')

if __name__ == "__main__":
    main()
