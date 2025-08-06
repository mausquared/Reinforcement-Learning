#!/usr/bin/env python3
"""
Quick test script for stable model evaluation
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import evaluate_model_comprehensive

if __name__ == "__main__":
    print("🧪 Testing stable model evaluation...")
    model_path = "./models/stable_autonomous_28_14000k_stable_continued_3000k.zip"
    
    if os.path.exists(model_path):
        print(f"✅ Model found: {model_path}")
        print("🔍 Running evaluation with 5 episodes...")
        
        try:
            evaluate_model_comprehensive(model_path, num_episodes=5, show_plots=False)
            print("✅ Evaluation completed successfully!")
        except Exception as e:
            print(f"❌ Evaluation failed with error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ Model not found: {model_path}")
