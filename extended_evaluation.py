#!/usr/bin/env python3
"""
Extended Evaluation - Reduce variance with more episodes
"""

import subprocess
import sys

PYTHON_PATH = "C:/Users/mdnva/OneDrive/Desktop/Projects/Reinforcement-Learning/.venv/Scripts/python.exe"

def extended_evaluation():
    """Run extended evaluation with more episodes for reduced variance."""
    print("🎯 EXTENDED EVALUATION FOR VARIANCE REDUCTION")
    print("=" * 60)
    print("🔬 Strategy: More episodes = more stable performance estimates")
    print("📊 Standard: 100 episodes (high variance)")
    print("🎯 Extended: 500-1000 episodes (lower variance)")
    print("⏱️  Time: ~5-10x longer but much more reliable")
    print("=" * 60)
    
    # This would be integrated into your train.py evaluation
    print("\n💡 To implement:")
    print("1. Modify train.py evaluation to accept episode count parameter")
    print("2. Use 500-1000 episodes for final model assessment")
    print("3. Use confidence intervals to show uncertainty ranges")
    print("4. Report both mean and standard deviation")

if __name__ == "__main__":
    extended_evaluation()
