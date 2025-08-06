#!/usr/bin/env python3
"""
Evaluation Wrapper Script
Handles Unicode encoding issues when calling train.py for evaluation
"""

import sys
import subprocess
import os

def main():
    """Run train.py with proper Unicode handling"""
    if len(sys.argv) < 4:
        print("Usage: python evaluate_wrapper.py <mode> <model_path> <episodes>")
        sys.exit(1)
    
    mode = sys.argv[1]
    model_path = sys.argv[2]
    episodes = sys.argv[3]
    
    # Set environment variables for proper Unicode handling
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONLEGACYWINDOWSSTDIO'] = '0'
    
    # Python path
    python_path = "C:/Users/mdnva/OneDrive/Desktop/Projects/Reinforcement-Learning/.venv/Scripts/python.exe"
    
    try:
        # Run train.py with the evaluation mode
        result = subprocess.run([
            python_path, "train.py", mode, model_path, episodes
        ], env=env, encoding='utf-8', errors='replace', text=True)
        
        sys.exit(result.returncode)
        
    except Exception as e:
        print(f"Error running evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
