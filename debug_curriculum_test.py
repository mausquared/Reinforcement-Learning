#!/usr/bin/env python3
"""
Debug script to test curriculum training with minimal parameters
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import train_curriculum_3d_matplotlib_ppo

if __name__ == "__main__":
    print("🔧 DEBUG: Testing curriculum training with minimal parameters")
    print("=" * 50)
    
    # Test with small timesteps to see the exact error
    test_timesteps = 1000  # Small number for quick test
    print(f"📊 Test timesteps: {test_timesteps}")
    print(f"🔧 Type: {type(test_timesteps)}")
    print(f"🔧 Is int: {isinstance(test_timesteps, int)}")
    
    try:
        train_curriculum_3d_matplotlib_ppo(
            difficulty='beginner',
            auto_progress=False,  # Disable auto-progression for test
            timesteps=test_timesteps
        )
        print("✅ SUCCESS: Curriculum training completed without errors!")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print(f"🔧 Error type: {type(e)}")
        import traceback
        traceback.print_exc()
