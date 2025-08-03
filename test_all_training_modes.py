#!/usr/bin/env python3
"""
Test all training modes to verify they work with balanced incentive system
"""

from train import (
    train_complex_3d_matplotlib_ppo,
    train_stable_3d_matplotlib_ppo, 
    train_curriculum_3d_matplotlib_ppo
)

def test_all_training_modes():
    """Test all training modes with minimal timesteps."""
    print("🧪 TESTING ALL TRAINING MODES")
    print("=" * 50)
    
    test_timesteps = 1000  # Small for quick testing
    
    print("1️⃣ Testing Complex 3D Training...")
    try:
        train_complex_3d_matplotlib_ppo(timesteps=test_timesteps)
        print("✅ Complex training: SUCCESS")
    except Exception as e:
        print(f"❌ Complex training: FAILED - {e}")
    
    print("\n2️⃣ Testing Stable Training...")
    try:
        train_stable_3d_matplotlib_ppo(timesteps=test_timesteps)
        print("✅ Stable training: SUCCESS")
    except Exception as e:
        print(f"❌ Stable training: FAILED - {e}")
    
    print("\n3️⃣ Testing Curriculum Training...")
    try:
        train_curriculum_3d_matplotlib_ppo(
            difficulty='beginner',
            auto_progress=False,
            timesteps=test_timesteps
        )
        print("✅ Curriculum training: SUCCESS")
    except Exception as e:
        print(f"❌ Curriculum training: FAILED - {e}")
    
    print("\n🎉 ALL TESTS COMPLETED!")

if __name__ == "__main__":
    test_all_training_modes()
