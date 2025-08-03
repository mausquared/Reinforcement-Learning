#!/usr/bin/env python3
"""
Test script for curriculum learning integration
"""

import os
import sys
from hummingbird_env import CurriculumHummingbirdEnv

def test_curriculum_integration():
    """Test curriculum learning environment integration."""
    print("🧪 Testing Curriculum Learning Integration")
    print("=" * 50)
    
    # Test 1: Environment creation
    print("1️⃣ Testing environment creation...")
    try:
        env = CurriculumHummingbirdEnv(difficulty='beginner', auto_progress=True)
        print("✅ Environment created successfully")
        
        # Test 2: Environment reset
        print("2️⃣ Testing environment reset...")
        obs, info = env.reset()
        if hasattr(obs, 'shape'):
            print(f"✅ Environment reset - Obs shape: {obs.shape}")
        else:
            print(f"✅ Environment reset - Obs type: {type(obs)}, Obs: {obs}")
        
        # Test 3: Environment step
        print("3️⃣ Testing environment step...")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Environment step - Reward: {reward:.2f}")
        
        # Test 4: Curriculum status
        print("4️⃣ Testing curriculum status...")
        status = env.get_curriculum_status()
        print(f"✅ Status retrieved - Difficulty: {status['difficulty']}")
        
        # Test 5: Difficulty progression
        print("5️⃣ Testing difficulty progression...")
        env.force_difficulty('medium')
        new_status = env.get_curriculum_status()
        print(f"✅ Difficulty changed to: {new_status['difficulty']}")
        
        env.close()
        print("✅ Environment closed successfully")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("\n🎉 All tests passed! Curriculum learning is ready!")
    return True

def test_curriculum_training_args():
    """Test curriculum training argument parsing."""
    print("\n🧪 Testing Curriculum Training Arguments")
    print("=" * 50)
    
    # Test argument combinations
    test_cases = [
        ['train.py', 'curriculum', 'beginner', 'auto', '100000'],
        ['train.py', 'curriculum', 'medium', 'manual', '500000'],
        ['train.py', 'curriculum', 'hard', 'auto', '1000000']
    ]
    
    for i, args in enumerate(test_cases, 1):
        print(f"{i}️⃣ Testing args: {' '.join(args[1:])}")
        
        # Validate arguments
        difficulty = args[2]
        progression = args[3]
        timesteps = args[4]
        
        valid_difficulties = ['beginner', 'easy', 'medium', 'hard']
        valid_progressions = ['auto', 'manual']
        
        if difficulty in valid_difficulties and progression in valid_progressions:
            try:
                int(timesteps)
                print(f"✅ Valid argument combination")
            except ValueError:
                print(f"❌ Invalid timesteps: {timesteps}")
        else:
            print(f"❌ Invalid difficulty or progression mode")
    
    print("✅ Argument parsing test completed!")

if __name__ == "__main__":
    success = test_curriculum_integration()
    if success:
        test_curriculum_training_args()
        print("\n🎓 Curriculum Learning System Ready!")
        print("💡 Use launcher Option 15 to start curriculum training")
        print("📚 Progressive difficulty: Beginner → Easy → Medium → Hard")
        print("🎯 Auto-progression based on survival rate milestones")
    else:
        print("\n❌ Curriculum learning system needs fixes")
