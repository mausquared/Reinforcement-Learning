#!/usr/bin/env python3
"""
Test the autonomous learning Phase 2 environment after removing engineered rewards.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hummingbird_env import ComplexHummingbird3DMatplotlibEnv
import numpy as np

def test_autonomous_environment():
    """Test that the environment works with autonomous learning (no engineered rewards)."""
    print("🤖 TESTING AUTONOMOUS LEARNING PHASE 2")
    print("=" * 50)
    print("🎯 Testing environment with minimal reward engineering")
    print("🧠 Agent must discover strategy autonomously")
    print("=" * 50)
    
    # Create environment
    env = ComplexHummingbird3DMatplotlibEnv(debug_mode=True, render_mode=None)
    
    print("\n✅ Environment created successfully!")
    
    # Test reset
    obs, info = env.reset()
    print(f"📊 Observation space: {len(obs)}D -> {obs}")
    print(f"📍 Starting position: [{obs[0]:.1f}, {obs[1]:.1f}, {obs[2]:.1f}]")
    print(f"🔋 Starting energy: {obs[3]:.1f}")
    
    # Test a few steps
    total_reward = 0
    nectar_collected = 0
    
    print("\n🎮 Testing 20 random actions...")
    for step in range(20):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 5 == 0:  # Print every 5 steps
            print(f"Step {step:2d}: Action={action}, Reward={reward:+6.3f}, "
                  f"Pos=[{obs[0]:4.1f},{obs[1]:4.1f},{obs[2]:4.1f}], Energy={obs[3]:5.1f}")
            
            if info.get('nectar_collected', 0) > nectar_collected:
                nectar_collected = info['nectar_collected']
                print(f"  🌸 Nectar collected! Total: {nectar_collected}")
        
        if terminated or truncated:
            print(f"  ⚰️  Episode ended at step {step}")
            break
    
    print(f"\n📊 Test Results:")
    print(f"   Total reward: {total_reward:+.3f}")
    print(f"   Nectar collected: {nectar_collected}")
    print(f"   Final energy: {obs[3]:.1f}")
    print(f"   Final position: [{obs[0]:.1f}, {obs[1]:.1f}, {obs[2]:.1f}]")
    
    # Test flower cooldown system
    print(f"\n🌸 Flower system test:")
    for i, flower in enumerate(env.flowers):
        cooldown = env.flower_cooldowns[i]
        available = cooldown == 0 and flower[3] > 0
        print(f"   Flower {i}: Pos=[{flower[0]:.1f},{flower[1]:.1f},{flower[2]:.1f}], "
              f"Nectar={flower[3]:.1f}, Cooldown={cooldown}, Available={available}")
    
    print("\n✅ AUTONOMOUS LEARNING ENVIRONMENT TEST COMPLETE!")
    print("🎯 Ready for genuine strategy discovery training!")
    
    env.close()

if __name__ == "__main__":
    test_autonomous_environment()
