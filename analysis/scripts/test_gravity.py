#!/usr/bin/env python3
"""Test the restored simple environment system."""

import hummingbird_env

def test_simple_environment():
    """Test the restored simple environment."""
    print("🐦 Testing Restored Simple Environment...")
    
    env = hummingbird_env.ComplexHummingbird3DMatplotlibEnv(debug_mode=True)
    obs, info = env.reset()
    
    print("✅ Environment initialized successfully")
    print(f"Agent obs shape: {obs['agent'].shape}")
    print(f"Flowers obs shape: {obs['flowers'].shape}")
    print(f"Initial agent observation: {obs['agent']}")
    print(f"Initial position: ({obs['agent'][0]:.1f}, {obs['agent'][1]:.1f}, {obs['agent'][2]:.1f})")
    print(f"Initial energy: {obs['agent'][3]:.1f}")
    print(f"Initial nectar collected: {obs['agent'][4]:.1f}")
    
    print("\n--- Testing Simple Movement ---")
    
    # Test 1: Up movement
    print("\n1. Testing UP action:")
    obs, reward, term, trunc, info = env.step(4)  # Up
    print(f"   After up - Position: ({obs['agent'][0]:.1f}, {obs['agent'][1]:.1f}, {obs['agent'][2]:.1f})")
    print(f"   Energy: {obs['agent'][3]:.1f}, Reward: {reward:.2f}")
    
    # Test 2: Down movement  
    print("\n2. Testing DOWN action:")
    obs, reward, term, trunc, info = env.step(5)  # Down
    print(f"   After down - Position: ({obs['agent'][0]:.1f}, {obs['agent'][1]:.1f}, {obs['agent'][2]:.1f})")
    print(f"   Energy: {obs['agent'][3]:.1f}, Reward: {reward:.2f}")
    
    # Test 3: Horizontal movement
    print("\n3. Testing FORWARD action:")
    obs, reward, term, trunc, info = env.step(0)  # Forward
    print(f"   After forward - Position: ({obs['agent'][0]:.1f}, {obs['agent'][1]:.1f}, {obs['agent'][2]:.1f})")
    print(f"   Energy: {obs['agent'][3]:.1f}, Reward: {reward:.2f}")
    
    # Test 4: Hover
    print("\n4. Testing HOVER action:")
    obs, reward, term, trunc, info = env.step(6)  # Hover
    print(f"   After hover - Position: ({obs['agent'][0]:.1f}, {obs['agent'][1]:.1f}, {obs['agent'][2]:.1f})")
    print(f"   Energy: {obs['agent'][3]:.1f}, Reward: {reward:.2f}")
    
    # Test 5: Flower observation
    print("\n5. Testing flower observations:")
    print(f"   Number of flowers: {len(obs['flowers'])}")
    print(f"   First flower: Position({obs['flowers'][0][0]:.1f}, {obs['flowers'][0][1]:.1f}, {obs['flowers'][0][2]:.1f})")
    print(f"                 Nectar: {obs['flowers'][0][3]:.1f}, Cooldown: {obs['flowers'][0][4]:.0f}, Available: {obs['flowers'][0][5]:.0f}")
    
    env.close()
    print("\n🎉 Simple environment test completed successfully!")
    
    print("\n📊 Environment Summary:")
    print(f"   - Movement: Simple discrete grid-based")
    print(f"   - Energy costs: Fixed per action type")  
    print(f"   - Flower info: Enhanced with cooldown and availability")
    print(f"   - No complex physics: Back to original fast system")

if __name__ == "__main__":
    test_simple_environment()
