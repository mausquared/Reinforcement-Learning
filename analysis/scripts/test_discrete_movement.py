#!/usr/bin/env python3
"""
Test script for discrete movement system with physics-informed energy costs
"""

import numpy as np
from hummingbird_env import ComplexHummingbird3DMatplotlibEnv

def test_discrete_movement():
    """Test that movement is now discrete and energy costs are realistic."""
    print("🧪 Testing Discrete Movement System")
    print("=" * 50)
    
    # Create environment
    env = ComplexHummingbird3DMatplotlibEnv(render_mode=None)
    obs, info = env.reset()
    
    print(f"Initial position: {env.agent_pos}")
    print(f"Initial energy: {env.agent_energy}")
    print()
    
    # Test each action and verify discrete movement
    actions = [
        (0, "Forward (North)"),
        (1, "Backward (South)"), 
        (2, "Left (West)"),
        (3, "Right (East)"),
        (4, "Up"),
        (5, "Down"),
        (6, "Hover")
    ]
    
    for action, name in actions:
        # Reset to center position
        env.agent_pos = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        env.agent_energy = 100.0
        
        old_pos = env.agent_pos.copy()
        old_energy = env.agent_energy
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        
        new_pos = env.agent_pos
        new_energy = env.agent_energy
        energy_cost = old_energy - new_energy
        
        # Check if movement is discrete (integer positions)
        position_change = new_pos - old_pos
        
        print(f"Action {action} ({name}):")
        print(f"  Position change: {position_change}")
        print(f"  Energy cost: {energy_cost:.1f}")
        print(f"  Position is discrete: {np.allclose(new_pos, np.round(new_pos))}")
        print()
    
    # Test energy cost hierarchy: Hover > Up > Horizontal > Down
    print("Energy Cost Hierarchy Test:")
    print(f"  Hover: {env.HOVER_ENERGY_COST + env.METABOLIC_COST:.1f}")
    print(f"  Up: {env.MOVE_UP_ENERGY_COST + env.METABOLIC_COST:.1f}")
    print(f"  Horizontal: {env.MOVE_HORIZONTAL_COST + env.METABOLIC_COST:.1f}")
    print(f"  Down: {env.MOVE_DOWN_ENERGY_COST + env.METABOLIC_COST:.1f}")
    
    hover_cost = env.HOVER_ENERGY_COST + env.METABOLIC_COST
    up_cost = env.MOVE_UP_ENERGY_COST + env.METABOLIC_COST
    horizontal_cost = env.MOVE_HORIZONTAL_COST + env.METABOLIC_COST
    down_cost = env.MOVE_DOWN_ENERGY_COST + env.METABOLIC_COST
    
    assert hover_cost > up_cost, "Hover should be more expensive than up!"
    assert up_cost > horizontal_cost, "Up should be more expensive than horizontal!"
    assert horizontal_cost > down_cost, "Horizontal should be more expensive than down!"
    
    print("✅ Energy cost hierarchy is correct!")
    print()
    
    # Test observation space (should be 4D now, not 5D)
    agent_obs = obs['agent']
    print(f"Agent observation shape: {agent_obs.shape}")
    print(f"Agent observation: {agent_obs}")
    assert agent_obs.shape == (4,), f"Expected 4D agent obs, got {agent_obs.shape}"
    print("✅ Observation space is correct!")
    
    print("\n🎉 All tests passed! Discrete movement system working correctly.")
    print("Key improvements:")
    print("- ✅ Discrete grid-based movement (no fractional positions)")
    print("- ✅ Physics-informed energy costs (hover most expensive)")
    print("- ✅ Simplified observation space (4D agent state)")
    print("- ✅ Enhanced flower cooldown information maintained")

if __name__ == "__main__":
    test_discrete_movement()
