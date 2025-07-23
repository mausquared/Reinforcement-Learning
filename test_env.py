"""
Test script for the HummingbirdEnv to verify it works correctly without the GUI.
"""
from hummingbird_env import HummingbirdEnv
import random


def test_environment():
    """Test the basic functionality of the HummingbirdEnv."""
    print("Testing HummingbirdEnv...")
    
    # Create environment
    env = HummingbirdEnv(grid_size=10)  # Smaller grid for testing
    
    # Test reset
    initial_state = env.reset()
    print(f"Initial state: {initial_state}")
    print(f"Agent position: {env.agent_pos}")
    print(f"Flower position: {env.flower_pos}")
    
    # Test that agent and flower are in different positions
    assert env.agent_pos != env.flower_pos, "Agent and flower should not start in the same position"
    
    # Test some random actions
    total_reward = 0
    steps = 0
    max_steps = 100
    
    while steps < max_steps:
        action = random.randint(0, 3)
        state, reward, done = env.step(action)
        total_reward += reward
        steps += 1
        
        action_names = ["Up", "Down", "Left", "Right"]
        print(f"Step {steps}: Action={action_names[action]}, State={state}, Reward={reward}, Done={done}")
        
        if done:
            print(f"Episode completed in {steps} steps with total reward: {total_reward}")
            break
    
    if steps >= max_steps:
        print(f"Episode ended after {max_steps} steps with total reward: {total_reward}")
    
    # Test boundary conditions
    print("\nTesting boundary conditions...")
    env.reset()
    env.agent_pos = [0, 0]  # Top-left corner
    
    # Try to move up (should hit boundary)
    state, reward, done = env.step(0)  # Up
    print(f"Move up from (0,0): reward={reward}, new_pos={env.agent_pos}")
    assert reward == -10, "Should get -10 reward for hitting boundary"
    assert env.agent_pos == [0, 0], "Position should not change when hitting boundary"
    
    # Try to move left (should hit boundary)
    state, reward, done = env.step(2)  # Left
    print(f"Move left from (0,0): reward={reward}, new_pos={env.agent_pos}")
    assert reward == -10, "Should get -10 reward for hitting boundary"
    assert env.agent_pos == [0, 0], "Position should not change when hitting boundary"
    
    # Test finding the flower
    print("\nTesting flower finding...")
    env.reset()
    env.agent_pos = env.flower_pos.copy()  # Place agent on flower
    state, reward, done = env.step(0)  # Any action
    print(f"Agent on flower: reward={reward}, done={done}")
    assert reward == 100, "Should get 100 reward for finding flower"
    assert done == True, "Episode should end when flower is found"
    
    print("\nAll tests passed! Environment is working correctly.")


if __name__ == "__main__":
    test_environment()
