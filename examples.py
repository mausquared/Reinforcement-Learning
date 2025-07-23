"""
Example script showing how to use the HummingbirdEnv with different RL approaches.
This demonstrates the flexibility of the Gymnasium interface.
"""

import numpy as np
from hummingbird_gymnasium import HummingbirdEnv


def random_policy_example():
    """Example using random policy."""
    print("=== Random Policy Example ===")
    
    env = HummingbirdEnv(grid_size=10, render_mode="human")
    
    for episode in range(3):
        observation, info = env.reset(seed=episode)
        print(f"\nEpisode {episode + 1}")
        print(f"Initial state: Agent={observation['agent']}, Flower={observation['flower']}")
        
        terminated = False
        step_count = 0
        total_reward = 0
        
        while not terminated and step_count < 100:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            env.render()
            
            if terminated:
                print(f"Success! Found flower in {step_count} steps with reward {total_reward}")
        
        if step_count >= 100:
            print(f"Episode timeout after {step_count} steps with reward {total_reward}")
    
    env.close()


def simple_heuristic_policy():
    """Example using a simple heuristic policy (move towards flower)."""
    print("\n=== Simple Heuristic Policy Example ===")
    
    env = HummingbirdEnv(grid_size=10, render_mode="human")
    
    for episode in range(3):
        observation, info = env.reset(seed=episode)
        print(f"\nEpisode {episode + 1}")
        
        terminated = False
        step_count = 0
        total_reward = 0
        
        while not terminated and step_count < 100:
            # Simple heuristic: move towards the flower
            agent_pos = observation['agent']
            flower_pos = observation['flower']
            
            # Calculate direction to flower
            diff = flower_pos - agent_pos
            
            # Choose action based on largest difference
            if abs(diff[0]) > abs(diff[1]):  # Move horizontally
                action = 3 if diff[0] > 0 else 2  # Right or Left
            else:  # Move vertically
                action = 1 if diff[1] > 0 else 0  # Down or Up
            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            env.render()
            
            if terminated:
                print(f"Heuristic success! Found flower in {step_count} steps with reward {total_reward}")
        
        if step_count >= 100:
            print(f"Heuristic timeout after {step_count} steps with reward {total_reward}")
    
    env.close()


def q_learning_example():
    """Example of a simple Q-learning implementation."""
    print("\n=== Simple Q-Learning Example ===")
    
    env = HummingbirdEnv(grid_size=5, render_mode=None)  # Smaller grid, no rendering for training
    
    # Q-table: state is (agent_x, agent_y, flower_x, flower_y), action is 0-3
    grid_size = 5
    q_table = np.zeros((grid_size, grid_size, grid_size, grid_size, 4))
    
    # Hyperparameters
    learning_rate = 0.1
    discount_factor = 0.95
    epsilon = 0.1
    episodes = 1000
    
    print(f"Training Q-learning agent for {episodes} episodes...")
    
    for episode in range(episodes):
        observation, info = env.reset()
        terminated = False
        step_count = 0
        
        while not terminated and step_count < 100:
            # Get state indices
            state = (*observation['agent'], *observation['flower'])
            
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            # Take action
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = (*next_observation['agent'], *next_observation['flower'])
            
            # Q-learning update
            current_q = q_table[state][action]
            max_next_q = np.max(q_table[next_state])
            new_q = current_q + learning_rate * (reward + discount_factor * max_next_q - current_q)
            q_table[state][action] = new_q
            
            observation = next_observation
            step_count += 1
        
        if episode % 100 == 0:
            print(f"Episode {episode}: {step_count} steps")
    
    print("Training completed! Testing trained agent...")
    
    # Test the trained agent
    env = HummingbirdEnv(grid_size=5, render_mode="human")
    
    for episode in range(3):
        observation, info = env.reset(seed=episode)
        print(f"\nTest Episode {episode + 1}")
        
        terminated = False
        step_count = 0
        total_reward = 0
        
        while not terminated and step_count < 100:
            # Use trained Q-table (no exploration)
            state = (*observation['agent'], *observation['flower'])
            action = np.argmax(q_table[state])
            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            env.render()
            
            if terminated:
                print(f"Trained agent success! Found flower in {step_count} steps with reward {total_reward}")
        
        if step_count >= 100:
            print(f"Trained agent timeout after {step_count} steps with reward {total_reward}")
    
    env.close()


def main():
    """Run all examples."""
    print("HummingbirdEnv - Gymnasium Examples")
    print("This script demonstrates different approaches to solving the environment.")
    print("Close Pygame windows to proceed to the next example.\n")
    
    try:
        # Run examples
        #random_policy_example()
        #simple_heuristic_policy()
        q_learning_example()
        
        print("\nAll examples completed successfully!")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nError running examples: {e}")


if __name__ == "__main__":
    main()
