import gymnasium as gym
import numpy as np
import sys
from hummingbird_gymnasium import HummingbirdEnv


def main():
    """
    Main function to run the Hummingbird environment simulation using Gymnasium.
    """
    # Create environment with Gymnasium interface
    env = HummingbirdEnv(grid_size=10, render_mode="human")
    
    print("Starting Hummingbird Environment with Gymnasium!")
    print("The blue circle is the hummingbird, the red circle is the flower.")
    print("The hummingbird will take random actions to find the flower.")
    print("Close the window to exit.")
    
    try:
        episode_count = 0
        # Main game loop
        running = True
        while running:
            episode_count += 1
            print(f"\n--- Episode {episode_count} ---")
            
            # Reset environment for new episode
            observation, info = env.reset(seed=episode_count)
            print(f"Starting new episode. Agent at {observation['agent']}, Flower at {observation['flower']}")
            
            terminated = False
            truncated = False
            step_count = 0
            total_reward = 0
            
            # Episode loop
            while not (terminated or truncated) and running:
                # Take random action using Gymnasium's action space
                action = env.action_space.sample()
                
                # Execute action in environment
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1
                
                # Print debug information
                action_names = ["Up", "Down", "Left", "Right"]
                if reward == 100:
                    print(f"Step {step_count}: Action={action_names[action]}, Reward={reward}, FOUND THE FLOWER!")
                
                # Render environment
                env.render()
                
                # Check if window was closed
                if env.screen is None:
                    print("Window closed by user.")
                    running = False
                    break
                
                # Optional: limit episode length
                if step_count >= 1000:
                    print("Episode truncated at 1000 steps.")
                    truncated = True
            
            if running:
                print(f"Episode {episode_count} completed in {step_count} steps with total reward: {total_reward}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # Clean up
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    main()
