"""
Train the HummingbirdEnv using Q-Learning algorithm.
Q-Learning is a classic, simple RL algorithm that works well for discrete state spaces.
"""

import numpy as np
import matplotlib.pyplot as plt
from hummingbird_gymnasium import HummingbirdEnv
import pickle
import os
from collections import defaultdict


class QLearningAgent:
    """Q-Learning agent for the HummingbirdEnv."""
    
    def __init__(self, grid_size, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize Q-Learning agent.
        
        Args:
            grid_size: Size of the grid
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Rate at which epsilon decays
            epsilon_min: Minimum epsilon value
        """
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: state -> action values
        # State: (agent_x, agent_y, flower_x, flower_y)
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.q_table = defaultdict(lambda: np.zeros(4))
        
        # Training statistics
        self.episode_rewards = []
        self.episode_steps = []
        self.success_rate = []
    
    def get_state(self, observation):
        """Convert observation to state tuple."""
        return tuple(observation['agent']) + tuple(observation['flower'])
    
    def choose_action(self, state, training=True):
        """Choose action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(4)  # Random action
        else:
            return np.argmax(self.q_table[state])  # Best action
    
    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning formula."""
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath):
        """Save the Q-table and training statistics."""
        model_data = {
            'q_table': dict(self.q_table),
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'success_rate': self.success_rate,
            'hyperparameters': {
                'grid_size': self.grid_size,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load the Q-table and training statistics."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: np.zeros(4), model_data['q_table'])
        self.episode_rewards = model_data['episode_rewards']
        self.episode_steps = model_data['episode_steps']
        self.success_rate = model_data['success_rate']
        
        # Load hyperparameters
        params = model_data['hyperparameters']
        self.grid_size = params['grid_size']
        self.learning_rate = params['learning_rate']
        self.discount_factor = params['discount_factor']
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.epsilon_min = params['epsilon_min']
        
        print(f"Model loaded from {filepath}")


def train_q_learning(episodes=5000, grid_size=8):
    """Train Q-Learning agent."""
    print(f"Training Q-Learning agent for {episodes} episodes...")
    
    # Create environment
    env = HummingbirdEnv(grid_size=grid_size, render_mode=None)
    
    # Create agent
    agent = QLearningAgent(grid_size=grid_size)
    
    # Training loop
    success_window = []  # For calculating rolling success rate
    
    for episode in range(episodes):
        observation, info = env.reset()
        state = agent.get_state(observation)
        
        terminated = False
        truncated = False
        step_count = 0
        episode_reward = 0
        max_steps = grid_size * grid_size  # Reasonable max steps
        
        while not (terminated or truncated) and step_count < max_steps:
            # Choose action
            action = agent.choose_action(state, training=True)
            
            # Take action
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = agent.get_state(next_observation)
            
            # Update Q-table
            agent.update_q_table(state, action, reward, next_state)
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            step_count += 1
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record statistics
        agent.episode_rewards.append(episode_reward)
        agent.episode_steps.append(step_count)
        
        # Track success rate
        success = terminated  # True if agent found the flower
        success_window.append(success)
        if len(success_window) > 100:
            success_window.pop(0)
        
        current_success_rate = sum(success_window) / len(success_window) * 100
        agent.success_rate.append(current_success_rate)
        
        # Print progress
        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(agent.episode_rewards[-100:])
            avg_steps = np.mean(agent.episode_steps[-100:])
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Avg Reward (last 100): {avg_reward:.1f}")
            print(f"  Avg Steps (last 100): {avg_steps:.1f}")
            print(f"  Success Rate (last 100): {current_success_rate:.1f}%")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Q-table size: {len(agent.q_table)}")
    
    env.close()
    
    # Save the trained model
    os.makedirs("models", exist_ok=True)
    agent.save_model("models/q_learning_hummingbird.pkl")
    
    return agent


def test_q_learning(model_path="models/q_learning_hummingbird.pkl", n_episodes=5):
    """Test the trained Q-Learning agent with visual rendering."""
    print(f"Testing Q-Learning agent...")
    print("🎮 Opening visual window - watch the trained hummingbird!")
    
    # Load agent
    agent = QLearningAgent(grid_size=8)  # Will be overwritten by load_model
    agent.load_model(model_path)
    
    # Create environment for testing (with rendering)
    env = HummingbirdEnv(grid_size=agent.grid_size, render_mode="human")
    
    # Test episodes
    for episode in range(n_episodes):
        observation, info = env.reset(seed=episode)
        state = agent.get_state(observation)
        print(f"\n--- Test Episode {episode + 1} ---")
        print(f"🐦 Agent starts at: {observation['agent']}")
        print(f"🌺 Flower is at: {observation['flower']}")
        print("Watch the blue circle (hummingbird) find the red circle (flower)!")
        
        terminated = False
        truncated = False
        step_count = 0
        total_reward = 0
        max_steps = 100
        
        while not (terminated or truncated) and step_count < max_steps:
            # Use trained agent (no exploration)
            action = agent.choose_action(state, training=False)
            
            # Execute action
            observation, reward, terminated, truncated, info = env.step(action)
            next_state = agent.get_state(observation)
            
            total_reward += reward
            step_count += 1
            state = next_state
            
            # Render environment (this shows the visual)
            env.render()
            
            # Add action name for better understanding
            action_names = ["⬆️ Up", "⬇️ Down", "⬅️ Left", "➡️ Right"]
            if step_count % 3 == 0:  # Print every 3 steps
                print(f"Step {step_count}: {action_names[action]} -> Agent at {observation['agent']}")
            
            if terminated:
                print(f"🎉 SUCCESS! Found flower in {step_count} steps with action {action_names[action]}")
                print("⏳ Starting next episode in 2 seconds...")
                import time
                time.sleep(2)  # Pause between episodes
                break
        
        if step_count >= max_steps:
            print(f"⏰ Episode timeout after {step_count} steps")
        
        print(f"Total reward: {total_reward}")
    
    print("\n🏁 Visual testing completed! Close the window when ready.")
    env.close()


def evaluate_q_learning(model_path="models/q_learning_hummingbird.pkl", n_episodes=100):
    """Evaluate the trained Q-Learning agent."""
    print(f"Evaluating Q-Learning agent over {n_episodes} episodes...")
    
    # Load agent
    agent = QLearningAgent(grid_size=8)
    agent.load_model(model_path)
    
    # Create environment (no rendering)
    env = HummingbirdEnv(grid_size=agent.grid_size, render_mode=None)
    
    success_count = 0
    total_steps = 0
    total_rewards = 0
    
    for episode in range(n_episodes):
        observation, info = env.reset()
        state = agent.get_state(observation)
        
        terminated = False
        truncated = False
        step_count = 0
        episode_reward = 0
        max_steps = 200
        
        while not (terminated or truncated) and step_count < max_steps:
            action = agent.choose_action(state, training=False)
            observation, reward, terminated, truncated, info = env.step(action)
            state = agent.get_state(observation)
            episode_reward += reward
            step_count += 1
            
            if terminated:
                success_count += 1
                break
        
        total_steps += step_count
        total_rewards += episode_reward
        
        if (episode + 1) % 20 == 0:
            print(f"Episodes {episode + 1}/{n_episodes} - Success rate: {success_count/(episode+1)*100:.1f}%")
    
    env.close()
    
    # Print results
    success_rate = success_count / n_episodes * 100
    avg_steps = total_steps / n_episodes
    avg_reward = total_rewards / n_episodes
    
    print(f"\n--- Q-Learning Evaluation Results ---")
    print(f"Success Rate: {success_rate:.1f}% ({success_count}/{n_episodes})")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Average Reward: {avg_reward:.1f}")


def plot_training_progress(model_path="models/q_learning_hummingbird.pkl"):
    """Plot training progress."""
    # Load agent to get training statistics
    agent = QLearningAgent(grid_size=8)
    agent.load_model(model_path)
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    episodes = range(1, len(agent.episode_rewards) + 1)
    
    # Plot episode rewards
    ax1.plot(episodes, agent.episode_rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Plot episode steps
    ax2.plot(episodes, agent.episode_steps)
    ax2.set_title('Episode Steps')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps to Complete')
    ax2.grid(True)
    
    # Plot success rate
    ax3.plot(episodes, agent.success_rate)
    ax3.set_title('Success Rate (Rolling 100 episodes)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Success Rate (%)')
    ax3.grid(True)
    
    # Plot moving average of rewards
    window_size = 100
    if len(agent.episode_rewards) >= window_size:
        moving_avg = np.convolve(agent.episode_rewards, 
                               np.ones(window_size)/window_size, mode='valid')
        ax4.plot(range(window_size, len(agent.episode_rewards) + 1), moving_avg)
        ax4.set_title(f'Moving Average Reward (window={window_size})')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Average Reward')
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/q_learning_training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Training progress plot saved to 'models/q_learning_training_progress.png'")


def main():
    """Main function for Q-Learning training and testing."""
    print("Q-Learning Training for HummingbirdEnv")
    print("=" * 40)
    
    choice = input("Choose option:\n1. Train new model\n2. Test existing model\n3. Evaluate model\n4. Plot training progress\n5. Train and test\nEnter choice (1-5): ")
    
    if choice == "1":
        train_q_learning()
    elif choice == "2":
        if os.path.exists("models/q_learning_hummingbird.pkl"):
            test_q_learning()
        else:
            print("No trained model found. Please train first (option 1).")
    elif choice == "3":
        if os.path.exists("models/q_learning_hummingbird.pkl"):
            evaluate_q_learning()
        else:
            print("No trained model found. Please train first (option 1).")
    elif choice == "4":
        if os.path.exists("models/q_learning_hummingbird.pkl"):
            plot_training_progress()
        else:
            print("No trained model found. Please train first (option 1).")
    elif choice == "5":
        agent = train_q_learning()
        print("\nTesting the trained model...")
        test_q_learning()
        evaluate_q_learning()
        plot_training_progress()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
