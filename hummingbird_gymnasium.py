import gymnasium as gym
import numpy as np
import pygame
import random
from gymnasium import spaces


class HummingbirdEnv(gym.Env):
    """
    Custom Gymnasium environment for a hummingbird seeking a flower in a 2D grid world.
    
    Observation Space:
        Dict with 'agent' and 'flower' keys, each containing (x, y) coordinates
    
    Action Space:
        Discrete(4): 0=Up, 1=Down, 2=Left, 3=Right
    
    Rewards:
        +100: Agent reaches the flower (episode ends)
        -1: Every other step (encourages efficiency)
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 10}
    
    def __init__(self, grid_size=20, render_mode=None):
        """
        Initialize the HummingbirdEnv.
        
        Args:
            grid_size (int): Size of the grid (grid_size x grid_size)
            render_mode (str): Rendering mode ('human' or None)
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        
        # Define observation space as a Dict with agent and flower positions
        self.observation_space = spaces.Dict({
            'agent': spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int64),
            'flower': spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int64)
        })
        
        # Define action space (4 discrete actions)
        self.action_space = spaces.Discrete(4)
        
        # Initialize positions
        self.agent_pos = None
        self.flower_pos = None
        
        # Pygame rendering attributes (lazy initialization)
        self.screen = None
        self.clock = None
        self.window_size = 600
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.
        
        Args:
            seed (int): Random seed for reproducibility
            options (dict): Additional options (unused)
            
        Returns:
            tuple: (observation, info)
        """
        # Handle seeding
        super().reset(seed=seed)
        
        # Set agent position to center of grid
        self.agent_pos = np.array([self.grid_size // 2, self.grid_size // 2], dtype=np.int64)
        
        # Randomly place flower, ensuring it's not on the agent
        while True:
            self.flower_pos = self.np_random.integers(0, self.grid_size, size=2, dtype=np.int64)
            if not np.array_equal(self.agent_pos, self.flower_pos):
                break
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action (int or np.ndarray): Action to take (0=Up, 1=Down, 2=Left, 3=Right)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Convert numpy array to integer if needed (for compatibility with different RL libraries)
        if hasattr(action, '__iter__') and not isinstance(action, str):
            action = int(action[0]) if len(action) > 0 else int(action)
        else:
            action = int(action)
        
        # Map action to direction
        action_to_direction = {
            0: np.array([0, -1]),  # Up
            1: np.array([0, 1]),   # Down
            2: np.array([-1, 0]),  # Left
            3: np.array([1, 0])    # Right
        }
        
        # Calculate new position
        direction = action_to_direction[action]
        new_pos = self.agent_pos + direction
        
        # Check boundaries and update position if valid
        if (0 <= new_pos[0] < self.grid_size and 
            0 <= new_pos[1] < self.grid_size):
            self.agent_pos = new_pos
        
        # Calculate reward and check termination
        if np.array_equal(self.agent_pos, self.flower_pos):
            reward = 100
            terminated = True
        else:
            reward = -1
            terminated = False
        
        truncated = False  # No time limit in this environment
        observation = self._get_observation()
        info = {}
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """
        Get the current observation.
        
        Returns:
            dict: Dictionary containing agent and flower positions
        """
        return {
            'agent': self.agent_pos.copy(),
            'flower': self.flower_pos.copy()
        }
    
    def render(self):
        """
        Render the environment using Pygame.
        """
        if self.render_mode == "human":
            return self._render_human()
    
    def _render_human(self):
        """
        Render the environment in human mode using Pygame.
        """
        # Lazy initialization of Pygame
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("🐦 Hummingbird Seeks Flower - Trained Agent")
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Colors
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        BLUE = (30, 144, 255)    # Dodger blue for hummingbird
        RED = (255, 69, 0)       # Orange red for flower
        GRAY = (200, 200, 200)   # Light gray for grid
        GREEN = (34, 139, 34)    # Forest green for success trail
        
        # Calculate cell size
        cell_size = self.window_size // self.grid_size
        
        # Fill background
        self.screen.fill(WHITE)
        
        # Draw grid lines
        for x in range(0, self.window_size + 1, cell_size):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, self.window_size), 1)
        for y in range(0, self.window_size + 1, cell_size):
            pygame.draw.line(self.screen, GRAY, (0, y), (self.window_size, y), 1)
        
        # Draw flower (red circle with petals effect)
        flower_center = (
            self.flower_pos[0] * cell_size + cell_size // 2,
            self.flower_pos[1] * cell_size + cell_size // 2
        )
        flower_radius = cell_size // 3
        # Draw flower petals (multiple circles)
        for offset in [(-3, -3), (3, -3), (-3, 3), (3, 3), (0, 0)]:
            petal_center = (flower_center[0] + offset[0], flower_center[1] + offset[1])
            pygame.draw.circle(self.screen, RED, petal_center, flower_radius // 2)
        # Main flower center
        pygame.draw.circle(self.screen, RED, flower_center, flower_radius)
        
        # Draw agent (blue circle with wings effect)
        agent_center = (
            self.agent_pos[0] * cell_size + cell_size // 2,
            self.agent_pos[1] * cell_size + cell_size // 2
        )
        agent_radius = cell_size // 4
        # Draw wings (side circles)
        wing_offset = agent_radius + 3
        left_wing = (agent_center[0] - wing_offset, agent_center[1])
        right_wing = (agent_center[0] + wing_offset, agent_center[1])
        pygame.draw.circle(self.screen, BLUE, left_wing, agent_radius // 2)
        pygame.draw.circle(self.screen, BLUE, right_wing, agent_radius // 2)
        # Main body
        pygame.draw.circle(self.screen, BLUE, agent_center, agent_radius)
        
        # Add text overlay for better visualization
        if hasattr(self, '_font') == False:
            pygame.font.init()
            self._font = pygame.font.Font(None, 24)
        
        # Show coordinates
        agent_text = self._font.render(f"🐦 Agent: {tuple(self.agent_pos)}", True, BLACK)
        flower_text = self._font.render(f"🌺 Flower: {tuple(self.flower_pos)}", True, BLACK)
        self.screen.blit(agent_text, (10, 10))
        self.screen.blit(flower_text, (10, 35))
        
        # Handle Pygame events to prevent window from becoming unresponsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        
        # Update display
        pygame.display.flip()
        self.clock.tick(8)  # Slower for better viewing (8 FPS)
    
    def close(self):
        """
        Clean up resources.
        """
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None


def main():
    """
    Test the HummingbirdEnv with random actions.
    """
    # Create environment
    env = HummingbirdEnv(grid_size=10, render_mode="human")
    
    print("Testing HummingbirdEnv with Gymnasium interface...")
    print("Running 5 episodes with random actions.")
    print("Blue circle = hummingbird, Red circle = flower")
    print("Close the Pygame window to exit early.")
    
    # Run test episodes
    for episode in range(5):
        print(f"\n--- Episode {episode + 1} ---")
        
        # Reset environment
        observation, info = env.reset(seed=episode)
        print(f"Initial observation: {observation}")
        
        terminated = False
        truncated = False
        step_count = 0
        total_reward = 0
        
        # Run episode
        while not (terminated or truncated):
            # Take random action
            action = env.action_space.sample()
            
            # Execute action
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Print step information
            action_names = ["Up", "Down", "Left", "Right"]
            print(f"Step {step_count}: Action={action_names[action]}, "
                  f"Agent={observation['agent']}, Flower={observation['flower']}, "
                  f"Reward={reward}, Done={terminated}")
            
            # Render environment
            env.render()
            
            # Check if window was closed
            if env.screen is None:
                print("Window closed by user.")
                return
            
            # Limit episode length for demonstration
            if step_count >= 100:
                print("Episode truncated at 100 steps.")
                break
        
        print(f"Episode {episode + 1} completed in {step_count} steps with total reward: {total_reward}")
    
    # Clean up
    env.close()
    print("\nTesting completed successfully!")


if __name__ == "__main__":
    main()
