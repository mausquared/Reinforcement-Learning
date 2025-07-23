import random
import pygame


class HummingbirdEnv:
    """
    A 2D Reinforcement Learning environment where a hummingbird seeks a flower in a grid world.
    """
    
    def __init__(self, grid_size=20):
        """
        Initialize the grid world environment.
        
        Args:
            grid_size (int): Number of cells in the grid (grid_size x grid_size)
        """
        self.grid_size = grid_size
        self.agent_pos = None
        self.flower_pos = None
        
        # Action space: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = 4
        
        # Initialize the environment
        self.reset()
    
    def reset(self):
        """
        Reset the environment for a new episode.
        
        Returns:
            tuple: Initial state (agent_x, agent_y, flower_x, flower_y)
        """
        # Place hummingbird at top-left corner
        self.agent_pos = [0, 0]
        
        # Randomly place flower, ensuring it's not on the agent's starting position
        while True:
            flower_x = random.randint(0, self.grid_size - 1)
            flower_y = random.randint(0, self.grid_size - 1)
            if [flower_x, flower_y] != self.agent_pos:
                self.flower_pos = [flower_x, flower_y]
                break
        
        return self._get_state()
    
    def step(self, action):
        """
        Execute one time step in the environment.
        
        Args:
            action (int): Action to take (0=Up, 1=Down, 2=Left, 3=Right)
            
        Returns:
            tuple: (next_state, reward, done)
        """
        # Store current position to check for boundary hits
        old_pos = self.agent_pos.copy()
        
        # Execute action
        if action == 0:  # Up
            self.agent_pos[1] -= 1
        elif action == 1:  # Down
            self.agent_pos[1] += 1
        elif action == 2:  # Left
            self.agent_pos[0] -= 1
        elif action == 3:  # Right
            self.agent_pos[0] += 1
        
        # Check for boundary collision
        if (self.agent_pos[0] < 0 or self.agent_pos[0] >= self.grid_size or
            self.agent_pos[1] < 0 or self.agent_pos[1] >= self.grid_size):
            # Hit boundary - revert position and apply penalty
            self.agent_pos = old_pos
            reward = -10
            done = False
        elif self.agent_pos == self.flower_pos:
            # Found the flower - big reward and episode ends
            reward = 100
            done = True
        else:
            # Valid move - small penalty to encourage efficiency
            reward = -1
            done = False
        
        return self._get_state(), reward, done
    
    def _get_state(self):
        """
        Get the current state of the environment.
        
        Returns:
            tuple: (agent_x, agent_y, flower_x, flower_y)
        """
        return (self.agent_pos[0], self.agent_pos[1], 
                self.flower_pos[0], self.flower_pos[1])
    
    def render(self, screen):
        """
        Render the environment using Pygame.
        
        Args:
            screen: Pygame surface to draw on
        """
        # Colors
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        BLUE = (0, 0, 255)
        RED = (255, 0, 0)
        GRAY = (128, 128, 128)
        
        # Get screen dimensions
        screen_width, screen_height = screen.get_size()
        cell_width = screen_width // self.grid_size
        cell_height = screen_height // self.grid_size
        
        # Fill background
        screen.fill(WHITE)
        
        # Draw grid lines
        for x in range(0, screen_width + 1, cell_width):
            pygame.draw.line(screen, GRAY, (x, 0), (x, screen_height))
        for y in range(0, screen_height + 1, cell_height):
            pygame.draw.line(screen, GRAY, (0, y), (screen_width, y))
        
        # Draw flower (red circle)
        flower_center_x = self.flower_pos[0] * cell_width + cell_width // 2
        flower_center_y = self.flower_pos[1] * cell_height + cell_height // 2
        flower_radius = min(cell_width, cell_height) // 3
        pygame.draw.circle(screen, RED, (flower_center_x, flower_center_y), flower_radius)
        
        # Draw hummingbird (blue circle)
        agent_center_x = self.agent_pos[0] * cell_width + cell_width // 2
        agent_center_y = self.agent_pos[1] * cell_height + cell_height // 2
        agent_radius = min(cell_width, cell_height) // 4
        pygame.draw.circle(screen, BLUE, (agent_center_x, agent_center_y), agent_radius)
