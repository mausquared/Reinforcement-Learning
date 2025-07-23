import gymnasium as gym
import numpy as np
import pygame
import random
from gymnasium import spaces


class ComplexHummingbirdEnv(gym.Env):
    """
    Complex Hummingbird environment with energy management and multiple flowers.
    
    Features:
    - Energy system (hovering, moving, and collecting nectar affects energy)
    - Multiple flowers with different nectar values
    - Flowers regenerate nectar over time
    - Agent must balance energy consumption with nectar collection
    - Death if energy reaches zero
    
    Observation Space:
        Dict with:
        - 'agent': [x, y, energy] 
        - 'flowers': [[x1, y1, nectar1], [x2, y2, nectar2], ...]
    
    Action Space:
        Discrete(5): 0=Up, 1=Down, 2=Left, 3=Right, 4=Hover
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 8}
    
    def __init__(self, grid_size=15, num_flowers=5, max_energy=100, render_mode=None):
        """
        Initialize the Complex HummingbirdEnv.
        
        Args:
            grid_size (int): Size of the grid
            num_flowers (int): Number of flowers in the environment
            max_energy (int): Maximum energy the hummingbird can have
            render_mode (str): Rendering mode
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.num_flowers = num_flowers
        self.max_energy = max_energy
        self.render_mode = render_mode
        
        # Energy costs
        self.MOVE_COST = 2      # Energy cost for moving
        self.HOVER_COST = 5     # Energy cost for hovering (expensive!)
        self.NECTAR_GAIN = 25   # Energy gained from collecting nectar
        
        # Flower mechanics
        self.MAX_NECTAR = 50    # Maximum nectar per flower
        self.NECTAR_REGEN_RATE = 2  # Nectar regenerated per step
        
        # Define observation space
        # Agent: [x, y, energy]
        # Flowers: [[x1, y1, nectar1], [x2, y2, nectar2], ...]
        self.observation_space = spaces.Dict({
            'agent': spaces.Box(
                low=np.array([0, 0, 0]), 
                high=np.array([grid_size-1, grid_size-1, max_energy]), 
                shape=(3,), 
                dtype=np.float32
            ),
            'flowers': spaces.Box(
                low=np.array([[0, 0, 0]] * num_flowers), 
                high=np.array([[grid_size-1, grid_size-1, self.MAX_NECTAR]] * num_flowers),
                shape=(num_flowers, 3), 
                dtype=np.float32
            )
        })
        
        # Define action space (5 actions including hover)
        self.action_space = spaces.Discrete(5)
        
        # Initialize state
        self.agent_pos = None
        self.agent_energy = None
        self.flowers = None  # List of [x, y, nectar_amount]
        self.total_nectar_collected = 0
        self.steps_taken = 0
        
        # Pygame rendering
        self.screen = None
        self.clock = None
        self.window_size = 600
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset agent
        self.agent_pos = np.array([self.grid_size // 2, self.grid_size // 2], dtype=np.float32)
        self.agent_energy = float(self.max_energy)
        
        # Reset flowers at random positions
        self.flowers = []
        occupied_positions = [tuple(self.agent_pos)]
        
        for _ in range(self.num_flowers):
            while True:
                flower_pos = self.np_random.integers(0, self.grid_size, size=2)
                if tuple(flower_pos) not in occupied_positions:
                    # Random nectar amount (20-50)
                    nectar = self.np_random.integers(20, self.MAX_NECTAR + 1)
                    self.flowers.append([float(flower_pos[0]), float(flower_pos[1]), float(nectar)])
                    occupied_positions.append(tuple(flower_pos))
                    break
        
        self.flowers = np.array(self.flowers, dtype=np.float32)
        self.total_nectar_collected = 0
        self.steps_taken = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step."""
        # Convert action if it's a numpy array
        if hasattr(action, '__iter__') and not isinstance(action, str):
            action = int(action[0]) if len(action) > 0 else int(action)
        else:
            action = int(action)
        
        self.steps_taken += 1
        
        # Action mapping
        action_to_direction = {
            0: np.array([0, -1]),  # Up
            1: np.array([0, 1]),   # Down
            2: np.array([-1, 0]),  # Left
            3: np.array([1, 0]),   # Right
            4: np.array([0, 0])    # Hover
        }
        
        # Calculate energy cost first
        if action == 4:  # Hover
            energy_cost = self.HOVER_COST
        else:  # Movement
            energy_cost = self.MOVE_COST
        
        # Apply movement (if not hovering)
        if action != 4:
            direction = action_to_direction[action]
            new_pos = self.agent_pos + direction
            
            # Check boundaries
            if (0 <= new_pos[0] < self.grid_size and 
                0 <= new_pos[1] < self.grid_size):
                self.agent_pos = new_pos
        
        # Consume energy
        self.agent_energy -= energy_cost
        
        # Check if agent is on a flower
        reward = 0
        on_flower_idx = self._check_flower_collision()
        
        if on_flower_idx is not None:
            # Collect nectar from flower
            nectar_available = self.flowers[on_flower_idx, 2]
            if nectar_available > 0:
                # Collect all available nectar
                nectar_collected = nectar_available
                self.flowers[on_flower_idx, 2] = 0  # Flower is now empty
                
                # Gain energy and reward
                self.agent_energy = min(self.max_energy, self.agent_energy + nectar_collected)
                self.total_nectar_collected += nectar_collected
                reward += nectar_collected  # Reward based on nectar collected
        
        # Regenerate nectar in all flowers
        self._regenerate_nectar()
        
        # Small negative reward for each step to encourage efficiency
        reward -= 0.5
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Death from energy depletion
        if self.agent_energy <= 0:
            terminated = True
            reward -= 100  # Large penalty for death
        
        # Episode too long
        if self.steps_taken >= 500:
            truncated = True
        
        # Bonus for survival and nectar collection
        if self.total_nectar_collected >= 200:  # Collected lots of nectar
            reward += 50
            
        info = {
            'total_nectar_collected': self.total_nectar_collected,
            'energy': self.agent_energy,
            'steps': self.steps_taken,
            'flowers_available': np.sum(self.flowers[:, 2] > 0)
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _check_flower_collision(self):
        """Check if agent is on a flower."""
        for i, flower in enumerate(self.flowers):
            if (abs(self.agent_pos[0] - flower[0]) < 0.1 and 
                abs(self.agent_pos[1] - flower[1]) < 0.1):
                return i
        return None
    
    def _regenerate_nectar(self):
        """Regenerate nectar in flowers over time."""
        for i in range(len(self.flowers)):
            if self.flowers[i, 2] < self.MAX_NECTAR:
                self.flowers[i, 2] = min(self.MAX_NECTAR, 
                                       self.flowers[i, 2] + self.NECTAR_REGEN_RATE)
    
    def _get_observation(self):
        """Get current observation."""
        agent_obs = np.array([self.agent_pos[0], self.agent_pos[1], self.agent_energy], dtype=np.float32)
        return {
            'agent': agent_obs,
            'flowers': self.flowers.copy()
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            return self._render_human()
    
    def _render_human(self):
        """Render in human mode."""
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size + 100))  # Extra space for UI
            pygame.display.set_caption("🐦 Complex Hummingbird - Energy & Multiple Flowers")
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Colors
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        BLUE = (30, 144, 255)
        GREEN = (34, 139, 34)
        RED = (255, 69, 0)
        YELLOW = (255, 215, 0)
        GRAY = (200, 200, 200)
        DARK_RED = (139, 0, 0)
        LIGHT_GREEN = (144, 238, 144)
        
        # Calculate cell size
        cell_size = self.window_size // self.grid_size
        
        # Fill background
        self.screen.fill(WHITE)
        
        # Draw grid
        for x in range(0, self.window_size + 1, cell_size):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, self.window_size), 1)
        for y in range(0, self.window_size + 1, cell_size):
            pygame.draw.line(self.screen, GRAY, (0, y), (self.window_size, y), 1)
        
        # Draw flowers
        for flower in self.flowers:
            flower_center = (
                int(flower[0] * cell_size + cell_size // 2),
                int(flower[1] * cell_size + cell_size // 2)
            )
            
            # Flower size based on nectar amount
            nectar_ratio = flower[2] / self.MAX_NECTAR
            base_radius = cell_size // 3
            flower_radius = int(base_radius * (0.3 + 0.7 * nectar_ratio))
            
            # Color based on nectar amount
            if nectar_ratio > 0.7:
                flower_color = RED
            elif nectar_ratio > 0.3:
                flower_color = YELLOW
            else:
                flower_color = DARK_RED
            
            # Draw flower with petals
            for offset in [(-4, -4), (4, -4), (-4, 4), (4, 4)]:
                petal_center = (flower_center[0] + offset[0], flower_center[1] + offset[1])
                pygame.draw.circle(self.screen, flower_color, petal_center, flower_radius // 2)
            
            # Main flower center
            pygame.draw.circle(self.screen, flower_color, flower_center, flower_radius)
            
            # Draw nectar amount as text
            if hasattr(self, '_small_font') == False:
                pygame.font.init()
                self._small_font = pygame.font.Font(None, 16)
            
            nectar_text = self._small_font.render(f"{int(flower[2])}", True, BLACK)
            text_rect = nectar_text.get_rect(center=(flower_center[0], flower_center[1] + flower_radius + 10))
            self.screen.blit(nectar_text, text_rect)
        
        # Draw hummingbird
        agent_center = (
            int(self.agent_pos[0] * cell_size + cell_size // 2),
            int(self.agent_pos[1] * cell_size + cell_size // 2)
        )
        
        # Energy-based coloring
        energy_ratio = self.agent_energy / self.max_energy
        if energy_ratio > 0.6:
            agent_color = BLUE
        elif energy_ratio > 0.3:
            agent_color = YELLOW
        else:
            agent_color = RED
        
        agent_radius = cell_size // 4
        
        # Wings
        wing_offset = agent_radius + 3
        left_wing = (agent_center[0] - wing_offset, agent_center[1])
        right_wing = (agent_center[0] + wing_offset, agent_center[1])
        pygame.draw.circle(self.screen, agent_color, left_wing, agent_radius // 2)
        pygame.draw.circle(self.screen, agent_color, right_wing, agent_radius // 2)
        
        # Body
        pygame.draw.circle(self.screen, agent_color, agent_center, agent_radius)
        
        # UI Panel
        ui_y = self.window_size + 10
        if hasattr(self, '_font') == False:
            pygame.font.init()
            self._font = pygame.font.Font(None, 24)
        
        # Energy bar
        energy_bar_width = 200
        energy_bar_height = 20
        energy_fill = int(energy_bar_width * (self.agent_energy / self.max_energy))
        
        # Energy bar background
        pygame.draw.rect(self.screen, GRAY, (10, ui_y, energy_bar_width, energy_bar_height))
        # Energy bar fill
        energy_color = GREEN if energy_ratio > 0.3 else RED
        pygame.draw.rect(self.screen, energy_color, (10, ui_y, energy_fill, energy_bar_height))
        
        # Text information
        energy_text = self._font.render(f"Energy: {int(self.agent_energy)}/{self.max_energy}", True, BLACK)
        nectar_text = self._font.render(f"Nectar Collected: {int(self.total_nectar_collected)}", True, BLACK)
        steps_text = self._font.render(f"Steps: {self.steps_taken}", True, BLACK)
        flowers_text = self._font.render(f"Active Flowers: {int(np.sum(self.flowers[:, 2] > 0))}/{self.num_flowers}", True, BLACK)
        
        self.screen.blit(energy_text, (220, ui_y))
        self.screen.blit(nectar_text, (10, ui_y + 25))
        self.screen.blit(steps_text, (250, ui_y + 25))
        self.screen.blit(flowers_text, (10, ui_y + 50))
        
        # Action costs legend
        legend_text = self._small_font.render("Move: -2 Energy, Hover: -5 Energy, Collect: +Nectar", True, BLACK)
        self.screen.blit(legend_text, (10, ui_y + 75))
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
    
    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None


# Test function for the complex environment
def test_complex_environment():
    """Test the complex hummingbird environment."""
    print("Testing Complex Hummingbird Environment...")
    
    env = ComplexHummingbirdEnv(grid_size=12, num_flowers=4, render_mode="human")
    
    for episode in range(3):
        observation, info = env.reset(seed=episode)
        print(f"\n--- Episode {episode + 1} ---")
        print(f"Agent: {observation['agent']}")
        print(f"Flowers: {observation['flowers']}")
        
        terminated = False
        truncated = False
        step_count = 0
        
        while not (terminated or truncated) and step_count < 200:
            # Random action for testing
            action = env.action_space.sample()
            
            observation, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            env.render()
            
            action_names = ["⬆️ Up", "⬇️ Down", "⬅️ Left", "➡️ Right", "🚁 Hover"]
            if step_count % 10 == 0:
                print(f"Step {step_count}: {action_names[action]} | Energy: {info['energy']:.1f} | Nectar: {info['total_nectar_collected']:.1f}")
            
            if terminated:
                if info['energy'] <= 0:
                    print(f"💀 Agent died from energy depletion after {step_count} steps!")
                break
        
        if not terminated:
            print(f"Episode completed: {step_count} steps, Energy: {info['energy']:.1f}, Nectar: {info['total_nectar_collected']:.1f}")
    
    env.close()
    print("Complex environment test completed!")


if __name__ == "__main__":
    test_complex_environment()
