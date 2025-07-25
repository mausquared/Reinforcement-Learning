import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from gymnasium import spaces
import time


class ComplexHummingbird3DMatplotlibEnv(gym.Env):
    """
    3D Complex Hummingbird environment with matplotlib 3D visualization.
    
    Features:
    - True 3D visualization using matplotlib
    - 3D movement (x, y, z coordinates)
    - Energy system with gravity effects
    - Multiple flowers at different heights
    - Real-time 3D plotting with interactive viewing
    """
    
    metadata = {"render_modes": ["human", "matplotlib"], "render_fps": 8}
    
    def __init__(self, grid_size=10, num_flowers=5, max_energy=100, max_height=8, render_mode=None):
        """Initialize the 3D matplotlib environment."""
        super().__init__()
        
        self.grid_size = grid_size
        self.num_flowers = num_flowers
        self.max_energy = max_energy
        self.max_height = max_height
        self.render_mode = render_mode
        
        # 3D Energy costs
        self.MOVE_HORIZONTAL_COST = 1.5    # Increased from 1
        self.MOVE_UP_COST = 4              # Increased from 3
        self.MOVE_DOWN_COST = 0.8          # Increased from 0.5
        self.HOVER_COST = 5                # Increased from 4
        self.GRAVITY_COST = 0.3            # Increased from 0.2
        self.NECTAR_GAIN = 25              # Reduced from 30
        
        # Flower mechanics
        self.MAX_NECTAR = 35               # Reduced from 40
        self.NECTAR_REGEN_RATE = 0.3       # Much slower: 0.3 per step instead of 1.0
        self.FLOWER_COOLDOWN_TIME = 15     # Steps before flower can be used again after being emptied
        
        # Observation space
        self.observation_space = spaces.Dict({
            'agent': spaces.Box(
                low=np.array([0, 0, 0, 0, -2]), 
                high=np.array([grid_size-1, grid_size-1, max_height, max_energy, 2]), 
                shape=(5,), 
                dtype=np.float32
            ),
            'flowers': spaces.Box(
                low=np.array([[0, 0, 0, 0]] * num_flowers), 
                high=np.array([[grid_size-1, grid_size-1, max_height, self.MAX_NECTAR]] * num_flowers),
                shape=(num_flowers, 4), 
                dtype=np.float32
            )
        })
        
        # Action space (7 actions for 3D movement)
        self.action_space = spaces.Discrete(7)
        
        # Initialize state
        self.agent_pos = None
        self.agent_energy = None
        self.agent_velocity_z = None
        self.flowers = None  # List of [x, y, z, nectar_amount]
        self.flower_cooldowns = None  # Track cooldown times for each flower
        self.total_nectar_collected = 0
        self.steps_taken = 0
        self.last_flower_visited = -1  # Track last flower to prevent camping
        
        # Matplotlib 3D setup
        self.fig = None
        self.ax = None
        self.agent_trail = []  # Track agent's path
        self.max_trail_length = 50
        
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset agent
        self.agent_pos = np.array([
            self.grid_size // 2, 
            self.grid_size // 2, 
            self.max_height // 2
        ], dtype=np.float32)
        self.agent_energy = float(self.max_energy)
        self.agent_velocity_z = 0.0
        
        # Reset flowers
        self.flowers = []
        occupied_positions = [tuple(self.agent_pos)]
        
        for _ in range(self.num_flowers):
            while True:
                flower_pos = np.array([
                    self.np_random.integers(1, self.grid_size - 1),
                    self.np_random.integers(1, self.grid_size - 1),
                    self.np_random.integers(1, self.max_height)
                ])
                
                if tuple(flower_pos) not in occupied_positions:
                    nectar = self.np_random.integers(20, self.MAX_NECTAR + 1)
                    self.flowers.append([
                        float(flower_pos[0]), 
                        float(flower_pos[1]), 
                        float(flower_pos[2]), 
                        float(nectar)
                    ])
                    occupied_positions.append(tuple(flower_pos))
                    break
        
        self.flowers = np.array(self.flowers, dtype=np.float32)
        self.flower_cooldowns = np.zeros(self.num_flowers, dtype=int)  # Initialize cooldowns
        self.total_nectar_collected = 0
        self.steps_taken = 0
        self.last_flower_visited = -1
        self.agent_trail = [self.agent_pos.copy()]
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step in 3D space."""
        if hasattr(action, '__iter__') and not isinstance(action, str):
            action = int(action[0]) if len(action) > 0 else int(action)
        else:
            action = int(action)
        
        self.steps_taken += 1
        
        # 3D Action mapping
        action_to_direction = {
            0: np.array([0, -1, 0]),   # Forward (North)
            1: np.array([0, 1, 0]),    # Backward (South)
            2: np.array([-1, 0, 0]),   # Left (West)
            3: np.array([1, 0, 0]),    # Right (East)
            4: np.array([0, 0, 1]),    # Up
            5: np.array([0, 0, -1]),   # Down
            6: np.array([0, 0, 0])     # Hover
        }
        
        # Calculate energy cost
        if action == 6:  # Hover
            energy_cost = self.HOVER_COST
        elif action == 4:  # Up
            energy_cost = self.MOVE_UP_COST
        elif action == 5:  # Down
            energy_cost = self.MOVE_DOWN_COST
        else:  # Horizontal
            energy_cost = self.MOVE_HORIZONTAL_COST
        
        energy_cost += self.GRAVITY_COST
        
        # Apply movement
        if action != 6:
            direction = action_to_direction[action]
            new_pos = self.agent_pos + direction
            
            # Boundary checks
            new_pos[0] = np.clip(new_pos[0], 0, self.grid_size - 1)
            new_pos[1] = np.clip(new_pos[1], 0, self.grid_size - 1)
            new_pos[2] = np.clip(new_pos[2], 0, self.max_height)
            
            # Update velocity
            if action == 4:  # Up
                self.agent_velocity_z = min(2, self.agent_velocity_z + 0.5)
            elif action == 5:  # Down
                self.agent_velocity_z = max(-2, self.agent_velocity_z - 0.5)
            else:
                self.agent_velocity_z *= 0.8
            
            self.agent_pos = new_pos
        else:
            self.agent_velocity_z *= 0.9
        
        # Update trail
        self.agent_trail.append(self.agent_pos.copy())
        if len(self.agent_trail) > self.max_trail_length:
            self.agent_trail.pop(0)
        
        # Energy consumption
        self.agent_energy -= energy_cost
        
        # Flower collision and nectar collection with anti-camping
        reward = 0
        on_flower_idx = self._check_flower_collision_3d()
        
        if on_flower_idx is not None:
            # Check if flower is available (not in cooldown)
            if self.flower_cooldowns[on_flower_idx] == 0:
                nectar_available = self.flowers[on_flower_idx, 3]
                if nectar_available > 0:
                    # Collect nectar (but not all at once to prevent infinite energy)
                    nectar_collected = min(nectar_available, 15)  # Max 15 nectar per visit
                    self.flowers[on_flower_idx, 3] -= nectar_collected
                    
                    # If flower is emptied, set cooldown
                    if self.flowers[on_flower_idx, 3] <= 0:
                        self.flowers[on_flower_idx, 3] = 0
                        self.flower_cooldowns[on_flower_idx] = self.FLOWER_COOLDOWN_TIME
                    
                    # Gain energy and reward
                    self.agent_energy = min(self.max_energy, self.agent_energy + nectar_collected)
                    self.total_nectar_collected += nectar_collected
                    reward += nectar_collected
                    
                    # Bonus for visiting new flowers (encourage exploration)
                    if on_flower_idx != self.last_flower_visited:
                        reward += 5  # Exploration bonus
                        self.last_flower_visited = on_flower_idx
                    else:
                        # Penalty for camping on same flower
                        reward -= 2
            else:
                # Penalty for trying to use flower in cooldown
                reward -= 1
        
        # Update flower cooldowns
        self.flower_cooldowns = np.maximum(0, self.flower_cooldowns - 1)
        
        # Regenerate nectar
        self._regenerate_nectar()
        
        # Rewards and penalties
        reward -= 0.1  # Step penalty
        
        energy_ratio = self.agent_energy / self.max_energy
        if energy_ratio > 0.5:
            reward += 0.5
        elif energy_ratio < 0.2:
            reward -= 1.0
        
        if 2 <= self.agent_pos[2] <= self.max_height - 2:
            reward += 0.1
        
        # Termination conditions
        terminated = False
        truncated = False
        
        if self.agent_energy <= 0:
            terminated = True
            reward -= 50
        
        if self.agent_pos[2] <= 0:
            terminated = True
            reward -= 30
        
        if self.steps_taken >= 400:
            truncated = True
        
        # Progressive bonuses
        if self.total_nectar_collected >= 100:
            reward += 20
        if self.total_nectar_collected >= 200:
            reward += 30
        
        info = {
            'total_nectar_collected': self.total_nectar_collected,
            'energy': self.agent_energy,
            'steps': self.steps_taken,
            'flowers_available': np.sum((self.flowers[:, 3] > 0) & (self.flower_cooldowns == 0)),
            'flowers_in_cooldown': np.sum(self.flower_cooldowns > 0),
            'altitude': self.agent_pos[2],
            'velocity_z': self.agent_velocity_z,
            'last_flower_visited': self.last_flower_visited
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _check_flower_collision_3d(self):
        """Check if agent is on a flower in 3D space."""
        for i, flower in enumerate(self.flowers):
            distance = np.sqrt(
                (self.agent_pos[0] - flower[0])**2 + 
                (self.agent_pos[1] - flower[1])**2 + 
                (self.agent_pos[2] - flower[2])**2
            )
            if distance < 1.2:
                return i
        return None
    
    def _regenerate_nectar(self):
        """Regenerate nectar in flowers over time, but only if not in cooldown."""
        for i in range(len(self.flowers)):
            # Only regenerate if flower is not in cooldown
            if self.flower_cooldowns[i] == 0 and self.flowers[i, 3] < self.MAX_NECTAR:
                self.flowers[i, 3] = min(self.MAX_NECTAR, 
                                       self.flowers[i, 3] + self.NECTAR_REGEN_RATE)
    
    def _get_observation(self):
        """Get current observation."""
        agent_obs = np.array([
            self.agent_pos[0], 
            self.agent_pos[1], 
            self.agent_pos[2], 
            self.agent_energy,
            self.agent_velocity_z
        ], dtype=np.float32)
        return {
            'agent': agent_obs,
            'flowers': self.flowers.copy()
        }
    
    def render(self):
        """Render the 3D environment using matplotlib."""
        if self.render_mode in ["human", "matplotlib"]:
            return self._render_matplotlib_3d()
    
    def _render_matplotlib_3d(self):
        """Render using matplotlib 3D plotting."""
        if self.fig is None:
            # Create 3D figure
            plt.ion()  # Turn on interactive mode
            self.fig = plt.figure(figsize=(12, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            
            # Set up the plot
            self.ax.set_xlim(0, self.grid_size - 1)
            self.ax.set_ylim(0, self.grid_size - 1)
            self.ax.set_zlim(0, self.max_height)
            self.ax.set_xlabel('X Position')
            self.ax.set_ylabel('Y Position')
            self.ax.set_zlabel('Z Position (Height)')
            self.ax.set_title('🐦 3D Hummingbird Environment - Matplotlib Visualization')
        
        # Clear the plot
        self.ax.clear()
        
        # Reset limits and labels (cleared with ax.clear())
        self.ax.set_xlim(0, self.grid_size - 1)
        self.ax.set_ylim(0, self.grid_size - 1)
        self.ax.set_zlim(0, self.max_height)
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_zlabel('Z Position (Height)')
        
        # Draw grid planes
        xx, yy = np.meshgrid(np.linspace(0, self.grid_size-1, self.grid_size),
                            np.linspace(0, self.grid_size-1, self.grid_size))
        
        # Ground plane (z=0)
        self.ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color='gray')
        
        # Draw 3D grid lines
        for i in range(self.grid_size):
            # Vertical lines
            self.ax.plot([i, i], [0, self.grid_size-1], [0, 0], 'k-', alpha=0.2, linewidth=0.5)
            self.ax.plot([0, self.grid_size-1], [i, i], [0, 0], 'k-', alpha=0.2, linewidth=0.5)
            
            # Height reference lines at corners
            if i == 0 or i == self.grid_size-1:
                for j in [0, self.grid_size-1]:
                    self.ax.plot([i, i], [j, j], [0, self.max_height], 'k--', alpha=0.3, linewidth=0.5)
        
        # Draw flowers as 3D points with size based on nectar and cooldown indicators
        for i, flower in enumerate(self.flowers):
            nectar_ratio = flower[3] / self.MAX_NECTAR
            is_in_cooldown = self.flower_cooldowns[i] > 0
            
            # Color based on nectar amount and cooldown status
            if is_in_cooldown:
                color = 'gray'  # Gray for flowers in cooldown
            elif nectar_ratio > 0.7:
                color = 'red'
            elif nectar_ratio > 0.3:
                color = 'orange'
            else:
                color = 'darkred'
            
            # Size based on nectar (smaller if in cooldown)
            base_size = 50 + nectar_ratio * 200
            size = base_size * 0.5 if is_in_cooldown else base_size
            
            # Plot flower
            marker = 'x' if is_in_cooldown else '*'  # Different marker for cooldown
            edge_color = None if is_in_cooldown else 'black'  # No edge for 'x' marker
            self.ax.scatter(flower[0], flower[1], flower[2], 
                          c=color, s=size, marker=marker, 
                          edgecolors=edge_color, linewidths=1,
                          alpha=0.5 if is_in_cooldown else 1.0)
            
            # Draw nectar amount and cooldown text
            if is_in_cooldown:
                text = f'CD:{self.flower_cooldowns[i]}'
                self.ax.text(flower[0], flower[1], flower[2] + 0.3, 
                            text, fontsize=8, ha='center', color='red')
            else:
                self.ax.text(flower[0], flower[1], flower[2] + 0.3, 
                            f'{int(flower[3])}', fontsize=8, ha='center')
            
            # Draw vertical line from ground to flower
            line_color = 'gray' if is_in_cooldown else 'black'
            line_alpha = 0.2 if is_in_cooldown else 0.3
            self.ax.plot([flower[0], flower[0]], [flower[1], flower[1]], 
                        [0, flower[2]], color=line_color, linestyle='--', 
                        alpha=line_alpha, linewidth=1)
        
        # Draw agent trail
        if len(self.agent_trail) > 1:
            trail_array = np.array(self.agent_trail)
            self.ax.plot(trail_array[:, 0], trail_array[:, 1], trail_array[:, 2], 
                        'b-', alpha=0.6, linewidth=2, label='Flight Path')
        
        # Draw hummingbird
        energy_ratio = self.agent_energy / self.max_energy
        if energy_ratio > 0.6:
            agent_color = 'blue'
        elif energy_ratio > 0.3:
            agent_color = 'yellow'
        else:
            agent_color = 'red'
        
        # Main body
        self.ax.scatter(self.agent_pos[0], self.agent_pos[1], self.agent_pos[2], 
                       c=agent_color, s=300, marker='o', 
                       edgecolors='black', linewidths=2,
                       label=f'Hummingbird (energy: {int(self.agent_energy)})')
        
        # Wings (simple representation)
        wing_offset = 0.3
        self.ax.scatter([self.agent_pos[0] - wing_offset, self.agent_pos[0] + wing_offset], 
                       [self.agent_pos[1], self.agent_pos[1]], 
                       [self.agent_pos[2], self.agent_pos[2]], 
                       c=agent_color, s=100, marker='_', alpha=0.7)
        
        # Draw vertical line from ground to agent
        self.ax.plot([self.agent_pos[0], self.agent_pos[0]], 
                    [self.agent_pos[1], self.agent_pos[1]], 
                    [0, self.agent_pos[2]], 'b--', alpha=0.5, linewidth=1)
        
        # Add text info near agent
        info_text = f'Energy: {int(self.agent_energy)}\nAltitude: {self.agent_pos[2]:.1f}\nNectar: {int(self.total_nectar_collected)}'
        self.ax.text(self.agent_pos[0], self.agent_pos[1], self.agent_pos[2] + 1, 
                    info_text, fontsize=9, ha='center', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # Set viewing angle for better 3D perspective
        self.ax.view_init(elev=20, azim=45)
        
        # Add title with current status
        title = f'Step {self.steps_taken} | Energy: {int(self.agent_energy)}/{self.max_energy} | Nectar: {int(self.total_nectar_collected)}'
        self.ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Update display
        plt.draw()
        plt.pause(0.1)  # Pause for animation effect
        
        return self.fig
    
    def close(self):
        """Clean up resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# Test function for the 3D matplotlib environment
def test_complex_3d_matplotlib_environment():
    """Test the 3D matplotlib hummingbird environment."""
    print("Testing 3D Complex Hummingbird Environment with Matplotlib...")
    
    env = ComplexHummingbird3DMatplotlibEnv(grid_size=8, num_flowers=4, max_height=6, 
                                          render_mode="matplotlib")
    
    for episode in range(1):
        observation, info = env.reset(seed=episode)
        print(f"\n--- 3D Matplotlib Episode {episode + 1} ---")
        print(f"Agent: Position {observation['agent'][:3]}, Energy {observation['agent'][3]:.1f}")
        print(f"Flowers: {len(observation['flowers'])} flowers at various heights")
        
        terminated = False
        truncated = False
        step_count = 0
        
        while not (terminated or truncated) and step_count < 200:
            # More strategic action selection for better visualization
            if step_count < 50:
                # Explore different heights initially
                action = env.action_space.sample()
            else:
                # Try to find flowers
                closest_flower_idx = np.argmin([
                    np.sqrt((observation['agent'][0] - flower[0])**2 + 
                           (observation['agent'][1] - flower[1])**2 + 
                           (observation['agent'][2] - flower[2])**2)
                    for flower in observation['flowers']
                ])
                
                closest_flower = observation['flowers'][closest_flower_idx]
                
                # Move towards closest flower
                dx = closest_flower[0] - observation['agent'][0]
                dy = closest_flower[1] - observation['agent'][1]
                dz = closest_flower[2] - observation['agent'][2]
                
                if abs(dx) > abs(dy) and abs(dx) > abs(dz):
                    action = 3 if dx > 0 else 2  # Right or Left
                elif abs(dy) > abs(dz):
                    action = 1 if dy > 0 else 0  # Backward or Forward
                else:
                    action = 4 if dz > 0 else 5  # Up or Down
            
            observation, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Render every few steps for better visualization
            if step_count % 3 == 0:
                env.render()
            
            action_names = ["⬆️ Forward", "⬇️ Backward", "⬅️ Left", "➡️ Right", "🔝 Up", "🔻 Down", "🚁 Hover"]
            if step_count % 20 == 0:
                print(f"Step {step_count}: {action_names[action]} | "
                      f"Energy: {info['energy']:.1f} | "
                      f"Altitude: {info['altitude']:.1f} | "
                      f"Nectar: {info['total_nectar_collected']:.1f}")
            
            if terminated:
                if info['energy'] <= 0:
                    print(f"💀 Agent died from energy depletion after {step_count} steps!")
                elif info['altitude'] <= 0:
                    print(f"💥 Agent crashed to the ground after {step_count} steps!")
                break
        
        if not terminated:
            print(f"3D Matplotlib Episode completed: {step_count} steps, Energy: {info['energy']:.1f}, "
                  f"Altitude: {info['altitude']:.1f}, Nectar: {info['total_nectar_collected']:.1f}")
        
        # Keep the final plot open for viewing
        print("Close the matplotlib window to continue...")
        plt.show()
    
    env.close()
    print("3D Matplotlib environment test completed!")


if __name__ == "__main__":
    test_complex_3d_matplotlib_environment()
