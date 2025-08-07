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
    
    def __init__(self, grid_size=10, num_flowers=5, max_energy=100, max_height=8, render_mode=None, debug_mode=False):
        """Initialize the 3D matplotlib environment."""
        super().__init__()
        
        self.grid_size = grid_size
        self.num_flowers = num_flowers
        self.max_energy = max_energy
        self.max_height = max_height
        self.render_mode = render_mode
        self._debug_mode = debug_mode  # For milestone celebration messages
        
        self.METABOLIC_COST = 0.2           # Reduced from 0.3
        self.MOVE_HORIZONTAL_COST = 0.8     # Reduced from 1.2  
        self.MOVE_UP_ENERGY_COST = 1.2      # Further reduced from 1.8 to 1.2 for better upper region access
        self.MOVE_DOWN_ENERGY_COST = 0.5    # Reduced from 0.8
        self.HOVER_ENERGY_COST = 2.0        # Reduced from 3.0

        self.NECTAR_GAIN = 40               # Energy gained from nectar

        # Flower mechanics
        self.MAX_NECTAR = 40               
        self.NECTAR_REGEN_RATE = 0.3       
        self.FLOWER_COOLDOWN_TIME = 15     
        
        # Memory and learning bonuses
        self.FLOWER_VISIT_BONUS = 20       
        # Removed engineered reward constants for autonomous learning  
        
        # Observation space
        self.observation_space = spaces.Dict({
            'agent': spaces.Box(
                low=np.array([0, 0, 0, 0]), 
                high=np.array([grid_size-1, grid_size-1, max_height, max_energy]), 
                shape=(4,), 
                dtype=np.float32
            ),
            'flowers': spaces.Box(
                low=np.array([[0, 0, 0, 0, 0, 0]] * num_flowers), 
                high=np.array([[grid_size-1, grid_size-1, max_height, self.MAX_NECTAR, self.FLOWER_COOLDOWN_TIME, 1]] * num_flowers),
                shape=(num_flowers, 6), 
                dtype=np.float32
            )
        })
        
        # Action space (7 actions for 3D movement)
        self.action_space = spaces.Discrete(7)
        
        # Initialize state
        self.agent_pos = None
        self.agent_energy = None
        self.flowers = None  # List of [x, y, z, nectar_amount]
        self.flower_cooldowns = None  # Track cooldown times for each flower
        self.total_nectar_collected = 0
        self.steps_taken = 0
        self.last_flower_visited = -1  # Track last flower to prevent camping
        
        # NEW: Memory assistance variables
        self.last_flower_position = None   # Remember last flower location
        # Removed tracking variables for engineered rewards
        self.flowers_found_this_episode = 0
        
        # NEW: Real-time reward tracking for visualization
        self.current_episode_reward = 0.0
        self.last_step_reward = 0.0
        self.reward_history = []  # Track reward over time
        self.max_reward_history = 100  # Keep last 100 rewards for display
        
        # NEW: Current action state tracking
        self.current_action = 6  # Start with hover
        self.action_names = {
            0: "Forward", 1: "Backward", 2: "Left", 3: "Right", 
            4: "Up", 5: "Down", 6: "Hovering"
        }
        
        # Matplotlib 3D setup
        self.fig = None
        self.ax = None
        self.agent_trail = []  # Track agent's path
        self.max_trail_length = 50

        # Milestone tracking for reward curriculum
        self.milestone_150_reached = False
        self.milestone_200_reached = False
        self.milestone_250_reached = False
        
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset agent - keep original height for optimal access to all regions
        self.agent_pos = np.array([
            self.grid_size // 2, 
            self.grid_size // 2, 
            self.max_height // 2  # Height 4 - optimal for reaching both low and high regions
        ], dtype=np.float32)
        self.agent_energy = float(self.max_energy)
        
        # Reset flowers with fair environment generation
        self.flowers = []
        self.flowers = self._generate_fair_flower_distribution()
        self.flower_cooldowns = np.zeros(self.num_flowers, dtype=int)  # Initialize cooldowns
        self.total_nectar_collected = 0
        self.steps_taken = 0
        self.last_flower_visited = -1
        self.agent_trail = [self.agent_pos.copy()]
        
        # NEW: Reset memory tracking
        self.last_flower_position = None
        # Removed tracking reset for autonomous learning
        self.flowers_found_this_episode = 0
        self.prev_dist_to_flower = float('inf')  # Initialize distance tracking
        
        # NEW: Reset reward tracking
        self.current_episode_reward = 0.0
        self.last_step_reward = 0.0
        self.reward_history = []
        self.current_action = 6  # Start with hover

        # Reset milestone flags for the new episode
        self.milestone_150_reached = False
        self.milestone_200_reached = False
        self.milestone_250_reached = False
        
        return self._get_observation(), {}
    
    def _generate_fair_flower_distribution(self):
        """
        Generate flowers with fair distribution considering:
        1. Minimum spacing between flowers
        2. Balanced distribution across 3D space
        3. Energy accessibility from agent start position
        """
        flowers = []
        MIN_FLOWER_DISTANCE = 3.0  # Minimum distance between flowers
        MIN_AGENT_DISTANCE = 2.0   # Minimum distance from agent start
        MAX_ATTEMPTS = 1000        # Prevent infinite loops
        
        # Define 3D grid regions for balanced distribution
        regions = self._create_distribution_regions()
        
        # NEW: Randomize region order for balanced distribution
        region_indices = list(range(len(regions)))
        self.np_random.shuffle(region_indices)  # Randomize the order
        
        # Calculate target flowers per region more fairly
        flowers_per_region = self.num_flowers // len(regions)  # Base amount (0 for 5 flowers, 8 regions)
        extra_flowers = self.num_flowers % len(regions)        # Remaining flowers (5 for 5 flowers, 8 regions)
        
        total_flowers_placed = 0
        
        for i, region_idx in enumerate(region_indices):
            region = regions[region_idx]
            
            # Stop if we've placed enough flowers
            if total_flowers_placed >= self.num_flowers:
                break
                
            # First 'extra_flowers' regions get 1 flower each, others get 0
            target_flowers = 1 if i < extra_flowers else 0
            
            if target_flowers == 0:
                continue  # Skip regions that don't get flowers in this distribution
                
            region_flowers = 0
            attempts = 0
            
            while region_flowers < target_flowers and attempts < MAX_ATTEMPTS:
                attempts += 1
                
                # Generate position within region bounds
                flower_pos = np.array([
                    self.np_random.uniform(region['x_min'], region['x_max']),
                    self.np_random.uniform(region['y_min'], region['y_max']),
                    self.np_random.uniform(region['z_min'], region['z_max'])
                ])
                
                # Check minimum distance from agent
                dist_to_agent = np.linalg.norm(flower_pos - self.agent_pos)
                if dist_to_agent < MIN_AGENT_DISTANCE:
                    continue
                
                # Check minimum distance from other flowers
                too_close = False
                for existing_flower in flowers:
                    dist = np.linalg.norm(flower_pos - existing_flower[:3])
                    if dist < MIN_FLOWER_DISTANCE:
                        too_close = True
                        break
                
                if too_close:
                    continue
                
                # Check energy accessibility (can agent reach this flower?)
                if not self._is_energy_accessible(flower_pos):
                    continue
                
                # Generate nectar amount
                nectar = self.np_random.integers(20, self.MAX_NECTAR + 1)
                
                flowers.append([
                    float(flower_pos[0]), 
                    float(flower_pos[1]), 
                    float(flower_pos[2]), 
                    float(nectar)
                ])
                region_flowers += 1
                total_flowers_placed += 1
        
        # If we couldn't place enough flowers with strict rules, fill remaining with relaxed rules
        while len(flowers) < self.num_flowers:
            attempts = 0
            while attempts < MAX_ATTEMPTS and len(flowers) < self.num_flowers:
                attempts += 1
                
                flower_pos = np.array([
                    self.np_random.uniform(1, self.grid_size - 1),
                    self.np_random.uniform(1, self.grid_size - 1),
                    self.np_random.uniform(1, self.max_height - 1)
                ])
                
                # Relaxed rules: only check agent distance and basic flower spacing
                dist_to_agent = np.linalg.norm(flower_pos - self.agent_pos)
                if dist_to_agent < MIN_AGENT_DISTANCE:
                    continue
                
                # Reduced minimum distance for remaining flowers
                too_close = False
                for existing_flower in flowers:
                    dist = np.linalg.norm(flower_pos - existing_flower[:3])
                    if dist < MIN_FLOWER_DISTANCE * 0.7:  # 30% reduction
                        too_close = True
                        break
                
                if not too_close:
                    nectar = self.np_random.integers(20, self.MAX_NECTAR + 1)
                    flowers.append([
                        float(flower_pos[0]), 
                        float(flower_pos[1]), 
                        float(flower_pos[2]), 
                        float(nectar)
                    ])
                    break
        
        # Safety check: ensure we don't return more flowers than requested
        if len(flowers) > self.num_flowers:
            flowers = flowers[:self.num_flowers]
        
        return np.array(flowers, dtype=np.float32)
    
    def _create_distribution_regions(self):
        """Create 3D regions for balanced flower distribution."""
        regions = []
        
        # Divide space into 8 3D regions (2x2x2 grid)
        x_mid = self.grid_size / 2
        y_mid = self.grid_size / 2
        z_mid = self.max_height / 2
        
        margin = 1  # Avoid exact boundaries
        
        for x_low in [True, False]:  # Front/Back
            for y_low in [True, False]:  # Left/Right
                for z_low in [True, False]:  # Bottom/Top
                    region = {
                        'x_min': margin if x_low else x_mid,
                        'x_max': x_mid if x_low else self.grid_size - margin,
                        'y_min': margin if y_low else y_mid,
                        'y_max': y_mid if y_low else self.grid_size - margin,
                        'z_min': margin if z_low else z_mid,
                        'z_max': z_mid if z_low else self.max_height - margin
                    }
                    regions.append(region)
        
        return regions
    
    def _is_energy_accessible(self, flower_pos):
        """
        Check if a flower position is reachable given energy constraints.
        Uses simplified pathfinding and energy estimation.
        """
        # Calculate 3D Manhattan distance (approximates energy cost)
        manhattan_dist = np.sum(np.abs(flower_pos - self.agent_pos))
        
        # Estimate energy cost for travel with proper up/down movement costs
        # Horizontal movement costs less than vertical
        horizontal_dist = np.sum(np.abs(flower_pos[:2] - self.agent_pos[:2]))
        vertical_dist = abs(flower_pos[2] - self.agent_pos[2])
        
        # Use appropriate vertical cost based on movement direction
        if flower_pos[2] > self.agent_pos[2]:
            # Moving upward - more expensive
            vertical_cost = vertical_dist * self.MOVE_UP_ENERGY_COST
        else:
            # Moving downward - less expensive
            vertical_cost = vertical_dist * self.MOVE_DOWN_ENERGY_COST
        
        estimated_cost = (horizontal_dist * self.MOVE_HORIZONTAL_COST + 
                         vertical_cost +  # Now uses proper up/down costs
                         manhattan_dist * self.METABOLIC_COST)  # Metabolic cost per step
        
        # Reduced safety margin for better upper region accessibility
        safety_margin = 1.0  # Further reduced from 1.2 to 1.0
        required_energy = estimated_cost * safety_margin
        
        # More generous energy allowance for flower accessibility
        max_reasonable_distance = self.max_energy * 0.8  # Further increased from 0.75 to 0.8
        
        return required_energy <= max_reasonable_distance

    def step(self, action):
        """Execute one step in 3D space."""
        # Handle different action formats (numpy arrays, lists, scalars)
        try:
            # If action is iterable and has length > 0, take first element
            if hasattr(action, '__len__') and len(action) > 0:
                action = int(action[0])
            else:
                action = int(action)
        except (TypeError, IndexError):
            # Fallback for numpy scalars or other edge cases
            action = int(action)
        
        self.steps_taken += 1
        
        # Store current action for display
        self.current_action = action
        
        # DISCRETE MOVEMENT WITH PHYSICS-INFORMED ENERGY COSTS
        base_energy_cost = self.METABOLIC_COST  # Always pay metabolic cost
        
        # Calculate new position with discrete grid movement
        new_pos = self.agent_pos.copy()
        
        # Movement and energy costs
        if action == 0:  # Forward (North)
            new_pos[1] -= 1
            base_energy_cost += self.MOVE_HORIZONTAL_COST
        elif action == 1:  # Backward (South)
            new_pos[1] += 1
            base_energy_cost += self.MOVE_HORIZONTAL_COST
        elif action == 2:  # Left (West)
            new_pos[0] -= 1
            base_energy_cost += self.MOVE_HORIZONTAL_COST
        elif action == 3:  # Right (East)
            new_pos[0] += 1
            base_energy_cost += self.MOVE_HORIZONTAL_COST
        elif action == 4:  # Up - Fight against gravity (expensive)
            new_pos[2] += 1
            base_energy_cost += self.MOVE_UP_ENERGY_COST
        elif action == 5:  # Down - Gravity assisted (cheap)
            new_pos[2] -= 1
            base_energy_cost += self.MOVE_DOWN_ENERGY_COST
        elif action == 6:  # Hover - Most expensive (fighting gravity with no momentum)
            # No position change, but highest energy cost
            base_energy_cost += self.HOVER_ENERGY_COST
        
        # Boundary checks
        new_pos[0] = np.clip(new_pos[0], 0, self.grid_size - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.grid_size - 1)
        new_pos[2] = np.clip(new_pos[2], 0, self.max_height)
        
        self.agent_pos = new_pos
        
        # Store energy cost for display
        self._last_energy_cost = base_energy_cost
        
        # Update trail
        self.agent_trail.append(self.agent_pos.copy())
        if len(self.agent_trail) > self.max_trail_length:
            self.agent_trail.pop(0)
        
        # STEP 6: Apply energy consumption (metabolic + action costs calculated earlier)
        self.agent_energy -= base_energy_cost
        
        # Flower collision and nectar collection with enhanced memory rewards
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
                    # CORE OBJECTIVE: Nectar collection (the only guidance allowed)
                    self.total_nectar_collected += nectar_collected
                    reward += nectar_collected  # Direct reward for nectar collection
                else:
                    # Removed penalties for empty flowers - agent must learn this autonomously
                    if hasattr(self, '_debug_mode') and self._debug_mode:
                        # Optional debug message
                        if hasattr(self, '_debug_mode') and self._debug_mode:
                            print(f"❌ Visited empty flower! Agent must learn from experience.")
            else:
                # NEW: Add a penalty for visiting a flower that is on cooldown.
                # This teaches the agent to not loiter at depleted flowers.
                reward -= 5.0
                
                # Agent must learn from consequences - no explicit penalty teaching
                if hasattr(self, '_debug_mode') and self._debug_mode:
                    cooldown_remaining = self.flower_cooldowns[on_flower_idx]
                    print(f"❌ Visited flower on cooldown! Remaining: {cooldown_remaining} steps")
        
        # Update flower cooldowns and regenerate nectar
        self.flower_cooldowns = np.maximum(0, self.flower_cooldowns - 1)
        self._regenerate_nectar()
        
        # =============================================================
        # AUTONOMOUS LEARNING: NO PROXIMITY GUIDANCE
        # =============================================================
        # Removed all proximity rewards - agent must discover flower locations autonomously
        
        # Removed exploration bonus - agent must develop own exploration strategy
        
        # Only keep core penalty
        reward -= 0.1  # Basic step penalty to encourage efficiency, increased from 0.05
        
        # NEW: Add a continuous reward for maintaining high energy to encourage survival
        # This reward is proportional to the current energy level.
        energy_ratio = self.agent_energy / self.max_energy
        reward += energy_ratio * 0.05  # Small bonus for high energy

        # Removed all remaining engineered rewards for autonomous learning

        # --- Survival Milestone Rewards ---
        # Check if the agent has crossed a new milestone
        if self.steps_taken >= 150 and not self.milestone_150_reached:
            reward += 25.0
            self.milestone_150_reached = True
        if self.steps_taken >= 200 and not self.milestone_200_reached:
            reward += 50.0
            self.milestone_200_reached = True
        if self.steps_taken >= 250 and not self.milestone_250_reached:
            reward += 75.0
            self.milestone_250_reached = True
        
        # Termination conditions
        terminated = False
        truncated = False
        
        # Death from energy depletion - the challenge!
        if self.agent_energy <= 0:
            terminated = True
            reward -= 50  # Big penalty for dying
        
        # Success condition: survive for 200 steps
        if self.steps_taken >= 200:
            truncated = True
            reward += 200  # Bonus for surviving the challenge!
        
        # Additional efficiency bonus for high collection rates
        # if self.steps_taken > 0:
        #     collection_rate = self.total_nectar_collected / self.steps_taken
        #     if collection_rate > 0.5:  # Very efficient collection
        #         reward += 3.0
        #     elif collection_rate > 0.3:  # Good collection rate
        #         reward += 1.0
        
        info = {
            'total_nectar_collected': self.total_nectar_collected,
            'energy': self.agent_energy,
            'steps': self.steps_taken,
            'flowers_available': np.sum((self.flowers[:, 3] > 0) & (self.flower_cooldowns == 0)),
            'flowers_in_cooldown': np.sum(self.flower_cooldowns > 0),
            'altitude': self.agent_pos[2],
            'last_flower_visited': self.last_flower_visited,
            # Energy cost information
            'energy_cost_breakdown': {
                'metabolic': self.METABOLIC_COST,
                'action_cost': base_energy_cost - self.METABOLIC_COST,
                'total': base_energy_cost
            },
            # Enhanced debugging information
            'flowers_found_this_episode': self.flowers_found_this_episode,
            'collection_rate': self.total_nectar_collected / max(1, self.steps_taken),
            'energy_ratio': self.agent_energy / self.max_energy,
            'nearest_flower_distance': min([np.linalg.norm(self.agent_pos - flower[:3]) 
                                           for flower in self.flowers]) if len(self.flowers) > 0 else float('inf'),
            'areas_explored': 0  # Removed for autonomous learning
        }
        
        # NEW: Track reward for real-time display
        self.last_step_reward = reward
        self.current_episode_reward += reward
        self.reward_history.append(reward)
        if len(self.reward_history) > self.max_reward_history:
            self.reward_history.pop(0)  # Keep only recent rewards
        
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
        """Get current observation with RAW data only - let agent learn patterns."""
        # RAW AGENT DATA ONLY - no engineered features
        agent_obs = np.array([
            self.agent_pos[0], 
            self.agent_pos[1], 
            self.agent_pos[2], 
            self.agent_energy
        ], dtype=np.float32)
        
        # Enhanced flower observations with availability status
        enhanced_flowers = []
        for i in range(len(self.flowers)):
            flower = self.flowers[i]
            cooldown_remaining = self.flower_cooldowns[i]
            nectar_amount = flower[3]
            is_available = 1.0 if (cooldown_remaining == 0 and nectar_amount > 0) else 0.0
            
            enhanced_flower = np.array([
                flower[0],          # x position
                flower[1],          # y position  
                flower[2],          # z position
                nectar_amount,      # current nectar amount
                cooldown_remaining, # cooldown timer (0 = ready)
                is_available        # binary availability flag
            ], dtype=np.float32)
            
            enhanced_flowers.append(enhanced_flower)
            
        enhanced_flowers = np.array(enhanced_flowers, dtype=np.float32)
        
        return {
            'agent': agent_obs,
            'flowers': enhanced_flowers
        }
    
    def _calculate_efficiency_bonus(self):
        """Calculate bonus for efficient flower visitation patterns."""
        bonus = 0.0
        
        # Bonus for visiting available flowers
        available_flowers = [i for i, flower in enumerate(self.flowers) 
                           if self.flower_cooldowns[i] == 0 and flower[3] > 0]
        
        # Simple efficiency check: if last visited flower was available, give bonus
        if hasattr(self, 'last_flower_visited') and self.last_flower_visited >= 0:
            if self.last_flower_visited in available_flowers:
                bonus += 2.0  # Bonus for visiting available flower
                
        # Extra bonus for finding flowers this episode
        if hasattr(self, 'flowers_found_this_episode') and self.flowers_found_this_episode > 0:
            bonus += 1.0 * self.flowers_found_this_episode
                
        return bonus

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
        
        # Calculate recent average reward for color coding
        recent_avg_reward = np.mean(self.reward_history[-10:]) if len(self.reward_history) >= 10 else (np.mean(self.reward_history) if self.reward_history else 0)
        
        # Add enhanced text info near agent with real-time rewards and physics
        reward_color = 'green' if self.last_step_reward > 0 else ('red' if self.last_step_reward < 0 else 'black')
        
        # Action indicators based on current action
        action_names = ["↑", "↓", "←", "→", "⬆️", "⬇️", "🚁"]
        action_indicator = action_names[self.current_action] if hasattr(self, 'current_action') else "🐦"
        
        info_text = (f'Energy: {int(self.agent_energy)}\n'
                    f'Altitude: {self.agent_pos[2]:.1f}\n'
                    f'Action: {action_indicator}\n'
                    f'Nectar: {int(self.total_nectar_collected)}\n'
                    f'Episode Reward: {self.current_episode_reward:.1f}\n'
                    f'Last Step: {self.last_step_reward:+.1f}')
        
        # Color-coded background based on recent performance
        bg_color = 'lightgreen' if recent_avg_reward > 0 else ('lightcoral' if recent_avg_reward < -1 else 'lightblue')
        
        self.ax.text(self.agent_pos[0], self.agent_pos[1], self.agent_pos[2] + 1, 
                    info_text, fontsize=8, ha='center', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_color, alpha=0.8, edgecolor=reward_color))
        
        # Set viewing angle for better 3D perspective
        self.ax.view_init(elev=20, azim=45)
        
        # Add title with current status including action and bird state
        current_state = self.action_names.get(self.current_action, "Unknown")
        energy_cost = getattr(self, '_last_energy_cost', 0)
        title = (f'Step {self.steps_taken} | Energy: {int(self.agent_energy)}/{self.max_energy} | '
                f'Nectar: {int(self.total_nectar_collected)} | {current_state} | Cost: {energy_cost:.1f} | '
                f'Reward: {self.current_episode_reward:.1f} ({self.last_step_reward:+.1f})')
        self.ax.set_title(title, fontsize=11, fontweight='bold')
        
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
    
    env = ComplexHummingbird3DMatplotlibEnv(grid_size=8, num_flowers=8, max_height=6, 
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
                else:
                    print(f"Episode terminated after {step_count} steps!")
                break
        
        if not terminated:
            if step_count >= 300:
                print(f"🏆 SUCCESS! Agent survived the 300-step challenge! Final stats:")
                print(f"   Steps: {step_count}, Energy: {info['energy']:.1f}, Nectar: {info['total_nectar_collected']:.1f}")
            else:
                print(f"3D Matplotlib Episode completed: {step_count} steps, Energy: {info['energy']:.1f}, "
                      f"Altitude: {info['altitude']:.1f}, Nectar: {info['total_nectar_collected']:.1f}")
        
        # Keep the final plot open for viewing
        print("Close the matplotlib window to continue...")
        plt.show()
    
    env.close()
    print("3D Matplotlib environment test completed!")


if __name__ == "__main__":
    test_complex_3d_matplotlib_environment()
