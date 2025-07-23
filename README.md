# Hummingbird Reinforcement Learning Environment

A 2D reinforcement learning environment implemented with **Gymnasium** and Pygame where a hummingbird agent seeks to find a flower in a grid world.

## Project Structure

- `hummingbird_gymnasium.py` - Gymnasium-based environment implementation (recommended)
- `main.py` - Main script that runs the Gymnasium environment
- `hummingbird_env.py` - Legacy standalone environment (kept for reference)
- `test_env.py` - Test script for the legacy environment
- `requirements.txt` - Python dependencies

## Environment Description

### Grid World
- 20x20 grid (configurable)
- Hummingbird starts at center of grid
- Flower is randomly placed at the start of each episode

### Actions (Gymnasium Discrete(4))
- 0: Move Up
- 1: Move Down  
- 2: Move Left
- 3: Move Right

### Rewards
- +100: Agent reaches the flower (episode ends)
- -1: Every other step (encourages efficient pathfinding)

### Observation Space (Gymnasium Dict)
```python
{
    'agent': Box(0, grid_size-1, (2,), int64),    # Agent (x, y) position
    'flower': Box(0, grid_size-1, (2,), int64)    # Flower (x, y) position
}
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Gymnasium environment:
```bash
python main.py
```

Or run the standalone Gymnasium implementation:
```bash
python hummingbird_gymnasium.py
```

## Usage

The environment now follows the standard Gymnasium API, making it compatible with popular RL libraries like Stable-Baselines3, Ray RLlib, and others.

### Basic Usage with Gymnasium

```python
import gymnasium as gym
from hummingbird_gymnasium import HummingbirdEnv

# Create environment
env = HummingbirdEnv(grid_size=20, render_mode="human")

# Reset environment
observation, info = env.reset()

# Take actions
for _ in range(100):
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

### Integration with RL Libraries

The environment can be easily used with popular RL frameworks:

```python
# Example with Stable-Baselines3
from stable_baselines3 import PPO
from hummingbird_gymnasium import HummingbirdEnv

env = HummingbirdEnv(grid_size=10)
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

### Key Features

- **Standard Gymnasium API**: Compatible with modern RL libraries
- **Flexible Observation Space**: Dict-based observations for easy feature extraction
- **Configurable Grid Size**: Adjust difficulty by changing grid dimensions
- **Proper Seeding**: Reproducible experiments with seed support
- **Clean Resource Management**: Proper initialization and cleanup

## Environment Specifications

- **Action Space**: `Discrete(4)` 
- **Observation Space**: `Dict` with 'agent' and 'flower' keys
- **Reward Range**: [-1, 100]
- **Episode Termination**: When agent reaches flower
- **Max Episode Length**: No limit (can be added if needed)

## Future Enhancements

- Add obstacles to the grid
- Implement multiple flowers
- Add wind effects or other environmental factors
- Support for continuous action spaces
- Multi-agent scenarios
- Curriculum learning with varying difficulty levels
