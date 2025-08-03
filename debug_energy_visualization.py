"""
Debug script to check energy visualization accuracy.
This will help us understand why the energy bar shows red when survival rates are high.
"""

import numpy as np
from hummingbird_env import ComplexHummingbird3DMatplotlibEnv
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

def debug_energy_visualization():
    """Test energy visualization with a trained model to see actual vs displayed energy."""
    
    # Load a successful model (Model #28)
    try:
        model_path = "models/autonomous_training_28_10000k_continued_100k.zip"
        model = PPO.load(model_path)
        print(f"Loaded model: {model_path}")
    except Exception as e:
        print(f"Could not load model: {e}")
        return
    
    # Create environment
    env = ComplexHummingbird3DMatplotlibEnv(
        grid_size=10,
        num_flowers=8,
        max_energy=100,
        max_height=8,
        render_mode="matplotlib"
    )
    
    print(f"Environment max_energy: {env.max_energy}")
    
    # Run a test episode and track energy
    obs, _ = env.reset()
    print(f"Initial energy: {env.agent_energy}/{env.max_energy} (ratio: {env.agent_energy/env.max_energy:.3f})")
    
    energy_history = []
    step_count = 0
    done = False
    
    while not done and step_count < 200:  # Limit steps to prevent infinite loops
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        current_energy = env.agent_energy
        energy_ratio = current_energy / env.max_energy
        energy_history.append((step_count, current_energy, energy_ratio))
        
        # Check color logic
        if energy_ratio > 0.6:
            expected_color = 'blue'
        elif energy_ratio > 0.3:
            expected_color = 'yellow'
        else:
            expected_color = 'red'
        
        # Print every 20 steps
        if step_count % 20 == 0:
            print(f"Step {step_count}: Energy {current_energy:.1f}/{env.max_energy} "
                  f"(ratio: {energy_ratio:.3f}) -> Color: {expected_color}")
        
        step_count += 1
    
    print(f"\nEpisode ended after {step_count} steps")
    print(f"Final energy: {env.agent_energy:.1f}/{env.max_energy} (ratio: {env.agent_energy/env.max_energy:.3f})")
    print(f"Episode survived: {not (env.agent_energy <= 0)}")
    print(f"Nectar collected: {env.total_nectar_collected}")
    
    # Plot energy over time
    if energy_history:
        steps, energies, ratios = zip(*energy_history)
        
        plt.figure(figsize=(12, 8))
        
        # Energy over time
        plt.subplot(2, 2, 1)
        plt.plot(steps, energies, 'b-', linewidth=2)
        plt.axhline(y=30, color='orange', linestyle='--', label='Yellow threshold (30%)')
        plt.axhline(y=60, color='blue', linestyle='--', label='Blue threshold (60%)')
        plt.xlabel('Step')
        plt.ylabel('Energy')
        plt.title('Energy Over Time')
        plt.legend()
        plt.grid(True)
        
        # Energy ratio over time
        plt.subplot(2, 2, 2)
        plt.plot(steps, ratios, 'g-', linewidth=2)
        plt.axhline(y=0.3, color='orange', linestyle='--', label='Yellow threshold')
        plt.axhline(y=0.6, color='blue', linestyle='--', label='Blue threshold')
        plt.xlabel('Step')
        plt.ylabel('Energy Ratio')
        plt.title('Energy Ratio Over Time')
        plt.legend()
        plt.grid(True)
        
        # Color regions
        plt.subplot(2, 2, 3)
        colors = []
        for _, _, ratio in energy_history:
            if ratio > 0.6:
                colors.append('blue')
            elif ratio > 0.3:
                colors.append('yellow')
            else:
                colors.append('red')
        
        color_counts = {'blue': colors.count('blue'), 'yellow': colors.count('yellow'), 'red': colors.count('red')}
        plt.bar(color_counts.keys(), color_counts.values(), color=['blue', 'yellow', 'red'])
        plt.title('Time Spent in Each Color State')
        plt.ylabel('Number of Steps')
        
        # Final statistics
        plt.subplot(2, 2, 4)
        final_ratio = energy_history[-1][2]
        survival_status = "SURVIVED" if env.agent_energy > 0 else "DIED"
        nectar = env.total_nectar_collected
        
        plt.text(0.1, 0.8, f"Episode Summary:", fontsize=14, fontweight='bold')
        plt.text(0.1, 0.7, f"Final Energy: {env.agent_energy:.1f}/{env.max_energy}", fontsize=12)
        plt.text(0.1, 0.6, f"Final Ratio: {final_ratio:.3f}", fontsize=12)
        plt.text(0.1, 0.5, f"Status: {survival_status}", fontsize=12, 
                color='green' if survival_status == "SURVIVED" else 'red')
        plt.text(0.1, 0.4, f"Nectar: {nectar}", fontsize=12)
        plt.text(0.1, 0.3, f"Steps: {step_count}", fontsize=12)
        plt.text(0.1, 0.2, f"Color Distribution:", fontsize=12)
        plt.text(0.1, 0.1, f"Blue: {color_counts['blue']}, Yellow: {color_counts['yellow']}, Red: {color_counts['red']}", fontsize=10)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('energy_visualization_debug.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Summary
        print(f"\nVisualization Analysis:")
        print(f"- Blue steps (>60%): {color_counts['blue']}")
        print(f"- Yellow steps (30-60%): {color_counts['yellow']}")
        print(f"- Red steps (<30%): {color_counts['red']}")
        print(f"- If agent shows red most of the time but survives, there might be a display bug")

if __name__ == "__main__":
    debug_energy_visualization()
