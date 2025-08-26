import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

def analyze_and_plot_trajectory(json_file_path):
    """
    Loads trajectory data from a JSON file, performs analysis, and generates
    3D trajectory plot and 2D plots for key metrics.
    
    Args:
        json_file_path (str): The path to the JSON trajectory file.
    """
    try:
        with open(json_file_path, 'r') as f:
            trajectory_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: The file at '{json_file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: Could not decode JSON from '{json_file_path}'. "
              "Please ensure the file is a valid JSON array.")
        return

    # --- 1. Extract Data from JSON ---
    steps = [d['step'] for d in trajectory_data]
    positions = np.array([d['position'] for d in trajectory_data])
    energy = [d['energy'] for d in trajectory_data]
    nectar_collected = [d['nectar_collected'] for d in trajectory_data]

    # --- 2. Calculate Cumulative Metrics ---
    # The JSON provides nectar_collected per step. We need the cumulative sum.
    cumulative_nectar = np.cumsum(nectar_collected)
    
    # --- 3. Plotting ---
    base_save_path = os.path.join(os.path.dirname(json_file_path), "trajectory_analysis")
    os.makedirs(base_save_path, exist_ok=True)
    
    # Use a clean style for better visual appeal
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Plot 1: 3D Trajectory ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the path of the agent
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            label='Agent Trajectory', color='blue', linewidth=2, marker='o', markersize=4, alpha=0.6)
    
    # Mark start and end points
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
               c='green', s=100, label='Start', marker='^', edgecolors='k')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
               c='red', s=100, label='End', marker='X', edgecolors='k')

    ax.set_title('3D Agent Trajectory', fontsize=16)
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_zlabel('Z Position', fontsize=12)
    ax.legend()
    ax.grid(True)
    
    # Set axis limits based on environment size (adjust if needed)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_save_path, '3d_trajectory.png'))
    print(f"✅ Saved 3D trajectory plot to: {os.path.join(base_save_path, '3d_trajectory.png')}")
    plt.close(fig)

    # --- Plot 2: Energy and Cumulative Nectar over Time ---
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Energy Plot (Primary Y-axis)
    ax1.plot(steps, energy, color='tab:red', linewidth=2)
    ax1.set_xlabel('Steps', fontsize=12)
    ax1.set_ylabel('Energy', color='tab:red', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Cumulative Nectar Plot (Secondary Y-axis)
    ax2 = ax1.twinx()
    ax2.plot(steps, cumulative_nectar, color='tab:green', linewidth=2)
    ax2.set_ylabel('Cumulative Nectar Collected', color='tab:green', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.set_ylim(0, max(1, max(cumulative_nectar) * 1.1)) # Adjust y-limit dynamically
    
    plt.title('Agent Energy and Cumulative Nectar Over Time', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_save_path, 'energy_and_nectar_over_time.png'))
    print(f"✅ Saved energy/nectar plot to: {os.path.join(base_save_path, 'energy_and_nectar_over_time.png')}")
    plt.close(fig)

    print("\n🎉 Analysis complete! Check the 'trajectory_analysis' folder for your plots.")

if __name__ == "__main__":
    # --- Instructions ---
    # 1. Place your JSON file in the same directory as this script.
    # 2. Update the `json_file_path` below with your file's name.
    
    # Example:
    # json_file_path = 'trajectory_run_100.json'
    
    # Or, if you want to make it an argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = 'trajectory_run_100.json'
        
    print(f"Using JSON file: {file_path}")
    analyze_and_plot_trajectory(file_path)
