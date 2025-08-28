#!/usr/bin/env python3
"""
Flower Distribution Analysis Tool
Analyze and visualize how flowers are distributed in the 3D environment
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hummingbird_env import ComplexHummingbird3DMatplotlibEnv

def analyze_flower_distribution():
    """Analyze the flower distribution mechanism and create visualizations."""
    
    print("🌸 FLOWER DISTRIBUTION ANALYSIS")
    print("=" * 50)
    
    # Create environment instance
    env = ComplexHummingbird3DMatplotlibEnv()
    
    print("\n📋 DISTRIBUTION PARAMETERS:")
    print("-" * 30)
    print(f"Number of Flowers: {env.num_flowers}")
    print(f"Grid Size: {env.grid_size}x{env.grid_size}x{env.max_height}")
    print(f"Max Nectar per Flower: {env.MAX_NECTAR}")
    print(f"Nectar Range: 20-{env.MAX_NECTAR}")
    print(f"Agent Start Position: [{env.grid_size//2}, {env.grid_size//2}, {env.max_height//2}]")
    
    print("\n🎯 PLACEMENT CONSTRAINTS:")
    print("-" * 30)
    print(f"Minimum Distance Between Flowers: 3.0 units")
    print(f"Minimum Distance from Agent Start: 2.0 units")
    print(f"Energy Accessibility Check: Yes")
    print(f"Regional Distribution: 8 regions (2x2x2 grid)")
    
    # Debug the energy accessibility for different regions
    debug_energy_accessibility(env)
    
    # Generate multiple distributions to analyze
    print("\n🔄 GENERATING SAMPLE DISTRIBUTIONS...")
    distributions = []
    for i in range(10):
        env.reset()
        distributions.append(env.flowers.copy())
    
    analyze_distribution_statistics(distributions, env)
    create_distribution_visualizations(distributions, env)
    
    env.close()

def debug_energy_accessibility(env):
    """Debug why certain regions aren't getting flowers."""
    
    print("\n🔍 ENERGY ACCESSIBILITY DEBUG:")
    print("-" * 40)
    
    # Test accessibility for each region
    regions = env._create_distribution_regions()
    agent_pos = np.array([env.grid_size//2, env.grid_size//2, env.max_height//2])
    
    print(f"Agent position: {agent_pos}")
    print(f"Max energy: {env.max_energy}")
    print(f"Energy costs: Horizontal={env.MOVE_HORIZONTAL_COST}, Up={env.MOVE_UP_ENERGY_COST}, Metabolic={env.METABOLIC_COST}")
    
    for i, region in enumerate(regions):
        print(f"\nRegion {i+1}:")
        print(f"  Bounds: X[{region['x_min']:.1f}-{region['x_max']:.1f}], Y[{region['y_min']:.1f}-{region['y_max']:.1f}], Z[{region['z_min']:.1f}-{region['z_max']:.1f}]")
        
        # Test center of region
        center_pos = np.array([
            (region['x_min'] + region['x_max']) / 2,
            (region['y_min'] + region['y_max']) / 2,
            (region['z_min'] + region['z_max']) / 2
        ])
        
        # Test corner positions (worst case)
        corner_pos = np.array([region['x_max'], region['y_max'], region['z_max']])
        
        # Calculate costs manually
        for pos, name in [(center_pos, "center"), (corner_pos, "corner")]:
            horizontal_dist = np.sum(np.abs(pos[:2] - agent_pos[:2]))
            vertical_dist = abs(pos[2] - agent_pos[2])
            manhattan_dist = np.sum(np.abs(pos - agent_pos))
            
            estimated_cost = (horizontal_dist * env.MOVE_HORIZONTAL_COST + 
                             vertical_dist * env.MOVE_UP_ENERGY_COST +
                             manhattan_dist * env.METABOLIC_COST)
            
            # Current accessibility check
            safety_margin = 1.2
            required_energy = estimated_cost * safety_margin
            max_allowance = env.max_energy * 0.75
            
            accessible = required_energy <= max_allowance
            
            print(f"    {name.capitalize()} [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]:")
            print(f"      Distance: H={horizontal_dist:.1f}, V={vertical_dist:.1f}, Total={manhattan_dist:.1f}")
            print(f"      Cost: {estimated_cost:.1f} → Required: {required_energy:.1f} / Allowed: {max_allowance:.1f}")
            print(f"      Accessible: {'✅ YES' if accessible else '❌ NO'}")
    
    # Test original accessibility check
    print(f"\n🧪 TESTING ORIGINAL vs MODIFIED ACCESSIBILITY:")
    test_pos = np.array([8.5, 8.5, 7.5])  # Upper region corner
    
    horizontal_dist = np.sum(np.abs(test_pos[:2] - agent_pos[:2]))
    vertical_dist = abs(test_pos[2] - agent_pos[2])
    manhattan_dist = np.sum(np.abs(test_pos - agent_pos))
    
    estimated_cost = (horizontal_dist * env.MOVE_HORIZONTAL_COST + 
                     vertical_dist * env.MOVE_UP_ENERGY_COST +
                     manhattan_dist * env.METABOLIC_COST)
    
    # Original settings
    orig_required = estimated_cost * 1.5
    orig_allowance = env.max_energy * 0.6
    orig_accessible = orig_required <= orig_allowance
    
    # Modified settings
    new_required = estimated_cost * 1.2
    new_allowance = env.max_energy * 0.75
    new_accessible = new_required <= new_allowance
    
    print(f"Test position: {test_pos}")
    print(f"Original: {orig_required:.1f} ≤ {orig_allowance:.1f} = {'✅ YES' if orig_accessible else '❌ NO'}")
    print(f"Modified: {new_required:.1f} ≤ {new_allowance:.1f} = {'✅ YES' if new_accessible else '❌ NO'}")

def analyze_distribution_statistics(distributions, env):
    """Analyze statistical properties of flower distributions."""
    
    print("\n📊 DISTRIBUTION STATISTICS (10 samples):")
    print("-" * 40)
    
    # Collect statistics
    all_distances = []
    all_heights = []
    all_nectar = []
    all_agent_distances = []
    region_counts = []
    
    for dist in distributions:
        flowers = dist
        
        # Inter-flower distances
        for i in range(len(flowers)):
            for j in range(i+1, len(flowers)):
                distance = np.linalg.norm(flowers[i][:3] - flowers[j][:3])
                all_distances.append(distance)
        
        # Heights
        all_heights.extend(flowers[:, 2])
        
        # Nectar amounts
        all_nectar.extend(flowers[:, 3])
        
        # Distances from agent start (agent starts at height 4, not 1)
        agent_start = np.array([env.grid_size//2, env.grid_size//2, env.max_height//2])
        for flower in flowers:
            dist_to_agent = np.linalg.norm(flower[:3] - agent_start)
            all_agent_distances.append(dist_to_agent)
        
        # Region distribution
        regions = count_flowers_per_region(flowers, env)
        region_counts.append(regions)
    
    # Print statistics
    print(f"Inter-flower Distance:")
    print(f"  Mean: {np.mean(all_distances):.2f} ± {np.std(all_distances):.2f}")
    print(f"  Min: {np.min(all_distances):.2f}")
    print(f"  Max: {np.max(all_distances):.2f}")
    
    print(f"\nFlower Heights:")
    print(f"  Mean: {np.mean(all_heights):.2f} ± {np.std(all_heights):.2f}")
    print(f"  Range: {np.min(all_heights):.1f} - {np.max(all_heights):.1f}")
    
    print(f"\nNectar Amounts:")
    print(f"  Mean: {np.mean(all_nectar):.1f} ± {np.std(all_nectar):.1f}")
    print(f"  Range: {np.min(all_nectar):.0f} - {np.max(all_nectar):.0f}")
    
    print(f"\nDistance from Agent Start:")
    print(f"  Mean: {np.mean(all_agent_distances):.2f} ± {np.std(all_agent_distances):.2f}")
    print(f"  Min: {np.min(all_agent_distances):.2f}")
    print(f"  Max: {np.max(all_agent_distances):.2f}")
    
    # Region distribution analysis
    mean_region_counts = np.mean(region_counts, axis=0)
    print(f"\nRegional Distribution (average flowers per region):")
    for i, count in enumerate(mean_region_counts):
        print(f"  Region {i+1}: {count:.1f} flowers")

def count_flowers_per_region(flowers, env):
    """Count how many flowers are in each of the 8 regions."""
    
    x_mid = env.grid_size / 2
    y_mid = env.grid_size / 2
    z_mid = env.max_height / 2
    
    region_counts = [0] * 8
    
    for flower in flowers:
        x, y, z = flower[:3]
        
        # Determine region (2x2x2 = 8 regions)
        x_high = x >= x_mid
        y_high = y >= y_mid
        z_high = z >= z_mid
        
        region_idx = (int(x_high) * 4 + int(y_high) * 2 + int(z_high))
        region_counts[region_idx] += 1
    
    return region_counts

def create_distribution_visualizations(distributions, env):
    """Create comprehensive visualizations of flower distribution."""
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 3D Scatter Plot of Multiple Distributions
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(distributions)))
    for i, (dist, color) in enumerate(zip(distributions[:5], colors[:5])):  # Show first 5
        flowers = dist
        ax1.scatter(flowers[:, 0], flowers[:, 1], flowers[:, 2], 
                   c=[color], alpha=0.7, s=60, label=f'Sample {i+1}')
    
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_zlabel('Z Coordinate (Height)')
    ax1.set_title('3D Flower Distributions\n(5 Random Samples)')
    ax1.legend()
    
    # 2. Height Distribution
    ax2 = fig.add_subplot(2, 3, 2)
    all_heights = np.concatenate([dist[:, 2] for dist in distributions])
    ax2.hist(all_heights, bins=15, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Height (Z Coordinate)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Flower Height Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. Inter-flower Distance Distribution
    ax3 = fig.add_subplot(2, 3, 3)
    all_distances = []
    for dist in distributions:
        flowers = dist
        for i in range(len(flowers)):
            for j in range(i+1, len(flowers)):
                distance = np.linalg.norm(flowers[i][:3] - flowers[j][:3])
                all_distances.append(distance)
    
    ax3.hist(all_distances, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(3.0, color='red', linestyle='--', label='Min Distance (3.0)')
    ax3.set_xlabel('Distance Between Flowers')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Inter-flower Distance Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Nectar Amount Distribution
    ax4 = fig.add_subplot(2, 3, 4)
    all_nectar = np.concatenate([dist[:, 3] for dist in distributions])
    ax4.hist(all_nectar, bins=15, alpha=0.7, color='orange', edgecolor='black')
    ax4.set_xlabel('Nectar Amount')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Nectar Amount Distribution')
    ax4.grid(True, alpha=0.3)
    
    # 5. Regional Distribution
    ax5 = fig.add_subplot(2, 3, 5)
    region_counts = []
    for dist in distributions:
        regions = count_flowers_per_region(dist, env)
        region_counts.append(regions)
    
    mean_region_counts = np.mean(region_counts, axis=0)
    std_region_counts = np.std(region_counts, axis=0)
    
    regions = [f'R{i+1}' for i in range(8)]
    bars = ax5.bar(regions, mean_region_counts, yerr=std_region_counts, 
                   alpha=0.7, color='purple', capsize=5)
    ax5.set_xlabel('Region')
    ax5.set_ylabel('Average Number of Flowers')
    ax5.set_title('Regional Distribution\n(8 3D Regions)')
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, mean_region_counts):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}', ha='center', va='bottom')
    
    # 6. Distance from Agent Start
    ax6 = fig.add_subplot(2, 3, 6)
    agent_start = np.array([env.grid_size//2, env.grid_size//2, env.max_height//2])
    all_agent_distances = []
    
    for dist in distributions:
        for flower in dist:
            dist_to_agent = np.linalg.norm(flower[:3] - agent_start)
            all_agent_distances.append(dist_to_agent)
    
    ax6.hist(all_agent_distances, bins=15, alpha=0.7, color='red', edgecolor='black')
    ax6.axvline(2.0, color='blue', linestyle='--', label='Min Distance (2.0)')
    ax6.set_xlabel('Distance from Agent Start')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Distance from Agent Start')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./models/flower_distribution_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Flower distribution analysis saved as flower_distribution_analysis.png")
    plt.show()

def visualize_distribution_algorithm():
    """Visualize the step-by-step distribution algorithm."""
    
    print("\n🔧 DISTRIBUTION ALGORITHM:")
    print("-" * 30)
    print("1. REGIONAL APPROACH:")
    print("   • Divide 3D space into 8 regions (2x2x2)")
    print("   • Distribute flowers evenly across regions")
    print("   • Each region gets ~0.6 flowers on average (5 flowers / 8 regions)")
    
    print("\n2. PLACEMENT CONSTRAINTS:")
    print("   • Minimum 3.0 units between flowers")
    print("   • Minimum 2.0 units from agent start position")
    print("   • Energy accessibility check (can agent reach it?)")
    print("   • Avoid grid boundaries (1-unit margin)")
    
    print("\n3. FALLBACK MECHANISM:")
    print("   • If regional placement fails, use relaxed rules")
    print("   • Reduce minimum distance to 2.1 units (70% of original)")
    print("   • Place anywhere in grid with basic constraints")
    
    print("\n4. ENERGY ACCESSIBILITY:")
    print("   • Calculate Manhattan distance to flower")
    print("   • Estimate energy cost (horizontal + vertical + metabolic)")
    print("   • Apply 20% safety margin (reduced from 50%)")
    print("   • Must be reachable within 75% of max energy (increased from 60%)")
    print("   • Agent starts at height 4 for balanced access to all regions")
    
    print("\n5. NECTAR RANDOMIZATION:")
    print("   • Random nectar amount: 20-35 units")
    print("   • Each flower gets different amount")
    print("   • Encourages exploration of multiple flowers")

if __name__ == "__main__":
    analyze_flower_distribution()
    visualize_distribution_algorithm()
