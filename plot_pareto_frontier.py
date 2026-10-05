"""
Create a Pareto frontier plot showing the tradeoff between GPU efficiency (GPU-hours)
and speed (total time) for different RL training strategies.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Change to the data directory
os.chdir('experiments-elastic/gpu-hour-baselines')

# Load the data
with open('sweep_results.json', 'r') as f:
    data = json.load(f)

# Extract data for each strategy
strategies = {
    'Synchronous': [],
    'Elastic': [],
    'One-Step Overlap': []
}

# Process sync data
for config in data['sync']:
    strategies['Synchronous'].append({
        'total_time': config['total_time'],
        'gpu_hours': config['gpu_hours'],
        'total_gpus': config['total_gpus'],
        'label': f"{config['total_gpus']} GPUs"
    })

# Process elastic data
for config in data['elastic']:
    strategies['Elastic'].append({
        'total_time': config['total_time'],
        'gpu_hours': config['gpu_hours'],
        'total_gpus': config['total_gpus'],
        'label': f"{config['total_gpus']}G ({config['num_dedicated_inference']}d+{config['num_elastic']}e)"
    })

# Process one-step overlap data
for config in data['one_step_overlap']:
    strategies['One-Step Overlap'].append({
        'total_time': config['total_time'],
        'gpu_hours': config['gpu_hours'],
        'total_gpus': config['total_gpus'],
        'label': f"{config['total_gpus']}G ({config['num_inference_gpus']}i+{config['num_training_gpus']}t)"
    })

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))

colors = {
    'Synchronous': '#2E86AB',
    'Elastic': '#A23B72',
    'One-Step Overlap': '#F18F01'
}

markers = {
    'Synchronous': 'o',
    'Elastic': 's',
    'One-Step Overlap': '^'
}

# Plot each strategy
for strategy_name, configs in strategies.items():
    times = [c['total_time'] / 3600 for c in configs]  # Convert to hours
    gpu_hours = [c['gpu_hours'] for c in configs]

    ax.scatter(times, gpu_hours,
              c=colors[strategy_name],
              marker=markers[strategy_name],
              s=100,
              alpha=0.6,
              label=strategy_name,
              edgecolors='black',
              linewidth=0.5)

# Find and plot the Pareto frontier
all_points = []
for strategy_name, configs in strategies.items():
    for config in configs:
        all_points.append({
            'time_hours': config['total_time'] / 3600,
            'gpu_hours': config['gpu_hours'],
            'strategy': strategy_name,
            'label': config['label']
        })

# Sort by time
all_points.sort(key=lambda x: x['time_hours'])

# Find Pareto frontier (points where no other point is better in both dimensions)
pareto_points = []
min_gpu_hours_so_far = float('inf')

for point in all_points:
    if point['gpu_hours'] < min_gpu_hours_so_far:
        pareto_points.append(point)
        min_gpu_hours_so_far = point['gpu_hours']

# Draw Pareto frontier line
if len(pareto_points) > 1:
    pareto_times = [p['time_hours'] for p in pareto_points]
    pareto_gpu_hours = [p['gpu_hours'] for p in pareto_points]
    ax.plot(pareto_times, pareto_gpu_hours,
           'k--',
           linewidth=2,
           alpha=0.5,
           label='Pareto Frontier',
           zorder=1)

    # Highlight Pareto optimal points
    for point in pareto_points:
        ax.scatter(point['time_hours'], point['gpu_hours'],
                  c=colors[point['strategy']],
                  marker=markers[point['strategy']],
                  s=200,
                  edgecolors='gold',
                  linewidth=3,
                  zorder=10)

# Add labels for interesting points (Pareto optimal)
for i, point in enumerate(pareto_points):
    offset_x = 5 if i % 2 == 0 else -5
    offset_y = 10 if i % 2 == 0 else -10
    ha = 'left' if i % 2 == 0 else 'right'

    ax.annotate(point['label'],
               xy=(point['time_hours'], point['gpu_hours']),
               xytext=(offset_x, offset_y),
               textcoords='offset points',
               fontsize=8,
               ha=ha,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1))

ax.set_xlabel('Total Time (hours)', fontsize=12, fontweight='bold')
ax.set_ylabel('GPU-Hours', fontsize=12, fontweight='bold')
ax.set_title('Pareto Frontier: GPU Efficiency vs Speed Tradeoff\n(Lower is better for both axes)',
            fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')

# Add a text box explaining the plot
textstr = 'Points on the Pareto frontier (gold outline)\nrepresent optimal tradeoffs between\nspeed and efficiency'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)

plt.tight_layout()
output_path = '../../pareto_frontier.png'  # Save in slime root
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved {os.path.abspath(output_path)}")

# Print Pareto optimal configurations
print("\n" + "="*60)
print("PARETO OPTIMAL CONFIGURATIONS")
print("="*60)
for point in pareto_points:
    print(f"\n{point['strategy']}: {point['label']}")
    print(f"  Time: {point['time_hours']:.2f} hours")
    print(f"  GPU-Hours: {point['gpu_hours']:.2f}")
    print(f"  Efficiency: {point['gpu_hours']/point['time_hours']:.2f} GPUs utilized")

plt.show()
