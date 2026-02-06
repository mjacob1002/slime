"""
Pareto Frontier Analysis: GPU-Hours vs Total Time Tradeoff

Visualizes all simulation configurations in the time-cost space and identifies
Pareto-optimal configurations that are not dominated on both metrics.
"""

import json
import matplotlib.pyplot as plt

# Load data
with open('sweep_results_token_based.json', 'r') as f:
    data = json.load(f)

# Extract all configurations
configs = []

# Sync
for entry in data['sync']:
    configs.append({
        'strategy': 'Sync',
        'total_gpus': entry['total_gpus'],
        'total_time': entry['total_time'],
        'gpu_hours': entry['gpu_hours'],
        'label': f"Sync-{entry['total_gpus']}GPU"
    })

# Elastic
for entry in data['elastic']:
    configs.append({
        'strategy': 'Elastic',
        'total_gpus': entry['total_gpus'],
        'total_time': entry['total_time'],
        'gpu_hours': entry['gpu_hours'],
        'num_dedicated': entry['num_dedicated_inference'],
        'num_elastic': entry['num_elastic'],
        'label': f"Elastic-{entry['total_gpus']}GPU({entry['num_dedicated_inference']}d/{entry['num_elastic']}e)"
    })

# Async
for entry in data['one_step_overlap']:
    configs.append({
        'strategy': 'Async',
        'total_gpus': entry['total_gpus'],
        'total_time': entry['total_time'],
        'gpu_hours': entry['gpu_hours'],
        'num_inference': entry['num_inference_gpus'],
        'num_training': entry['num_training_gpus'],
        'label': f"Async-{entry['total_gpus']}GPU({entry['num_inference_gpus']}i/{entry['num_training_gpus']}t)"
    })

# Compute Pareto frontier
def is_pareto_optimal(config, all_configs):
    """Check if config is not dominated by any other config."""
    for other in all_configs:
        if other == config:
            continue
        # Other dominates if it's better or equal on both metrics
        if (other['total_time'] <= config['total_time'] and
            other['gpu_hours'] <= config['gpu_hours'] and
            (other['total_time'] < config['total_time'] or other['gpu_hours'] < config['gpu_hours'])):
            return False
    return True

pareto_configs = [c for c in configs if is_pareto_optimal(c, configs)]
pareto_configs.sort(key=lambda x: x['total_time'])

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Plot all configurations
strategy_colors = {'Sync': '#808080', 'Elastic': '#1f77b4', 'Async': '#ff7f0e'}
strategy_markers = {'Sync': 'o', 'Elastic': 's', 'Async': '^'}

for strategy in ['Sync', 'Elastic', 'Async']:
    strategy_configs = [c for c in configs if c['strategy'] == strategy]
    times = [c['total_time'] / 3600 for c in strategy_configs]  # Convert to hours
    gpu_hours = [c['gpu_hours'] for c in strategy_configs]
    sizes = [c['total_gpus'] * 50 for c in strategy_configs]

    ax.scatter(times, gpu_hours,
              c=strategy_colors[strategy],
              marker=strategy_markers[strategy],
              s=sizes,
              alpha=0.6,
              label=strategy,
              edgecolors='black',
              linewidth=0.5)

# Plot Pareto frontier
pareto_times = [c['total_time'] / 3600 for c in pareto_configs]
pareto_gpu_hours = [c['gpu_hours'] for c in pareto_configs]
ax.plot(pareto_times, pareto_gpu_hours, 'k--', linewidth=2, alpha=0.7, label='Pareto Frontier')
ax.scatter(pareto_times, pareto_gpu_hours, c='red', s=200, marker='*',
          edgecolors='black', linewidth=1, zorder=10, label='Pareto Optimal')

# Annotate key configurations
annotations = []

# Find fastest
fastest = min(configs, key=lambda x: x['total_time'])
annotations.append((fastest, 'Fastest'))

# Find most GPU-efficient
most_efficient = min(configs, key=lambda x: x['gpu_hours'])
annotations.append((most_efficient, 'Most GPU-Efficient'))

# Find sweet spot (4-5 GPU elastic configs)
sweet_spots = [c for c in configs if c['strategy'] == 'Elastic' and 4 <= c['total_gpus'] <= 5]
if sweet_spots:
    best_sweet = min(sweet_spots, key=lambda x: x['gpu_hours'])
    annotations.append((best_sweet, 'Sweet Spot'))

for config, label_text in annotations:
    ax.annotate(
        f"{label_text}\n{config['label']}\n{config['gpu_hours']:.0f} GPU-hrs, {config['total_time']/3600:.1f}h",
        xy=(config['total_time']/3600, config['gpu_hours']),
        xytext=(20, 20), textcoords='offset points',
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=1.5)
    )

# Formatting
ax.set_xlabel('Total Time (hours)', fontsize=12, fontweight='bold')
ax.set_ylabel('GPU-Hours', fontsize=12, fontweight='bold')
ax.set_title('GPU-Hours vs Total Time Tradeoff\n(Token-Based Simulation - 3000 Rollouts)',
            fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)

# Add size legend
legend_elements = [plt.scatter([], [], s=n*50, c='gray', alpha=0.6, edgecolors='black', linewidth=0.5)
                   for n in [2, 4, 6, 8]]
size_legend = ax.legend(legend_elements, ['2 GPUs', '4 GPUs', '6 GPUs', '8 GPUs'],
                       loc='lower left', title='GPU Count', fontsize=9)
ax.add_artist(size_legend)

# Add config info
config_text = (
    f"Configuration:\n"
    f"• Batch size: {data['config']['global_batch_size']}\n"
    f"• Inference: {data['config']['gpu_inference_throughput_tokens']} tokens/sec/GPU\n"
    f"• Training: {data['config']['gpu_training_throughput_tokens']:.0f} tokens/sec/GPU"
)
ax.text(0.98, 0.02, config_text, transform=ax.transAxes,
       fontsize=8, verticalalignment='bottom', horizontalalignment='right',
       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig('pareto_frontier_analysis.png', dpi=200, bbox_inches='tight')
print("Saved: pareto_frontier_analysis.png")

# Print Pareto-optimal configurations
print("\n" + "="*80)
print("PARETO-OPTIMAL CONFIGURATIONS")
print("="*80)
print(f"{'Strategy':<10} {'GPUs':<6} {'Config':<20} {'Time (hrs)':<12} {'GPU-Hours':<12}")
print("-"*80)
for config in pareto_configs:
    config_str = ""
    if config['strategy'] == 'Elastic':
        config_str = f"{config['num_dedicated']}d/{config['num_elastic']}e"
    elif config['strategy'] == 'Async':
        config_str = f"{config['num_inference']}i/{config['num_training']}t"

    print(f"{config['strategy']:<10} {config['total_gpus']:<6} {config_str:<20} "
          f"{config['total_time']/3600:<12.1f} {config['gpu_hours']:<12.1f}")
print("="*80)
