"""
Create combined subplot showing GPU-hours vs dedicated inference GPUs.

Generates a single figure with 2x4 subplots (7 GPU counts + 1 config panel)
comparing elastic and async strategies against the sync baseline.
"""

import json
import matplotlib.pyplot as plt

# Load results
with open('sweep_results_token_based.json', 'r') as f:
    data = json.load(f)

# Create subplot grid (2x4 for 7 GPU counts + 1 legend)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('GPU-Hours vs Dedicated Inference GPUs (Token-Based Simulation)',
             fontsize=16, fontweight='bold')

axes = axes.flatten()

# Create a plot for each GPU count
for idx, total_gpus in enumerate(range(2, 9)):
    ax = axes[idx]

    # Get sync baseline
    sync_entry = next(x for x in data['sync'] if x['total_gpus'] == total_gpus)
    sync_gpu_hours = sync_entry['gpu_hours']

    # Extract elastic data
    elastic_configs = [x for x in data['elastic'] if x['total_gpus'] == total_gpus]
    elastic_configs.sort(key=lambda x: x['num_dedicated_inference'])
    elastic_x = [c['num_dedicated_inference'] for c in elastic_configs]
    elastic_y = [c['gpu_hours'] for c in elastic_configs]

    # Extract async data
    async_configs = [x for x in data['one_step_overlap'] if x['total_gpus'] == total_gpus]
    async_configs.sort(key=lambda x: x['num_inference_gpus'])
    async_x = [c['num_inference_gpus'] for c in async_configs]
    async_y = [c['gpu_hours'] for c in async_configs]

    # Plot
    ax.plot(elastic_x, elastic_y, 'o-', label='Elastic', linewidth=2, markersize=6, color='#1f77b4')
    ax.plot(async_x, async_y, 's-', label='Async', linewidth=2, markersize=6, color='#ff7f0e')
    ax.axhline(y=sync_gpu_hours, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # Formatting
    ax.set_xlabel('Dedicated Inference GPUs', fontsize=10)
    ax.set_ylabel('GPU-Hours', fontsize=10)
    ax.set_title(f'{total_gpus} Total GPUs\n(Sync: {sync_gpu_hours:.0f} GPU-hrs)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, total_gpus))

    # Add legend only to first plot
    if idx == 0:
        ax.legend(fontsize=9, loc='best')

# Hide the last subplot (8th position in 2x4 grid)
axes[7].axis('off')

# Add configuration info to the empty subplot
config_text = (
    f"Configuration:\n"
    f"• Batch size: {data['config']['global_batch_size']} samples\n"
    f"• Inference: {data['config']['gpu_inference_throughput_tokens']:.0f} tokens/sec/GPU\n"
    f"• Training: {data['config']['gpu_training_throughput_tokens']:.0f} tokens/sec/GPU\n"
    f"• Switching costs: {data['config']['inference_to_training_cost']:.2f}s + {data['config']['training_to_inference_cost']:.2f}s\n"
    f"\nElastic: Dedicated + Elastic GPUs (work-stealing)\n"
    f"Async: Fixed inference + training GPUs\n"
    f"Gray dashed line: Synchronous baseline"
)
axes[7].text(0.1, 0.5, config_text, transform=axes[7].transAxes,
             fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()

# Save combined plot
output_file = 'gpu_hours_combined.png'
plt.savefig(output_file, dpi=200, bbox_inches='tight')
print(f"Saved: {output_file}")
plt.close()

print("Combined plot saved!")
