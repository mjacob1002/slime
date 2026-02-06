"""
Create individual plots showing GPU-hours vs dedicated inference GPUs.

Generates 7 separate plots (one for each GPU count from 2-8) comparing
elastic and async strategies against the sync baseline.
"""

import json
import matplotlib.pyplot as plt

# Load results
with open('sweep_results_token_based.json', 'r') as f:
    data = json.load(f)

# Create a plot for each GPU count
for total_gpus in range(2, 9):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get sync baseline for this GPU count
    sync_entry = next(x for x in data['sync'] if x['total_gpus'] == total_gpus)
    sync_gpu_hours = sync_entry['gpu_hours']

    # Extract elastic data for this GPU count
    elastic_configs = [x for x in data['elastic'] if x['total_gpus'] == total_gpus]
    elastic_configs.sort(key=lambda x: x['num_dedicated_inference'])

    elastic_x = [c['num_dedicated_inference'] for c in elastic_configs]
    elastic_y = [c['gpu_hours'] for c in elastic_configs]

    # Extract async data for this GPU count
    async_configs = [x for x in data['one_step_overlap'] if x['total_gpus'] == total_gpus]
    async_configs.sort(key=lambda x: x['num_inference_gpus'])

    async_x = [c['num_inference_gpus'] for c in async_configs]
    async_y = [c['gpu_hours'] for c in async_configs]

    # Plot
    ax.plot(elastic_x, elastic_y, 'o-', label='Elastic', linewidth=2, markersize=8)
    ax.plot(async_x, async_y, 's-', label='Async (One-Step Overlap)', linewidth=2, markersize=8)
    ax.axhline(y=sync_gpu_hours, color='gray', linestyle='--', label=f'Sync Baseline ({sync_gpu_hours:.0f} GPU-hrs)', linewidth=1.5)

    # Formatting
    ax.set_xlabel('Number of Dedicated Inference GPUs', fontsize=12)
    ax.set_ylabel('GPU-Hours', fontsize=12)
    ax.set_title(f'GPU-Hours vs Dedicated Inference GPUs ({total_gpus} Total GPUs)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set integer ticks on x-axis
    ax.set_xticks(range(0, total_gpus))

    # Add configuration labels
    textstr = f'Config: {data["config"]["global_batch_size"]} samples, {data["config"]["gpu_inference_throughput_tokens"]:.0f}/{data["config"]["gpu_training_throughput_tokens"]:.0f} tokens/sec'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save plot
    output_file = f'gpu_hours_{total_gpus}gpus.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

print("\nAll individual plots saved!")
