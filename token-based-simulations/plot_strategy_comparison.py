"""
Strategy Efficiency Comparison: GPU-Hours vs Total Time

Alternative visualization showing strategy comparison with efficiency ratio lines.
Helps identify patterns in resource utilization across different approaches.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('sweep_results_token_based.json', 'r') as f:
    data = json.load(f)

fig, ax = plt.subplots(figsize=(12, 8))

# Extract configs by strategy
strategy_colors = {'Sync': '#808080', 'Elastic': '#1f77b4', 'Async': '#ff7f0e'}

for strategy_name, data_key in [('Sync', 'sync'), ('Elastic', 'elastic'), ('Async', 'one_step_overlap')]:
    times = [entry['total_time'] / 3600 for entry in data[data_key]]
    gpu_hours = [entry['gpu_hours'] for entry in data[data_key]]
    sizes = [entry['total_gpus'] * 40 for entry in data[data_key]]

    ax.scatter(times, gpu_hours, c=strategy_colors[strategy_name],
              s=sizes, alpha=0.6, label=strategy_name, edgecolors='black', linewidth=0.5)

# Add efficiency lines (constant GPU-hours/time ratio)
time_range = np.linspace(60, 500, 100)
for efficiency_ratio in [1.0, 1.5, 2.0, 2.5]:
    gpu_hours_line = time_range * efficiency_ratio
    ax.plot(time_range, gpu_hours_line, 'k:', alpha=0.3, linewidth=0.8)
    ax.text(time_range[-1], gpu_hours_line[-1], f'{efficiency_ratio:.1f}x',
           fontsize=8, alpha=0.5)

ax.set_xlabel('Total Time (hours)', fontsize=12, fontweight='bold')
ax.set_ylabel('GPU-Hours', fontsize=12, fontweight='bold')
ax.set_title('GPU Resource Efficiency Comparison\n(Dotted lines show constant GPU-hours/time ratios)',
            fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('strategy_efficiency_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: strategy_efficiency_comparison.png")
