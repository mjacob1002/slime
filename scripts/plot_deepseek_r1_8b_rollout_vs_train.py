"""
Plot rollout vs training time percentage breakdown for DeepSeek-R1-8B
at two max response lengths (8k and 32k).
"""

import matplotlib.pyplot as plt
import numpy as np

# --- Data ---
steps = [0, 1, 2, 3, 4]

# 8k run (TP=1, 2 engines, max-response-len=8092)
rollout_8k = [168.20, 155.12, 163.08, 149.88, 158.39]
train_8k = [140.33, 127.55, 132.31, 123.61, 135.00]

# 32k run (TP=2, 1 engine, max-response-len=32768)
rollout_32k = [349.40, 306.18, 408.72, 309.08, 286.35]
train_32k = [256.4, 221.5, 251.9, 218.2, 216.2]

# Compute percentages
def pct(rollout, train):
    total = np.array(rollout) + np.array(train)
    return np.array(rollout) / total * 100, np.array(train) / total * 100

rollout_pct_8k, train_pct_8k = pct(rollout_8k, train_8k)
rollout_pct_32k, train_pct_32k = pct(rollout_32k, train_32k)

# --- Plot ---
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

configs = [
    ("8k max response length (TP=1, 2 engines)", rollout_pct_8k, train_pct_8k),
    ("32k max response length (TP=2, 1 engine)", rollout_pct_32k, train_pct_32k),
]

color_rollout = '#2196F3'
color_train = '#4CAF50'

for ax, (title, r_pct, t_pct) in zip(axes, configs):
    y = np.arange(len(steps))
    height = 0.5

    bars_r = ax.barh(y, r_pct, height, label='Rollout', color=color_rollout)
    bars_t = ax.barh(y, t_pct, height, left=r_pct, label='Training', color=color_train)

    # Annotate percentages
    for i in range(len(steps)):
        # Rollout label
        ax.text(r_pct[i] / 2, i, f'{r_pct[i]:.1f}%',
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        # Train label
        ax.text(r_pct[i] + t_pct[i] / 2, i, f'{t_pct[i]:.1f}%',
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    ax.set_yticks(y)
    ax.set_yticklabels([f'Step {s}' for s in steps])
    ax.set_xlim(0, 100)
    ax.set_xlabel('Percentage of Time (%)')
    ax.set_title(title, fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    # Summary in subtitle
    mean_r = np.mean(r_pct)
    mean_t = np.mean(t_pct)
    ax.set_title(f'{title}\n(avg: rollout {mean_r:.1f}% / train {mean_t:.1f}%)', fontsize=12)

fig.suptitle('DeepSeek-R1-8B: Rollout vs Training Time Breakdown', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plotting_scripts/deepseek_r1_8b_rollout_vs_train.png', dpi=150, bbox_inches='tight')
plt.savefig('plotting_scripts/deepseek_r1_8b_rollout_vs_train.pdf', bbox_inches='tight')
print("Saved to plotting_scripts/deepseek_r1_8b_rollout_vs_train.{png,pdf}")
