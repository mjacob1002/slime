"""
Plot per-rollout training vs inference time for DeepSeek-R1-Distill-Llama-8B
synchronous colocated training run.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data extracted from logs/session_2026-03-05_09-55-40_112226_367442
steps = [0, 1, 2, 3, 4]

# From RolloutManager perf logs
rollout_time = [168.20, 155.12, 163.08, 149.88, 158.39]

# From MegatronTrainRayActor perf logs (train_time includes ref_log_probs + log_probs + actor_train)
train_time = [140.33, 127.55, 132.31, 123.61, 135.00]

# Sub-components of training
ref_log_probs_time = [30.58, 21.13, 22.27, 20.77, 22.60]
log_probs_time = [20.62, 21.19, 22.19, 20.38, 22.57]
actor_train_time = [87.96, 84.11, 86.95, 81.58, 89.01]

# Overhead: sleep (offload) + wake_up (onload) + update_weights
sleep_time = [52.75, 35.71, 3.58, 3.11, 3.64]
wake_up_time = [1.66, 2.41, 2.44, 2.62, 2.62]
update_weights_time = [0.73, 0.68, 1.94, 0.73, 0.67]

step_time = [312.52, 323.55, 305.48, 282.08, 303.00]

# Mean response lengths per step (from data.py rollout logs, max=8092)
mean_response_length = [5366, 5507, 5757, 5297, 5864]
truncated_ratio = [0.40, 0.35, 0.41, 0.35, 0.36]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot 1: Stacked bar chart of time breakdown ---
ax = axes[0]
x = np.arange(len(steps))
width = 0.6

bars_rollout = ax.bar(x, rollout_time, width, label='Inference (rollout)', color='#2196F3')
bars_ref = ax.bar(x, ref_log_probs_time, width, bottom=rollout_time, label='Ref log probs', color='#FF9800')
bars_logp = ax.bar(x, log_probs_time, width,
                   bottom=np.array(rollout_time) + np.array(ref_log_probs_time),
                   label='Log probs', color='#FFC107')
bars_train = ax.bar(x, actor_train_time, width,
                    bottom=np.array(rollout_time) + np.array(ref_log_probs_time) + np.array(log_probs_time),
                    label='Actor train (backprop)', color='#4CAF50')
overhead = np.array(sleep_time) + np.array(wake_up_time) + np.array(update_weights_time)
bars_overhead = ax.bar(x, overhead, width,
                       bottom=np.array(rollout_time) + np.array(ref_log_probs_time) + np.array(log_probs_time) + np.array(actor_train_time),
                       label='Offload/onload overhead', color='#9E9E9E')

ax.set_xlabel('Rollout Step', fontsize=12)
ax.set_ylabel('Time (s)', fontsize=12)
ax.set_title('DeepSeek-R1-8B: Per-Step Time Breakdown\n(2 GPU colocated, synchronous)', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(steps)
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# --- Plot 2: Side-by-side inference vs training ---
ax2 = axes[1]
width2 = 0.35
x2 = np.arange(len(steps))

bars1 = ax2.bar(x2 - width2/2, rollout_time, width2, label='Inference', color='#2196F3', edgecolor='white')
bars2 = ax2.bar(x2 + width2/2, train_time, width2, label='Training', color='#4CAF50', edgecolor='white')

# Add value labels
for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{bar.get_height():.0f}s', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{bar.get_height():.0f}s', ha='center', va='bottom', fontsize=9)


ax2.set_xlabel('Rollout Step', fontsize=12)
ax2.set_ylabel('Time (s)', fontsize=12)
ax2.set_title('Inference vs Training Time per Step\n(synchronous, max response len = 8092)', fontsize=13)
ax2.set_xticks(x2)
ax2.set_xticklabels(steps)
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3)

# Add summary stats
mean_rollout = np.mean(rollout_time)
mean_train = np.mean(train_time)
mean_step = np.mean(step_time)
ax2.text(0.98, 0.65, f'Avg inference: {mean_rollout:.0f}s\n'
                       f'Avg training: {mean_train:.0f}s\n'
                       f'Avg step: {mean_step:.0f}s\n'
                       f'Inference/Step: {mean_rollout/mean_step:.0%}',
         transform=ax2.transAxes, ha='right', va='top', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('plotting_scripts/deepseek_r1_8b_timing.png', dpi=150, bbox_inches='tight')
plt.savefig('plotting_scripts/deepseek_r1_8b_timing.pdf', bbox_inches='tight')
print("Saved to plotting_scripts/deepseek_r1_8b_timing.png")
