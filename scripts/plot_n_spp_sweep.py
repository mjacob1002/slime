"""
Plot n_samples_per_prompt sweep results: inference and training time per rollout.

Usage:
    python scripts/plot_n_spp_sweep.py sweep_results/n_spp_sweep_20260304_022116
    python scripts/plot_n_spp_sweep.py sweep_results/n_spp_sweep_fixedgbs256_20260304_070814
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_n_spp_sweep.py <sweep_results_dir>")
        sys.exit(1)

    sweep_dir = Path(sys.argv[1])
    summary = json.loads((sweep_dir / "summary.json").read_text())

    n_spp_vals = []
    gbs_vals = []
    inference_times = []
    training_times = []

    for trial in summary["trials"]:
        if trial["status"] != "completed":
            continue
        n_spp_vals.append(trial["n_samples_per_prompt"])
        gbs_vals.append(trial["global_batch_size"])
        inference_times.append(trial["timing"]["mean_rollout_time"])
        training_times.append(trial["timing"]["mean_training_time"])

    n_spp_vals = np.array(n_spp_vals)
    gbs_vals = np.array(gbs_vals)
    inference_times = np.array(inference_times)
    training_times = np.array(training_times)

    # Determine sweep mode from config
    sweep_config = summary["sweep_config"]
    fixed_gbs = sweep_config.get("fixed_gbs")
    if fixed_gbs:
        mode_label = f"Fixed GBS={fixed_gbs}, batch_size=GBS/n_spp"
    else:
        rbs = sweep_config.get("rollout_batch_size", 32)
        mode_label = f"Fixed batch_size={rbs} prompts, GBS={rbs}×n_spp"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: absolute times
    ax1.plot(n_spp_vals, inference_times, "o-", color="tab:blue", label="Inference (rollout)", linewidth=2, markersize=8)
    ax1.plot(n_spp_vals, training_times, "s-", color="tab:orange", label="Training", linewidth=2, markersize=8)

    # Annotate GBS on each point
    for i, (ns, gbs, inf_t) in enumerate(zip(n_spp_vals, gbs_vals, inference_times)):
        ax1.annotate(f"gbs={gbs}", (ns, inf_t), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=8, color="tab:blue")

    ax1.set_xlabel("n_samples_per_prompt", fontsize=12)
    ax1.set_ylabel("Time per rollout (s)", fontsize=12)
    ax1.set_title("Inference & Training Time vs n_samples_per_prompt", fontsize=13)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(n_spp_vals)
    ax1.set_xticklabels([str(int(x)) for x in n_spp_vals])
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Right plot: training / inference ratio
    ratio = training_times / inference_times
    ax2.plot(n_spp_vals, ratio, "D-", color="tab:green", linewidth=2, markersize=8)
    for i, (ns, r) in enumerate(zip(n_spp_vals, ratio)):
        ax2.annotate(f"{r:.3f}", (ns, r), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)

    ax2.set_xlabel("n_samples_per_prompt", fontsize=12)
    ax2.set_ylabel("Training / Inference ratio", fontsize=12)
    ax2.set_title("Training-to-Inference Time Ratio", fontsize=13)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(n_spp_vals)
    ax2.set_xticklabels([str(int(x)) for x in n_spp_vals])
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"n_samples_per_prompt Sweep — {mode_label}\n"
        f"Global batch size = n_samples_per_prompt × num_prompts. "
        f"Since num_prompts is fixed, GBS increases linearly with n_spp.",
        fontsize=10, y=1.02,
    )

    plt.tight_layout()
    out_path = sweep_dir / "n_spp_sweep_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")

    out_pdf = sweep_dir / "n_spp_sweep_plot.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved plot to {out_pdf}")


if __name__ == "__main__":
    main()
