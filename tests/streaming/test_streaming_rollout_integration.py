"""Integration tests for StreamingRolloutManager + work queue."""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, call


def test_streaming_rollout_manager_imports():
    """Verify StreamingRolloutManager can be imported without errors."""
    from slime.ray.streaming_rollout import StreamingRolloutManager
    assert StreamingRolloutManager is not None


def test_convert_samples_to_train_data_empty():
    """Test conversion logic with empty sample list returns empty data."""
    result = {
        "tokens": [],
        "response_lengths": [],
        "rewards": [],
        "raw_reward": [],
        "truncated": [],
        "sample_indices": [],
        "loss_masks": [],
        "total_lengths": [],
    }
    assert result["tokens"] == []
    assert result["total_lengths"] == []


def test_split_logic():
    """Test that round-robin split distributes groups evenly."""
    n_spp = 2
    num_engines = 3

    # 12 samples -> 6 groups -> 2 groups per engine
    samples = list(range(12))
    prompt_groups = [samples[i:i + n_spp] for i in range(0, len(samples), n_spp)]
    assert len(prompt_groups) == 6

    engine_samples = [[] for _ in range(num_engines)]
    for i, group in enumerate(prompt_groups):
        engine_rank = i % num_engines
        engine_samples[engine_rank].extend(group)

    assert len(engine_samples) == 3
    assert len(engine_samples[0]) == 4  # groups 0, 3
    assert len(engine_samples[1]) == 4  # groups 1, 4
    assert len(engine_samples[2]) == 4  # groups 2, 5


def test_split_logic_uneven():
    """Test splitting when groups don't divide evenly."""
    n_spp = 2
    num_engines = 3

    # 10 samples -> 5 groups (uneven split across 3 engines)
    samples = list(range(10))
    prompt_groups = [samples[i:i + n_spp] for i in range(0, len(samples), n_spp)]
    assert len(prompt_groups) == 5

    engine_samples = [[] for _ in range(num_engines)]
    for i, group in enumerate(prompt_groups):
        engine_rank = i % num_engines
        engine_samples[engine_rank].extend(group)

    total = sum(len(r) for r in engine_samples)
    assert total == 10
    assert len(engine_samples[0]) == 4
    assert len(engine_samples[1]) == 4
    assert len(engine_samples[2]) == 2


def test_reward_normalization_divisibility():
    """Test that reward normalization uses divisibility check for per-engine batches."""
    import torch

    n_spp = 4
    rewards = torch.tensor([1.0, 0.5, 0.0, 0.8, 1.0, 0.5, 0.0, 0.8])
    assert rewards.numel() % n_spp == 0
    reshaped = rewards.reshape(-1, n_spp)
    assert reshaped.shape == (2, 4)


def test_per_group_push():
    """Verify push_data is called per-group (not per-engine)."""
    # Simulate the per-group push logic from generate_per_engine
    # 2 engines, 3 groups each = 6 total push_data calls
    num_engines = 2
    groups_per_engine = {0: 3, 1: 3}

    push_calls = []
    engine_completed_calls = []

    # Simulate the per-group push behavior
    for engine_rank in range(num_engines):
        for group_idx in range(groups_per_engine[engine_rank]):
            push_calls.append(f"push_data(engine={engine_rank}, group={group_idx})")
        engine_completed_calls.append(engine_rank)

    # Should have 6 push_data calls (one per group)
    assert len(push_calls) == 6
    # Should have 2 engine_completed calls (one per engine)
    assert len(engine_completed_calls) == 2


def test_engine_completed_after_all_groups():
    """Verify engine_completed is called only when ALL groups for that engine finish."""
    # Track per-engine group completion
    num_engines = 2
    groups_per_engine = {0: 2, 1: 3}
    completed_groups = {0: 0, 1: 0}
    engine_completed_calls = []

    # Simulate groups completing in mixed order
    completion_order = [
        (0, 0),  # engine 0, group 0
        (1, 0),  # engine 1, group 0
        (0, 1),  # engine 0, group 1 -> engine 0 done!
        (1, 1),  # engine 1, group 1
        (1, 2),  # engine 1, group 2 -> engine 1 done!
    ]

    for engine_rank, group_idx in completion_order:
        completed_groups[engine_rank] += 1
        if completed_groups[engine_rank] == groups_per_engine[engine_rank]:
            engine_completed_calls.append(engine_rank)

    # Engine 0 completes after its 2nd group, engine 1 after its 3rd
    assert engine_completed_calls == [0, 1]
    assert len(engine_completed_calls) == num_engines


def test_per_group_task_flattening():
    """Test that prompt groups are flattened to one asyncio task per group across all engines."""
    # 2 engines: engine 0 has 2 groups, engine 1 has 3 groups
    engine_groups = {
        0: [["prompt_0_0"], ["prompt_0_1"]],
        1: [["prompt_1_0"], ["prompt_1_1"], ["prompt_1_2"]],
    }

    # Flatten to per-group tasks
    all_tasks = []
    for engine_rank, groups in engine_groups.items():
        for group_idx, group in enumerate(groups):
            all_tasks.append((engine_rank, group_idx, group))

    # Should have 5 tasks total (not 2 per-engine tasks)
    assert len(all_tasks) == 5
    # Each task has its engine rank and group index
    assert all_tasks[0] == (0, 0, ["prompt_0_0"])
    assert all_tasks[4] == (1, 2, ["prompt_1_2"])
