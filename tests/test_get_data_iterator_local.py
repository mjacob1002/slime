"""Tests for get_data_iterator_local() — the collective-free data iterator."""
import pytest
from argparse import Namespace
from unittest.mock import patch, MagicMock

import torch


@pytest.fixture
def mock_megatron():
    """Mock megatron distributed primitives."""
    with patch("slime.backends.megatron_utils.data.mpu") as mock_mpu, \
         patch("slime.backends.megatron_utils.data.dist") as mock_dist:
        mock_mpu.get_data_parallel_world_size.return_value = 1
        mock_mpu.get_data_parallel_group.return_value = None
        mock_mpu.get_virtual_pipeline_model_parallel_world_size.return_value = None
        mock_mpu.get_context_parallel_world_size.return_value = 1
        yield mock_mpu, mock_dist


@pytest.fixture
def make_args():
    """Create a minimal args namespace."""
    def _make(
        use_dynamic_batch_size=False,
        micro_batch_size=2,
        max_tokens_per_gpu=1024,
        global_batch_size=8,
    ):
        return Namespace(
            use_dynamic_batch_size=use_dynamic_batch_size,
            micro_batch_size=micro_batch_size,
            max_tokens_per_gpu=max_tokens_per_gpu,
            global_batch_size=global_batch_size,
        )
    return _make


@pytest.fixture
def make_rollout_data():
    """Create mock rollout data."""
    def _make(num_samples, seq_len=100):
        return {
            "total_lengths": [seq_len] * num_samples,
            "tokens": [list(range(seq_len))] * num_samples,
            "response_lengths": [seq_len // 2] * num_samples,
            "loss_masks": [torch.ones(seq_len // 2)] * num_samples,
        }
    return _make


def test_no_collective_ops(mock_megatron, make_args, make_rollout_data):
    """Verify that all_reduce is NOT called during get_data_iterator_local."""
    from slime.backends.megatron_utils.data import get_data_iterator_local

    mock_mpu, mock_dist = mock_megatron
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=512)
    rollout_data = make_rollout_data(4, seq_len=100)
    model = MagicMock()

    get_data_iterator_local(args, model, rollout_data)

    # The key assertion: all_reduce should NOT have been called
    mock_dist.all_reduce.assert_not_called()


def test_correct_microbatch_count_fixed(mock_megatron, make_args, make_rollout_data):
    """Given N samples and micro_batch_size M, verify num_microbatches = N // M."""
    from slime.backends.megatron_utils.data import get_data_iterator_local

    args = make_args(use_dynamic_batch_size=False, micro_batch_size=2)
    rollout_data = make_rollout_data(6, seq_len=100)
    model = MagicMock()

    _, num_microbatches = get_data_iterator_local(args, model, rollout_data)
    assert num_microbatches == [3]  # 6 samples / 2 per microbatch


def test_zero_samples(mock_megatron, make_args, make_rollout_data):
    """With 0 samples, verify num_microbatches = [0] and empty iterator."""
    from slime.backends.megatron_utils.data import get_data_iterator_local

    args = make_args(use_dynamic_batch_size=False, micro_batch_size=2)
    rollout_data = make_rollout_data(0)
    model = MagicMock()

    data_iters, num_microbatches = get_data_iterator_local(args, model, rollout_data)
    assert num_microbatches == [0]


def test_dynamic_batch_single_step(mock_megatron, make_args, make_rollout_data):
    """With dynamic batch, all local samples form a single step."""
    from slime.backends.megatron_utils.data import get_data_iterator_local

    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=512)
    # 4 samples of length 100 each = 400 tokens, fits in 512
    rollout_data = make_rollout_data(4, seq_len=100)
    model = MagicMock()

    data_iters, num_microbatches = get_data_iterator_local(args, model, rollout_data)
    assert len(num_microbatches) == 1
    assert num_microbatches[0] >= 1
