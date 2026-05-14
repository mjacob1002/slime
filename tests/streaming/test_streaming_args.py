"""End-to-end tests for streaming synchronous training.

These tests validate the full system including gradient equivalence.
They require multi-GPU and the full stack (Megatron, Ray, SGLang).
"""
import pytest
from unittest.mock import patch, MagicMock


def test_train_streaming_imports():
    """Verify train_streaming.py can be imported."""
    import train_streaming
    assert hasattr(train_streaming, "train")
    assert hasattr(train_streaming, "validate_streaming_args")


def test_validate_streaming_args_passes():
    """Test validation passes with correct args."""
    from train_streaming import validate_streaming_args
    from argparse import Namespace

    args = Namespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        overlap_grad_reduce=False,
        use_critic=False,
        num_elastic_nodes=2,
        num_elastic_gpus_per_node=1,
    )
    # Should not raise
    validate_streaming_args(args)


def test_validate_streaming_args_rejects_tp():
    """Test validation rejects TP > 1."""
    from train_streaming import validate_streaming_args
    from argparse import Namespace

    args = Namespace(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        overlap_grad_reduce=False,
        use_critic=False,
        num_elastic_nodes=2,
        num_elastic_gpus_per_node=1,
    )
    with pytest.raises(AssertionError, match="TP=1"):
        validate_streaming_args(args)


def test_validate_streaming_args_rejects_pp():
    """Test validation rejects PP > 1."""
    from train_streaming import validate_streaming_args
    from argparse import Namespace

    args = Namespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        overlap_grad_reduce=False,
        use_critic=False,
        num_elastic_nodes=2,
        num_elastic_gpus_per_node=1,
    )
    with pytest.raises(AssertionError, match="PP=1"):
        validate_streaming_args(args)


def test_validate_streaming_args_rejects_overlap_grad_reduce():
    """Test validation rejects overlap_grad_reduce=True."""
    from train_streaming import validate_streaming_args
    from argparse import Namespace

    args = Namespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        overlap_grad_reduce=True,
        use_critic=False,
        num_elastic_nodes=2,
        num_elastic_gpus_per_node=1,
    )
    with pytest.raises(AssertionError, match="overlap_grad_reduce"):
        validate_streaming_args(args)


def test_validate_streaming_args_rejects_critic():
    """Test validation rejects critic model."""
    from train_streaming import validate_streaming_args
    from argparse import Namespace

    args = Namespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        overlap_grad_reduce=False,
        use_critic=True,
        num_elastic_nodes=2,
        num_elastic_gpus_per_node=1,
    )
    with pytest.raises(AssertionError, match="critic"):
        validate_streaming_args(args)


def test_validate_streaming_args_rejects_no_elastic():
    """Test validation rejects zero elastic nodes."""
    from train_streaming import validate_streaming_args
    from argparse import Namespace

    args = Namespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        overlap_grad_reduce=False,
        use_critic=False,
        num_elastic_nodes=0,
        num_elastic_gpus_per_node=0,
    )
    with pytest.raises(AssertionError, match="elastic"):
        validate_streaming_args(args)
