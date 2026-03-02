"""Tests for StreamingMegatronTrainRayActor.

These tests verify the key properties:
- Lightweight sleep/wake calls the right functions
- finalize_model_grads is suppressed during local forward+backward
- finalize_model_grads is called during sync phase
"""
import pytest
import sys
from unittest.mock import patch, MagicMock

# Import the module under test — we need to handle the heavy __init__.py
import slime.backends.megatron_utils.streaming_actor as streaming_actor_module


def test_sleep_wake_lightweight_calls():
    """Test that sleep_lightweight and wake_up_lightweight call the right functions."""
    with patch.object(streaming_actor_module, "torch_memory_saver") as mock_tms, \
         patch.object(streaming_actor_module, "clear_memory"), \
         patch.object(streaming_actor_module, "print_memory"):

        actor = streaming_actor_module.StreamingMegatronTrainRayActor.__new__(
            streaming_actor_module.StreamingMegatronTrainRayActor
        )

        # Test sleep_lightweight calls pause() only
        actor.sleep_lightweight()
        mock_tms.pause.assert_called_once()

        # Test wake_up_lightweight calls resume() only
        actor.wake_up_lightweight()
        mock_tms.resume.assert_called_once()


def test_finalize_grads_suppressed_during_local_train():
    """Verify that config.finalize_model_grads_func is set to None during fwd+bwd."""
    with patch.object(streaming_actor_module, "get_args") as mock_get_args, \
         patch.object(streaming_actor_module, "get_model_config") as mock_get_config, \
         patch.object(streaming_actor_module, "get_forward_backward_func") as mock_get_fwdbwd, \
         patch.object(streaming_actor_module, "get_data_iterator_local") as mock_get_iter, \
         patch.object(streaming_actor_module, "compute_advantages_and_returns"):

        StreamingActor = streaming_actor_module.StreamingMegatronTrainRayActor

        # Setup mocks
        mock_args = MagicMock()
        mock_args.compute_advantages_and_returns = False
        mock_args.data_pad_size_multiplier = 128
        mock_args.qkv_format = "thd"
        mock_args.seq_length = 2048
        mock_args.micro_batch_size = 1
        mock_args.decoder_seq_length = None
        mock_get_args.return_value = mock_args

        mock_config = MagicMock()
        original_func = MagicMock()
        mock_config.finalize_model_grads_func = original_func
        mock_get_config.return_value = mock_config

        mock_fwdbwd_func = MagicMock(return_value=[])
        mock_get_fwdbwd.return_value = mock_fwdbwd_func

        mock_get_iter.return_value = ([MagicMock()], [2])

        actor = StreamingActor.__new__(StreamingActor)
        actor.model = [MagicMock()]
        actor.optimizer = MagicMock()
        actor._active_model_tag = "actor"
        actor._get_rollout_data = MagicMock(return_value={"total_lengths": [100, 200]})

        # Track config state during forward_backward_func call
        captured_finalize_func = []

        def capture_config(*args, **kwargs):
            captured_finalize_func.append(mock_config.finalize_model_grads_func)
            return []

        mock_fwdbwd_func.side_effect = capture_config

        # Run local forward+backward
        result = actor.train_forward_backward_local(0, MagicMock())

        # During fwd+bwd, finalize_model_grads_func should have been None
        assert len(captured_finalize_func) == 1
        assert captured_finalize_func[0] is None

        # After fwd+bwd, it should be restored
        assert mock_config.finalize_model_grads_func is original_func


def test_finalize_grads_called_during_sync():
    """Verify finalize_model_grads_with_empty_cache IS called during sync."""
    with patch.object(streaming_actor_module, "get_args") as mock_get_args, \
         patch.object(streaming_actor_module, "finalize_model_grads_with_empty_cache") as mock_finalize:

        StreamingActor = streaming_actor_module.StreamingMegatronTrainRayActor

        mock_args = MagicMock()
        mock_args.check_for_nan_in_loss_and_grad = True
        mock_args.global_batch_size = 256
        mock_get_args.return_value = mock_args

        actor = StreamingActor.__new__(StreamingActor)
        actor.model = [MagicMock()]
        actor.optimizer = MagicMock()
        actor.optimizer.prepare_grads.return_value = False
        actor.optimizer.step.return_value = (True, 1.0, 0)
        actor.opt_param_scheduler = MagicMock()
        actor.weights_backuper = MagicMock()

        actor.sync_gradients_and_step(rollout_id=0)

        # finalize_model_grads_with_empty_cache should be called
        mock_finalize.assert_called_once_with(actor.model)
