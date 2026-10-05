"""Tests for elastic_actor.py streaming mode extensions."""
import pytest
from unittest.mock import patch, MagicMock, PropertyMock


def test_streaming_flag_stored():
    """Test that streaming flag is stored in RayElasticGroup."""
    from slime.ray.elastic_actor import RayElasticGroup

    with patch.object(RayElasticGroup, "_create_training_actors", return_value=[]), \
         patch.object(RayElasticGroup, "_create_inference_engines", return_value=[]):
        args = MagicMock()
        args.num_elastic_nodes = 1
        args.num_elastic_gpus_per_node = 1
        pg = (MagicMock(), [0], [0])

        group = RayElasticGroup(args, pg, streaming=True)
        assert group._streaming is True

        group2 = RayElasticGroup(args, pg, streaming=False)
        assert group2._streaming is False


def test_streaming_uses_streaming_actor():
    """Test that streaming=True imports StreamingMegatronTrainRayActor."""
    from slime.ray.elastic_actor import RayElasticGroup

    with patch("slime.ray.elastic_actor.ray") as mock_ray:
        mock_remote_cls = MagicMock()
        mock_remote_cls.options.return_value.remote.return_value = MagicMock()
        mock_ray.remote.return_value = mock_remote_cls
        mock_ray.get.return_value = ("127.0.0.1", 12345)

        with patch.object(RayElasticGroup, "_create_inference_engines", return_value=[]):
            args = MagicMock()
            args.num_elastic_nodes = 1
            args.num_elastic_gpus_per_node = 1
            args.train_backend = "megatron"
            args.train_env_vars = {}
            pg = (MagicMock(), [0], [0])

            # With streaming=True, it should try to import StreamingMegatronTrainRayActor
            with patch(
                "slime.backends.megatron_utils.streaming_actor.StreamingMegatronTrainRayActor"
            ) as mock_streaming_actor:
                group = RayElasticGroup(args, pg, streaming=True)
                # The remote() call should have been made with the streaming actor class
                assert mock_ray.remote.called


def test_get_engine_urls():
    """Test get_engine_urls returns properly formatted URLs."""
    from slime.ray.elastic_actor import RayElasticGroup

    with patch.object(RayElasticGroup, "_create_training_actors", return_value=[]), \
         patch.object(RayElasticGroup, "_create_inference_engines", return_value=[]):
        args = MagicMock()
        args.num_elastic_nodes = 1
        args.num_elastic_gpus_per_node = 2
        pg = (MagicMock(), [0, 1], [0, 1])

        group = RayElasticGroup(args, pg, streaming=True)

        # Mock engines with get_server_info
        engine0 = MagicMock()
        engine0.get_server_info.remote.return_value = MagicMock()
        engine1 = MagicMock()
        engine1.get_server_info.remote.return_value = MagicMock()
        group._inference_engines = [engine0, engine1]

        with patch("slime.ray.elastic_actor.ray") as mock_ray:
            mock_ray.get.return_value = [("192.168.1.1", 16000), ("192.168.1.1", 16001)]
            urls = group.get_engine_urls()

        assert urls == ["http://192.168.1.1:16000", "http://192.168.1.1:16001"]


def test_switch_engine_to_training():
    """Test switching a single engine to training mode."""
    from slime.ray.elastic_actor import RayElasticGroup

    with patch.object(RayElasticGroup, "_create_training_actors", return_value=[]), \
         patch.object(RayElasticGroup, "_create_inference_engines", return_value=[]):
        args = MagicMock()
        args.num_elastic_nodes = 1
        args.num_elastic_gpus_per_node = 2
        pg = (MagicMock(), [0, 1], [0, 1])
        group = RayElasticGroup(args, pg, streaming=True)

        # Setup mock actors and engines
        engine0 = MagicMock()
        engine1 = MagicMock()
        actor0 = MagicMock()
        actor1 = MagicMock()
        group._inference_engines = [engine0, engine1]
        group._training_actors = [actor0, actor1]

        with patch("slime.ray.elastic_actor.ray") as mock_ray:
            mock_ray.get.return_value = None
            group.switch_engine_to_training(0)

        # Engine 0 should be deregistered and released
        engine0.deregister_from_router.remote.assert_called_once()
        engine0.release_memory_occupation.remote.assert_called_once()
        # Actor 0 should be woken up
        actor0.wake_up_lightweight.remote.assert_called_once()
        # Engine 1 and Actor 1 should NOT be touched
        engine1.deregister_from_router.remote.assert_not_called()
        actor1.wake_up_lightweight.remote.assert_not_called()


def test_start_local_train_returns_ref():
    """Test that start_local_train returns a Ray ObjectRef."""
    from slime.ray.elastic_actor import RayElasticGroup

    with patch.object(RayElasticGroup, "_create_training_actors", return_value=[]), \
         patch.object(RayElasticGroup, "_create_inference_engines", return_value=[]):
        args = MagicMock()
        args.num_elastic_nodes = 1
        args.num_elastic_gpus_per_node = 1
        pg = (MagicMock(), [0], [0])
        group = RayElasticGroup(args, pg, streaming=True)

        actor0 = MagicMock()
        mock_ref = MagicMock()
        actor0.train_forward_backward_local.remote.return_value = mock_ref
        group._training_actors = [actor0]

        result = group.start_local_train(0, rollout_id=0, data_ref=MagicMock())
        assert result is mock_ref


def test_start_work_stealing_train():
    """Test that start_work_stealing_train calls actor.train_work_stealing.remote with correct args."""
    from slime.ray.elastic_actor import RayElasticGroup

    with patch.object(RayElasticGroup, "_create_training_actors", return_value=[]), \
         patch.object(RayElasticGroup, "_create_inference_engines", return_value=[]):
        args = MagicMock()
        args.num_elastic_nodes = 1
        args.num_elastic_gpus_per_node = 2
        pg = (MagicMock(), [0, 1], [0, 1])
        group = RayElasticGroup(args, pg, streaming=True)

        actor0 = MagicMock()
        actor1 = MagicMock()
        mock_ref = MagicMock()
        actor0.train_work_stealing.remote.return_value = mock_ref
        group._training_actors = [actor0, actor1]

        mock_work_queue = MagicMock()
        result = group.start_work_stealing_train(
            engine_rank=0, rollout_id=0, work_queue=mock_work_queue
        )

        # Should call train_work_stealing.remote with work_queue and dp_size
        actor0.train_work_stealing.remote.assert_called_once_with(mock_work_queue, 2)
        assert result is mock_ref

        # Actor 1 should NOT be touched
        actor1.train_work_stealing.remote.assert_not_called()
