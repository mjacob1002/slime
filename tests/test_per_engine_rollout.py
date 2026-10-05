"""Tests for per_engine_rollout.py — per-engine generation and completion detection."""
import asyncio
import copy
import pytest
from argparse import Namespace
from unittest.mock import patch, MagicMock, AsyncMock

from slime.utils.types import Sample


@pytest.fixture
def base_args():
    """Create minimal args for testing."""
    args = Namespace(
        sglang_router_ip="original-host",
        sglang_router_port=30000,
        n_samples_per_prompt=2,
        rollout_temperature=0.8,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_max_response_len=1024,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=True,
        sglang_server_concurrency=64,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        ci_test=False,
        group_rm=False,
        use_rollout_routing_replay=False,
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        custom_generate_function_path=None,
        sglang_enable_deterministic_inference=False,
        use_slime_router=False,
        hf_checkpoint="test",
        sglang_dp_size=1,
    )
    return args


def test_args_shallow_copy(base_args):
    """Verify original args are not mutated when overriding sglang_router_ip/port."""
    from slime.rollout.per_engine_rollout import generate_for_single_engine

    original_ip = base_args.sglang_router_ip
    original_port = base_args.sglang_router_port

    # We'll just verify the copy logic by patching the actual generation
    with patch("slime.rollout.per_engine_rollout.generate_and_rm_group", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = []

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                generate_for_single_engine(
                    base_args,
                    "http://new-host:12345",
                    [],
                    {},
                )
            )
        finally:
            loop.close()

        # Original args should NOT be modified
        assert base_args.sglang_router_ip == original_ip
        assert base_args.sglang_router_port == original_port


def test_completion_order_detection():
    """Mock 3 engines with different latencies, verify on_engine_complete is called in fastest-first order."""
    from slime.rollout.per_engine_rollout import generate_per_engine_streaming

    completion_order = []

    def on_complete(engine_rank, results):
        completion_order.append(engine_rank)

    # Mock generate_for_single_engine to have different sleep times per engine
    async def mock_gen_engine(args, url, groups, params):
        # Parse engine rank from URL
        port = int(url.split(":")[-1])
        # Engine 0 (port 10000): 0.3s, Engine 1 (port 10001): 0.1s, Engine 2 (port 10002): 0.2s
        delays = {10000: 0.3, 10001: 0.1, 10002: 0.2}
        await asyncio.sleep(delays.get(port, 0.1))
        return []

    args = MagicMock()
    engine_urls = ["http://host:10000", "http://host:10001", "http://host:10002"]
    all_prompt_groups = [[], [], []]  # Empty groups for test
    sampling_params = {}

    with patch("slime.rollout.per_engine_rollout.generate_for_single_engine", side_effect=mock_gen_engine):
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                generate_per_engine_streaming(
                    args, engine_urls, all_prompt_groups, sampling_params, on_complete
                )
            )
        finally:
            loop.close()

    # Engine 1 should finish first (0.1s), then engine 2 (0.2s), then engine 0 (0.3s)
    assert completion_order == [1, 2, 0]


def test_url_parsing():
    """Verify URL parsing extracts host and port correctly."""
    from slime.rollout.per_engine_rollout import generate_for_single_engine

    with patch("slime.rollout.per_engine_rollout.generate_and_rm_group", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = [MagicMock()]

        captured_args = {}

        async def capture_args(args, group, params, evaluation=False):
            captured_args["ip"] = args.sglang_router_ip
            captured_args["port"] = args.sglang_router_port
            return group

        mock_gen.side_effect = capture_args

        sample = MagicMock(spec=Sample)
        args = MagicMock()
        args.sglang_router_ip = "original"
        args.sglang_router_port = 99999

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                generate_for_single_engine(
                    args,
                    "http://192.168.1.1:16000",
                    [[sample]],
                    {},
                )
            )
        finally:
            loop.close()

        assert captured_args["ip"] == "192.168.1.1"
        assert captured_args["port"] == 16000
