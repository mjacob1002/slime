"""Pin the DEFAULT of the sleep_lightweight host-cache gate.

`SLIME_SLEEP_SKIP_HOST_CACHE` exists so the expensive `torch._C._host_emptyCache()`
call inside `sleep_lightweight` can be A/B'd without a code change. Its whole value
depends on the default being byte-identical to the pre-gate behaviour, because
`sleep_lightweight` is on the hot path of EVERY streaming and async_overlapped run --
DAPO math included (elastic_actor.py:823, overlapped_rl_elastic_group.py).

These tests exist so a future edit cannot silently flip that default. They call the
real method with `clear_memory` / `torch_memory_saver` / logging patched out, so no
GPU, no Ray, no Megatron.

Run: PYTHONPATH=/root/Megatron-LM python3 -m pytest \
         tests/streaming/test_sleep_host_cache_gate_unit.py -v

NOTE the PYTHONPATH. streaming_actor imports megatron.training, so without it every
test here SKIPS -- and a skipped guard pins nothing. If you see 8 skipped, the default
is unprotected, not verified.
"""
from __future__ import annotations

import sys
import types
from unittest import mock

import pytest


def _invoke_sleep(monkeypatch, env_value):
    """Call StreamingMegatronTrainRayActor.sleep_lightweight unbound, recording the
    clear_memory calls. Unbound + a stub self avoids constructing a real actor
    (which needs Megatron global state and a GPU)."""
    mod = pytest.importorskip(
        "slime.backends.megatron_utils.streaming_actor",
        reason="megatron/torch_memory_saver not importable in this environment",
    )
    if env_value is None:
        monkeypatch.delenv(mod.StreamingMegatronTrainRayActor.SKIP_HOST_CACHE_ENV, raising=False)
    else:
        monkeypatch.setenv(mod.StreamingMegatronTrainRayActor.SKIP_HOST_CACHE_ENV, env_value)

    calls = []
    monkeypatch.setattr(mod, "clear_memory", lambda **kw: calls.append(kw), raising=True)
    monkeypatch.setattr(mod, "print_memory", lambda *a, **k: None, raising=True)
    saver = types.SimpleNamespace(pause=lambda: None, resume=lambda: None)
    monkeypatch.setattr(mod, "torch_memory_saver", saver, raising=True)

    stub = mock.Mock()
    stub._log_memory = lambda *a, **k: None
    stub._host_free_gb = lambda: 123.4
    stub.SKIP_HOST_CACHE_ENV = mod.StreamingMegatronTrainRayActor.SKIP_HOST_CACHE_ENV

    fn = mod.StreamingMegatronTrainRayActor.sleep_lightweight
    fn = getattr(fn, "__wrapped__", fn)      # strip @timer
    fn(stub)
    return calls


class TestHostCacheGateDefault:
    def test_unset_still_frees_the_host_cache(self, monkeypatch):
        """THE load-bearing assertion: no env var -> identical to pre-gate behaviour."""
        calls = _invoke_sleep(monkeypatch, None)
        assert calls, "clear_memory was never called"
        assert calls[0] == {"clear_host_memory": True}, (
            "default changed! every streaming run, DAPO math included, would be affected"
        )

    @pytest.mark.parametrize("val", ["0", "", "false", "no", "2"])
    def test_only_exactly_1_opts_in(self, monkeypatch, val):
        """Anything other than "1" must keep the old behaviour — no truthiness games."""
        calls = _invoke_sleep(monkeypatch, val)
        assert calls[0] == {"clear_host_memory": True}

    def test_set_to_1_skips_the_host_cache(self, monkeypatch):
        calls = _invoke_sleep(monkeypatch, "1")
        assert calls[0] == {"clear_host_memory": False}

    def test_second_clear_is_unconditional(self, monkeypatch):
        """pause() must still be followed by a plain clear_memory() in BOTH modes —
        that is the one that hands GPU blocks back to SGLang."""
        for val in (None, "1"):
            calls = _invoke_sleep(monkeypatch, val)
            assert len(calls) == 2, f"expected 2 clear_memory calls, got {len(calls)}"
            assert calls[1] == {}, "post-pause clear_memory must take no kwargs"
