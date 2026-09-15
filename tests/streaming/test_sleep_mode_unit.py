"""Pin the DEFAULT of the sleep_lightweight mode knob.

`SLIME_SLEEP_MODE` exists so the two expensive parts of `sleep_lightweight` -- the
host-cache free and `torch_memory_saver.pause()` -- can be A/B'd without a code
change. Its whole value depends on the default being byte-identical, because
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


def _invoke(monkeypatch, env_value, which="sleep", record=None):
    """Call StreamingMegatronTrainRayActor.sleep_lightweight unbound, recording the
    clear_memory calls. Unbound + a stub self avoids constructing a real actor
    (which needs Megatron global state and a GPU)."""
    mod = pytest.importorskip(
        "slime.backends.megatron_utils.streaming_actor",
        reason="megatron/torch_memory_saver not importable in this environment",
    )
    ENV = mod.StreamingMegatronTrainRayActor.SLEEP_MODE_ENV
    if env_value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, env_value)

    calls = []
    monkeypatch.setattr(mod, "clear_memory", lambda **kw: calls.append(kw), raising=True)
    monkeypatch.setattr(mod, "print_memory", lambda *a, **k: None, raising=True)
    rec = [] if record is None else record
    saver = types.SimpleNamespace(
        pause=lambda: rec.append("pause"), resume=lambda: rec.append("resume")
    )
    monkeypatch.setattr(mod, "torch_memory_saver", saver, raising=True)

    stub = mock.Mock()
    stub._log_memory = lambda *a, **k: None
    stub._host_free_gb = lambda: 123.4
    stub.SLEEP_MODE_ENV = ENV
    stub.SLEEP_MODES = mod.StreamingMegatronTrainRayActor.SLEEP_MODES
    stub._assert_resident_fits = lambda: None

    stub._sleep_mode = lambda: mod.StreamingMegatronTrainRayActor._sleep_mode(stub)
    fn = getattr(mod.StreamingMegatronTrainRayActor, f"{which}_lightweight")
    fn = getattr(fn, "__wrapped__", fn)      # strip @timer
    fn(stub)
    return calls


def _invoke_sleep(monkeypatch, env_value, paused=None):
    return _invoke(monkeypatch, env_value, "sleep", paused)


class TestSleepModeDefault:
    def test_unset_is_the_old_behaviour(self, monkeypatch):
        """THE load-bearing assertion: no env var -> identical to pre-knob behaviour.
        Frees the host cache AND pauses."""
        paused = []
        calls = _invoke_sleep(monkeypatch, None, paused)
        assert calls, "clear_memory was never called"
        assert calls[0] == {"clear_host_memory": True}, (
            "default changed! every streaming run, DAPO math included, would be affected"
        )
        assert paused == ["pause"], "default must still call torch_memory_saver.pause()"

    def test_explicit_full_matches_unset(self, monkeypatch):
        paused = []
        calls = _invoke_sleep(monkeypatch, "full", paused)
        assert calls[0] == {"clear_host_memory": True}
        assert paused == ["pause"]

    def test_no_host_cache_still_pauses(self, monkeypatch):
        paused = []
        calls = _invoke_sleep(monkeypatch, "no_host_cache", paused)
        assert calls[0] == {"clear_host_memory": False}
        assert paused == ["pause"], "no_host_cache must NOT change the offload"

    def test_resident_skips_the_pause(self, monkeypatch):
        paused = []
        calls = _invoke_sleep(monkeypatch, "resident", paused)
        assert calls[0] == {"clear_host_memory": False}
        assert paused == [], "resident must skip torch_memory_saver.pause()"

    @pytest.mark.parametrize("val", ["1", "", "true", "RESIDENT", "off"])
    def test_unknown_value_raises_rather_than_falling_back(self, monkeypatch, val):
        """A typo must not silently read as 'full' — that would make an A/B look
        like a null result, which is exactly how a wrong conclusion gets published."""
        with pytest.raises(ValueError):
            _invoke_sleep(monkeypatch, val)

    def test_sleep_and_wake_are_symmetric(self, monkeypatch):
        """THE regression this exists for. pause() and resume() must be called the
        same number of times: skipping the offload but still resuming aborts the
        worker with "Cannot resume allocation that is not paused", which Ray reports
        only as an opaque ActorDiedError. Cost: one failed 4-rollout run."""
        for mode in (None, "full", "no_host_cache", "resident"):
            rec = []
            _invoke(monkeypatch, mode, "sleep", rec)
            _invoke(monkeypatch, mode, "wake_up", rec)
            assert rec.count("pause") == rec.count("resume"), (
                f"mode={mode}: pause/resume asymmetry {rec} — this kills the actor"
            )
            expected = [] if mode == "resident" else ["pause", "resume"]
            assert rec == expected, f"mode={mode}: got {rec}, expected {expected}"

    def test_second_clear_is_unconditional(self, monkeypatch):
        """The post-pause clear_memory() hands GPU blocks back to SGLang and must
        run in every mode."""
        for val in (None, "no_host_cache", "resident"):
            calls = _invoke_sleep(monkeypatch, val)
            assert len(calls) == 2, f"mode={val}: expected 2 clear_memory calls, got {len(calls)}"
            assert calls[1] == {}, "post-pause clear_memory must take no kwargs"
