"""Offline tests for the SkyRL Text2SQL tool loop — no GPU, no Ray, no SGLang.

Drives `examples/skyrl_text2sql/generate_with_sql.py` with a fake /generate endpoint so
the token/mask bookkeeping, observation splicing and resume-after-migration paths can be
checked deterministically.

Run inside the slime container:
    SLIME_T2S_DB_PATH=/workspace/slime/text2sql_data/db_files/data \
      python -m pytest tests/test_text2sql_tool_loop.py -v
"""
import asyncio
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "skyrl_text2sql"))

import generate_with_sql as G  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

DB_PATH = os.environ.get("SLIME_T2S_DB_PATH")
pytestmark = pytest.mark.skipif(not DB_PATH, reason="SLIME_T2S_DB_PATH not set")


class FakeTokenizer:
    """Deterministic byte-level tokenizer: one token per character."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 50000 for c in text]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class FakeState:
    def __init__(self, args):
        self.tokenizer = FakeTokenizer()


def _pick_db():
    """Any database that the prep script actually extracted."""
    root = os.path.join(DB_PATH, "SynSQL-2.5M", "databases")
    for db_id in sorted(os.listdir(root)):
        if os.path.exists(os.path.join(root, db_id, f"{db_id}.sqlite")):
            return db_id
    raise RuntimeError(f"no extracted databases under {root}")


def make_sample(db_id):
    return Sample(
        prompt="SYSTEM+USER PROMPT",
        label="SELECT 1;",
        metadata={"db_id": db_id, "data": "synsql"},
    )


def fake_post_factory(turn_texts):
    """Return a `post` stub that replays `turn_texts`, one per call."""
    calls = {"n": 0, "rids": [], "payloads": []}

    async def fake_post(url, payload):
        i = calls["n"]
        calls["n"] += 1
        calls["rids"].append(payload["rid"])
        calls["payloads"].append(payload)
        text = turn_texts[min(i, len(turn_texts) - 1)]
        tok = FakeTokenizer()(text)["input_ids"]
        return {
            "text": text,
            "meta_info": {
                "output_token_logprobs": [[-0.1, t, None] for t in tok],
                "finish_reason": {"type": "stop"},
            },
        }

    return fake_post, calls


class LenientArgs(types.SimpleNamespace):
    """Namespace that reports unset flags as falsy, so we only pin what matters."""

    def __getattr__(self, name):
        return None


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(G, "GenerateState", FakeState)
    monkeypatch.setenv("SLIME_T2S_DB_PATH", DB_PATH)
    monkeypatch.setenv("SLIME_T2S_MAX_TURNS", "4")
    return LenientArgs(sglang_router_ip="127.0.0.1", sglang_router_port=1)


def run(coro):
    return asyncio.run(coro)


def test_single_turn_solution_terminates(patched):
    """A trajectory that answers immediately runs exactly one turn and is COMPLETED."""
    db_id = _pick_db()
    sample = make_sample(db_id)
    fake_post, calls = fake_post_factory(["<think>easy</think><solution>SELECT 1;</solution>"])
    G.post = fake_post

    out = run(G.generate(patched, sample, {"max_new_tokens": 100}))

    assert calls["n"] == 1
    assert out.status == Sample.Status.COMPLETED
    assert out.metadata[G.K_TURN] == 1
    assert out.metadata[G.K_DONE] is True
    # No tool ran, so no observation was spliced.
    assert "<observation>" not in out.response
    assert len(out.loss_mask) == out.response_length
    assert set(out.loss_mask) == {1}


def test_tool_turn_masks_observation(patched):
    """A <sql> turn runs the tool, splices an observation, and masks it out of the loss."""
    db_id = _pick_db()
    sample = make_sample(db_id)
    fake_post, calls = fake_post_factory(
        [
            "<think>look</think><sql>SELECT name FROM sqlite_master LIMIT 1;</sql>",
            "<think>done</think><solution>SELECT 1;</solution>",
        ]
    )
    G.post = fake_post

    out = run(G.generate(patched, sample, {"max_new_tokens": 100}))

    assert calls["n"] == 2
    assert out.status == Sample.Status.COMPLETED
    assert "<observation>" in out.response
    assert "<reminder>" in out.response

    # Invariants that the training path asserts on.
    assert len(out.loss_mask) == out.response_length
    assert len(out.tokens) == out.response_length + len("SYSTEM+USER PROMPT")
    assert out.rollout_log_probs is not None
    assert len(out.rollout_log_probs) == out.response_length

    # Nothing inside an <observation> may be trained on.
    trained = "".join(
        c for c, m in zip(out.response, out.loss_mask, strict=True) if m == 1
    )
    assert "<observation>" not in trained
    assert "<reminder>" not in trained

    # Each turn got its own rid.
    assert len(set(calls["rids"])) == 2


def test_observation_byte_matches_env(patched):
    """The spliced observation is exactly what skyrl_gym's tool produced."""
    import skyrl_gym
    from skyrl_gym.envs.sql.env import Text2SQLEnvConfig

    db_id = _pick_db()
    action = "<think>x</think><sql>SELECT name FROM sqlite_master LIMIT 1;</sql>"

    env = skyrl_gym.make(
        "text2sql",
        env_config=Text2SQLEnvConfig(db_path=DB_PATH),
        extras={
            "db_id": db_id,
            "data": "synsql",
            "reward_spec": {"method": "rule", "ground_truth": "SELECT 1;"},
            "max_turns": 4,
        },
    )
    expected = env.step(action)["observations"][0]["content"]

    sample = make_sample(db_id)
    fake_post, _ = fake_post_factory([action, "<think>d</think><solution>SELECT 1;</solution>"])
    G.post = fake_post
    out = run(G.generate(patched, sample, {"max_new_tokens": 100}))

    assert expected in out.response, "observation text drifted from skyrl_gym's output"


def test_max_turns_ends_episode(patched):
    """Never emitting <solution> runs exactly max_turns and ends the episode.

    SkyRL's `SQLEnv._is_done` returns True once `turns >= max_turns`, so exhausting the
    turn budget is a normal terminal state (scored -1.0 for the missing <solution>), not
    a length truncation. Marking it TRUNCATED would misreport it to slime's overlong
    filtering and truncation stats.
    """
    db_id = _pick_db()
    sample = make_sample(db_id)
    fake_post, calls = fake_post_factory(["<think>a</think><sql>SELECT 1;</sql>"])
    G.post = fake_post

    out = run(G.generate(patched, sample, {"max_new_tokens": 100}))

    assert calls["n"] == 4  # SLIME_T2S_MAX_TURNS
    assert out.status == Sample.Status.COMPLETED
    assert out.metadata[G.K_DONE] is True
    assert out.metadata[G.K_TURN] == 4
    assert len(out.loss_mask) == out.response_length
    # The final turn is terminal, so the env returns no observation for it.
    assert out.response.count("<observation>") == 3


def test_context_cap_truncates(patched):
    """Exhausting the context budget (not the turn budget) marks TRUNCATED."""
    db_id = _pick_db()
    sample = make_sample(db_id)
    fake_post, calls = fake_post_factory(["<think>a</think><sql>SELECT 1;</sql>"])
    G.post = fake_post
    # Prompt is 18 chars/tokens; one turn's action + observation blows past this.
    os.environ["SLIME_T2S_MAX_CONTEXT"] = "60"
    try:
        out = run(G.generate(patched, sample, {"max_new_tokens": 100}))
    finally:
        os.environ.pop("SLIME_T2S_MAX_CONTEXT", None)

    assert out.status == Sample.Status.TRUNCATED
    assert calls["n"] >= 1
    assert len(out.loss_mask) == out.response_length


def test_migrate_flag_stops_at_turn_boundary(patched):
    """Setting the cancel flag stops the loop cleanly, with masks still consistent."""
    db_id = _pick_db()
    sample = make_sample(db_id)
    fake_post, calls = fake_post_factory(["<think>a</think><sql>SELECT 1;</sql>"])

    async def flagging_post(url, payload):
        result = await fake_post(url, payload)
        # Router sets the flag while the trajectory is mid-flight.
        sample.metadata[G.K_MIGRATE] = True
        return result

    G.post = flagging_post
    out = run(G.generate(patched, sample, {"max_new_tokens": 100}))

    assert out.status == Sample.Status.ABORTED
    assert calls["n"] == 1
    # The observation for turn 1 was still appended before we bailed: a <sql> must never
    # be left without its <observation>.
    assert "<observation>" in out.response
    assert out.metadata[G.K_TURN] == 1
    assert len(out.loss_mask) == out.response_length


def test_resume_after_migration_restart(patched):
    """Re-entering the generate fn continues the trajectory instead of restarting it."""
    db_id = _pick_db()
    sample = make_sample(db_id)

    fake_post, calls = fake_post_factory(["<think>a</think><sql>SELECT 1;</sql>"])

    async def flagging_post(url, payload):
        result = await fake_post(url, payload)
        sample.metadata[G.K_MIGRATE] = True
        return result

    G.post = flagging_post
    out = run(G.generate(patched, sample, {"max_new_tokens": 100}))
    assert out.status == Sample.Status.ABORTED
    tokens_before = len(out.tokens)
    response_before = out.response

    # What the router does on migration: clear the flag, reset status, drop the rid.
    out.metadata.pop(G.K_MIGRATE)
    out.status = Sample.Status.PENDING
    out.rid = None

    fake_post2, calls2 = fake_post_factory(["<think>done</think><solution>SELECT 1;</solution>"])
    G.post = fake_post2
    out2 = run(G.generate(patched, out, {"max_new_tokens": 100}))

    assert out2.status == Sample.Status.COMPLETED
    assert calls2["n"] == 1, "resume should not replay turns already completed"
    assert out2.metadata[G.K_TURN] == 2
    assert len(out2.tokens) > tokens_before
    assert out2.response.startswith(response_before), "prior trajectory must be preserved"
    assert len(out2.loss_mask) == out2.response_length
    # The resumed POST continues from the accumulated buffer.
    assert calls2["payloads"][0]["input_ids"] == out.tokens[: len(calls2["payloads"][0]["input_ids"])]


def test_stale_migrate_flag_would_livelock(patched):
    """Documents why the router MUST clear `migrate_requested` before re-dispatching.

    `_execute_migration` sets the flag so a mid-tool-call trajectory can stop at its next
    turn boundary. If it is still set when the group is re-dispatched, the resumed
    coroutine bails at its first check and reports ABORTED again -- forever, since the
    router would keep re-dispatching a group that never progresses. This test pins that
    behaviour so the clear in streaming_router.py can't be dropped silently.
    """
    db_id = _pick_db()
    sample = make_sample(db_id)
    fake_post, calls = fake_post_factory(["<think>a</think><sql>SELECT 1;</sql>"])

    async def flagging_post(url, payload):
        result = await fake_post(url, payload)
        sample.metadata[G.K_MIGRATE] = True
        return result

    G.post = flagging_post
    out = run(G.generate(patched, sample, {"max_new_tokens": 100}))
    assert out.status == Sample.Status.ABORTED

    # Router resets status/rid but (incorrectly) leaves the flag set.
    out.status = Sample.Status.PENDING
    out.rid = None
    fake_post2, calls2 = fake_post_factory(["<think>d</think><solution>SELECT 1;</solution>"])
    G.post = fake_post2
    out2 = run(G.generate(patched, out, {"max_new_tokens": 100}))

    assert calls2["n"] == 0, "stale flag should short-circuit before any /generate"
    assert out2.status == Sample.Status.ABORTED, "no progress is possible with a stale flag"

    # And with the flag cleared, as the router actually does, it resumes and finishes.
    out2.metadata.pop(G.K_MIGRATE)
    out2.status = Sample.Status.PENDING
    fake_post3, calls3 = fake_post_factory(["<think>d</think><solution>SELECT 1;</solution>"])
    G.post = fake_post3
    out3 = run(G.generate(patched, out2, {"max_new_tokens": 100}))
    assert calls3["n"] == 1
    assert out3.status == Sample.Status.COMPLETED


def test_done_idempotent_on_reentry(patched):
    """A finished trajectory that gets re-dispatched must not generate again."""
    db_id = _pick_db()
    sample = make_sample(db_id)
    fake_post, _ = fake_post_factory(["<think>d</think><solution>SELECT 1;</solution>"])
    G.post = fake_post
    out = run(G.generate(patched, sample, {"max_new_tokens": 100}))
    assert out.status == Sample.Status.COMPLETED
    snapshot = (list(out.tokens), out.response, list(out.loss_mask))

    # Router resets status and re-dispatches (the race window in the plan).
    out.status = Sample.Status.PENDING
    out.rid = None
    fake_post2, calls2 = fake_post_factory(["SHOULD NOT BE USED"])
    G.post = fake_post2
    out2 = run(G.generate(patched, out, {"max_new_tokens": 100}))

    assert calls2["n"] == 0, "re-entry after done must not issue another /generate"
    assert out2.status == Sample.Status.COMPLETED
    assert (list(out2.tokens), out2.response, list(out2.loss_mask)) == snapshot


def test_reward_matches_skyrl(patched):
    """Our reward wrapper agrees with SkyRL's scorer on the same trajectory string."""
    from skyrl_gym.envs.sql.utils import compute_score_single

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "skyrl_text2sql"))
    import sql_reward

    db_id = _pick_db()
    db_file = os.path.join(DB_PATH, "SynSQL-2.5M", "databases", db_id, f"{db_id}.sqlite")

    # Missing <think> -> format violation -> -1.0
    bad = "<solution>SELECT 1;</solution>"
    # Well-formed, gold matches itself -> 1.0
    good = "<think>reason</think><solution>SELECT 1;</solution>"

    for traj, expected in ((bad, -1.0), (good, 1.0)):
        sample = make_sample(db_id)
        sample.response = traj
        sample.label = "SELECT 1;"
        got = run(sql_reward.reward_func(patched, sample))
        ref = float(compute_score_single(traj, "SELECT 1;", db_file))
        assert got == ref, f"wrapper {got} != skyrl {ref}"
        assert got == expected, f"unexpected score {got} for {traj!r}"
