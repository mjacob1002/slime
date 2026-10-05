"""RollPacker's stream-trainer work queue, re-implemented for slime.

Enabled by `--grab-policy rollpacker_prefetch --rollpacker-faithful-queue`. Without the
flag, `rollpacker_prefetch` keeps its earlier behaviour (`RollPackerPrefetchPolicy` in
grab_policy.py), so every previously measured run stays reproducible.

Reference: github.com/Farrrrland/RollPacker @ 1dc8aae739ec4e5651929bd0c9c225b41dcb30f1
  [bw]  roll/pipeline/base_worker.py                          (train_step_second_half_with_func, train_step_full)
  [ms]  roll/distributed/scheduler/multi_async_generate_scheduler.py   (prefetch_completed_requests, get_batch)
  [dec] roll/distributed/scheduler/decorator.py               (DP_MP_DISPATCH_FIRST)
  [pl]  roll/pipeline/rlvr/rlvr_pipeline_async.py

What RollPacker does, and what this module reproduces
------------------------------------------------------
Streaming phase -- only the scaled-down ("second half") DP replicas take part:

  * ONE coordinator rank polls `prefetch_completed_requests` [bw:368,389-420]. The batch it
    gets back is shuffled, `chunk(pg_world_size)`-ed into equal sample-count slices and
    broadcast, and EVERY scaled-down replica trains its slice between two barriers
    [bw:466-546]. The next poll is issued only after that. Here the work-queue actor plays
    the coordinator: a grab is cut into one share per scaled-down train group, and no new
    grab is cut until every one of those groups has trained its share (`_round_in_flight`).
  * A poll returns the completed, not-yet-trained prompt groups in prompt-id order
    [ms:729], at most `scaling_down_train_batch_size` of them; from the second grab on also
    at most `pg_prompt_count = 2 * per_device_train_batch_size * pg_world_size //
    num_return_sequences_in_group` when that is > 0 [bw:356,370,548], and truncated from
    the front to a multiple of it [ms:741-745].
  * Streaming stops for the rest of the rollout once `B - actor_train.world_size` prompts
    have been handed out (checked on entry only, so one grab may overshoot) [ms:704], once
    all but one prompt have completed [ms:719], or once generation has finished [ms:698].

Final step -- every DP replica:

  * `get_batch` returns the prompts that were never prefetched, in prompt-id order
    [ms:614-622]; `train_step_full` is `DP_MP_DISPATCH_FIRST`, which splits that residual
    into `dp_size` contiguous equal sample-count slices, one per replica [bw:297, dec:215-219].
    The pipeline only reaches it after the streaming loop has returned [pl:913-919], so the
    final cut here also waits for the round in flight.

Not reproduced: the coordinator's poll is a `ray.get(..., timeout=2)` [bw:389-420], and a
timeout leaves `batch_available_state` at its default of 2, which ends streaming for the
rollout. That is an RPC artifact, not a queue rule; dropping it can only help RollPacker.

Deliberate adaptations (slime differs structurally)
---------------------------------------------------
  * A "DP replica" is a slime train group (one Megatron TP group). `pg_world_size` is the
    number of train groups the StreamTrainer scale-down emptied (G_free).
  * RollPacker computes rewards/advantages on the coordinator before the sample-level
    split. slime normalises GRPO rewards per prompt group when a group is converted
    (StreamingRolloutManager._post_process_rewards), before it reaches the queue, so
    splitting a prompt group's samples across train groups is equally safe.
  * `per_device_train_batch_size` has no slime counterpart (slime batches by
    --max-tokens-per-gpu). It only enters through `pg_prompt_count`; the default is 1, the
    value in RollPacker's stream-trainer config
    (examples/stream_trainer_table3/rlvr_config_stream_trainer_7B.yaml:85).

This class is plain Python on purpose (no Ray, no torch): `StreamingWorkQueue` owns an
instance and forwards to it, and tests drive it directly.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# One share entry: (item, sample_indices). `sample_indices is None` means the whole item.
ShareEntry = tuple[Any, "list[int] | None"]


@dataclass(frozen=True)
class RollPackerScatterConfig:
    expected_items_per_rollout: int        # batch_size_of_all_domains (prompt groups per rollout)
    scaling_down_train_batch_size: int     # per-grab prompt-group bound
    train_world_size: int                  # actor_train.world_size (training GPUs)
    num_train_groups: int                  # cluster.dp_size
    n_samples_per_prompt: int              # num_return_sequences_in_group
    per_device_train_batch_size: int = 1
    seed: int = 0

    def __post_init__(self):
        if self.scaling_down_train_batch_size <= 0:
            raise ValueError(
                "scaling_down_train_batch_size must be > 0 "
                "(RollPacker raises NotImplementedError otherwise, [ms:746])"
            )
        if self.num_train_groups < 1:
            raise ValueError(f"num_train_groups must be >= 1, got {self.num_train_groups}")
        if self.n_samples_per_prompt < 1:
            raise ValueError(f"n_samples_per_prompt must be >= 1, got {self.n_samples_per_prompt}")
        if self.per_device_train_batch_size < 1:
            raise ValueError(
                f"per_device_train_batch_size must be >= 1, got {self.per_device_train_batch_size}"
            )

    def pg_prompt_count(self, pg_world_size: int) -> int:
        """[bw:356] -- the steady-state grab size AND the divisibility multiplier."""
        return 2 * self.per_device_train_batch_size * pg_world_size // self.n_samples_per_prompt

    @property
    def max_prefetch(self) -> int:
        """[ms:334] max_number_of_preftch_completed_prompts."""
        return self.expected_items_per_rollout - self.train_world_size


@dataclass
class _Item:
    ref: Any            # opaque payload handed back to the trainer (a Box in production)
    num_samples: int
    order_key: int      # prompt id (submission order)
    seq: int            # arrival order; tie-break only


def _split_sizes(total: int, parts: int) -> list[int]:
    """Chunk sizes of `np.array_split(range(total), parts)` (DataProto.chunk)."""
    base, extra = divmod(total, parts)
    return [base + 1 if i < extra else base for i in range(parts)]


class RollPackerScatterCoordinator:
    def __init__(self, cfg: RollPackerScatterConfig):
        self.cfg = cfg
        # Not reseeded per rollout: every rollout gets a different permutation, and the
        # whole run is still reproducible from the seed.
        self._rng = random.Random(cfg.seed)
        self.reset()

    # ------------------------------------------------------------------ rollout state
    def reset(self) -> None:
        self._pending: list[_Item] = []            # completed, not yet assigned to a share
        self._seq = 0
        # group -> (share, sample count), assigned by a cut but not yet picked up
        self._mailbox: dict[int, tuple[list[ShareEntry], int]] = {}
        self._busy: set[int] = set()               # groups training a delivered share
        self._members: list[int] = []              # scaled-down train groups (G_free)
        self._generation_complete = False
        self._stream_closed: str | None = None     # reason streaming stopped, if it has
        self._prefetched = 0                       # number_of_prefetch_completed_prompts
        self._rounds = 0                           # streamed grabs cut so far
        self._final_cut = False
        self._samples_assigned: dict[int, int] = {}
        self.last_delivered_samples = 0            # size of the share the last grab() returned

    def push(self, ref: Any, num_samples: int, order_key: int | None = None) -> None:
        if num_samples is None or int(num_samples) < 0:
            raise ValueError(
                f"the faithful RollPacker queue needs each item's sample count, got {num_samples!r}"
            )
        if self._final_cut:
            raise RuntimeError("push after the final cut: generation was already marked complete")
        seq = self._seq
        self._seq += 1
        self._pending.append(
            _Item(ref=ref, num_samples=int(num_samples),
                  order_key=seq if order_key is None else int(order_key), seq=seq)
        )

    def set_stream_members(self, train_groups: list[int]) -> None:
        """The train groups a scale-down emptied -- RollPacker's `second_half_ranks`."""
        self._members = sorted(set(self._members) | set(train_groups))

    def mark_generation_complete(self) -> None:
        self._generation_complete = True

    # ------------------------------------------------------------------ observability
    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def items_assigned(self) -> int:
        return self._seq - len(self._pending)

    def snapshot(self) -> dict:
        return {
            "pending": len(self._pending),
            "prefetched": self._prefetched,
            "rounds": self._rounds,
            "members": list(self._members),
            "busy": sorted(self._busy),
            "undelivered": sorted(self._mailbox),
            "stream_closed": self._stream_closed,
            "final_cut": self._final_cut,
            "generation_complete": self._generation_complete,
            "samples_assigned": dict(sorted(self._samples_assigned.items())),
        }

    # ------------------------------------------------------------------ trainer API
    def grab(self, train_group: int) -> list[ShareEntry]:
        """One poll by `train_group`. Returns its share, or [] for "nothing for you now".

        A train group polls only between chunks (the faithful mode disables grab-ahead), so
        a poll that finds no undelivered share also means the group has finished training
        whatever it was last given.
        """
        share = self._deliver(train_group)
        if share:
            return share
        self._busy.discard(train_group)
        if self._final_cut:
            return []
        # Barrier [bw:513,545]: no new grab while a scaled-down group is still training its
        # slice of the current one. The final step waits for it too [pl:913-919].
        if self._round_in_flight():
            return []
        if self._generation_complete:
            self._cut_final()
            return self._deliver(train_group)
        # Only the scaled-down groups stream; the others join at the final step.
        if train_group not in self._members or self._stream_closed is not None:
            return []
        selected = self._select_streamed()
        if not selected:
            return []
        self._cut(selected, self._members, shuffle=True, kind="STREAM")
        self._prefetched += len(selected)
        self._rounds += 1
        return self._deliver(train_group)

    def is_done_for(self, train_group: int) -> bool:
        """True once `train_group` has been handed everything it will ever get."""
        return self._final_cut and train_group not in self._mailbox

    def is_done(self) -> bool:
        return self._final_cut and not self._mailbox

    # ------------------------------------------------------------------ internals
    def _deliver(self, train_group: int) -> list[ShareEntry]:
        share = self._mailbox.pop(train_group, None)
        if not share:
            self.last_delivered_samples = 0
            return []
        entries, n_samples = share
        self._busy.add(train_group)
        self.last_delivered_samples = n_samples
        return entries

    def _round_in_flight(self) -> bool:
        return bool(self._mailbox) or bool(self._busy)

    def _ordered_pending(self) -> list[_Item]:
        return sorted(self._pending, key=lambda it: (it.order_key, it.seq))

    def _close_stream(self, reason: str) -> None:
        self._stream_closed = reason
        logger.info(
            f"[RP-SCATTER] streaming closed ({reason}): prefetched={self._prefetched} "
            f"pending={len(self._pending)} max_prefetch={self.cfg.max_prefetch} "
            f"expected={self.cfg.expected_items_per_rollout}; the rest waits for the final step"
        )

    def _select_streamed(self) -> list[_Item]:
        """`prefetch_completed_requests` [ms:695-760], for one coordinator poll."""
        cfg = self.cfg
        # [ms:704] global cap -- tested on entry only, so the grab below may overshoot it.
        if self._prefetched >= cfg.max_prefetch:
            self._close_stream("RP_CAP")
            return []
        # [ms:719] near-end guard. total_valid_prompts counts every fully completed prompt,
        # including the ones already prefetched.
        total_valid = self._prefetched + len(self._pending)
        if cfg.expected_items_per_rollout - total_valid <= 1:
            self._close_stream("RP_NEAREND")
            return []
        pg_prompt_count = cfg.pg_prompt_count(len(self._members))
        limit = cfg.scaling_down_train_batch_size
        # prefetch_prompt_count is -1 until the first batch has been trained [bw:370,548].
        if self._rounds > 0 and pg_prompt_count > 0:
            limit = min(limit, pg_prompt_count)
        selected = self._ordered_pending()[:limit]
        # [ms:741-745] div_multipler = pg_prompt_count on every call, popping from the front.
        if pg_prompt_count > 0:
            selected = selected[len(selected) % pg_prompt_count:]
        return selected

    def _cut_final(self) -> None:
        self._final_cut = True
        if self._stream_closed is None:
            self._stream_closed = "GENERATION_COMPLETE"
        residual = self._ordered_pending()
        if residual:
            self._cut(residual, list(range(self.cfg.num_train_groups)), shuffle=False, kind="FINAL")
        else:
            logger.info("[RP-SCATTER] FINAL: no residual to train")

    def _cut(self, items: list[_Item], groups: list[int], shuffle: bool, kind: str) -> None:
        """Split `items` into equal sample-count shares, one per group, into the mailboxes."""
        chosen = {id(it) for it in items}
        self._pending = [it for it in self._pending if id(it) not in chosen]
        units = [(it, i) for it in items for i in range(it.num_samples)]
        if shuffle:
            self._rng.shuffle(units)      # [bw:476] np.random.permutation(indices)
        sizes = _split_sizes(len(units), len(groups))
        start = 0
        shares: dict[int, int] = {}
        for group, size in zip(groups, sizes):
            chunk = units[start:start + size]
            start += size
            if not chunk:
                continue
            by_item: dict[int, tuple[_Item, list[int]]] = {}
            for it, idx in chunk:
                by_item.setdefault(id(it), (it, []))[1].append(idx)
            entries: list[ShareEntry] = []
            for it, idxs in by_item.values():
                entries.append((it.ref, None if len(idxs) == it.num_samples else sorted(idxs)))
            assert group not in self._mailbox, f"train group {group} already has an undelivered share"
            self._mailbox[group] = (entries, len(chunk))
            shares[group] = len(chunk)
            self._samples_assigned[group] = self._samples_assigned.get(group, 0) + len(chunk)
        logger.info(
            f"[RP-SCATTER] {kind} round={self._rounds if kind == 'STREAM' else 'final'} "
            f"groups={groups} items={len(items)} samples={len(units)} shares={shares} "
            f"prefetched_before={self._prefetched} pending_after={len(self._pending)}"
        )
