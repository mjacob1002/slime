"""ThresholdTuner — sets the migration batch threshold (B) automatically.

Third member of the streaming control plane, alongside the two existing pieces:
  • `MigrationPolicy`        decides **what work moves where** (rollout-manager actor)
  • `GroupSwitchController`  decides **when G_train membership may change** (driver)
  • `ThresholdTuner`         decides **how aggressive migration should be** (driver)

B (`--migration-batch-threshold`) is hand-picked today and does not transfer between
workloads: the measured optimum is ~32 on Text2SQL (Qwen3-8B) and ~96 on DAPO-math
(DeepSeek-R1-Distill-Llama-8B), a 3x spread on identical code. Each candidate costs a
~2-hour run, and the run-to-run noise floor (4.4% wall on byte-identical config) means a
single run cannot separate neighbouring values anyway.

The tuner runs in the DRIVER, once per rollout, because that is where the signal lives:
in the streaming path every `get_tracer().emit()` happens driver-side (training actors
never call `init_tracer`, so their emits no-op), which means the driver already holds the
complete per-GPU span set for the rollout that just finished. `collect_observation()`
reads it and computes a metric byte-identical to what `perf_analysis/compare_gpu_time.py`
reports offline — the control signal and the validation number are the same quantity.

B itself lives on the policy instance inside the rollout-manager actor and is read fresh
at every decision (`migration_policy.py`, `cumulative_batch_threshold`), so retuning is
one field write between rollouts. The driver pushes it via
`StreamingRolloutManager.set_migration_threshold()`.

Default is `FixedTuner`, which never changes B, so runs are unchanged unless
`--threshold-tuner` is set to something else.
"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Per-GPU tracer rows are pid 100..163 (see slime/utils/perfetto_tracer.py::_device_pid).
_ENGINE_PID_LO = 100
_ENGINE_PID_HI = 164


@dataclass
class TunerObservation:
    """One rollout's control signals. Everything here is computable ONLINE.

    Deliberately small, mirroring `FlipDecisionContext`. In particular it carries no
    perfetto handle and no actor references: `update()` must be a pure function of this
    struct so a recorded stream of observations can be replayed against different tuners
    offline, with no GPU time.

    `idle_ratio` is the fraction of training-phase GPU-time in which a GPU was flipped
    into training but not actually computing — i.e. work-stealing scaffolding plus
    starvation. It is `1 - busy/span`, the same quantity compare_gpu_time.py reports as
    (ws_* + residual) / training-span.
    """

    rollout_id: int
    threshold: int                  # B in effect DURING this rollout
    idle_ratio: float               # [0,1]; None-safe callers should skip if span==0
    training_span_gpu_s: float
    busy_gpu_s: float
    wall_s: float
    migrations: int = 0

    # --- idle decomposition -------------------------------------------------
    # `idle_ratio` sums two physically different things, and measurement shows one
    # of them dominates in the regime where B is healthy:
    #
    #   TRAILING  time after this GPU's last chunk ended, up to the end of the
    #             `training` span. That span end is a SINGLE GLOBAL timestamp
    #             (train_streaming.py: `training_done_time`, taken after
    #             ray.get(all_refs) across ALL groups), so every group that
    #             finished early is charged the wait for the last one. It is
    #             therefore last-chunk granularity -- who happened to grab the
    #             final chunk and how big it was -- which the GRAB policy governs
    #             and B barely influences.
    #
    #   INTERIOR  gaps BETWEEN chunks, inside the window where this GPU was
    #             actively polling for work and getting none. The work-stealing
    #             loop only records chunk_*/ws_* stats inside `if buffer:`
    #             (streaming_actor.py), so a starved iteration -- grab_available
    #             returns [], 50 ms sleep, retry -- emits nothing and shows up
    #             here by omission. THIS is the term migration aggressiveness
    #             actually causes.
    #
    # Measured over whole runs (per-rollout interior ratio):
    #     B=64  healthy   0.00037 - 0.00051   (98.3% of its idle is trailing)
    #     B=96             0.00043 - 0.00071, plus one 0.0189 starvation rollout
    #     none  healthy   0.00040 - 0.00079
    #     B=128 cliff     0.00048 - 0.30429   (58.6% of its idle is interior)
    #
    # The healthy band is tight (<0.0008) and real starvation is >=0.0145, a ~25x
    # gap, which is why `InteriorIdleTuner` can use a plain absolute threshold
    # where `IdleThresholdTuner` on the combined signal cannot.
    #
    # Default None, NOT 0.0: a caller that predates the decomposition (an old
    # replay stream, a hand-built test observation) must read as "no signal" and
    # make the tuner HOLD. Defaulting to 0.0 would read as "zero starvation" and
    # ramp B upward every single rollout -- the exact silent-monotone-ramp failure
    # that already wasted a live eval when a target was mis-set.
    interior_idle_gpu_s: float | None = None
    trailing_idle_gpu_s: float | None = None

    @property
    def interior_idle_ratio(self) -> float | None:
        """Interior (between-chunk starvation) idle as a fraction of training span."""
        if self.interior_idle_gpu_s is None or self.training_span_gpu_s <= 0:
            return None
        return self.interior_idle_gpu_s / self.training_span_gpu_s

    @property
    def trailing_idle_ratio(self) -> float | None:
        """Trailing (post-last-chunk barrier wait) idle as a fraction of training span."""
        if self.trailing_idle_gpu_s is None or self.training_span_gpu_s <= 0:
            return None
        return self.trailing_idle_gpu_s / self.training_span_gpu_s


class ThresholdTuner(ABC):
    """Chooses B for the next rollout.

    Subclasses implement `_propose()`. `update()` is concrete and applies the safety
    rails on top, so an experimental tuner cannot wedge a multi-hour run by returning
    something absurd — the rails are not each subclass's responsibility to remember.
    """

    def __init__(
        self,
        initial: int,
        b_min: int = 8,
        b_max: int = 256,
        step: int = 16,
        max_up_step: int | None = None,
        max_down_step: int | None = None,
    ):
        if b_min < 1:
            raise ValueError(f"b_min must be >= 1, got {b_min}")
        if b_max < b_min:
            raise ValueError(f"b_max ({b_max}) must be >= b_min ({b_min})")
        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")
        for name, v in (("max_up_step", max_up_step), ("max_down_step", max_down_step)):
            if v is not None and v < 1:
                raise ValueError(f"{name} must be >= 1 or None, got {v}")
        self.b_min = b_min
        self.b_max = b_max
        self.step = step
        # Asymmetric rails. Both default to `step`, so every tuner that predates this
        # (fixed / idle_ratio / idle_threshold / interior_idle) keeps the exact symmetric
        # one-step-per-rollout cap it has always had. A control law whose whole shape
        # lives in the size of its moves -- CUBIC's multiplicative decrease and cubic
        # growth -- would be flattened into a staircase by that cap, so it widens them.
        self.max_up_step = step if max_up_step is None else max_up_step
        self.max_down_step = step if max_down_step is None else max_down_step
        self.current = self._clamp(initial)
        self._reason = "initial"

    # ---------------------------------------------------------------- rails

    def _clamp(self, b: int) -> int:
        """Bound B, and never return 0.

        B=0 would disable migration entirely; that has to be an explicit
        `--migration-policy none`, not something a controller can stumble into.
        """
        return max(self.b_min, min(self.b_max, int(b)))

    def update(self, obs: TunerObservation) -> int:
        """Return B for the NEXT rollout, with rails applied. Do not override."""
        proposed = self._propose(obs)
        if proposed is None:
            proposed = self.current
        # One step per rollout, whatever the subclass asked for. Bounds the blast radius
        # of a single bad reading -- important because the per-rollout signal has a
        # measured CV of 26-36% at fixed B.
        delta = int(proposed) - self.current
        cap = self.max_up_step if delta > 0 else self.max_down_step
        if abs(delta) > cap:
            proposed = self.current + int(math.copysign(cap, delta))
        self.current = self._clamp(proposed)
        return self.current

    # ------------------------------------------------------------- subclass API

    @abstractmethod
    def _propose(self, obs: TunerObservation) -> int | None:
        """Propose B for the next rollout. `None` means "leave it alone"."""

    @abstractmethod
    def reset(self) -> None:
        """Reset per-RUN state. Called once before the rollout loop, not per rollout."""

    def explain(self) -> str:
        """Human-readable reason for the last decision, written to the decision log."""
        return self._reason


class FixedTuner(ThresholdTuner):
    """Never changes B. The inert default, and the control arm for experiments."""

    def reset(self) -> None:
        self._reason = "fixed"

    def _propose(self, obs: TunerObservation) -> int | None:
        self._reason = "fixed"
        return None


class IdleRatioTuner(ThresholdTuner):
    """Drive B from training-phase idle as a fraction of training-phase GPU-time.

    Heuristic: below baseline there is headroom, so migrate more aggressively (raise B);
    above baseline training is starving, so back off (lower B).

        calibrate:  first `warmup` rollouts at the starting B -> baseline = EWMA(ratio)
        thereafter: r = EWMA(ratio)
                    r outside baseline*(1 +/- dead) for `patience` CONSECUTIVE
                    rollouts -> step B in the indicated direction; any in-band
                    reading resets the streak

    Three properties of the signal, measured on the committed ladders, dictate the shape:

    1. **The baseline MUST be calibrated per run, never a constant.** Absolute levels are
       workload-specific: Text2SQL sits at 4-7%, DAPO-math at 2.7-2.9%. A fixed target is
       unsatisfiable on one and trivially met on the other.

    2. **The raw per-rollout value is too noisy to act on.** CV is 26-36% at fixed B
       (t128: 78%, individual rollouts spanning 1.89-15.32%). Reacting to a single
       rollout is a random walk, not control -- hence the EWMA.

    3. **The calibrated baseline is itself noisy, and that biases the loop.** With CV
       ~36%, a `warmup`-sample mean carries ~36%/sqrt(warmup) of standard error. On the
       committed DAPO t64 stream the first 3-6 samples read 5-11% BELOW the true mean, so
       every later reading looks "above baseline" and B ratchets down -- measured drift of
       64 -> 32 on a stream where B should not have moved at all. `patience` is the fix:
       requiring consecutive confirmation kills drift from a mis-calibrated setpoint
       without widening the dead band, which would blunt cliff detection. (The t128 cliff
       is 2.36x baseline, so it clears any sane band and confirms on every rollout.)

    Honest scope: the signal is a sharp OVER-aggression detector (the t128 cliff is ~4
    sigma) but a weak gradient (t64-vs-t96 is 0.42 points against a 0.90-point stdev,
    i.e. half the noise). So this converges to *the most aggressive B that does not trip
    the guard*, not to the wall-clock optimum. That is the intended v1 goal.
    """

    def __init__(
        self,
        initial: int,
        b_min: int = 8,
        b_max: int = 256,
        step: int = 16,
        warmup: int = 3,
        alpha: float = 0.4,
        dead_band: float = 0.20,
        patience: int = 2,
    ):
        super().__init__(initial, b_min=b_min, b_max=b_max, step=step)
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0,1], got {alpha}")
        if dead_band < 0.0:
            raise ValueError(f"dead_band must be >= 0, got {dead_band}")
        if warmup < 1:
            raise ValueError(f"warmup must be >= 1, got {warmup}")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        self.warmup = warmup
        self.alpha = alpha
        self.dead_band = dead_band
        self.patience = patience
        self.reset()

    def reset(self) -> None:
        self._ewma: float | None = None
        self._baseline: float | None = None
        self._n = 0
        self._streak = 0        # consecutive out-of-band readings, signed
        self._reason = "reset"

    def _propose(self, obs: TunerObservation) -> int | None:
        r = obs.idle_ratio
        if r is None or not math.isfinite(r):
            self._reason = "no signal (span=0 or non-finite); holding"
            return None

        # Rollout 0 is never representative: it carries one-time startup (cuda-graph
        # capture, cold radix cache, first weight sync) that never recurs. Measured, it
        # runs 17-52% slower than steady state and its idle_ratio was the highest of the
        # run in BOTH live tests (0.0290 vs 0.0123-0.0211; 0.0252 vs 0.0187). Folding it
        # into the baseline inflates the setpoint, which makes the controller believe it
        # has headroom and climb -- a bias in the opposite direction to the one `patience`
        # guards against. Skip it entirely rather than let the EWMA dilute it.
        if obs.rollout_id == 0:
            self._reason = "rollout 0 skipped (startup-inflated, not representative)"
            return None

        self._ewma = r if self._ewma is None else self.alpha * r + (1 - self.alpha) * self._ewma
        self._n += 1

        if self._n < self.warmup:
            self._reason = f"calibrating {self._n}/{self.warmup} (ewma={self._ewma:.4f})"
            return None
        if self._n == self.warmup:
            self._baseline = self._ewma
            self._reason = f"baseline={self._baseline:.4f} from {self.warmup} rollouts"
            return None

        lo = self._baseline * (1.0 - self.dead_band)
        hi = self._baseline * (1.0 + self.dead_band)
        band = f"[{lo:.4f},{hi:.4f}]"

        if self._ewma < lo:
            direction, want = +1, "headroom, more aggressive"
        elif self._ewma > hi:
            direction, want = -1, "starving, less aggressive"
        else:
            # Any in-band reading breaks the streak. This is what stops a baseline that
            # was calibrated slightly low/high from ratcheting B in one direction forever.
            self._streak = 0
            self._reason = f"ewma={self._ewma:.4f} within {band} -> hold"
            return None

        # Same direction as the running streak? extend it; otherwise start a new one.
        self._streak = self._streak + direction if self._streak * direction > 0 else direction
        if abs(self._streak) < self.patience:
            self._reason = (
                f"ewma={self._ewma:.4f} outside {band} but only "
                f"{abs(self._streak)}/{self.patience} consecutive -> hold"
            )
            return None

        self._streak = 0        # consume the confirmation, re-arm for the next move
        self._reason = (
            f"ewma={self._ewma:.4f} outside {band} for {self.patience} consecutive "
            f"(baseline {self._baseline:.4f}) -> {want}"
        )
        return self.current + direction * self.step


# --------------------------------------------------------------- signal extractors
#
# A tuner makes TWO independent choices: WHICH scalar it watches (the signal) and HOW
# it reacts (the control law). Keeping them apart is what makes a new tuner cheap --
# `InteriorIdleTuner` is the same bang-bang law as `IdleThresholdTuner` pointed at a
# different field, and it costs three class attributes rather than a copied _propose().
#
# An extractor returns None to mean "this observation does not carry that signal",
# which every control law below treats as HOLD. Adding a signal here makes it
# available to every existing law for free.
SIGNAL_REGISTRY: dict[str, "Callable[[TunerObservation], float | None]"] = {
    "idle_ratio": lambda o: o.idle_ratio,
    "interior_idle_ratio": lambda o: o.interior_idle_ratio,
    "trailing_idle_ratio": lambda o: o.trailing_idle_ratio,
    "interior_idle_gpu_s": lambda o: o.interior_idle_gpu_s,
}


class BangBangTuner(ThresholdTuner):
    """Bang-bang control on ONE scalar signal against an ABSOLUTE target.

        if signal > target:  B -= step   # starving      -> migrate less
        else:                B += step   # has headroom  -> migrate more

    Every rollout moves B by exactly one step; there is no hold state except when the
    signal is missing. Reacting in ONE rollout instead of ~4 matters on a 10-rollout
    run, and the cost is no noise rejection: on a stationary workload sitting near
    `target` this oscillates by +/-step rather than converging. That limit cycle is
    inherent to bang-bang control, not a bug.

    SWAPPING THE SIGNAL — subclass and set three class attributes, nothing else:

        class MyTuner(BangBangTuner):
            signal         = "interior_idle_ratio"   # key into SIGNAL_REGISTRY
            default_target = 0.005                   # epsilon when the CLI omits one
            target_arg     = "tuner_interior_target" # argparse dest holding it

    then add it to THRESHOLD_TUNER_REGISTRY and to the `choices=` tuple in
    slime/utils/arguments.py (a drift-guard test fails if those two disagree).

    Each subclass owns its OWN target argument on purpose. Sharing one flag across
    tuners whose sane epsilons differ by ~6x invites carrying a stale value into a new
    arm -- which is exactly how one live eval ran `idle_threshold` at target=0.30,
    roughly 10x above any real reading, and degenerated into a monotone ramp that
    proved nothing.
    """

    # ---- subclass contract ----
    signal: str = "idle_ratio"
    default_target: float = 0.03
    target_arg: str = "tuner_idle_target"
    # Ratio signals are fractions and validate in (0,1); an absolute-seconds signal
    # (e.g. "interior_idle_gpu_s") only has to be positive.
    target_is_ratio: bool = True

    def __init__(
        self,
        initial: int,
        b_min: int = 8,
        b_max: int = 256,
        step: int = 16,
        target: float | None = None,
        skip_first: bool = True,
    ):
        super().__init__(initial=initial, b_min=b_min, b_max=b_max, step=step)
        if self.signal not in SIGNAL_REGISTRY:
            raise ValueError(
                f"{type(self).__name__}.signal={self.signal!r} is not in "
                f"SIGNAL_REGISTRY. Choices: {sorted(SIGNAL_REGISTRY)}"
            )
        if target is None:
            target = self.default_target
        target = float(target)
        if self.target_is_ratio:
            if not 0.0 < target < 1.0:
                raise ValueError(f"target must be in (0, 1), got {target}")
        elif target <= 0.0:
            raise ValueError(f"target must be > 0, got {target}")
        self.target = target
        self.skip_first = skip_first

    def reset(self) -> None:
        self._reason = "initial"

    def _propose(self, obs: TunerObservation) -> int | None:
        if self.skip_first and obs.rollout_id == 0:
            self._reason = "rollout 0 skipped (startup-inflated, not representative)"
            return None
        r = SIGNAL_REGISTRY[self.signal](obs)
        # HOLD, never assume zero. An observation built before this signal existed
        # carries None; reading that as 0 would look like "no starvation at all" and
        # ramp B to b_max over the run.
        if r is None or not math.isfinite(r):
            self._reason = f"no {self.signal} signal on this observation; holding"
            return None
        # `>` not `>=`: exactly at target counts as headroom.
        if r > self.target:
            self._reason = (
                f"{self.signal}={r:.5f} > target={self.target:.5f} "
                f"-> starving, less aggressive"
            )
            return self.current - self.step
        self._reason = (
            f"{self.signal}={r:.5f} <= target={self.target:.5f} "
            f"-> headroom, more aggressive"
        )
        return self.current + self.step


class IdleThresholdTuner(BangBangTuner):
    """Bang-bang control on an ABSOLUTE idle-ratio threshold. No EWMA, no baseline.

        idle_ratio = 1 - (chunk_* + ws_* busy GPU-s) / (training-span GPU-s)

        if idle_ratio > target:  B -= step   # training is starving -> migrate less
        else:                    B += step   # there is headroom  -> migrate more

    Every rollout moves B by exactly one step; there is no hold state. That is the
    point of this tuner -- it is the deliberately simple counterpart to
    `IdleRatioTuner`, which smooths (EWMA), calibrates its own baseline from the run,
    and requires consecutive confirmation before moving.

    The trade this makes, stated plainly so results are read correctly:

    * It reacts in ONE rollout instead of ~4, so a 10-rollout run yields ~9 decisions
      instead of ~6. On short runs that is a real advantage.
    * It has no noise rejection whatsoever. Per-rollout idle_ratio has a measured CV of
      26-36% at fixed B, so on a stationary workload sitting near `target` this
      oscillates B by +/-step every rollout rather than converging. Expect a limit
      cycle, not a fixed point -- that is inherent to bang-bang control, not a bug.
    * `target` is an ABSOLUTE number and is workload-specific. Measured means: ~2.6% on
      DAPO-math (DeepSeek-R1-8B) and ~4-7% on Text2SQL (Qwen3-8B). A target that is
      sane on one is unsatisfiable or trivial on the other, so it MUST be set per
      workload -- unlike `IdleRatioTuner`, which calibrates its own reference.

    Rollout 0 is skipped: its idle_ratio is inflated by startup (measured 0.0209-0.0290
    against a 0.0189-0.0322 steady-state band, on a rollout that ran 771s vs ~560s).
    """

    signal = "idle_ratio"
    default_target = 0.03
    target_arg = "tuner_idle_target"


class InteriorIdleTuner(BangBangTuner):
    """Bang-bang on INTERIOR idle only — the starvation B actually causes.

    Same control law as `IdleThresholdTuner`, pointed at `interior_idle_ratio` instead
    of the combined `idle_ratio`. That one substitution is the whole point, because the
    combined signal is mostly measuring something else.

    Decomposing the committed traces (see `TunerObservation`) shows the combined
    idle_ratio at a healthy B is ~98% TRAILING — the wait after a group's last chunk
    for the global `training_done_time` barrier. Trailing is set by which group grabbed
    the final chunk and how big it was, i.e. by the GRAB policy's tail split, not by
    migration aggressiveness. So on the combined signal:

      * the run-to-run spread blamed on noise (CV 26-36%) is largely real variation in
        last-chunk landing, a quantity B does not control; and
      * a single absolute epsilon cannot work, because the trailing floor is
        workload-specific (DAPO ~2.6%, Text2SQL ~4-7%).

    Interior has neither problem. Measured whole-run interior ratios:

        B=64  healthy      0.00037 - 0.00051
        B=96               0.00043 - 0.00071  (+ one 0.0189 starvation rollout)
        none  healthy      0.00040 - 0.00079
        B=128 over-agg.    0.00048 - 0.30429

    The healthy ceiling is 0.00079 and the smallest genuine starvation event is 0.0145
    — a ~25x gap with nothing in between. `default_target = 0.005` sits ~6x above the
    healthy ceiling (so noise never triggers a needless back-off) and ~3x below the
    smallest real event (so genuine starvation always trips it).

    Because the healthy floor is nearly identical across three different runs AND two
    migration policies, this epsilon is far more likely to transfer between workloads
    than the combined signal's — though that is a prediction from four traces on one
    model family, not a measured cross-workload result, and Text2SQL should be
    confirmed before it is treated as settled.

    A train group that flipped into training and received ZERO chunks is charged its
    entire span to interior: it polled an empty queue the whole time, which is the
    extreme case of starvation rather than a tail wait. That case appears in 0/10
    rollouts at B=64 and 8/10 at B=128, so it discriminates strongly.
    """

    signal = "interior_idle_ratio"
    default_target = 0.005
    target_arg = "tuner_interior_target"


class CubicTuner(ThresholdTuner):
    """TCP CUBIC (RFC 9438) ported to B, with `t` in ROLLOUTS instead of seconds.

    Design doc: perf_analysis/CUBIC_TUNER_DESIGN.md. Read §5.1 and §13 before changing a
    default -- several are deliberately NOT the RFC's.

        W(t) = C * (t - K)^3 + W_max        K = cbrt(W_max * (1 - beta) / C)

    `W_max` is the B in effect at the last starvation, i.e. the last known-bad point;
    `t` counts rollouts since then. The curve is concave below K (fast recovery), flat
    near K, convex above it (probing for a new ceiling), so the controller spends most of
    its time parked just under the last boundary. That is the entire reason to prefer it
    to `InteriorIdleTuner`, whose bang-bang law has no hold state, moves B every rollout,
    and measurably walked B from 28 to the b_min floor on a false alarm (§13.3).

    WHAT IS DELIBERATELY NOT FAITHFUL TO THE RFC

    * **No Reno-friendly region** (RFC 9438 §4.3). It bundles inter-flow fairness -- there
      is one controller and one workload, so it transfers to nothing -- with a growth floor
      for when the cubic curve is slow. The convex region already IS that floor, and adding
      a second mechanism would partly undo the plateau we chose CUBIC for.
    * **C defaults to 1.0, not 0.4.** C survives the seconds->rollouts clock change
      numerically but not the change in HORIZON: TCP sees thousands of RTTs between
      congestion events, a 10-rollout arm gives ~9 decisions. Escaping a `W_max` set 33%
      too low takes t=7 at C=0.4 and t=4 at C=1.6; C=1.0 keeps a 2-3 rollout plateau while
      halving the escape time. Use C=0.4 only for runs of >= 50 rollouts.
    * **`fast_convergence` defaults OFF.** The RFC applies it (§4.7) to release bandwidth
      to a competing flow -- irrelevant here. We keep the mechanism only because our safe B
      genuinely drifts DOWN as the policy learns (144 -> 96 -> 48 over 50 rollouts), so a
      falling ceiling is real signal. But it deliberately OVER-reacts to a lower `W_max`,
      which is exactly wrong when the congestion event was a false alarm -- and until the
      zero-chunk bug in `collect_observation` is fixed, some of them are (§13.3). Turn it
      on once the signal is trustworthy, as its own arm.
    * **`gamma = 1.0` holds B instead of growing it.** It is the control arm for "does
      slow start earn anything", so it must not grow by a fixed `step` -- that would
      reintroduce the constant-step behaviour CUBIC replaces and make the comparison
      meaningless. With `gamma = 1` the controller sits at `b_init` until the first
      starvation, which is the right shape for a short run started near the boundary.
    * **Slow start is entered once and never re-entered.** Tahoe restarts it on timeout
      because capacity may have grown; our measured drift is downward, so re-probing
      exponentially would climb into a boundary that just moved down. This is the
      Reno/CUBIC fast-recovery behaviour and the deviation is data-driven.

    RAILS. The base class caps |dB| at `step` symmetrically, which would flatten both the
    multiplicative decrease and the cubic growth into a staircase. This class widens both
    caps to the full B range by default; `b_min`/`b_max` still bound the result.

    **`step` is not a knob for this tuner.** Once the first congestion event establishes
    `W_max`, every move size comes from the curve -- large during concave recovery, ~0 on
    the plateau, accelerating during convex probing (measured, W_max=144: -40, +24, +16,
    +8, +8, +24, +48). `step` is accepted only because the base class requires it, and it
    is never read: `gamma > 1` grows multiplicatively and `gamma = 1` holds.

    LATTICE. Under `--migration-count-unit groups` the policy can only fire at multiples of
    `n_samples_per_prompt`, so a sub-quantum move is a no-op and the plateau would freeze.
    `snap_quantum` projects onto that lattice. Under `samples` B is continuous and the
    quantum should be 1 (the identity).
    """

    signal: str = "interior_idle_ratio"

    def __init__(
        self,
        initial: int,
        b_min: int = 8,
        b_max: int = 256,
        step: int = 16,
        C: float = 1.0,
        beta: float = 0.7,
        gamma: float = 2.0,
        epsilon: float = 0.005,
        fast_convergence: bool = False,
        snap_quantum: int = 1,
        skip_first: bool = True,
    ):
        # Rails widened to the full range: the curve, not a per-rollout cap, decides the
        # move. _clamp still bounds the result to [b_min, b_max].
        super().__init__(
            initial, b_min=b_min, b_max=b_max, step=step,
            max_up_step=max(1, b_max), max_down_step=max(1, b_max),
        )
        if C <= 0:
            raise ValueError(f"C must be > 0, got {C}")
        if not 0.0 < beta < 1.0:
            raise ValueError(f"beta must be in (0,1), got {beta}")
        if gamma < 1.0:
            raise ValueError(f"gamma must be >= 1 (1.0 = no slow start), got {gamma}")
        if not 0.0 < epsilon < 1.0:
            raise ValueError(f"epsilon must be in (0,1), got {epsilon}")
        if snap_quantum < 1:
            raise ValueError(f"snap_quantum must be >= 1, got {snap_quantum}")
        if self.signal not in SIGNAL_REGISTRY:
            raise ValueError(
                f"{type(self).__name__}.signal={self.signal!r} is not in SIGNAL_REGISTRY. "
                f"Choices: {sorted(SIGNAL_REGISTRY)}"
            )
        self.C = C
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.fast_convergence = fast_convergence
        self.snap_quantum = snap_quantum
        self.skip_first = skip_first
        self.reset()

    def reset(self) -> None:
        self.W_max: float | None = None      # None until the first congestion event
        self.W_last_max: float | None = None
        self.ssthresh: float = float(self.b_max)
        self.t: int = 0                      # rollouts since the last congestion event
        self.in_slow_start: bool = True
        self._reason = "initial"

    # ------------------------------------------------------------------ helpers

    def _snap(self, b: float) -> int:
        """Project onto the actionable lattice, rounding to the nearest reachable B.

        Never returns the CURRENT value when the curve asked to move: at the plateau the
        per-rollout increment can be smaller than the quantum, and silently snapping back
        would freeze the controller for several rollouts. Nudge one quantum instead.
        """
        q = self.snap_quantum
        snapped = max(self.b_min, int(round(b / q)) * q) if q > 1 else int(round(b))
        if q > 1 and snapped == self.current and abs(b - self.current) > 1e-9:
            snapped = self.current + (q if b > self.current else -q)
        return snapped

    def K(self) -> float:
        """Rollouts the curve takes to climb from beta*W_max back to W_max."""
        if self.W_max is None:
            return 0.0
        return (self.W_max * (1.0 - self.beta) / self.C) ** (1.0 / 3.0)

    # ------------------------------------------------------------------ control law

    def _propose(self, obs: TunerObservation) -> int | None:
        if self.skip_first and obs.rollout_id == 0:
            self._reason = "rollout 0 skipped (startup-inflated, not representative)"
            return None
        r = SIGNAL_REGISTRY[self.signal](obs)
        # HOLD, never assume 0. An observation predating this signal reads None; treating
        # that as "no starvation" would ramp B to b_max over the run.
        if r is None or not math.isfinite(r):
            self._reason = f"no {self.signal} signal on this observation; holding"
            return None

        if r > self.epsilon:
            return self._on_congestion(r)

        self.t += 1
        if self.in_slow_start:
            if self.gamma <= 1.0:
                # gamma = 1.0 means NO slow start: hold B at its initial value until the
                # first congestion event establishes W_max, then let the curve take over.
                # This is the control arm for "does slow start earn anything", and it is
                # deliberately a HOLD rather than additive growth -- growing by a fixed
                # `step` would reintroduce exactly the constant-step behaviour CUBIC
                # exists to replace, and would make the arm untestable as a control.
                self._reason = (
                    f"{self.signal}={r:.5f} <= eps={self.epsilon:.5f}; gamma=1 so no slow "
                    f"start -- holding B={self.current} until the first congestion event"
                )
                return None
            nxt = min(self.ssthresh, self.current * self.gamma)
            self._reason = (
                f"{self.signal}={r:.5f} <= eps={self.epsilon:.5f}; slow start "
                f"x{self.gamma:g} -> {nxt:.1f} (ssthresh={self.ssthresh:.1f})"
            )
            return self._snap(nxt)

        k = self.K()
        nxt = self.C * (self.t - k) ** 3 + self.W_max
        region = "concave recovery" if self.t < k else ("plateau" if self.t < k + 1.5 else "convex probing")
        self._reason = (
            f"{self.signal}={r:.5f} <= eps={self.epsilon:.5f}; cubic t={self.t} "
            f"K={k:.1f} W_max={self.W_max:.1f} -> {nxt:.1f} ({region})"
        )
        return self._snap(nxt)

    def _on_congestion(self, r: float) -> int:
        """A starvation event: record the ceiling, drop to beta*W_max, restart the clock."""
        W = float(self.current)
        prev = self.W_max
        if prev is not None and W < prev and self.fast_convergence:
            # Ceiling appears to be falling -- pull W_max down further so the plateau
            # forms below the old boundary rather than at it.
            self.W_last_max, self.W_max = prev, W * (1.0 + self.beta) / 2.0
            how = f"fast convergence: W<{prev:.1f} so W_max={self.W_max:.1f}"
        else:
            self.W_last_max, self.W_max = prev, W
            how = f"W_max={self.W_max:.1f}"
        self.ssthresh = max(float(self.b_min), self.beta * self.W_max)
        self.in_slow_start = False
        self.t = 0
        nxt = max(float(self.b_min), self.beta * self.W_max)
        self._reason = (
            f"{self.signal}={r:.5f} > eps={self.epsilon:.5f} STARVED; {how}, "
            f"B -> {nxt:.1f} (beta={self.beta:g})"
        )
        return self._snap(nxt)


# ---------------------------------------------------------------------- observation


def collect_observation(
    rollout_id: int,
    threshold: int,
    wall_s: float,
    migrations: int = 0,
) -> TunerObservation | None:
    """Build a `TunerObservation` from the driver's in-memory tracer events.

    Requires NO new instrumentation. In the streaming path every emit is driver-side
    (training actors never `init_tracer`), so by the end of a rollout the driver's event
    list already holds that rollout's `training`, `chunk_*` and `ws_*` spans. Reading the
    private `_events` under `_lock` follows existing precedent (`train.py` reads
    `get_tracer()._wall_epoch`).

    Returns None when the tracer is disabled or the rollout emitted no training spans —
    callers should treat that as "no signal" and leave B alone.

    Deliberately does NOT use `overlap_time` / `last_engine_done_time` from the driver:
    despite their names those derive from the last group *flip*, not last inference
    completion, so a holding GroupSwitchController inflates both.
    """
    from slime.utils.perfetto_tracer import get_tracer

    tracer = get_tracer()
    events = getattr(tracer, "_events", None)
    if not events:
        return None
    lock = getattr(tracer, "_lock", None)
    if lock is not None:
        with lock:
            snapshot = list(events)
    else:
        snapshot = list(events)

    span_by_pid: dict[int, list[tuple[float, float]]] = {}
    busy_by_pid: dict[int, list[tuple[float, float]]] = {}
    for e in snapshot:
        if e.get("ph") != "X" or e.get("dur") is None:
            continue
        pid = e.get("pid", -1)
        if not (_ENGINE_PID_LO <= pid < _ENGINE_PID_HI):
            continue
        if (e.get("args") or {}).get("rollout_id") != rollout_id:
            continue
        name = e.get("name", "")
        iv = (e["ts"], e["ts"] + e["dur"])
        if name == "training":
            span_by_pid.setdefault(pid, []).append(iv)
        elif name.startswith("chunk_") or name.startswith("ws_"):
            busy_by_pid.setdefault(pid, []).append(iv)

    span = sum(_union_len(v) for v in span_by_pid.values())
    if span <= 0:
        return None
    # Union per pid, never a naive sum: the tracer is known to emit duplicate spans
    # (verified on the canonical benchmark, engines 2 and 5 at rollout 0).
    busy = sum(_union_len(v) for v in busy_by_pid.values())

    # Split the idle residual into interior (between-chunk starvation) and trailing
    # (post-last-chunk wait for the global barrier). See TunerObservation for why the
    # two are physically different and why only interior tracks migration aggression.
    # `busy` above is left EXACTLY as it was so `idle_ratio` stays byte-identical to
    # what perf_analysis/compare_gpu_time.py reports; interior/trailing are additional.
    interior = trailing = 0.0
    for pid, ivs in span_by_pid.items():
        span_len = _union_len(ivs)
        if span_len <= 0:
            continue
        busy_ivs = busy_by_pid.get(pid)
        if not busy_ivs:
            # Flipped into training, received zero chunks: polled an empty queue for
            # the whole span. Extreme starvation, not a tail wait -- charge it all to
            # interior. Measured: 0/10 rollouts at B=64, 8/10 at B=128.
            interior += span_len
            continue
        tail = max(0.0, max(e for _, e in ivs) - max(e for _, e in busy_ivs))
        trailing += tail
        # Clamped at 0: a chunk span can in principle overhang the training span.
        interior += max(0.0, span_len - _union_len(busy_ivs) - tail)

    return TunerObservation(
        rollout_id=rollout_id,
        threshold=threshold,
        idle_ratio=max(0.0, 1.0 - busy / span),
        training_span_gpu_s=span / 1e6,
        busy_gpu_s=busy / 1e6,
        wall_s=wall_s,
        migrations=migrations,
        interior_idle_gpu_s=interior / 1e6,
        trailing_idle_gpu_s=trailing / 1e6,
    )


def _union_len(intervals: list[tuple[float, float]]) -> float:
    """Total length covered by merged [start, end) intervals. Same units in/out."""
    if not intervals:
        return 0.0
    iv = sorted(intervals)
    total = 0.0
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return total + (ce - cs)


# ------------------------------------------------------------------------- registry

# Single source of truth for `--threshold-tuner` name -> class, mirroring
# MIGRATION_POLICY_REGISTRY.
THRESHOLD_TUNER_REGISTRY: dict[str, type[ThresholdTuner]] = {
    "fixed": FixedTuner,
    "idle_ratio": IdleRatioTuner,
    "idle_threshold": IdleThresholdTuner,
    "interior_idle": InteriorIdleTuner,
    "cubic": CubicTuner,
}


def make_threshold_tuner(args) -> ThresholdTuner:
    """Build the tuner named by `--threshold-tuner`, seeded with the CLI's B."""
    name = getattr(args, "threshold_tuner", "fixed") or "fixed"
    if name not in THRESHOLD_TUNER_REGISTRY:
        raise ValueError(
            f"Unknown threshold tuner: {name!r}. "
            f"Choices: {sorted(THRESHOLD_TUNER_REGISTRY)}"
        )
    cls = THRESHOLD_TUNER_REGISTRY[name]
    initial = int(getattr(args, "migration_batch_threshold", 8) or 8)
    kwargs = dict(
        initial=initial,
        b_min=int(getattr(args, "tuner_b_min", 8)),
        b_max=int(getattr(args, "tuner_b_max", 256)),
        step=int(getattr(args, "tuner_step", 16)),
    )
    if issubclass(cls, CubicTuner):
        # snap_quantum is DERIVED, not a flag. Under `groups` the policy can only fire at
        # multiples of n_samples_per_prompt, so a sub-quantum move is a no-op; under
        # `samples` B is a raw live count and the lattice is the identity.
        unit = str(getattr(args, "migration_count_unit", "groups") or "groups")
        q = 1 if unit == "samples" else max(1, int(getattr(args, "n_samples_per_prompt", 1) or 1))
        eps = getattr(args, "tuner_interior_target", None)
        kwargs.update(
            C=float(getattr(args, "tuner_cubic_c", 1.0)),
            beta=float(getattr(args, "tuner_cubic_beta", 0.7)),
            gamma=float(getattr(args, "tuner_cubic_gamma", 2.0)),
            epsilon=0.005 if eps is None else float(eps),
            fast_convergence=bool(int(getattr(args, "tuner_cubic_fast_convergence", 0) or 0)),
            snap_quantum=q,
            skip_first=bool(int(getattr(args, "tuner_skip_first", 1))),
        )
    elif issubclass(cls, BangBangTuner):
        # Each bang-bang tuner names its own target arg, so adding one needs no edit
        # here. `None` -> the subclass's default_target, which keeps a tuner whose flag
        # was never passed on its own calibrated epsilon instead of another's.
        target = getattr(args, cls.target_arg, None)
        kwargs.update(
            target=None if target is None else float(target),
            skip_first=bool(int(getattr(args, "tuner_skip_first", 1))),
        )
    elif issubclass(cls, IdleRatioTuner):
        # Fallbacks here must match the class defaults, or a caller that constructs args
        # by hand (tests, sweeps) silently gets different control behaviour than the CLI.
        kwargs.update(
            warmup=int(getattr(args, "tuner_warmup", 3)),
            alpha=float(getattr(args, "tuner_ewma_alpha", 0.4)),
            dead_band=float(getattr(args, "tuner_dead_band", 0.20)),
            patience=int(getattr(args, "tuner_patience", 2)),
        )
    return cls(**kwargs)
