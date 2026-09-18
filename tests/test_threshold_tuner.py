"""Tests for slime/router/threshold_tuner.py — no GPU, no Ray, no tracer.

Run:  python3 tests/test_threshold_tuner.py

`ThresholdTuner.update()` is a pure function of a `TunerObservation`, which is exactly
what makes this testable: a synthetic observation stream stands in for a run. The cases
below pin the properties that matter for a controller nobody watches for two hours:

  rails          a subclass cannot move B more than one step per rollout, escape
                 [b_min,b_max], or reach 0 (which would silently disable migration)
  calibration    the baseline is learned per-run, never assumed -- absolute idle_ratio
                 levels are workload-specific (Text2SQL 4-7%, DAPO-math 2.7-2.9%)
  dead band      noise inside the band must NOT move B; the measured per-rollout CV is
                 26-36% at fixed B, so a controller without a dead band random-walks
  cliff          a sustained spike (the t128 signature, ~2.3x baseline) must retreat
  degenerate     missing/non-finite signal holds rather than crashing or lurching
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slime.router.threshold_tuner import (  # noqa: E402
    SIGNAL_REGISTRY,
    CubicTuner,
    THRESHOLD_TUNER_REGISTRY,
    BangBangTuner,
    FixedTuner,
    IdleRatioTuner,
    IdleThresholdTuner,
    InteriorIdleTuner,
    ThresholdTuner,
    TunerObservation,
    _union_len,
    make_threshold_tuner,
)

FAILURES = []


def obs(ratio, rollout_id=0, threshold=64, span=100.0, wall=500.0):
    busy = span * (1.0 - ratio) if ratio is not None and math.isfinite(ratio) else span
    return TunerObservation(
        rollout_id=rollout_id, threshold=threshold, idle_ratio=ratio,
        training_span_gpu_s=span, busy_gpu_s=busy, wall_s=wall,
    )


def drive(tuner, ratios, start=1):
    """Feed a ratio stream, return the B trajectory.

    Starts at rollout_id 1 by default: rollout 0 is deliberately skipped by the tuner as
    startup-inflated, so numbering from 0 would silently drop the first sample of every
    test stream.
    """
    return [tuner.update(obs(r, rollout_id=i)) for i, r in enumerate(ratios, start=start)]


def check(label, got, want):
    ok = got == want if not isinstance(want, float) else abs(got - want) < 1e-9
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<52} got={got!r} want={want!r}")
    if not ok:
        FAILURES.append(label)


def case(t):
    print(f"\n{t}")


# ------------------------------------------------------------------------- rails


def test_rails_clamp_and_step():
    case("rails — clamp, one step per rollout, never zero")

    class Runaway(ThresholdTuner):
        """Proposes an absurd value every time; the base class must contain it."""
        def reset(self): pass
        def _propose(self, o): return 100000

    t = Runaway(initial=64, b_min=8, b_max=128, step=16)
    check("one step per rollout despite absurd proposal", t.update(obs(0.02)), 80)
    check("second step", t.update(obs(0.02)), 96)
    check("third step", t.update(obs(0.02)), 112)
    check("clamps at b_max", t.update(obs(0.02)), 128)
    check("stays at b_max", t.update(obs(0.02)), 128)

    class Crasher(ThresholdTuner):
        def reset(self): pass
        def _propose(self, o): return -9999

    t2 = Crasher(initial=24, b_min=8, b_max=128, step=16)
    check("descends one step", t2.update(obs(0.5)), 8)
    check("clamps at b_min, never 0", t2.update(obs(0.5)), 8)


def test_rails_validate_construction():
    case("rails — constructor validation")
    for kwargs, why in [
        (dict(initial=64, b_min=0), "b_min < 1"),
        (dict(initial=64, b_min=64, b_max=8), "b_max < b_min"),
        (dict(initial=64, step=0), "step < 1"),
    ]:
        try:
            FixedTuner(**kwargs)
            check(f"rejects {why}", "no raise", "ValueError")
        except ValueError:
            check(f"rejects {why}", "ValueError", "ValueError")
    for kwargs, why in [
        (dict(initial=64, alpha=0.0), "alpha=0"),
        (dict(initial=64, alpha=1.5), "alpha>1"),
        (dict(initial=64, dead_band=-0.1), "negative dead_band"),
        (dict(initial=64, warmup=0), "warmup<1"),
        (dict(initial=64, patience=0), "patience<1"),
    ]:
        try:
            IdleRatioTuner(**kwargs)
            check(f"rejects {why}", "no raise", "ValueError")
        except ValueError:
            check(f"rejects {why}", "ValueError", "ValueError")


def test_fixed_is_inert():
    case("FixedTuner — never moves B (the control arm)")
    t = FixedTuner(initial=64, b_min=8, b_max=128, step=16)
    traj = drive(t, [0.001, 0.5, 0.02, 0.9, 0.03])
    check("B constant across wild swings", traj, [64] * 5)


# ------------------------------------------------------------------ IdleRatioTuner


def test_calibrates_then_holds_at_baseline():
    case("IdleRatioTuner — calibrates a per-run baseline, then holds on flat input")
    t = IdleRatioTuner(initial=64, b_min=8, b_max=128, step=16, warmup=3, dead_band=0.15)
    traj = drive(t, [0.03] * 8)
    check("no movement during warmup or on flat input", traj, [64] * 8)
    check("baseline learned", round(t._baseline, 4), 0.03)


def test_low_ratio_climbs():
    case("IdleRatioTuner — sustained low ratio -> more aggressive")
    t = IdleRatioTuner(initial=64, b_min=8, b_max=256, step=16, warmup=3, dead_band=0.15)
    drive(t, [0.04] * 3)                       # baseline 0.04
    traj = drive(t, [0.01] * 6)                # well under the band, sustained
    check("climbs once per `patience` rollouts", traj, [64, 80, 80, 96, 96, 112])


# Per-rollout idle_ratio measured from the committed traces. These are the real signal
# the controller has to survive, not synthetic smooth curves.
DAPO_T64 = [0.0154, 0.0184, 0.0347, 0.0252, 0.0177, 0.0213, 0.0443, 0.0211, 0.0293, 0.0202]
DAPO_T128 = [0.0460, 0.0189, 0.1042, 0.1532, 0.0375, 0.0239, 0.0513, 0.0388, 0.1340, 0.0232]
T2S_T32 = [0.0204, 0.0229, 0.0354, 0.0220, 0.0244, 0.0169, 0.0245, 0.0197, 0.0322,
           0.0264, 0.0380, 0.0187, 0.0298, 0.0252, 0.0162]


def test_rollout_zero_is_excluded_from_the_baseline():
    """Rollout 0 carries one-time startup and must not set the setpoint.

    Measured live in both eval runs: r0's idle_ratio was the highest of the run
    (0.0290 vs 0.0123-0.0211, and 0.0252 vs 0.0187), and its wall was 771s vs ~560s
    steady state. A baseline inflated by r0 makes the controller believe it has headroom
    and climb -- the opposite bias to the one `patience` guards against.
    """
    case("IdleRatioTuner — rollout 0 excluded from calibration")
    t = IdleRatioTuner(initial=64, warmup=2, dead_band=0.20, patience=1)
    t.update(obs(0.10, rollout_id=0))                 # anomalous startup rollout
    check("r0 does not move B", t.current, 64)
    check("r0 not folded into the EWMA", t._ewma, None)
    check("reason names it", "rollout 0 skipped" in t.explain(), True)
    for i, r in enumerate([0.02, 0.02], start=1):     # warmup=2 real rollouts
        t.update(obs(r, rollout_id=i))
    check("baseline comes from real rollouts only", round(t._baseline, 4), 0.02)


def test_no_drift_on_real_stationary_streams():
    """The regression this whole `patience` mechanism exists for.

    B must not move on a stream where the config never changed. Before `patience`, the
    DAPO t64 stream drove B from 64 down to 32 purely on noise: the baseline calibrated
    from the first few rollouts landed 5-11% below the stream's true mean, so every later
    reading read "above baseline" and B ratcheted down.
    """
    case("IdleRatioTuner — no drift on REAL stationary streams (regression)")
    for name, stream, start in [("DAPO t64", DAPO_T64, 64), ("T2S t32", T2S_T32, 32)]:
        t = IdleRatioTuner(initial=start, b_min=8, b_max=256, step=16)
        traj = drive(t, stream)
        check(f"{name}: B never moves", set(traj), {start})


def test_cliff_retreats_with_a_sane_baseline():
    """Cliff detection, in the sequence a real run actually produces.

    NOTE the calibration order. Feeding the t128 stream from rollout 0 does NOT trigger a
    retreat, and that is correct behaviour rather than a bug: the baseline would be
    calibrated FROM the cliff, so the cliff becomes "normal". A real run calibrates at a
    sane B first and only then climbs into trouble, which is what this reproduces.
    """
    case("IdleRatioTuner — retreats from the real t128 cliff once a baseline exists")
    t = IdleRatioTuner(initial=96, b_min=8, b_max=256, step=16)
    traj = drive(t, DAPO_T64[:5] + DAPO_T128)
    check("baseline phase does not move B", set(traj[:5]), {96})
    check("retreats after the cliff arrives", traj[-1] < 96, True)
    check("retreat is gradual, not a lurch to b_min", traj[-1] >= 32, True)


def test_baseline_calibrated_at_the_cliff_is_a_known_limitation():
    """Documents the failure mode so nobody rediscovers it as a bug.

    A self-calibrating controller started AT a bad B measures the cliff and treats it as
    normal, so it never retreats. Whether it then holds or climbs further depends on where
    the warmup window happens to land relative to the stream's own peaks -- so only the
    robust claim is asserted here: it will NOT come back down on its own.

    No baseline derived from a cliff can detect that cliff -- arithmetic, not a tuning
    failure. The mitigation is operational: start B conservatively so calibration happens
    somewhere healthy. Pinned here so the behaviour is documented rather than discovered
    mid-run.
    """
    case("IdleRatioTuner — baseline calibrated AT the cliff cannot detect it")
    t = IdleRatioTuner(initial=128, b_min=8, b_max=256, step=16)
    traj = drive(t, DAPO_T128)
    check("never retreats (cannot see its own baseline as bad)", min(traj) >= 128, True)


def test_ewma_damps_a_single_outlier():
    case("IdleRatioTuner — one outlier rollout does not swing B")
    t = IdleRatioTuner(initial=64, b_min=8, b_max=256, step=16, warmup=3,
                       alpha=0.4, dead_band=0.15)
    drive(t, [0.03] * 3)
    before = t.current
    t.update(obs(0.15))                        # 5x spike, single rollout
    after_spike = t.current
    check("single spike moves B by at most one step", abs(after_spike - before) <= 16, True)


def test_degenerate_signal_holds():
    case("IdleRatioTuner — missing / non-finite signal holds")
    t = IdleRatioTuner(initial=64, b_min=8, b_max=128, step=16, warmup=2)
    check("None ratio holds", t.update(obs(None)), 64)
    check("NaN ratio holds", t.update(obs(float("nan"))), 64)
    check("inf ratio holds", t.update(obs(float("inf"))), 64)
    check("reason explains", "no signal" in t.explain(), True)


def test_reset_clears_baseline():
    case("IdleRatioTuner — reset() clears per-run state")
    t = IdleRatioTuner(initial=64, warmup=2)
    drive(t, [0.05] * 4)
    check("baseline set before reset", t._baseline is not None, True)
    t.reset()
    check("baseline cleared", t._baseline is None, True)
    check("ewma cleared", t._ewma is None, True)


# --------------------------------------------------------------------- plumbing


def test_registry_and_factory():
    case("registry + make_threshold_tuner")
    check("registry names", sorted(THRESHOLD_TUNER_REGISTRY),
          ["cubic", "fixed", "idle_ratio", "idle_threshold", "interior_idle"])

    class A:
        threshold_tuner = "idle_ratio"
        migration_batch_threshold = 64
        tuner_b_min, tuner_b_max, tuner_step = 16, 96, 8
        tuner_warmup, tuner_ewma_alpha, tuner_dead_band = 2, 0.5, 0.2
        tuner_patience = 3

    t = make_threshold_tuner(A())
    check("class", type(t).__name__, "IdleRatioTuner")
    check("seeded from --migration-batch-threshold", t.current, 64)
    check("rails plumbed", (t.b_min, t.b_max, t.step), (16, 96, 8))
    check("tuner args plumbed", (t.warmup, t.alpha, t.dead_band), (2, 0.5, 0.2))
    # Regression: patience was parsed by the CLI but dropped by the factory, so
    # --tuner-patience 3 silently ran with the default 2.
    check("patience plumbed (regression)", t.patience, 3)

    class D:
        threshold_tuner = "idle_ratio"       # no tuner_* attrs -> factory fallbacks
    dflt = make_threshold_tuner(D())
    check("factory fallbacks match class defaults",
          (dflt.warmup, dflt.alpha, dflt.dead_band, dflt.patience),
          (IdleRatioTuner(initial=8).warmup, IdleRatioTuner(initial=8).alpha,
           IdleRatioTuner(initial=8).dead_band, IdleRatioTuner(initial=8).patience))

    class B:
        pass  # no attrs at all -> must default to the inert tuner

    check("defaults to FixedTuner", type(make_threshold_tuner(B())).__name__, "FixedTuner")

    class C:
        threshold_tuner = "nope"

    try:
        make_threshold_tuner(C())
        check("unknown name raises", "no raise", "ValueError")
    except ValueError as e:
        check("unknown name raises and lists choices", "idle_ratio" in str(e), True)


def test_union_len():
    case("_union_len — merge semantics (tracer emits duplicate spans)")
    check("disjoint", _union_len([(0, 1), (2, 3)]), 2)
    check("overlapping", _union_len([(0, 2), (1, 3)]), 3)
    check("exact duplicate counted once", _union_len([(0, 5), (0, 5)]), 5)
    check("nested", _union_len([(0, 10), (2, 4)]), 10)
    check("empty", _union_len([]), 0.0)


# ------------------------------------------------------- IdleThresholdTuner (bang-bang)


def drive_thresh(ratios, target=0.03, initial=64, start=1, **kw):
    """Run a ratio stream through IdleThresholdTuner, return the B trajectory.

    `start=1` by default for the same reason `drive()` does it: rollout 0 is skipped.
    """
    t = IdleThresholdTuner(initial=initial, target=target, **kw)
    return [t.update(obs(r, rollout_id=i)) for i, r in enumerate(ratios, start=start)]


def test_bangbang_direction():
    case("IdleThresholdTuner — absolute threshold, one step every rollout")
    check("above target decreases", drive_thresh([0.09] * 3), [48, 32, 16])
    check("below target increases", drive_thresh([0.001] * 3), [80, 96, 112])
    # `>` not `>=`: exactly at target counts as headroom. Pinned so a refactor cannot
    # flip the boundary silently.
    check("exactly at target increases", drive_thresh([0.03]), [80])


def test_bangbang_never_holds():
    case("IdleThresholdTuner — no hold state (the defining difference vs idle_ratio)")
    traj = drive_thresh([0.02, 0.05, 0.02, 0.05])
    check("B moves every rollout", all(a != b for a, b in zip([64] + traj, traj)), True)
    check("alternating input alternates B", traj, [80, 64, 80, 64])


def test_bangbang_skips_rollout_zero():
    case("IdleThresholdTuner — rollout 0 is startup-inflated and must not move B")
    check("r0 held, r1 acts", drive_thresh([0.09, 0.09], start=0), [64, 48])
    check("skip_first=False acts on r0", drive_thresh([0.09], start=0, skip_first=False), [48])


def test_bangbang_respects_rails():
    case("IdleThresholdTuner — base-class rails still bound it")
    check("clamps at b_min", drive_thresh([0.09] * 10, b_min=32)[-1], 32)
    check("clamps at b_max", drive_thresh([0.0] * 10, b_max=96)[-1], 96)


def test_bangbang_oscillates_on_stationary_stream():
    """Documents the known limit cycle so it reads as expected, not as a bug.

    Real measured DAPO t64 idle_ratios with the target set at their own mean. With no
    EWMA and no dead band there is no fixed point -- bang-bang control oscillates by
    construction. This is the cost paid for reacting in 1 rollout instead of ~4.
    """
    case("IdleThresholdTuner — limit cycle on a stationary real stream (expected)")
    traj = drive_thresh(DAPO_T64[1:], target=0.026)
    check("does not converge", len(set(traj)) > 1, True)
    check("stays bounded (no ratchet)", max(traj) - min(traj) <= 4 * 16, True)


def test_bangbang_validates_target():
    case("IdleThresholdTuner — target must be a fraction in (0, 1)")
    for bad in (0.0, 1.0, -0.1, 1.5):
        try:
            IdleThresholdTuner(initial=64, target=bad)
            check(f"target={bad} raises", "no raise", "ValueError")
        except ValueError:
            check(f"target={bad} raises", "ValueError", "ValueError")


def test_bangbang_factory_forwards_target():
    """A dropped kwarg silently reverts to the default target -- the exact bug class
    that already bit --tuner-patience."""
    case("IdleThresholdTuner — factory forwards --tuner-idle-target")

    class A:
        threshold_tuner = "idle_threshold"
        migration_batch_threshold = 64
        tuner_b_min, tuner_b_max, tuner_step = 16, 160, 16
        tuner_idle_target = 0.055

    t = make_threshold_tuner(A())
    check("builds IdleThresholdTuner", type(t).__name__, "IdleThresholdTuner")
    check("target forwarded", t.target, 0.055)
    check("rails forwarded", (t.b_min, t.b_max, t.step, t.current), (16, 160, 16, 64))


# ------------------------------------------------------ InteriorIdleTuner (interior)

# Real measured whole-run interior ratios (see TunerObservation for provenance).
INTERIOR_HEALTHY_T64 = [0.00051, 0.00047, 0.00039, 0.00037, 0.00039,
                        0.00044, 0.00047, 0.00044, 0.00041, 0.00048]
# B=128, the over-aggression cliff: rollout 0 and 8 are fine, the rest starve.
INTERIOR_CLIFF_T128 = [0.00048, 0.08588, 0.30429, 0.02777, 0.01922,
                       0.01449, 0.16836, 0.09421, 0.00060, 0.06850]


def iobs(interior_ratio, rollout_id=1, threshold=64, span=100.0):
    """Observation carrying ONLY an interior signal, built the way the driver does.

    interior is stored in GPU-seconds and divided by the span, so this exercises the
    property rather than smuggling a ratio straight in.
    """
    return TunerObservation(
        rollout_id=rollout_id, threshold=threshold, idle_ratio=0.02,
        training_span_gpu_s=span, busy_gpu_s=span * 0.98, wall_s=500.0,
        interior_idle_gpu_s=None if interior_ratio is None else interior_ratio * span,
        trailing_idle_gpu_s=span * 0.019,
    )


def drive_interior(ratios, initial=64, start=1, **kw):
    t = InteriorIdleTuner(initial=initial, **kw)
    return [t.update(iobs(r, rollout_id=i)) for i, r in enumerate(ratios, start=start)]


def test_interior_direction():
    case("InteriorIdleTuner — below epsilon climbs, above it retreats")
    check("healthy interior climbs", drive_interior([0.0005] * 3), [80, 96, 112])
    check("starving interior retreats", drive_interior([0.05] * 3), [48, 32, 16])
    # Same `>` boundary as the parent law; pinned so a refactor cannot flip it.
    check("exactly at epsilon climbs", drive_interior([0.005]), [80])


def test_interior_separates_the_measured_regimes():
    """The point of the tuner: at the default epsilon the real healthy stream must
    climb monotonically and the real cliff stream must retreat overall."""
    case("InteriorIdleTuner — real measured streams separate at the default epsilon")
    healthy = drive_interior(INTERIOR_HEALTHY_T64, b_max=256)
    check("healthy stream never backs off",
          all(b > a for a, b in zip([64] + healthy, healthy)), True)
    # 8 of the 10 cliff rollouts are above epsilon and 2 below, so the stream is driven
    # into the b_min floor rather than to a net -6 steps: it reaches 8 by rollout 6, the
    # two sub-epsilon readings lift it one step each, and every other reading re-floors
    # it. Pinning the floor (not the arithmetic net) is what actually matters.
    cliff = drive_interior(INTERIOR_CLIFF_T128, b_min=8, b_max=256)
    check("cliff stream is driven to b_min", cliff[-1], 8)
    check("cliff reaches the floor by rollout 6", cliff[5], 8)
    check("cliff never climbs above its start", max(cliff) <= 80, True)


def test_interior_ignores_trailing_noise():
    """The whole reason this tuner exists: trailing swings must not move B.

    Both observations carry the SAME interior signal but wildly different trailing
    idle, which is what drives idle_ratio's 26-36% CV. B must be identical.
    """
    case("InteriorIdleTuner — trailing variation does not move B")
    quiet = TunerObservation(
        rollout_id=1, threshold=64, idle_ratio=0.009, training_span_gpu_s=100.0,
        busy_gpu_s=99.1, wall_s=500.0,
        interior_idle_gpu_s=0.05, trailing_idle_gpu_s=0.85)
    noisy = TunerObservation(
        rollout_id=1, threshold=64, idle_ratio=0.042, training_span_gpu_s=100.0,
        busy_gpu_s=95.8, wall_s=500.0,
        interior_idle_gpu_s=0.05, trailing_idle_gpu_s=4.15)
    a = InteriorIdleTuner(initial=64).update(quiet)
    b = InteriorIdleTuner(initial=64).update(noisy)
    check("4.7x trailing swing, same B", (a, b), (80, 80))
    # Same two observations through the combined signal DO diverge -- that contrast is
    # the justification for the new tuner, so pin it.
    c = IdleThresholdTuner(initial=64, target=0.026).update(quiet)
    d = IdleThresholdTuner(initial=64, target=0.026).update(noisy)
    check("combined signal diverges on the same pair", (c, d), (80, 48))


def test_interior_missing_signal_holds_not_ramps():
    """A stale observation must HOLD. If `interior_idle_gpu_s` defaulted to 0.0 this
    would read as 'zero starvation' and ramp B to b_max over a run -- the silent
    monotone ramp that already wasted one live eval."""
    case("InteriorIdleTuner — missing decomposition holds instead of ramping")
    stale = TunerObservation(
        rollout_id=1, threshold=64, idle_ratio=0.03,
        training_span_gpu_s=100.0, busy_gpu_s=97.0, wall_s=500.0)
    check("field defaults to None", stale.interior_idle_gpu_s, None)
    check("ratio property is None", stale.interior_idle_ratio, None)
    check("holds on 10 stale rollouts", drive_interior([None] * 10)[-1], 64)
    t = InteriorIdleTuner(initial=64)
    check("B unchanged", t.update(stale), 64)
    check("reason names the signal", "no interior_idle_ratio signal" in t.explain(), True)
    check("span=0 also holds",
          TunerObservation(rollout_id=1, threshold=64, idle_ratio=0.0,
                           training_span_gpu_s=0.0, busy_gpu_s=0.0, wall_s=0.0,
                           interior_idle_gpu_s=1.0).interior_idle_ratio, None)


def test_interior_skips_rollout_zero_and_respects_rails():
    case("InteriorIdleTuner — inherits rollout-0 skip and the base-class rails")
    check("r0 held, r1 acts", drive_interior([0.05, 0.05], start=0), [64, 48])
    check("clamps at b_min", drive_interior([0.05] * 10, b_min=32)[-1], 32)
    check("clamps at b_max", drive_interior([0.0] * 10, b_max=96)[-1], 96)
    check("one step per rollout", drive_interior([0.0005], initial=64)[0] - 64, 16)


def test_interior_defaults_and_factory():
    case("InteriorIdleTuner — calibrated default epsilon + its OWN target flag")
    check("default epsilon", InteriorIdleTuner(initial=64).target, 0.005)
    check("signal", InteriorIdleTuner(initial=64).signal, "interior_idle_ratio")

    class A:
        threshold_tuner = "interior_idle"
        migration_batch_threshold = 64
        tuner_b_min, tuner_b_max, tuner_step = 16, 160, 16
        tuner_interior_target = 0.002

    t = make_threshold_tuner(A())
    check("builds InteriorIdleTuner", type(t).__name__, "InteriorIdleTuner")
    check("target forwarded", t.target, 0.002)
    check("rails forwarded", (t.b_min, t.b_max, t.step, t.current), (16, 160, 16, 64))

    class B:
        threshold_tuner = "interior_idle"       # no target flag -> class default

    check("unset flag uses class default", make_threshold_tuner(B()).target, 0.005)

    # The cross-tuner footgun: --tuner-idle-target must NOT leak into interior_idle.
    class C:
        threshold_tuner = "interior_idle"
        tuner_idle_target = 0.03

    check("does not inherit --tuner-idle-target",
          make_threshold_tuner(C()).target, 0.005)


def test_bangbang_base_is_swappable():
    """A new bang-bang tuner must cost three class attributes and nothing else."""
    case("BangBangTuner — swapping the signal needs no method override")

    class TrailingTuner(BangBangTuner):
        signal = "trailing_idle_ratio"
        default_target = 0.03
        target_arg = "tuner_trailing_target"

    check("no _propose override", "_propose" in TrailingTuner.__dict__, False)
    t = TrailingTuner(initial=64)
    check("reads its own signal", t.update(iobs(0.9, rollout_id=1)), 80)  # trailing=1.9%
    check("reason names its signal", "trailing_idle_ratio" in t.explain(), True)

    class Absolute(BangBangTuner):
        signal = "interior_idle_gpu_s"
        default_target = 10.0
        target_arg = "tuner_abs_target"
        target_is_ratio = False

    check("non-ratio target accepted", Absolute(initial=64).target, 10.0)
    check("absolute signal decreases above target",
          Absolute(initial=64).update(iobs(0.5, rollout_id=1)), 48)  # 50 GPU-s > 10

    class Bogus(BangBangTuner):
        signal = "not_a_signal"

    try:
        Bogus(initial=64)
        check("unknown signal raises", "no raise", "ValueError")
    except ValueError as e:
        check("unknown signal raises and lists choices",
              "interior_idle_ratio" in str(e), True)

    check("every registry signal resolves on a full observation",
          sorted(SIGNAL_REGISTRY),
          sorted(["idle_ratio", "interior_idle_ratio",
                  "trailing_idle_ratio", "interior_idle_gpu_s"]))
    full = iobs(0.001)
    for nm, fn in SIGNAL_REGISTRY.items():
        check(f"signal {nm} is numeric", isinstance(fn(full), float), True)


def test_interior_decomposition_from_events():
    """`collect_observation`'s split, exercised through the same arithmetic it uses.

    Two GPUs: one with a normal trailing gap, one that got ZERO chunks. The zero-chunk
    GPU is charged entirely to interior -- that case appears in 8/10 rollouts on the
    measured cliff run and 0/10 healthy, so it must not be filed as a barrier wait.
    """
    case("idle decomposition — zero-chunk group is interior, not trailing")
    # GPU A: span [0,100], chunks covering [0,60] and [70,90] -> interior 10, trail 10.
    span_a, busy_a = [(0.0, 100.0)], [(0.0, 60.0), (70.0, 90.0)]
    tail_a = max(0.0, max(e for _, e in span_a) - max(e for _, e in busy_a))
    int_a = max(0.0, _union_len(span_a) - _union_len(busy_a) - tail_a)
    check("GPU with chunks: trailing", tail_a, 10.0)
    check("GPU with chunks: interior", int_a, 10.0)
    # GPU B: span [0,100], no chunks at all -> all 100 is interior.
    check("zero-chunk GPU: all interior", _union_len([(0.0, 100.0)]), 100.0)

    o = TunerObservation(
        rollout_id=1, threshold=64, idle_ratio=0.6,
        training_span_gpu_s=200.0, busy_gpu_s=80.0, wall_s=1.0,
        interior_idle_gpu_s=int_a + 100.0, trailing_idle_gpu_s=tail_a)
    check("interior ratio", round(o.interior_idle_ratio, 4), 0.55)
    check("trailing ratio", round(o.trailing_idle_ratio, 4), 0.05)



# ------------------------------------------------------------------------- cubic


def cobs(interior, rollout_id=1, threshold=32, span=100.0):
    """Observation carrying a given interior_idle_ratio."""
    return TunerObservation(
        rollout_id=rollout_id, threshold=threshold, idle_ratio=0.03,
        training_span_gpu_s=span, busy_gpu_s=span * 0.97, wall_s=500.0,
        interior_idle_gpu_s=interior * span, trailing_idle_gpu_s=0.0,
    )


def test_cubic_slow_start_then_congestion_sets_wmax():
    case("cubic — slow start grows x gamma, congestion sets W_max and drops to beta*W_max")
    t = CubicTuner(initial=10, b_min=4, b_max=200, C=1.0, beta=0.7, gamma=2.0,
                   epsilon=0.005, skip_first=False)
    check("slow start doubles", t.update(cobs(0.001, rollout_id=0)), 20)
    check("and again", t.update(cobs(0.001, rollout_id=1)), 40)
    check("in slow start", t.in_slow_start, True)
    # starvation at B=40 -> W_max=40, B -> 0.7*40 = 28
    check("congestion drops to beta*W_max", t.update(cobs(0.30, rollout_id=2)), 28)
    check("W_max recorded", t.W_max, 40.0)
    check("slow start exited for good", t.in_slow_start, False)
    check("clock restarted", t.t, 0)


def test_cubic_curve_recovers_then_plateaus_then_probes():
    case("cubic — concave recovery, plateau near W_max, then convex probing")
    t = CubicTuner(initial=40, b_min=4, b_max=500, C=1.0, beta=0.7,
                   epsilon=0.005, skip_first=False)
    t.update(cobs(0.30, rollout_id=0))              # congestion: W_max=40, B=28
    traj = [t.update(cobs(0.001, rollout_id=i)) for i in range(1, 6)]
    print(f"     trajectory after W_max=40: {traj}")
    check("recovers toward W_max, never past it early", all(x <= 41 for x in traj[:2]), True)
    check("reaches the plateau at ~W_max", abs(traj[2] - 40) <= 2, True)
    check("then probes ABOVE W_max", traj[-1] > 40, True)
    check("monotone non-decreasing while healthy", traj == sorted(traj), True)


def test_cubic_fast_convergence_flag():
    case("cubic — fast convergence pulls W_max down only when enabled")
    for fc, want in ((False, 40.0), (True, 34.0)):
        t = CubicTuner(initial=60, b_min=4, b_max=500, C=1.0, beta=0.7,
                       epsilon=0.005, fast_convergence=fc, skip_first=False)
        t.update(cobs(0.30, rollout_id=0))          # W_max=60
        t.current = 40                              # pretend we are back down at 40
        t.update(cobs(0.30, rollout_id=1))          # congestion BELOW the old W_max
        check(f"fast_convergence={fc} -> W_max", t.W_max, want)


def test_cubic_rails_allow_big_moves_but_clamp():
    case("cubic — rails widened (curve decides the move) but b_min/b_max still bound it")
    t = CubicTuner(initial=100, b_min=20, b_max=120, step=16, C=1.0, beta=0.7,
                   epsilon=0.005, skip_first=False)
    # a 30-unit drop in one rollout would be impossible under the symmetric `step` cap
    check("multiplicative decrease is not capped at step", t.update(cobs(0.30, rollout_id=0)), 70)
    check("max_up_step widened", t.max_up_step > t.step, True)
    t2 = CubicTuner(initial=100, b_min=20, b_max=110, C=1.0, epsilon=0.005, skip_first=False)
    t2.update(cobs(0.30, rollout_id=0))
    for i in range(1, 12):
        t2.update(cobs(0.001, rollout_id=i))
    check("convex probing still clamped at b_max", t2.current, 110)


def test_cubic_snap_quantum_never_freezes():
    case("cubic — lattice snapping moves at least one quantum when the curve asks to move")
    t = CubicTuner(initial=96, b_min=8, b_max=256, C=1.0, beta=0.7, epsilon=0.005,
                   snap_quantum=8, skip_first=False)
    t.update(cobs(0.30, rollout_id=0))              # W_max=96 -> B=67.2 -> snaps to 64
    check("congestion snapped to the lattice", t.current % 8, 0)
    prev = t.current
    moved = False
    for i in range(1, 6):
        nxt = t.update(cobs(0.001, rollout_id=i))
        check(f"r{i} on lattice", nxt % 8, 0)
        if nxt != prev:
            moved = True
        prev = nxt
    check("did not freeze on the plateau", moved, True)


def test_cubic_missing_signal_holds():
    case("cubic — missing/non-finite signal holds, never assumes zero")
    t = CubicTuner(initial=32, b_min=8, b_max=256, epsilon=0.005, skip_first=False)
    o = TunerObservation(rollout_id=1, threshold=32, idle_ratio=0.03,
                         training_span_gpu_s=100.0, busy_gpu_s=97.0, wall_s=500.0)
    check("interior None -> hold", t.update(o), 32)
    # gamma=1.0 HOLDS until the first congestion event -- it must not grow by `step`,
    # which would reintroduce the constant-step behaviour CUBIC exists to replace and
    # make it useless as a control arm.
    t1 = CubicTuner(initial=32, step=5, gamma=1.0, epsilon=0.005, skip_first=False)
    check("gamma=1.0 holds while healthy", t1.update(cobs(0.001, rollout_id=1)), 32)
    check("still holding", t1.update(cobs(0.001, rollout_id=2)), 32)
    check("congestion still works", t1.update(cobs(0.30, rollout_id=3)), 22)
    check("and the curve takes over after", t1.update(cobs(0.001, rollout_id=4)) > 22, True)


def test_cubic_construction_validation():
    case("cubic — constructor validation")
    for kw, why in [(dict(C=0), "C=0"), (dict(beta=1.0), "beta=1"), (dict(beta=0.0), "beta=0"),
                    (dict(gamma=0.5), "gamma<1"), (dict(epsilon=0), "epsilon=0"),
                    (dict(snap_quantum=0), "snap_quantum=0")]:
        try:
            CubicTuner(initial=32, **kw)
            check(f"rejects {why}", "no raise", "ValueError")
        except ValueError:
            check(f"rejects {why}", "ValueError", "ValueError")


def test_registry_matches_argparse_choices():
    """The registry and the CLI `choices=` tuple are declared in two files. A tuner in
    only one of them is either uninvokable or an argparse crash at launch."""
    case("registry <-> argparse choices (drift guard)")
    import re

    src = open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "slime", "utils", "arguments.py")).read()
    m = re.search(r'"--threshold-tuner".*?choices=\(([^)]*)\)', src, re.DOTALL)
    check("found choices tuple", bool(m), True)
    if m:
        choices = set(re.findall(r'"([^"]+)"', m.group(1)))
        check("choices == registry", sorted(choices), sorted(THRESHOLD_TUNER_REGISTRY))


def main():
    tests = [
        test_rails_clamp_and_step, test_rails_validate_construction, test_fixed_is_inert,
        test_calibrates_then_holds_at_baseline, test_low_ratio_climbs,
        test_rollout_zero_is_excluded_from_the_baseline,
        test_no_drift_on_real_stationary_streams, test_cliff_retreats_with_a_sane_baseline,
        test_baseline_calibrated_at_the_cliff_is_a_known_limitation,
        test_ewma_damps_a_single_outlier,
        test_degenerate_signal_holds, test_reset_clears_baseline,
        test_registry_and_factory, test_union_len,
        test_bangbang_direction, test_bangbang_never_holds,
        test_bangbang_skips_rollout_zero, test_bangbang_respects_rails,
        test_bangbang_oscillates_on_stationary_stream,
        test_bangbang_validates_target, test_bangbang_factory_forwards_target,
        test_interior_direction, test_interior_separates_the_measured_regimes,
        test_interior_ignores_trailing_noise,
        test_interior_missing_signal_holds_not_ramps,
        test_interior_skips_rollout_zero_and_respects_rails,
        test_interior_defaults_and_factory, test_bangbang_base_is_swappable,
        test_interior_decomposition_from_events,
        test_cubic_slow_start_then_congestion_sets_wmax,
        test_cubic_curve_recovers_then_plateaus_then_probes,
        test_cubic_fast_convergence_flag,
        test_cubic_rails_allow_big_moves_but_clamp,
        test_cubic_snap_quantum_never_freezes,
        test_cubic_missing_signal_holds,
        test_cubic_construction_validation,
        test_registry_matches_argparse_choices,
    ]
    # Guard: this list is maintained by hand, so a test added but not registered would
    # silently never run -- the same class of drift the registry<->choices guard covers.
    registered = {f.__name__ for f in tests}
    defined = {k for k in globals() if k.startswith("test_") and callable(globals()[k])}
    missing = sorted(defined - registered)
    if missing:
        FAILURES.append(f"tests defined but not registered in main(): {missing}")
        print(f"  FAIL  unregistered tests: {missing}")
    for fn in tests:
        fn()
    print("\n" + "=" * 74)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
