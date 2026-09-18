# `CubicTuner` — design plan

A TCP CUBIC-derived controller for **B** (`--migration-batch-threshold`), to be added to
`slime/router/threshold_tuner.py` as the fifth entry in `THRESHOLD_TUNER_REGISTRY`.

**Status: plan only. Nothing here is implemented.**

Every number below is tagged with where it comes from. Where a claim is an inference
rather than a measurement it says so. One figure quoted earlier in this line of work (a
"4.7x" group-vs-live ratio) turned out to be inflated by oversampling — the corrected
value is 3.0x — so the provenance tags are not decoration.

Primary data source, referred to throughout as **the 50-rollout run**:
`experiments/long_rl_training/deepseek_r1_8b/results_streaming_interior_tuner_50_step_dapo_8gpu/batch_thresh_agg_64_mc0/`
(DeepSeek-R1-Distill-Llama-8B, DAPO-math, 8 GPU, train TP 2 → 4 train groups, 8 engines at
infer TP 1, `rollout_batch_size` 128 × `n_samples_per_prompt` 8 = 1024 samples/rollout,
`global_batch_size` 1024, natural generation, `interior_idle` bang-bang tuner with
step 16 / rails [32,256] / eps 0.005, 33,089 s wall).

---

## 1. TCP CUBIC, stated precisely

CUBIC is specified in **RFC 8312** (Informational, Feb 2018) and **RFC 9438** (Aug 2023),
which obsoletes 8312. The core window function is unchanged between them:

```
W_cubic(t) = C · (t − K)³ + W_max
K          = cbrt( W_max · (1 − β) / C )
```

* `t` — time elapsed since the last congestion event (seconds, in TCP).
* `W_max` — the window in effect at that congestion event, i.e. the last known-bad point.
* `β` — the multiplicative decrease factor, **0.7**. On a loss, `cwnd ← β · cwnd`.
  Note this is the *retained* fraction: CUBIC keeps 70%, where Reno keeps 50%.
* `C` — the cubic scaling constant, **0.4**. Governs how aggressively the window grows
  once it is far from `W_max`.

`W(0) = C(−K)³ + W_max = −W_max(1−β) + W_max = β·W_max`, so the curve is continuous with
the multiplicative decrease by construction.

The shape is the entire point: **concave** (fast recovery) while `t < K`, **flat** near
`t = K` where `W ≈ W_max`, and **convex** (accelerating probe for a new ceiling) for
`t > K`. It spends most of its time near the last known-bad point.

Three parts people routinely omit when porting CUBIC:

1. **Slow start.** Before the first congestion event `W_max` is undefined, so CUBIC uses
   standard slow start (exponential growth) exactly as Reno does. The cubic function only
   takes over after the first loss establishes `W_max`.

2. **Fast convergence.** When a new congestion event occurs at a window *lower* than the
   previous `W_max` — evidence the available capacity has shrunk — CUBIC reduces `W_max`
   further before applying the cubic curve:

   ```
   if W_max < W_last_max:            # capacity appears to have dropped
       W_last_max = W_max
       W_max      = W_max · (1 + β) / 2
   else:
       W_last_max = W_max
   ```
   With β = 0.7 that multiplies `W_max` by 0.85. The purpose in TCP is to release
   bandwidth faster so a new flow can claim it.

3. **The Reno-friendly region** (RFC 9438 §4.3; "TCP-friendly region" in RFC 8312).
   CUBIC tracks the window a Reno flow would have achieved over the same period, `W_est`,
   and uses `max(W_cubic, W_est)`. This guarantees CUBIC is never *less* aggressive than
   Reno on short-RTT/low-BDP paths where the cubic curve would be slower.
   **We DROP this — see §5.1 for why, and why `C` replaces it.**

(Verified against RFC 9438 on 2026-09-18: fast convergence is §4.7, the Reno-friendly
region is §4.3, `beta_cubic = 0.7` is §4.6, `C = 0.4` is §5.1. Fast convergence is
genuinely part of the specification, not an embellishment.)

**Uncertainty flags.** I am confident of the window function, `K`, β = 0.7, C = 0.4, the
fast-convergence formula, and the existence and purpose of slow start and the
Reno-friendly region. I am **not** confident of the exact variable naming RFC 9438 adopted
(it reorganised the pseudocode and renamed several state variables relative to 8312), nor
of 9438's precise document status. Anyone implementing against the letter of the spec
should read RFC 9438 directly rather than trust this summary on those points. None of the
design decisions below depend on them.

---

## 2. Equivalence table: TCP → this setting

| TCP concept | Our analogue | Transfers? |
|---|---|---|
| `cwnd` (congestion window) | **B**, the migration batch threshold | **By analogy only, and the direction is inverted.** A larger `cwnd` means *more* data in flight; a larger B means migration fires *earlier* (while more work remains), i.e. more aggressive migration. Both are "how hard am I pushing", and both have a cliff above which the system degrades — that is the whole basis of the port. But B is not a quantity of in-flight work; it is a *threshold on* in-flight work. |
| one segment (MSS) | one prompt group (`n_samples_per_prompt` samples) | **Under `--migration-count-unit groups`, cleanly** — B is quantized to multiples of q, exactly as `cwnd` is counted in MSS. **Under `samples`, not at all** — B becomes a continuous integer. See §7. |
| packet loss / ECN mark | a **starvation event** — training GPUs polling an empty work queue because migration moved work away too eagerly | **By analogy.** Both are binary, both signal "you pushed too hard". But a TCP loss is a discrete, unambiguous event; starvation is a *thresholded continuous measurement* (§3), so its labelling depends on a chosen epsilon. |
| RTT (the control interval) | **one rollout** | **By analogy, with a severe scale disanalogy.** TCP sees thousands of RTTs between congestion events; the 50-rollout run has **1-10 rollouts** between starvations. Our controller gets ~2-3 orders of magnitude fewer decisions to converge with. |
| `ssthresh` | the remembered safe B | **Cleanly.** This is the single most valuable part of the port — see §5. |
| bandwidth-delay product | the true safe B | **By analogy, and worse behaved.** A path's BDP is approximately stationary over a connection. Our safe B is **stochastic** (the same B is both healthy and starved within one era) **and drifts** (halves over 50 rollouts). See §5. |
| competing flows / fairness | *nothing* | **Does not transfer.** There is one controller and one workload. Fast convergence's stated purpose (release bandwidth for a new flow) is irrelevant; we keep the mechanism for an entirely different reason (§5). The Reno-friendly region is likewise moot and should be **omitted**. |
| cost of one loss | cost of one starved rollout | **Does not transfer.** A TCP loss costs roughly one RTT of retransmission. A starved rollout costs **+5% to +17%** of a ~550-800 s rollout (era-controlled, 50-rollout run: era A +11%, era B +5%, era C +17%; +11% pooled). Relative to the control interval our "loss" is far cheaper than a TCP loss is relative to an RTT — which argues for CUBIC's gentle β = 0.7 over Reno's 0.5, and against panic-grade backoff. |
| `cwnd` lower bound | `b_min`, never 0 | **Cleanly, and already enforced.** `ThresholdTuner._clamp` refuses to return 0 because B = 0 disables migration entirely, which must be an explicit `--migration-policy none`. |

---

## 3. The congestion signal

**Binary event**: `starved(obs) := signal(obs) > epsilon`, where `signal` is a key into the
existing `SIGNAL_REGISTRY` (`slime/router/threshold_tuner.py:338`). Reusing the registry
keeps the signal swappable and costs three class attributes, exactly as `BangBangTuner`
already documents at `threshold_tuner.py:~355-375`.

A missing signal (`None`) must mean **HOLD**, never "not starved". The existing code makes
this point emphatically (`TunerObservation`, `threshold_tuner.py:~95-105`): defaulting to
0.0 reads as "zero starvation" and ramps B upward every rollout — a failure that already
cost a live eval.

### Which ratio

The user asked for "idle_training_time_ratio". That quantity exists in two forms on
`TunerObservation`, and the choice matters more than the control law:

* `idle_ratio` — the combined signal, `1 − busy/span` over the training phase.
* `interior_idle_ratio` — the **interior decomposition** of the same quantity: the
  between-chunk gaps where a training GPU polled an empty work queue, excluding the
  trailing wait after a group's last chunk for the global `training_done_time` barrier.

**Recommendation: `interior_idle_ratio`, default epsilon 0.005.** Stated plainly, because
this is a substitution on the user's request: this *is* the idle-training-time ratio, but
only the component that migration aggressiveness actually causes. The reason is recorded
in `InteriorIdleTuner`'s docstring and in commit `84ff2041`: at a healthy B the combined
`idle_ratio` is **~98% trailing**, and trailing is set by which train group grabbed the
final chunk and how large it was — governed by the GRAB policy, not by B. Thresholding the
combined signal would mostly threshold a quantity B does not control.

**Separation, measured on the 50-rollout run** (not the older committed traces the
docstring cites), excluding rollout 0 as startup-inflated:

| | n | range |
|---|---|---|
| healthy (r > 0) | 23 | 0.00037 … **0.00086** |
| starved | 26 | **0.00536** … 0.59993 |
| rollout 0 (startup) | 1 | 0.00258 |

The healthy ceiling is 0.00086 and the starved floor is 0.00536 — a **6.2x** gap with
nothing in between. `eps = 0.005` sits 5.8x above the healthy ceiling.

**This is a weaker separation than the `InteriorIdleTuner` docstring claims** (it cites
healthy < 0.0008 vs starvation ≥ 0.0145, a ~25x gap, from earlier traces). On this run
exactly one of 26 starvation events (r47, 0.00536) lands within 2x of epsilon; drop that
single marginal event and the gap is 0.00086 → 0.03191 = 37x. So the signal is still
strongly bimodal, but the epsilon is not as comfortably centred as previously documented,
and r47 is effectively a coin-flip label. **Any implementation should treat the labelling
of near-epsilon events as uncertain**, which is an argument for the severity-weighted
variant in §12.

Rollout 0 at 0.00258 sits between the healthy band and epsilon. It is labelled healthy at
eps = 0.005 but would flip at eps = 0.002. Keep `skip_first` (default 1) available.

---

## 4. Why CUBIC rather than bang-bang or AIMD

### The bang-bang arm's measured failure

On the 50-rollout run the `interior_idle` bang-bang tuner:

* **moved B on all 50 rollouts** — verified, `b_before != b_effective` for every row. It
  has no hold state by construction, so it cannot converge; it can only oscillate.
* **starved in 26 of 50 rollouts (52%)**.

The mechanism is simple: bang-bang steps toward the cliff until it falls off, steps back,
and repeats. Half the run is spent paying the starvation penalty.

### The boundary is stochastic

Within *every* era, the same B appears both healthy and starved (50-rollout run,
eps = 0.005, `b_before` values):

| era | rollouts | healthy B | starved B |
|---|---|---|---|
| A | r0-r17 | 64, 80, 96, 112, 128 | 80, 96, 112, 128, 144 |
| B | r18-r35 | 32, 48, 64, 80 | 48, 64, 80, 96 |
| C | r36-r49 | 32, 48, 64 | 48, 64, 80 |

Every era has a 3-4 value overlap. There is no B that is *safe*; there is only a B with an
acceptable starvation probability. No controller can find a fixed point that does not
exist — the honest goal is to manage exposure near a probabilistic edge.

### The boundary drifts DOWN

The same table read vertically: the healthy band falls from 64-128 to 32-64 across the
run, roughly halving. Responses lengthen as reward improves, so the safe B shrinks. A
controller that converges is *wrong* by construction; it must track.

### What CUBIC gives that AIMD does not

1. **The plateau parks where information is cheapest.** Most of the trajectory sits within
   a few percent of `W_max` — the last known-bad point. That is precisely where an extra
   observation is most informative about a stochastic boundary, and where the expected
   cost of being wrong is smallest (you are just below the edge, not deep into starvation).
   AIMD's linear ramp spends its time uniformly across the range instead.

2. **Fast convergence is a falling-ceiling tracker.** When the new congestion point is
   below the previous one, CUBIC pulls `W_max` down by an extra factor of `(1+β)/2 = 0.85`.
   Its *stated* purpose in TCP (releasing bandwidth for competing flows) is irrelevant
   here — but the *mechanism* is exactly right for a boundary that drifts downward, which
   is our measured drift direction. AIMD has nothing that anticipates drift; it re-learns
   from scratch after every event.

3. **β = 0.7 matches the measured cost asymmetry.** Overshoot costs +5..17%, not the
   +30..100% a casual reading of the worst rollouts suggests. Reno's β = 0.5 would give
   back 2-3 extra rollouts of climbing for no benefit.

**Honest counterweight.** CUBIC was designed for thousands of RTTs between losses; we have
1-10 rollouts. The curve is sampled at only a handful of integer `t` values per epoch, so
much of its careful shape is invisible (§6, §7). What survives at our sampling rate is:
fast concave recovery, a multi-rollout plateau at `W_max`, and slow convex probing above
it. That is still the right shape — but this is a *port of the shape*, not a claim that
CUBIC's asymptotic fairness or scalability properties carry over. They do not, and they
are not what we want.

---

## 5. The clock: `t` in rollouts

**`t` = rollouts since the last starvation event**, not wall-seconds.

Justification: the thing that moves the boundary is *policy learning* — responses lengthen
as reward improves. Learning advances one optimizer step per rollout in this config
(`global_batch_size` 1024 = samples per rollout 1024, so exactly one step). The drift clock
*is* the step clock. Wall-seconds would conflate it with rollout duration, which itself
varied 485-1094 s on the 50-rollout run — a 2.3x range — so a seconds-based `t` would
advance the curve faster during precisely the slow rollouts that starvation causes.

### 5.1 Dropping the Reno-friendly region, and why `C` replaces it

The Reno-friendly region bundles two purposes. Separating them is what settles the question:

1. **Inter-flow fairness** — never be less aggressive than a competing Reno flow. There is
   no competing flow. One controller, one workload. This transfers to nothing.
2. **A growth floor** when the cubic curve is slow — which it is, near the plateau, where
   we intend to live.

Purpose 2 is the only one worth preserving, and **the convex region already provides it**.
For `t > K` the curve accelerates cubically and escapes on its own; it does not need a
linear floor underneath it. Adding one would mean a second mechanism controlling something
`C` already controls, while partially undoing the plateau — and the plateau is the entire
reason for choosing CUBIC over bang-bang (§4). So: **omit the Reno-friendly region, and
treat `C` as the plateau-dwell knob.**

### Re-deriving C for a rollout clock AND for our run lengths

`C = 0.4` is tuned for `t` in seconds with `cwnd` in segments. It survives the change of
clock numerically — both TCP's `cwnd` (tens to hundreds of segments) and our B are
similar in magnitude, and both want `K` of a few time units — but it does **not** survive
the change in *horizon*. TCP has thousands of RTTs between congestion events; a 10-rollout
benchmark gives ~9 decisions total.

The quantity that matters is **how many healthy rollouts it takes to escape a `W_max` that
was set too low**. At our operating point (samples unit, `W_max = 30`, the equivalent of
groups B = 96):

```
C=0.4  K=2.8   21.0  27.6  29.8  30.0  30.7  34.1  42.8  59.1   -> >W_max+10% at t=5
C=1.0  K=2.1   21.0  28.7  30.0  30.8  37.1  54.9  90.2 149.1   -> >W_max+10% at t=4
C=1.6  K=1.8   21.0  29.2  30.0  32.9  47.5  83.5 150.4 257.8   -> >W_max+10% at t=4
```

And if `W_max = 30` was set by a false congestion event when the true safe B was 45:
**C = 0.4 reaches 45 at t = 7; C = 1.6 reaches it at t = 4.** On a 10-rollout arm, seven
rollouts is the whole run.

**Recommendation: `C = 1.0` as the default, not the RFC's 0.4.** It preserves a 2-3
rollout plateau (the caution that motivates CUBIC) while halving the escape time. `C = 0.4`
is right for a >= 50-rollout run and wrong for a 10-rollout one, so `C` should be chosen
with the horizon in mind and the docstring must say so. This supersedes the trajectories
below, which were computed at C = 0.4 and are retained because they show the shape:

```
W_max=144  K=4.8   101 123 136 142 144 144 145 148 158 174 201 241
W_max= 80  K=3.9    56  70  77  80  80  81  84  92 107 133 170 222
W_max= 48  K=3.3    34  43  47  48  48  50  56  68  89 122 168 230
```

This is the desired shape at our sampling rate: ~3-5 rollouts of concave recovery, ~3
rollouts of plateau within 1% of `W_max`, then slow convex probing. Compare `C = 1.6`
(`W_max=144`): `101 131 142 144 146 157 187 246 …` — the plateau is one rollout wide and it
is 70% above `W_max` by t=7. `C ≥ 1.6` is too aggressive for a 10-minute control interval.

**`K ∝ W_max^(1/3)` is a feature, not an accident.** Smaller ceilings recover in fewer
rollouts (K = 4.8 at W_max 144, 3.3 at 48) with no extra tuning. Since the boundary drifts
*down* over a run, the controller automatically becomes more responsive exactly as the
safe B shrinks.

**Parameterisation choice.** Keep `C` as the CLI knob (faithful to the spec, and the
`W_max^(1/3)` scaling is desirable) rather than exposing `K` directly. Report the implied
`K` in `explain()` so the decision log is readable — "K=4.8 rollouts to regain W_max=144"
is the interpretable quantity, and an operator should not have to compute a cube root to
understand what the controller intends.

---

## 6. B₀, slow start, and the lattice

### B₀ = 1, as a constructor parameter

`b_init: int = 1`, exposed on `__init__` and via CLI, per explicit request. Slow start runs
from there: `B ← min(ssthresh, γ·B)` on each healthy rollout, γ default 2.

**The dead zone must be documented, not discovered.** Under `--migration-count-unit groups`
(the default), `cumulative_batch` sums whole prompt groups, so it only takes values
`{0, q, 2q, …}` where `q = n_samples_per_prompt` (verified: all 6400 group dispatches in
the 50-rollout run logged "8 samples"). The trigger fires when `cumulative_batch < B`.
Therefore:

* **B ≤ q can only fire when `cumulative_batch == 0`** — the train group has nothing in
  flight, so the migration loop iterates an empty list and issues **zero** decisions.
* Worse, it is not a harmless no-op. `_triggered_groups.add(src_group)`
  (`migration_policy.py:438`) happens *before* the destination scan, and the base
  `_should_latch` (`migration_policy.py:587`) returns `True` unconditionally. So the firing
  **burns the one-shot latch** and excludes that train group from migration for the rest of
  the rollout.

With q = 8 and γ = 2, `B₀ = 1` gives `1 → 2 → 4 → 8 → 16 → …`: **four rollouts in the dead
zone**, three of which actively damage the rollout by burning latches. `b_init = 1` is
therefore a knob whose default should be overridden in any real config; recommend configs
set `b_init = 2q` (16 here), derived from `n_samples_per_prompt`, which is the smallest
value that can migrate anything.

### The lattice

Equivalence classes are `B ∈ [kq+1, (k+1)q]` — all fire at `in-flight ≤ kq`. So B = 97,
101 and 104 are literally the same policy; 96 and 97 are different. The cubic trajectory
must **snap to the class representative** `(k+1)q`, both so the decision log is honest and
so `explain()` does not claim a precision the policy lacks.

Snapping the W_max=144 trajectory at q=8:

```
raw      101 123 136 142 144 144 145 148 158 174
snapped  104 128 136 144 144 152 152 152 160 176
```

**The lattice costs CUBIC most of its fine structure, and the cost is worst where it hurts
most.** The gentle probing above `W_max` (144 → 145 → 148) all lands in one class and
becomes a single 8-unit step. At W_max = 144 that step is 5.6% of B — tolerable. At
W_max = 48 (era C's regime) it is 17% of B, so the controller degenerates into a coarse
stepper in exactly the low-B regime the run drifts into. This is a real limitation of
CUBIC under the `groups` unit and should not be papered over.

---

## 7. Interaction with `--migration-count-unit` — and why it dominates the sequencing

Commit `932a5f7b` ("migration: --migration-count-unit, B measured in live samples not whole
groups") adds `--migration-count-unit {groups,samples}`, default `groups`. Two consequences
for this tuner, one expected and one not:

**Expected — the scale shifts ~3x.** Measured over 135 independent trigger firings on the
50-rollout run: median group-implied 56 vs 17 samples actually generating, a 3.0x median
ratio (3.2x mean). So `W_max`, `b_min`/`b_max`, `b_init`, and `step` calibrated under
`groups` do **not** carry to `samples`; divide by ~3. The epsilon does not move (it is a
ratio of training-phase GPU-time, independent of B's unit).

Note the divisor is not constant — the bias runs ~3.3x at low B and ~2.2x at high B, since
groups are fuller earlier in the drain. A single division is an approximation good enough
to seed a controller that then adapts, and not good enough to port a *tuned* value.

**Unexpected, and more important — the lattice disappears under `samples`.** Under that
unit `_cumulative_batch` returns `sum(ctx.live_samples[e] for e in siblings)`
(`migration_policy.py`, `_cumulative_batch`), a raw count of not-yet-done asyncio tasks.
That is any integer in `[0, 256]`, not a multiple of q. **The quantization in §6 is a
`groups`-unit artifact.**

This inverts the priority between the two changes. Under `groups`, CUBIC's plateau and its
convex probing region are quantized to ~6-17% steps and most of the curve's shape is
unrepresentable. Under `samples`, B is continuous and the cubic trajectory means what it
says.

**Recommendation: calibrate and evaluate `CubicTuner` against `--migration-count-unit
samples`, and land the unit change first.** Building CUBIC on the `groups` unit would tune
a fine-grained controller against a scale that cannot express it, and would then require a
full recalibration when the unit flips. The dead zone in §6 also vanishes under `samples`
(B = 1 becomes meaningful, firing when ≤ 0 samples are live — still degenerate, but no
longer latch-burning at B = 2..8), which incidentally makes `b_init = 1` far less harmful.

---

## 8. The rails

`ThresholdTuner.update()` (`slime/router/threshold_tuner.py`, `update`) currently caps the
per-rollout change symmetrically:

```python
delta = int(proposed) - self.current
if abs(delta) > self.step:
    proposed = self.current + int(math.copysign(self.step, delta))
```

This blocks **both** halves of CUBIC: the concave recovery moves B by 22 in one rollout
(101 → 123 at W_max 144), and the multiplicative decrease moves it by 43 (144 → 101). With
`step = 16` neither is expressible, and fighting the rail inside `_propose()` would be
worse — the rails exist precisely so an experimental subclass cannot wedge a multi-hour run.

**Proposed change — split the cap, defaulting both to `step`:**

```python
def __init__(self, initial, b_min=8, b_max=256, step=16,
             max_up_step=None, max_down_step=None):
    ...
    self.step = step
    # Default to `step` in BOTH directions, so fixed / idle_ratio / idle_threshold /
    # interior_idle are byte-identical. Only a subclass that asks gets asymmetry.
    self.max_up_step = step if max_up_step is None else max_up_step
    self.max_down_step = step if max_down_step is None else max_down_step

def update(self, obs):
    proposed = self._propose(obs)
    if proposed is None:
        proposed = self.current
    delta = int(proposed) - self.current
    cap = self.max_up_step if delta > 0 else self.max_down_step
    if abs(delta) > cap:
        proposed = self.current + int(math.copysign(cap, delta))
    self.current = self._clamp(proposed)
    return self.current
```

`CubicTuner` sets both to `b_max` (i.e. unrestricted), because the cubic function and β
*are* its rate limits; `_clamp` still bounds the range. Add validation (`max_up_step >= 1`,
`max_down_step >= 1`) alongside the existing `step >= 1` check, and a test asserting the
four existing tuners produce identical trajectories before and after the change.

---

## 9. Algorithm

State: `W_max` (None until the first starvation), `W_last_max`, `t` (rollouts since the
last starvation), `ssthresh`, `in_slow_start`.

```
on observation obs:
    if skip_first and obs.rollout_id == 0:      return None      # startup-inflated
    r = SIGNAL_REGISTRY[signal](obs)
    if r is None or not isfinite(r):            return None      # HOLD, never assume 0

    if r > epsilon:                                    # ---- congestion event ----
        W = self.current
        if W_max is not None and W < W_max and fast_convergence:
            W_last_max, W_max = W, W * (1 + beta) / 2          # falling ceiling
        else:
            W_last_max, W_max = W, W
        ssthresh      = max(b_min, beta * W_max)
        in_slow_start = False                          # never re-enter (see below)
        t             = 0
        return snap(max(b_min, beta * W_max))

    t += 1                                             # ---- healthy ----
    if in_slow_start:
        return snap(min(ssthresh, gamma * self.current))
    K = cbrt(W_max * (1 - beta) / C)
    return snap(C * (t - K)**3 + W_max)
```

**Slow start is entered once and never re-entered.** TCP Tahoe restarts slow start on
timeout because capacity may have grown. Our measured drift is *downward* (§4), so
re-probing exponentially after a starvation would climb into a boundary that just moved
down. This is the Reno/CUBIC fast-recovery behaviour, and the deviation from Tahoe is
deliberate and data-driven.

**`snap()` is the lattice projection** from §6 under `groups`, and the identity under
`samples`. It should read `q` from `n_samples_per_prompt` at construction — derived, not a
new hyperparameter.

Parameters and defaults: `beta=0.7`, `C=0.4`, `gamma=2.0`, `b_init=1`, `epsilon=0.005`,
`signal="interior_idle_ratio"`, `fast_convergence=True`. **`gamma=1.0` degenerates to pure
additive growth during slow start**, which is the control arm for "does slow start earn
anything"; `fast_convergence=False` is the control arm for the drift-tracking claim.

---

## 10. Integration points

1. **`THRESHOLD_TUNER_REGISTRY`** (`threshold_tuner.py`, near the bottom): add
   `"cubic": CubicTuner`.
2. **`slime/utils/arguments.py:297`** — extend the `choices=` tuple to
   `("fixed", "idle_ratio", "idle_threshold", "interior_idle", "cubic")`. A drift guard at
   `tests/test_threshold_tuner.py:598` (`test_registry_matches_argparse_choices`) parses
   this tuple out of the source and asserts it equals the registry keys, so the two files
   cannot disagree. It will fail until both are updated.
3. **`make_threshold_tuner()`** — `CubicTuner` is not a `BangBangTuner` subclass, so it
   needs its own `elif issubclass(cls, CubicTuner)` branch plumbing `beta`, `C`, `gamma`,
   `b_init`, `epsilon`, `signal`, `fast_convergence`, `max_up_step`, `max_down_step`.
   **The existing fallback defaults must match the class defaults exactly** — the file
   already warns that a mismatch silently gives hand-built args (tests, sweeps) different
   control behaviour than the CLI.
4. **New CLI flags**, in the repo's verbose style: `--tuner-cubic-beta`,
   `--tuner-cubic-c`, `--tuner-cubic-gamma`, `--tuner-b-init`, `--tuner-cubic-epsilon`,
   `--tuner-cubic-fast-convergence`. Per the `BangBangTuner` docstring's warning, give
   CUBIC its **own** epsilon flag rather than reusing `--tuner-interior-target`: sharing a
   flag across tuners whose sane values differ is how one live eval ran at target 0.30,
   ~10x above any real reading, and degenerated into a monotone ramp that proved nothing.
5. **`perf_analysis/replay_threshold_tuner.py`** — add a `--cubic-*` argument group
   mirroring the `--target` / `--skip-first` handling, so a replay runs the same control
   config as the live run it is compared against (the file notes this already caused one
   spurious trajectory discrepancy).
6. **Decision logging** — `train_streaming.py` writes a `tuner_decision` row per rollout.
   Add `w_max`, `ssthresh`, `t_since_event`, `K`, `in_slow_start`, and `starved` so the
   trajectory is reconstructible offline. These are additive JSONL fields; existing
   analysis keys are untouched.

---

## 11. Offline validation

### Why the existing replay is not enough

`perf_analysis/replay_threshold_tuner.py` feeds recorded `TunerObservation`s to any tuner,
and its own docstring states the limit: the observations were generated under **one** B
trajectory, so replaying a different one is counterfactual — a rollout that would have
happened at B=96 is scored with data recorded at B=64. It can reject a broken control law
(oscillation, drift on a stationary stream, rail violations) but cannot say the resulting B
is fast.

### Proposed stochastic-boundary simulator

The 50-rollout run yields 50 labelled `(rollout_id, B, starved?)` rows from
`.../batch_thresh_agg_64_mc0/train_metrics/*.jsonl` (`phase == "tuner_decision"`, using
`b_before` as the B in effect and `interior_idle_ratio > 0.005` as the label). From those:

1. **Fit `P(starve | B, era)`** — e.g. logistic in B with an era-varying intercept, or a
   monotone fit with a drifting midpoint. The three eras in §4 give the drift; the
   healthy/starved overlap within each era gives the slope.
2. **Monte-Carlo each candidate controller** over a synthetic 50-rollout run, sampling
   starvation from the fitted probability at whatever B the controller chose.
3. **Score in expected wall-seconds**, applying the measured era-controlled penalty
   (+11% era A, +5% era B, +17% era C; +11% pooled) to starved rollouts.
4. **Sweep** `(C, β, γ, fast_convergence on/off, epsilon)` and compare against the
   bang-bang baseline and a fixed-B arm.

**What this can prove**: that a control law does not oscillate, does not ratchet, retreats
from a cliff, tracks a downward-drifting boundary, and that one parameterisation dominates
another *under the fitted model*. That is enough to reject bad `(C, β)` choices for seconds
of compute instead of 9 GPU-hours each.

**What it cannot prove**: anything the model does not contain. It inherits the fitted
boundary's functional form; it cannot see inference-side migration re-prefill cost (which
commit `84ff2041` notes no train-phase idle signal can observe); it assumes the wall
penalty is independent of *how far* past the boundary B went, which §12 shows is false; and
it is fit to 50 points from one run on one model and one workload. It is a filter, not a
verdict. Survivors still need a real paired run.

---

## 12. Failure modes and open questions

**1. `W_max` from a single noisy event, versus fast convergence — these pull opposite ways
and the algorithm cannot tell them apart.** CUBIC sets `W_max` from the window at *one*
congestion event. Our boundary is stochastic, so that event may be noise. The clearest case
is **r45: starved at B = 80 with interior = 0.59993 and wall 1093.6 s** — in era C, where
B = 64 was healthy and where B = 80 had been healthy at r44 one rollout earlier. Faithful
CUBIC slams `W_max` to 80; fast convergence then pulls it to 68, and the controller spends
several rollouts recovering from what may have been a fluke.

Fast convergence makes this strictly worse — it *deliberately* over-reacts to a lower
`W_max`. That is correct for genuine drift (our measured case) and wrong for a noise event,
and nothing in the algorithm distinguishes them.

**UPDATE 2026-09-18 — a large share of these "noise events" are a BUG, not stochasticity.**
See §13.3. `collect_observation()` charges a training GPU's *entire* span to `interior`
whenever it produced no `chunk_*`/`ws_*` spans. That condition is not "polled an empty queue
the whole time"; it is "this train group flipped in after the last chunk had already been
grabbed", which costs ~0 wall-clock. Two of eight GPUs hitting it moved interior from 0.0027
to 0.083 — 16x over epsilon. **CubicTuner reads the same signal through the same function and
would set `W_max` from these false events, with `fast_convergence=True` amplifying the
error.** Fixing the zero-chunk rule is a prerequisite, not an optional cleanup. Candidate mitigations, none free:
EWMA over recent congestion points; require two consecutive events before moving `W_max`
(risky given real drift); or severity-weighting (below). **Recommend shipping faithful
CUBIC with `fast_convergence` as a flag**, so the two arms are separable and the result is
attributable, rather than pre-emptively adding a heuristic we cannot yet justify.

**2. Should severity modulate β?** CUBIC treats every loss identically, but our signal is
graded and severity tracks cost. Measured on the 50-rollout run: wall regresses **+31 s per
e-fold of interior idle** among starved rollouts, and severe starvation (≥ 0.05) is more
expensive than mild (eps..0.05) in two of three eras — era A 632 s severe vs 585 s mild,
era B 688 vs 642 — but **era C inverts** (815 severe vs 821 mild, on n=5 and n=2), so the
trend is directionally right and not universal. A
severity-scaled β (mild → 0.85, severe → 0.5) would use information faithful CUBIC discards,
and would also soften failure mode 1 by making marginal events like r47 (0.00536) barely
move `W_max`. **Deliberately out of scope for v1** — keep the port faithful so results are
attributable, then test severity-weighting as a second arm.

**3. A workload whose boundary drifts UP.** Everything above is designed against a falling
ceiling. If responses *shorten* (a workload where reward improves via brevity, or a KV
regime change), then: never re-entering slow start (§9) leaves the controller climbing only
via the convex region, and fast convergence actively suppresses `W_max` growth. Convex
probing does eventually find a higher ceiling — `W_max=144, t=10 → 201` — so the controller
is not stuck, just slow. **Unmeasured and untested**; a bounded re-probe after K consecutive
healthy rollouts pinned at `W_max` is the obvious mitigation, and should be **off by
default** since on the one long run we have it would only hurt.

**4. The evidence base is one run.** Every number here comes from a single 50-rollout DAPO
run on one model (DeepSeek-R1-Distill-Llama-8B), one grab policy
(`graduated_tail_split`), one migration policy
(`train_group_batch_threshold_aggressive`), and one topology (8 GPU, train TP 2). The era
structure, the drift direction, the +5..17% penalty, and the 6.2x signal separation are all
properties of *that run*. Text2SQL is known to behave differently — commit `84ff2041`
records that its dominant cost at higher B is inference-side migration re-prefill, which no
train-phase idle signal can see at all, so `CubicTuner` may be blind to the thing that
actually matters there.

**5. Few decisions per run.** A 10-rollout benchmark gives ~9 decisions, and slow start
plus one recovery epoch consumes most of them. CUBIC's shape only becomes visible over
≥ 30 rollouts. **`CubicTuner` should be evaluated on long runs (≥ 50 rollouts) only**; on a
10-rollout arm it will mostly be measuring its own slow start.

**6. The epsilon is not as safely centred as documented.** §3: one of 26 starvation events
sits within 2x of eps = 0.005, and rollout 0's 0.00258 would flip label at eps = 0.002. The
25x gap in the `InteriorIdleTuner` docstring is from earlier traces, not this run, where it
is 6.2x. Worth re-deriving epsilon under the `samples` unit before trusting it.

---

## 13. Results of the 2026-09-17 controlled experiments

All 8 GPU, DeepSeek-R1-Distill-Llama-8B, DAPO-math, rb 128 x nspp 8 = 1024 samples = global
batch, TP2 train / TP1 infer, natural generation, same box, same session. Wall figures are
TRUE per-rollout totals (see `perf_analysis/arm_totals.sh`: `train.py:239` prints
generation only, `train_streaming.py:703` prints the whole rollout — conflating them
overstates colocate's speed by ~40%).

### 13.1 The count unit is throughput-neutral at fixed B

10 rollouts, no tuner, matched firing point (groups B=96 fires at ~28 live; samples B=32
fires at 31 live):

| arm | total | mean | migrations | mean resp len |
|---|---|---|---|---|
| fixed groups96 | 5606 s | 561 s | 312 | 7419 |
| fixed samples32 | 5619 s | 562 s | 306 | 7380 |

**+0.23%**, paired per-rollout deltas scattering both directions (mean +0.26%, stdev 2.27,
t ~ 0.36). The units are interchangeable. This validates §7's prerequisite: adopting the
`samples` unit to give CUBIC a continuous B costs nothing in throughput.

### 13.2 Fixed B beats the bang-bang tuner — CubicTuner's real target

Per-rollout, vs the same-session colocate baseline (663 s/step):

| arm | s/step | vs colocate |
|---|---|---|
| colocate | 663 | -- |
| tuner_groups (bang-bang, B0=64) | 583 | -12.1% |
| tuner_samples (bang-bang, B0=18) | 651 | -1.8% |
| **fixed B=96 groups** | **571** | **-13.9%** |
| **fixed B=32 samples** | **572** | **-13.8%** |

**`CubicTuner` must beat 571 s/step, not merely beat bang-bang.** The existing tuner is
*costing* ~2 points against a well-chosen constant.

There is also evidence the optimum sits ABOVE B=96: at r4 `tuner_groups` was at B=128 and
hit -15.0%, where fixed B=96 got -11.7%. A controller that can reliably HOLD a high B
should beat fixed B — that, and not "beat bang-bang", is the case for building this.

### 13.3 The bang-bang collapse was driven by a signal bug, not by B dynamics

`tuner_samples`, 5 rollouts. The zero-chunk override (§12.1) fired on exactly the two
rollouts where the tuner read starvation, and nowhere else:

| rollout | B | zero-chunk GPUs | interior | action | wall |
|---|---|---|---|---|---|
| r0 | 18 | 0 | 0.00269 | up -> 28 | 622 |
| r1 | 28 | **2** | **0.02977** | down -> 18 | 629 |
| r2 | 18 | **2** | **0.08328** | down -> 11 (floor) | 690 |
| r3 | 11 | 0 | 0.00075 | up -> 21 | 663 |
| r4 | 21 | 0 | 0.00062 | up -> 31 | 652 |

`tuner_groups` had 0 zero-chunk events in 5 rollouts; `fixed samples32` had 0 in 10. At r1
the tuner was sitting at B=28 — essentially the optimum (fixed B=32 gives 572 s/step) — and
a false alarm knocked it to the floor. By r3 it was 6.6% SLOWER than colocate: migration had
effectively been switched off.

The bang-bang law has no memory and cannot recover from this. CUBIC would recover, but only
after the plateau/convex escape (§5.1) — 4-7 rollouts depending on `C` — and would first
have anchored `W_max` to the false value.

### 13.4 Revised sequencing

1. **Fix the zero-chunk rule in `collect_observation()`.** Prerequisite. It corrupts the
   only signal either tuner has, and it affects the `groups` path too (it fired once in
   `fixed groups96`). Any experiment using `--threshold-tuner interior_idle` is exposed.
2. **Re-derive epsilon** under the corrected signal (§12.6 already flags the 6.2x
   separation, weaker than the 25x in the `InteriorIdleTuner` docstring).
3. **Then** implement `CubicTuner`, with `C = 1.0` (§5.1), no Reno-friendly region, and
   `fast_convergence` defaulted OFF until the signal is trustworthy — it amplifies exactly
   the failure mode that killed the bang-bang arm.
4. Evaluate at >= 30 rollouts (§12.5), against the 571 s/step fixed-B target.
