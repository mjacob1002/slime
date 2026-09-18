# B-tuning investigation — findings, 2026-09-16 to 2026-09-18

Compaction-proof record. Every number here was measured in-session; provenance is given
for each. Raw transcript: `~/.claude/projects/-m-coriander-coriander-mjacob2-slime/ef740dd7-6456-441d-ab8f-954073723be2.jsonl`
(searchable with `/searchchat` and `/readchat`).

**Convention warning.** Speedup = `colocate_time / streaming_time`. Time reduction =
`streaming/colocate - 1`. They are NOT interchangeable and this session initially reported
the latter while calling it the former. A "-13.9% time reduction" is a "1.161x speedup".
The repo's memory notes use the reduction form.

---

## 1. Headline results (all 8 GPU, DeepSeek-R1-Distill-8B, DAPO-math, rb128 x nspp8 = 1024
samples = global batch, TP2 train / TP1 infer, natural generation, groups unit)

| arm | rollouts | speedup vs colocate | s/step | source |
|---|---|---|---|---|
| **fixed B=96, no tuner** | 5 | **1.161x** | 571 | `results_fixedB_probe_3roll`, `results_fixedB_10roll` |
| bang-bang interior_idle tuner | 5 | 1.138x | 583 | `results_count_unit_controlled_5roll/tuner_groups` |
| CubicTuner B0=144 | 5 | 1.131x | 586 | `results_cubic_5roll_8gpu` |
| CubicTuner B0=32, gamma=1.5 | 20 | 1.120x (1.131x excl. r0) | 603 | `results_cubic_20roll_smallB0` |
| historical bang-bang 10-roll | 10 | **1.184x** | — | `results_streaming_interior_tuner_10_step_dapo_8gpu` |
| historical bang-bang 50-roll | 50 | 1.152x | — | `results_streaming_interior_tuner_50_step_dapo_8gpu` |

**A well-chosen constant B beats every tuner we have.** The 1.184x historical figure is a
10-rollout window; the same run's advantage falls to 1.152x over 50 rollouts because
response lengths grow, so window length matters when comparing.

---

## 2. THE central finding: interior_idle is the wrong control signal

From the 20-rollout CUBIC run (n=20, `results_cubic_20roll_smallB0`):

| relationship | Pearson | Spearman |
|---|---|---|
| **B vs speedup** | **+0.641** | +0.592 |
| migrations vs speedup | +0.679 | +0.690 |
| B vs migrations | +0.991 | — |
| **interior idle vs speedup** | **+0.204** | +0.194 |
| starved(0/1) vs speedup | **+0.367** | — |

- B's correlation survives both confounds: partial corr controlling for rollout index
  **+0.649**, controlling for mean response length **+0.679**.
- **Partial corr(interior, speedup | B) = +0.167** — interior carries essentially no
  information beyond B, and the sign is POSITIVE (higher idle -> faster).
- Starved rollouts averaged **1.178x** vs healthy **1.106x**.
- Within the B=96-112 band (B nearly constant): **starved 1.214x (n=4) vs healthy
  1.139x (n=7)** — at the same B, alarm-tripping rollouts were 6.6% FASTER.

**Mechanism.** Starvation occurs when B is high; high B is where the speedup is. The cost
of a starvation is not paid in that rollout — it is paid in the 2-3 rollouts AFTER, once
the tuner cuts B: r16 1.175x -> r17 1.032x -> r18 1.081x.

**Implication:** do not tune the controller; change what it optimizes. Either drive B from
measured wall-clock, or demote interior_idle to a hard safety rail (fire at ~0.05, where
genuine collapse lives) rather than a control signal at 0.005.

Corroborating: training time is flat (259-449s) across all 20 rollouts regardless of
starvation, while inference swings 410-738s. The objective lives in inference; the signal
measures training.

---

## 3. `--migration-count-unit` (commit `cac3dd6a`)

B is denominated in SAMPLES but was measured by summing whole prompt groups (a group is
held at full weight until its SLOWEST sample lands). Measured over **135 independent
trigger firings** on the 50-rollout run: median group-implied **56** vs **17** samples
actually generating = **3.0x** (mean 3.2x). Bias is NOT constant: ~3.3x at low B, ~2.2x at
high B. At a fixed trigger value, true remaining work spans 2.6-4.1x.

**Verdict: throughput-neutral.** 10-rollout fixed-B, matched firing point (groups B=96
fires at ~28 live; samples B=32 at 31):

| arm | total | migrations |
|---|---|---|
| fixed groups96 | 5606s | 312 |
| fixed samples32 | 5619s | 306 |

**+0.23%**, paired per-rollout deltas scattering both ways (mean +0.26%, stdev 2.27,
t~0.36). Response lengths within 0.5%.

Default is `groups` (byte-identical to the original expression — verified against
`cac3dd6a^`). Default-path overhead measured at **15.3 ms/rollout (0.0026%)**. Only the new
experiment drivers pass the flag; no pre-existing test/sweep touches it.

**The apparent samples regression was the tuner, not the unit.** `tuner_samples` hit only
1.019x vs `tuner_groups` 1.138x — but the samples arm's tuner collapsed B to the b_min
floor after early starvations, ending with 98 migrations vs 156 and mean running batch
49.2 vs 55.2.

---

## 4. CubicTuner (commits `875517cb`, `3f54be54`)

TCP CUBIC (RFC 9438) with `t` in ROLLOUTS. Design doc: `perf_analysis/CUBIC_TUNER_DESIGN.md`.

Deliberate deviations from the RFC, all justified in the class docstring:
- **No Reno-friendly region** (9438 4.3) — bundles inter-flow fairness (irrelevant, one
  controller) with a growth floor the convex region already provides.
- **C = 1.0, not 0.4** — C survives the seconds->rollouts clock change numerically but not
  the change in HORIZON. Escaping a W_max set 33% too low takes t=7 at C=0.4, t=4 at C=1.6.
- **fast_convergence OFF by default** — its RFC purpose (yield bandwidth to a competing
  flow) does not exist here; it deliberately over-reacts to a lower W_max.
- **gamma=1.0 HOLDS B** until the first congestion event (does not grow by `step`).
- **`step` is never read** once W_max exists; the curve sets every move. Measured moves at
  W_max=144: -40, +24, +16, +8, +8, +24, +48.

Rails split into `max_up_step`/`max_down_step`, both defaulting to `step`, so the four
pre-existing tuners are byte-identical (their tests pass unchanged).

**Verified working**: 5-roll arm tracked the predicted curve exactly — 144 (held) -> 144
(starved 0.177) -> 104 -> 128 -> 144 -> 144.

**Known bug, NOT yet fixed**: `_snap()`'s anti-freeze nudge fires on rounding noise. When
the curve wants 111.99 and B is already 112, it nudges DOWN a full quantum, producing
+/-8 oscillation at exactly the plateau CUBIC exists to provide. Observed 3x in the
20-rollout run; one instance (r6) changed where W_max anchored. **Correct fix: keep B as a
continuous internal value and snap only on output**, so lattice rounding never re-enters
the controller state.

---

## 5. Errors made and corrected (do not repeat)

1. **Generation-only vs whole-rollout timing.** `train.py:239` prints GENERATION ONLY
   ("Rollout N took"), with training (`:265`) and weight update (`:298`) separate;
   `train_streaming.py:703` prints the WHOLE rollout. Grepping both into one column makes
   colocate look ~40% faster. Cost: a false "colocate is 30% faster" alarm. Guard:
   `perf_analysis/arm_totals.sh`.
2. **Counting abort lines as firings.** `[MIGRATION] aborting` fires ~8x per trigger
   firing, all carrying the same trigger value. Weighting by them oversamples the deep
   tail and inflated the group/live ratio to a spurious 4.7x (true: 3.0x over 135
   independent firings).
3. **Flat rescaling of the rails.** The group->live bias is not constant; a flat /3.56 was
   wrong at high B. Then "correcting" step 5->10 DOUBLED the controller's step in work
   terms (step 5 = 1.9 groups, matching groups' step 16 = 2 groups exactly).
4. **Trusting an unverified subagent claim.** Accepted "zero-chunk groups flipped in AFTER
   the last chunk" and built two messages on it. Direct check: all 10 zero-chunk GPUs
   flipped in 17-20s BEFORE the last chunk was grabbed. The zero-chunk rule is CORRECT;
   those starvations are real. (The agent's `significance` and `trigger-timing` verifiers
   had died on a token limit and never ran.)
5. **`pkill -f <pat>` matches its own `docker exec` shell** and kills it — produces NO
   output and leaves targets alive. Use `pkill -f "[s]glang"`. Cost: two runs colliding.
6. **`common.sh` assigns `ROLLOUT_BATCH=256` / `GLOBAL_BATCH=1024` unconditionally** (no
   `${VAR:-}` guard), so `X=${X:-...}` after sourcing silently keeps its value.

---

## 6. Open items

1. **Fix `_snap()`** — continuous internal B, snap on output only. (~10 lines)
2. **Re-derive epsilon, or demote the signal.** Given section 2, 0.005 is roughly 10x too
   sensitive; backing off costs more than the starvation does.
3. **Extend fixed B=96 to 20 rollouts.** Its 1.161x was measured to r10 only, and B=96
   starved severely at r16 (interior 0.1397). Neither fixed B's advantage nor the tuner's
   disadvantage has been tested past r10.
4. **Destination-ranking count unit** — `ctx.in_flight_count` is still group-denominated;
   `MIN_LOAD_*` constants cannot be rescaled by a single factor. Separate arm.

## 7. Where the data lives

All under `experiments/long_rl_training/deepseek_r1_8b/`:
`results_count_unit_controlled_5roll/` (colocate + tuner_groups + tuner_samples),
`results_fixedB_probe_3roll/`, `results_fixedB_10roll/`, `results_cubic_5roll_8gpu/`,
`results_cubic_20roll_smallB0/`, `results_tuner_10roll_samples_unit_8gpu/`.
Baselines: `results_colocated_50_step_dapo_8gpu/` (50 rollouts),
`results_streaming_interior_tuner_{10,50}_step_dapo_8gpu/`.

Tooling: `perf_analysis/arm_totals.sh`, `compare_three_arms.py`, `compare_count_unit.py`,
`cubic_20roll_trajectory.png`, `CUBIC_TUNER_DESIGN.md`.
