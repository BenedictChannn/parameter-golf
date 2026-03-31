# Agent Lab Findings

This file records the durable conclusions the lab currently believes, with direct links back to evidence.

Use it for the question: what is probably true right now?

Use [`state.md`](./state.md) for the live dashboard, [`ideas.md`](./ideas.md) for candidate hypotheses, and [`experiments.tsv`](./experiments.tsv) for the exact ledger.

## F-20260329-001: Width Rescue Was About Spending Bytes Better

- Claim: the cleanest way to save bytes on the width-biased line was reducing `MLP_MULT` from `3` to `2`, not trimming the whole model uniformly.
- Confidence: high
- Evidence:
- [`AL-20260329-012`](./experiments.tsv) beat the dim trims and became the new valid winner.
- [`AL-20260329-011`](./experiments.tsv) and [`AL-20260329-013`](./experiments.tsv) show that global dim trims can work, but they are less efficient.
- [`AL-20260329-014`](./experiments.tsv) shows that dropping one layer was also a weaker byte-saving move.
- Counterevidence:
- [`AL-20260329-015`](./experiments.tsv) says combined light cuts are a respectable backup, not a dead path.
- Next falsification:
- if a future architecture family shows global dim trims beating local structural trims, this finding becomes family-specific rather than general.

## F-20260329-002: More Steps Helped More Than Simple LR Increases

- Claim: on the surviving `9L / MLP2` family, getting more optimizer steps mattered much more than raising `MATRIX_LR`.
- Confidence: high
- Evidence:
- [`AL-20260329-016`](./experiments.tsv) was a major jump from lowering `TRAIN_BATCH_TOKENS` to `98304`.
- [`AL-20260329-017`](./experiments.tsv) and [`AL-20260329-018`](./experiments.tsv) show the `MATRIX_LR=0.065` bump did not beat the step-only win.
- [`AL-20260329-020`](./experiments.tsv) shows the same LR bump also hurt the fallback line.
- Counterevidence:
- none strong in this family so far
- Next falsification:
- test a more targeted optimizer change instead of a broad matrix-LR increase.

## F-20260329-003: The Current Frontier Prefers Fewer, Wider Query Heads

- Claim: on the post-tranche-D frontier, fewer wider query heads were better than the old `q8` setup.
- Confidence: medium-high
- Evidence:
- [`AL-20260329-021`](./experiments.tsv) `q4/kv2` beat the prior `q8/kv2` frontier.
- [`AL-20260329-022`](./experiments.tsv) shows the opposite direction, `q16/kv2`, failed badly.
- Counterevidence:
- `q4/kv2` changes both query width and the query-to-kv sharing ratio, so this is not a perfectly isolated result.
- [`AL-20260329-023`](./experiments.tsv) says less KV sharing alone was not enough.
- Next falsification:
- run `q4/kv4` or `q4/kv1` to separate “wider heads” from “different KV sharing ratio”.

## F-20260329-004: Output Path Is a First-Class Frontier Family

- Claim: the output path was one of the strongest underexplored levers in the current harness.
- Confidence: high
- Evidence:
- [`AL-20260329-026`](./experiments.tsv) untied outputs gave a large jump.
- [`AL-20260329-027`](./experiments.tsv) tighter `LOGIT_SOFTCAP=20` improved again.
- [`AL-20260329-030`](./experiments.tsv) faster `HEAD_LR=0.012` improved again on top.
- Counterevidence:
- [`AL-20260329-028`](./experiments.tsv) and [`AL-20260329-029`](./experiments.tsv) show not every output-path change wins.
- Next falsification:
- if a narrow local output tranche flattens, the next gains may come from residual or skip-path simplification instead.

## F-20260330-005: The Local Output Optimum Looks Mostly Mapped

- Claim: the narrow local neighborhood around `LOGIT_SOFTCAP=20` and `HEAD_LR=0.012` looks mostly exhausted for now.
- Confidence: medium-high
- Evidence:
- [`AL-20260330-001`](./experiments.tsv) effectively tied the best line at 4 decimals, but did not beat it and came in slightly larger.
- [`AL-20260330-002`](./experiments.tsv) shows that relaxing softcap above `20` is worse.
- [`AL-20260330-003`](./experiments.tsv) and [`AL-20260330-004`](./experiments.tsv) show that moving `HEAD_LR` slightly below or above `0.012` is also worse.
- [`AL-20260330-005`](./experiments.tsv) says even the strongest local combo stayed just behind the anchor.
- Counterevidence:
- the `AL-20260330-001` tie means there may still be noise-scale room in the family, just not a clear new winner.
- Next falsification:
- if a later output-path tranche adds a new mechanism rather than a nearby scalar retune and wins again, this finding becomes “the local scalar neighborhood is mapped” rather than “the family is done.”

## F-20260330-006: `resid_mix` Is Still Earning Its Keep

- Claim: learned input-stream mixing is still a useful part of the current frontier rather than removable baggage.
- Confidence: medium-high
- Evidence:
- [`AL-20260330-006`](./experiments.tsv) removed `resid_mix` and regressed sharply from `1.3564` to `1.3763`.
- The regression was large enough that this does not look like local noise.
- Counterevidence:
- none yet; this is the first direct `resid_mix` ablation on the current best family.
- Next falsification:
- if a later broader residual simplification package wins while still removing `resid_mix`, then the interaction story matters more than `resid_mix` alone.

## F-20260330-007: Learned Residual Scales Also Still Help

- Claim: the per-channel learned residual scales are still useful on the current frontier, even if they matter less than `resid_mix`.
- Confidence: medium
- Evidence:
- [`AL-20260330-007`](./experiments.tsv) removed both learned residual scales and regressed from `1.3564` to `1.3666`.
- The result stayed valid and somewhat smaller, but the quality loss was still too large to call a win.
- Counterevidence:
- the regression was smaller than the `resid_mix` ablation, so this family may still have interaction effects rather than being purely indispensable one knob at a time.
- Next falsification:
- test whether skip-path changes or a broader simplification package can recover the lost quality while still reducing residual-control complexity.

## F-20260330-008: Skip Topology May Matter More Than Learned Skip Weights

- Claim: the skip-path topology itself may be doing the real work, while the learned skip-weight parameterization is at most a second-order effect.
- Confidence: medium
- Evidence:
- [`AL-20260330-008`](./experiments.tsv) replaced learned skip weights with unit skip additions and nearly tied the frontier at `1.3568`.
- This was much stronger than the `resid_mix` and learned-scale ablations.
- Counterevidence:
- it still did not beat the anchor, so learned skip weights might still be worth a small amount.
- Next falsification:
- run the stronger topology ablation (`SKIP_MODE=off`). If that loses clearly while `SKIP_MODE=unit` nearly ties, the topology is the important part.

## F-20260330-009: Skip Topology Is Useful, Even If Learned Skip Weights Are Not Essential

- Claim: the skip connections themselves are helping on the current frontier, and the weaker part of the design is the learned skip weighting rather than the topology.
- Confidence: medium-high
- Evidence:
- [`AL-20260330-009`](./experiments.tsv) removed skip topology entirely and regressed to `1.3610`.
- This was much worse than [`AL-20260330-008`](./experiments.tsv), which kept the topology but used unit skip additions and nearly tied the frontier.
- Counterevidence:
- `SKIP_MODE=off` is still far better than the `resid_mix` ablation, so not every residual-control removal is equally damaging.
- Next falsification:
- run the broader simplification package (`H5`). If that still fails, then residual simplification likely is not the next frontier family.

## F-20260330-010: Residual Simplification Does Not Become Good Just Because It Is Combined

- Claim: the residual-control family does not hide a broad simplification win that only appears when multiple control removals are stacked together.
- Confidence: high
- Evidence:
- [`AL-20260330-010`](./experiments.tsv) combined the main simplifications and regressed badly to `1.3807`.
- This was worse than every other tranche-H run except the strongest single `resid_mix` removal, so the package did not unlock a hidden clean regime.
- Counterevidence:
- none meaningful in the current frontier family.
- Next falsification:
- only revisit this family if a future architecture change fundamentally alters the role of the residual paths.

## F-20260330-011: Tranche H Mostly Closed the Residual-Simplification Family

- Claim: tranche H says residual simplification is not the next major frontier family, except that fixed unit skip weights may preserve most of the skip benefit.
- Confidence: high
- Evidence:
- [`AL-20260330-006`](./experiments.tsv) and [`AL-20260330-007`](./experiments.tsv) show `resid_mix` and learned residual scales still help.
- [`AL-20260330-008`](./experiments.tsv) shows unit skip weights nearly tie the frontier.
- [`AL-20260330-009`](./experiments.tsv) shows removing skip topology itself is meaningfully worse.
- [`AL-20260330-010`](./experiments.tsv) rules out a hidden combo win.
- Counterevidence:
- `AL-20260330-008` means one narrow simplification path is still alive if we later want a cleaner skip parameterization.
- Next falsification:
- revisit only if a more radical architecture change makes the residual system play a different role.

## F-20260330-012: Mild All-Layer Latent-KV Is Viable but Not a Drop-In Win

- Claim: mild all-layer latent-KV compression is technically viable in this repo, but it does not preserve enough quality to challenge the current frontier as a direct replacement for full K/V attention.
- Confidence: medium
- Evidence:
- [`AL-20260330-011`](./experiments.tsv) trained cleanly, stayed under the size cap at 15.83 MB, and finished the full TTT pipeline.
- The final score still regressed clearly from `1.3564` to `1.3718`, so the first mild latent-KV form is not close enough to call a near miss.
- Counterevidence:
- this result only tests all-layer compression at `LATENT_KV_DIM=128`; it does not yet tell us whether the real issue is full-stack compression rather than the latent-KV mechanism itself.
- Next falsification:
- test whether stronger compression simply makes the loss worse (`I2`) or whether localized upper/lower placement (`I3`/`I4`) preserves more quality than all-layer compression.

## F-20260330-013: The First Latent-KV Form Did Not Buy a Cleaner Eval Path

- Claim: the first latent-KV design does not appear to improve the expensive TTT evaluation path, which weakens its value proposition for this challenge.
- Confidence: low-medium
- Evidence:
- [`AL-20260330-011`](./experiments.tsv) took `801s` for TTT eval, which sits in the same slow band as the recent frontier runs rather than obviously improving evaluation throughput.
- Counterevidence:
- this is only one latent-KV placement and one latent width; the evaluation cost may depend on where compression is applied rather than the general idea.
- Next falsification:
- compare TTT eval times and scores for the stronger all-layer and partial-layer latent-KV runs before concluding the whole family is evaluation-unfriendly.

## F-20260330-014: Stronger All-Layer Latent-KV Makes the Trade Worse

- Claim: if latent-KV is applied across every layer, making the bottleneck stronger buys size headroom and a few extra steps, but the quality trade degrades rather than improving.
- Confidence: medium-high
- Evidence:
- [`AL-20260330-012`](./experiments.tsv) improved on `I1` in efficiency terms: it reached `1684` steps instead of `1647` and shrank the artifact from `15.83 MB` to `14.73 MB`.
- Despite those savings, the final score regressed from [`AL-20260330-011`](./experiments.tsv) `1.3718` to `1.3865`.
- Counterevidence:
- this only rules against naive full-stack compression; it does not yet say whether upper-only or lower-only placement can make better use of the latent bottleneck.
- Next falsification:
- run the localized placement tests (`I3` and `I4`). If those recover quality, the mechanism is still alive but placement-sensitive.

## F-20260330-015: Latent-KV Is Placement-Sensitive, Not Fully Dead

- Claim: latent-KV is not a useful drop-in replacement for full attention across the whole stack, but it becomes materially better when localized to the upper layers.
- Confidence: medium-high
- Evidence:
- [`AL-20260330-013`](./experiments.tsv) improved from `1.3865` to `1.3685` relative to the all-layer latent64 baseline.
- [`AL-20260330-014`](./experiments.tsv) also improved over all-layer latent64, but not as much, which makes the upper-only placement the best version of the family so far.
- Counterevidence:
- even the best localized result is still clearly behind the frontier at `1.3564`, so the family is not yet submission-competitive.
- Next falsification:
- if a second-generation upper-only latent-KV design still cannot close the remaining gap, the family should be deprioritized behind other bold architecture ideas.

## F-20260330-016: Upper-Layer Compression Is Cleaner Than Lower-Layer Compression

- Claim: if K/V compression is used at all, the upper half of the stack is the better place to use it.
- Confidence: medium
- Evidence:
- [`AL-20260330-013`](./experiments.tsv) (`1.3685`) beat [`AL-20260330-014`](./experiments.tsv) (`1.3737`) while staying in the same size band.
- Both runs improved over all-layer latent64, but the upper-only one recovered more quality.
- Counterevidence:
- the difference is meaningful but not huge; more placement patterns would still be needed before calling this fully settled.
- Next falsification:
- test a narrower upper-only pattern or a mixed upper-middle pattern if the latent-KV family is revisited.

## F-20260330-017: Extra Depth Does Not Rescue Naive All-Layer Latent-KV

- Claim: the saved headroom from strong all-layer latent-KV does not automatically become useful when spent on one more layer.
- Confidence: medium-high
- Evidence:
- [`AL-20260330-015`](./experiments.tsv) lost to both the all-layer latent64 baseline and the localized placements.
- It gave back much of the saved headroom, dropped from `1684` to `1506` steps, and had the slowest TTT eval in the tranche.
- Counterevidence:
- this only tests one reinvestment style: another full-attention layer on top of the same all-layer compression recipe.
- Next falsification:
- if reinvestment is revisited, try spending the savings on a different form of capacity rather than plain extra depth.

## F-20260331-018: Lower-Stack Hybrid Mixers Are a Real Frontier Move

- Claim: the strongest architectural win in the overnight program was replacing the lower four attention layers with sequence mixers.
- Confidence: high
- Evidence:
- [`AL-20260330-104`](./experiments.tsv) improved the frontier from `1.3564` to `1.3488` while staying valid and smaller than the old best line.
- [`AL-20260330-103`](./experiments.tsv) also stayed ahead of the old frontier, so the hybrid family is not a one-off lucky placement.
- [`AL-20260330-101`](./experiments.tsv), [`AL-20260330-102`](./experiments.tsv), and [`AL-20260330-105`](./experiments.tsv) map the failure modes and show the lower stack is the cleanest place to swap attention out.
- Counterevidence:
- none strong inside tranche J; the main open question is how much more headroom the winning placement still has.
- Next falsification:
- vary mixer width, kernel, or exact lower-stack placement around [`AL-20260330-104`](./experiments.tsv). If those all flatten or regress, the family is real but already locally mapped.

## F-20260331-019: Output-Head Architecture Was Not the Bottleneck

- Claim: after tranche F, the dense untied output head remained the right output-head architecture; the next win did not come from low-rank or richer nonlinear heads.
- Confidence: high
- Evidence:
- [`AL-20260330-106`](./experiments.tsv) and [`AL-20260330-107`](./experiments.tsv) show low-rank heads regressed badly.
- [`AL-20260330-108`](./experiments.tsv) and [`AL-20260330-110`](./experiments.tsv) show richer nonlinear heads also failed badly.
- [`AL-20260330-109`](./experiments.tsv) was the best K-family variant, but still clearly worse than the dense untied frontier.
- Counterevidence:
- [`AL-20260330-109`](./experiments.tsv) suggests the untied win may be partly about a corrective residual head, so the family is not conceptually dead forever.
- Next falsification:
- only revisit output-head architecture if a later backbone change creates a different representation regime; do not spend another blind tranche here on the current line.

## F-20260331-020: Local-Global Attention Is Still Untested, Not Falsified

- Claim: the local-global attention family is currently unresolved rather than disproven.
- Confidence: high
- Evidence:
- [`AL-20260330-111`](./experiments.tsv) crashed before producing a metric, so tranche L never became a fair comparison against the frontier.
- The family therefore has zero trustworthy score evidence so far.
- Counterevidence:
- none yet; the family has not actually been tested.
- Next falsification:
- debug the lower-stack local-window crash first, then rerun the planned L tranche before making any scientific claim about local-global attention.

## F-20260331-021: Routing Redesign Can Stack Into a Useful Secondary Frontier

- Claim: skip/residual redesign was worth the compute, and the best version is a coherent cheap-routing package rather than a single isolated simplification.
- Confidence: medium-high
- Evidence:
- [`AL-20260330-116`](./experiments.tsv) shared scalar skip gates already beat the old pre-program frontier.
- [`AL-20260330-120`](./experiments.tsv) improved further to `1.3534`, making it the second-best overnight result after the hybrid mixer win.
- [`AL-20260330-117`](./experiments.tsv), [`AL-20260330-118`](./experiments.tsv), and [`AL-20260330-119`](./experiments.tsv) stayed close but weaker, which supports the idea that the routing components interact.
- Counterevidence:
- the routing package still lost to the hybrid-mixer winner by a clear margin.
- Next falsification:
- test whether the cheap-routing package stacks with the hybrid-mixer frontier or whether both are buying the same underlying effect.

## F-20260331-022: Mechanism-Specific Learning Rates Are Supportive, Not Transformative

- Claim: more structured learning-rate splits can keep up with the old frontier, but they did not create a new breakthrough on their own.
- Confidence: medium
- Evidence:
- [`AL-20260330-121`](./experiments.tsv), [`AL-20260330-124`](./experiments.tsv), and [`AL-20260330-125`](./experiments.tsv) all landed in the old-best band around `1.3561` to `1.3562`.
- [`AL-20260330-122`](./experiments.tsv) and [`AL-20260330-123`](./experiments.tsv) show the wrong directional splits regress.
- Counterevidence:
- the positive N-family results are so small that they may mostly be support tools for future architectures rather than useful standalone tranches.
- Next falsification:
- apply the best N-family LR splits to the hybrid-mixer or cheap-routing winners and see whether they unlock extra headroom there.

## F-20260331-023: Naive Quantization-Aware Warmdown Mostly Hurts

- Claim: the first pass at quantization-aware warmdown did not help the submitted compressed model, even though some variants reduced artifact size sharply.
- Confidence: medium-high
- Evidence:
- [`AL-20260330-126`](./experiments.tsv) and [`AL-20260330-127`](./experiments.tsv) both regressed badly despite much smaller artifacts.
- [`AL-20260330-130`](./experiments.tsv) shows the combined tail also failed badly.
- [`AL-20260330-128`](./experiments.tsv) and [`AL-20260330-129`](./experiments.tsv) stayed close, but still did not beat the old frontier.
- Counterevidence:
- [`AL-20260330-129`](./experiments.tsv) was close enough that output-path-specific cooldown might still matter in a different architecture regime.
- Next falsification:
- if quantization-aware scheduling is revisited, focus on output-path-sensitive finishing or architecture-specific tails rather than broad colder schedules.
