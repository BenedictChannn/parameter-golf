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


## F-20260331-024: The Hybrid Mixer Frontier Still Has Local Headroom

- Claim: the lower-stack hybrid-mixer family was not locally exhausted at [`AL-20260330-104`](./experiments.tsv); refining the mechanism still produced meaningful gains.
- Confidence: high
- Evidence:
- [`AL-20260331-001`](./experiments.tsv) improved the frontier further with lower-three mixers.
- [`AL-20260331-004`](./experiments.tsv) improved again and became the new best valid run at `1.3451`.
- [`AL-20260331-005`](./experiments.tsv) also beat the old best via a wider mixer kernel.
- Counterevidence:
- [`AL-20260331-002`](./experiments.tsv) shows the win does not simply scale with “more mixer layers.”
- [`AL-20260331-003`](./experiments.tsv) stayed flat, so not every refinement direction pays.
- Next falsification:
- test whether the cheap-routing package stacks with the stronger hybrid winner, or whether both families are buying the same underlying effect.

## F-20260331-025: The Lower Stack Wants Mixers, But Not Too Many

- Claim: the lower stack remains the right place to replace attention, but pushing the replacement from four lower layers to five goes too far on the current frontier.
- Confidence: medium-high
- Evidence:
- [`AL-20260331-002`](./experiments.tsv) was smaller and faster than the anchor, but still regressed in quality.
- [`AL-20260331-001`](./experiments.tsv) and [`AL-20260331-004`](./experiments.tsv) both beat the old best without needing a fifth lower mixer layer.
- Counterevidence:
- the quality loss at five mixers was modest, not catastrophic, so the family still tolerates aggressive lower-stack replacement better than many alternatives.
- Next falsification:
- test whether five lower mixers only works when paired with a stronger mixer or a complementary routing change.

## F-20260331-026: Broad Warmdown Is Still The Wrong Schedule Story On The Hybrid Winner

- Claim: even on the stronger hybrid-mixer backbone, broad warmdown remains the wrong architecture-specific schedule story.
- Confidence: medium-high
- Evidence:
- [`AL-20260331-011`](./experiments.tsv) and [`AL-20260331-012`](./experiments.tsv) both regressed clearly.
- [`AL-20260331-015`](./experiments.tsv) shows the broader mild-tail combo also failed.
- Counterevidence:
- [`AL-20260331-013`](./experiments.tsv) and [`AL-20260331-014`](./experiments.tsv) stayed close enough that some narrow end-of-training interventions may still matter.
- Next falsification:
- if schedule work is revisited, focus on output-path-sensitive cooldown or very narrow stabilization rather than global colder tails.

## F-20260331-027: Head-Focused Cooldown Is The Only Live Schedule Hint

- Claim: among the architecture-specific schedule ideas tested so far, only output-path-sensitive cooldown stayed in the noise band of the hybrid frontier.
- Confidence: medium
- Evidence:
- [`AL-20260331-014`](./experiments.tsv) finished at `1.3489`, effectively tying the older hybrid winner.
- It beat the broader warmdown and combined-tail variants cleanly.
- Counterevidence:
- it still did not beat the actual hybrid frontier, so this is a hint rather than a win.
- Next falsification:
- if schedule work is resumed later, apply head-focused cooldown on top of the stronger `AL-20260331-004` backbone or after another architecture change.

## F-20260331-028: Late-Layer AttnRes-lite Is The Wrong Routing Shape On This Frontier

- Claim: replacing fixed routing with AttnRes-lite across late layers is strongly misaligned with the current hybrid-mixer frontier.
- Confidence: high
- Evidence:
- [`AL-20260331-007`](./experiments.tsv) and [`AL-20260331-009`](./experiments.tsv) both regressed catastrophically into the `1.50+` band.
- [`AL-20260331-010`](./experiments.tsv) shows the failure persists even when stacked on top of the cheap-routing package.
- Counterevidence:
- the tranche still lacks a clean rerun of `Q1`, so the exact three-source late-layer number is missing.
- Next falsification:
- only revisit late-layer depth routing if the mechanism itself is redesigned substantially rather than simply rerun.

## F-20260331-029: If Dynamic Depth Routing Helps At All, It Belongs Only At The Top

- Claim: the only AttnRes-lite placement that looked even remotely viable was restricting the mechanism to the top two layers.
- Confidence: medium
- Evidence:
- [`AL-20260331-008`](./experiments.tsv) at `1.3499` was vastly better than the other completed Q runs and stayed close to the old hybrid anchor.
- All broader late-layer placements were dramatically worse.
- Counterevidence:
- [`AL-20260331-008`](./experiments.tsv) still lost clearly to the current best [`AL-20260331-004`](./experiments.tsv), so this is a faint hint rather than a live frontier result.
- Next falsification:
- if AttnRes-lite is revisited, test an even lighter top-of-stack-only version or a more constrained top-layer gate rather than broad late-stack routing.

## F-20260331-030: Cheap Fixed Routing And Dynamic Depth Routing Did Not Stack

- Claim: the best cheap-routing package and the first AttnRes-lite mechanism are not complementary in their current forms.
- Confidence: medium-high
- Evidence:
- [`AL-20260331-010`](./experiments.tsv) stayed in the same catastrophic regime as the other broad late-layer AttnRes-lite runs.
- It did not recover toward the hybrid frontier even though the cheap-routing package was itself a real secondary winner.
- Counterevidence:
- this only tests one combo form; a future top-only or much lighter depth-routing design could still interact differently.
- Next falsification:
- if routing combo work is revisited, combine cheap routing only with top-of-stack dynamic routing, not the broad late-layer AttnRes-lite design.

## F-20260401-031: Shared Skip Gates Are The Cleanest Stacking Win On The Hybrid Frontier

- Claim: the strongest routing change to pair with the widened lower-four-mixer winner is shared scalar skip gating, not the full cheap-routing package.
- Confidence: high
- Evidence:
- [`AL-20260331-017`](./experiments.tsv) is the current best valid run at `1.3429`.
- [`AL-20260331-016`](./experiments.tsv) also won, but the full routing package was slightly worse than skip gates alone.
- Counterevidence:
- [`AL-20260331-018`](./experiments.tsv) shows scalar `resid_mix` is also compatible, so the story is not exclusively about skip gates.
- Next falsification:
- test whether the `AL-20260331-017` line still wants any additional routing simplification, or whether skip gates already capture nearly all of the available gain.

## F-20260401-032: Local Attention On The Remaining Attention Stack Is Mostly The Wrong Trade

- Claim: once the lower stack is already mixer-heavy, localizing the remaining attention layers hurts more than it helps.
- Confidence: medium-high
- Evidence:
- [`AL-20260331-022`](./experiments.tsv) failed badly with four upper local-attention layers.
- [`AL-20260331-023`](./experiments.tsv) was the best repaired local-window run, but still lost clearly at `1.3545`.
- [`AL-20260331-024`](./experiments.tsv) and [`AL-20260331-025`](./experiments.tsv) also lost cleanly.
- Counterevidence:
- [`AL-20260331-021`](./experiments.tsv) crashed before producing the lightest top-two-local `256` result, so that exact point remains unmeasured.
- Next falsification:
- only revisit this family if there is a new theory beyond plain windowed local attention on the surviving attention layers.

## F-20260401-033: Naive Low-Rank Factorization Is Not Yet A Viable Compression-Native Backbone

- Claim: the first compression-native branch failed because straightforward low-rank factorization removes too much useful structure from the current hybrid winner.
- Confidence: high
- Evidence:
- [`AL-20260331-026`](./experiments.tsv) and [`AL-20260331-027`](./experiments.tsv) both lost clearly despite saving bytes.
- [`AL-20260331-028`](./experiments.tsv) and [`AL-20260331-029`](./experiments.tsv) show that MLP factorization is far more destructive still.
- [`AL-20260331-030`](./experiments.tsv) confirms that combining the two does not rescue the idea.
- Counterevidence:
- the branch only tested one compression-native mechanism family; it does not rule out other quantization-aware or structure-aware designs.
- Next falsification:
- the next compression-native tranche should test a qualitatively different mechanism, not just another rank sweep.

## F-20260401-034: Top-Only Dynamic Routing Is A Real Secondary Family

- Claim: dynamic depth routing is only viable on this frontier when it is kept extremely narrow: top-only, low-source, and preferably paired with clean fixed routing.
- Confidence: medium-high
- Evidence:
- [`AL-20260331-031`](./experiments.tsv) and [`AL-20260331-032`](./experiments.tsv) both beat the older hybrid anchor.
- [`AL-20260331-035`](./experiments.tsv) at `1.3433` became the second-best run in the whole `S` through `X` queue.
- [`AL-20260331-033`](./experiments.tsv) and [`AL-20260331-034`](./experiments.tsv) show that coarse shared routing and extra source complexity both make the top-only line worse.
- Counterevidence:
- even the best top-only result still lost to [`AL-20260331-017`](./experiments.tsv), so the family is secondary rather than dominant.
- Next falsification:
- if revisited, keep the router top-only and light; do not return to broad late-layer routing.

## F-20260401-035: The Broad MLP Family Still Belongs To ReLU-Squared

- Claim: on the current hybrid frontier, `relu^2` remains the right broad MLP family and the smooth replacements are not competitive.
- Confidence: high
- Evidence:
- [`AL-20260331-036`](./experiments.tsv) reproduced the frontier class under the new MLP-mode surface.
- [`AL-20260331-037`](./experiments.tsv), [`AL-20260331-038`](./experiments.tsv), and [`AL-20260331-039`](./experiments.tsv) all regressed clearly.
- Counterevidence:
- [`AL-20260331-040`](./experiments.tsv) suggests a gated MLP could still be interesting if its size cost is brought under control.
- Next falsification:
- if the MLP family is revisited soon, focus on size-controlled gated variants rather than plain smooth activations.

## F-20260401-036: Mixed Linear-Plus-Quadratic MLPs Are The Only Live Polynomial Variant

- Claim: inside the polynomial family, the only variant that stayed near the frontier was mixing linear and quadratic behavior; cubic structure mostly hurt.
- Confidence: medium
- Evidence:
- [`AL-20260331-043`](./experiments.tsv) at `1.3444` was the best non-anchor MLP result in the polynomial tranche.
- [`AL-20260331-042`](./experiments.tsv) and [`AL-20260331-044`](./experiments.tsv) show that cubic-heavy variants regress clearly.
- [`AL-20260331-045`](./experiments.tsv) stayed close but still did not beat the plain `relu^2` baseline.
- Counterevidence:
- the gains over the anchor are still small, so this is a later refinement path rather than a proven new frontier family.
- Next falsification:
- if polynomial MLPs are revisited, treat `relu + quadratic` as the anchor and ignore cubic-heavy forms unless a new theory justifies them.

## F-20260401-037: The Current Frontier Still Needs Real Dense FFNs In Most Layers

- Claim: broad removal or strong shrinking of the expand-project MLP structure is misaligned with the current hybrid frontier.
- Confidence: high
- Evidence:
- [`AL-20260401-046`](./experiments.tsv) and [`AL-20260401-048`](./experiments.tsv) show that no-expand tokenwise MLP replacements collapse far away from the frontier.
- [`AL-20260401-047`](./experiments.tsv) shows even halving dense FFN width loses clearly.
- [`AL-20260401-050`](./experiments.tsv) shows keeping full MLPs only at the top is also too weak.
- Counterevidence:
- [`AL-20260401-049`](./experiments.tsv) stayed relatively close, which means some depth-specific MLP simplification may still be possible.
- Next falsification:
- if FFN structure is revisited, do not retry broad no-expand minimalism; only test lower-light or other narrow stage-specific simplifications.

## F-20260401-038: Naive Cross-Layer Weight Sharing Is Not A Viable Compression-Native Win

- Claim: the current hybrid backbone is not overparameterized in a way that simple block sharing can exploit cleanly.
- Confidence: medium-high
- Evidence:
- [`AL-20260401-061`](./experiments.tsv) was the best sharing run, and it still lost clearly to the frontier.
- [`AL-20260401-062`](./experiments.tsv) and [`AL-20260401-063`](./experiments.tsv) show that both lower-stage and top-stage sharing degrade quality materially.
- [`AL-20260401-064`](./experiments.tsv) and [`AL-20260401-065`](./experiments.tsv) show that reinvesting the saved bytes does not rescue the sharing branch.
- Counterevidence:
- `AB1` was only modestly worse while saving bytes, so sharing is not catastrophic in the same way as naive low-rank MLP factorization.
- Next falsification:
- if compression-native design is revisited, move away from whole-block sharing and test mechanisms that preserve more layer individuality.

## F-20260401-039: The Upper Stack Can Probably Be Simplified By Interleaving, Not By Collapsing

- Claim: the remaining upper attention stack is still necessary, but a periodic global-refresh pattern may be a viable simplification path.
- Confidence: medium
- Evidence:
- [`AL-20260401-059`](./experiments.tsv) at `1.3454` was only `+0.0025` off the frontier and dramatically better than the other AA variants.
- [`AL-20260401-057`](./experiments.tsv) and [`AL-20260401-060`](./experiments.tsv) show that top-two-only and single-final-reasoner collapse the upper stack too aggressively.
- [`AL-20260401-058`](./experiments.tsv) shows that adding top-only routing does not rescue an overly thinned upper-attention stack.
- Counterevidence:
- [`AL-20260401-059`](./experiments.tsv) still lost to the frontier, so interleaving is only a near-survivor, not a winner.
- Next falsification:
- if the upper-attention family is resumed, focus on interleaved or periodic-refresh designs, not further collapse to only one or two global layers.
