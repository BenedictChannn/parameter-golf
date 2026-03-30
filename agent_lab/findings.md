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
