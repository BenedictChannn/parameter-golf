# Agent Lab State

This is the first-read dashboard for autonomous research. Read this file for the high-level picture, then drill down into the linked tranche, idea bank, experiment ledger, and build log.

## Current Best Valid

- Experiment: [`AL-20260329-030`](./experiments.tsv)
- Commit: `a8333e1`
- Primary metric: `final_int8_ttt_lora`
- Best `val_bpb`: `1.3564`
- Winning branch shape: `9` layers, `MLP_MULT=2`, `MODEL_DIM=512`, `NUM_HEADS=4`, `NUM_KV_HEADS=2`, `TRAIN_BATCH_TOKENS=98304`, `TIE_EMBEDDINGS=0`, `LOGIT_SOFTCAP=20`, `HEAD_LR=0.012`
- Total artifact size: `15,797,032` bytes int8+zlib

## Best Invalid Near-Miss

- Experiment: [`AL-20260329-010`](./experiments.tsv)
- Commit: `fe477f9`
- Primary metric: `final_int8_ttt_lora`
- Best invalid `val_bpb`: `1.3899`
- Shape: `9` layers, `MLP_MULT=3`, `NUM_KV_HEADS=2`, `TRAIN_BATCH_TOKENS=131072`
- Why invalid: `17,680,105` bytes int8+zlib exceeds the `16,000,000` byte cap

## Active Tranche

- Tranche: [`T-20260330-H`](./tranches.md#t-20260330-h-residual-control-simplification)
- Goal: test whether the current best line is carrying unnecessary residual-control and skip-path complexity
- Status: completed

## Working Beliefs

- The mar29 system runtime is stronger than the earlier mar28 local anchor, so new baselines must be compared against the refreshed local frontier, not the older numbers.
- `10` layers are useful only when they are not step-starved.
- Smaller batches recovered enough optimizer steps to turn the `10`-layer branch from a loser into the current winner.
- The frontier is flattening: `196608` and `131072` batch settings are very close.
- Cheaper attention via `NUM_KV_HEADS=1` was not a better trade than keeping `NUM_KV_HEADS=2`.
- At fixed `10`-layer depth, shrinking the MLP to `MLP_MULT=1` bought steps and artifact headroom, but not enough quality to beat the `MLP_MULT=2` anchor.
- At fixed `10`-layer depth, widening to `MLP_MULT=3` was clearly bad: fewer steps, worse `val_bpb`, and a 17.43 MB artifact that breaks the cap.
- Reallocating width into one extra layer via `11L / MLP1` did not rescue the idea; it stayed within size and got good step count, but still lost on quality.
- The first promising width branch is `9L / MLP3`: it improved a lot over `10L / MLP3`, but it still trails the anchor and is slightly too large at 16.18 MB.
- Width clearly benefits from step recovery: `9L / MLP3 / 131072` became the best raw scorer at `1.3899`, but the artifact grew even further over the cap.
- A mild global dim trim to `DIM480` is enough to make the width winner valid on size, but not enough to keep it competitive on score.
- The cleaner byte cut is not a global dim trim. `9L / MLP2 / 131072` kept the high-step regime, stayed well under the size cap, and moved the valid frontier to `1.3838`.
- If a global dim trim is necessary, `DIM448` is the meaningful point, not `DIM480`: it recovered most of the lost score and stayed very small, but it still did not beat the `9L / MLP2` survivor.
- Dropping one layer is not the cleaner size move here. `8L / MLP3 / 131072` kept decent quality, but it was still over the cap, so the layer cut saved fewer useful bytes than the MLP-notch cut.
- The best fallback after the winner is the combined-light-cuts line, `8L / MLP3 / DIM480 / 131072`. It is valid and stronger than the other fallback trims, but it still sits well behind `9L / MLP2 / 131072`.
- The winner was still substantially step-limited. Dropping batch to `98304` on `9L / MLP2` was a major gain, not a marginal one, and it still stayed inside the artifact cap.
- A simple LR bump is not interchangeable with extra steps. `MATRIX_LR=0.065` at `131072` batch fell back to the old frontier band instead of following the `98304` win.
- The combo test also failed to beat the step-only win. `98304 + MATRIX_LR=0.065` stayed strong, but it still lost clearly to `98304` with the default LR.
- The fallback line is also step-limited. `8L / MLP3 / DIM480 / 98304` improved a lot over its `131072` version, but it still remains secondary to the `9L / MLP2 / 98304` frontier.
- The LR bump helps neither survivor. It regressed both the main line and the fallback line, so the current best-tested setting keeps the default `MATRIX_LR=0.06`.
- The next good question is no longer optimization. The best line is now stable enough that another worthwhile tranche should attack a different component family.
- Attention is the cleanest next component family because it has several env-exposed geometry knobs: query-head count, KV sharing, and QK sharpness at initialization.
- The first attention-geometry probe already moved the frontier. `q4/kv2` beat `q8/kv2`, which is a strong hint that the current model wants fewer, wider query heads.
- The opposite head-direction failed hard. `q16/kv2` regressed badly, which means the E1 result is directional evidence, not a generic “head changes help” result.
- Reducing KV sharing is not the main win. `q8/kv4` improved on the old `q8/kv2` line, but it still fell short of `q4/kv2`.
- Softer QK init is not the main win either. `QK_GAIN_INIT=1.0` stayed competitive, but it still did not beat the wider-head direction.
- Sharper QK init also stayed secondary. `QK_GAIN_INIT=2.0` beat the softer bracket, but neither QK-gain change beat the `q4/kv2` head-geometry win.
- The next good question is no longer attention. The best line is now `q4/kv2`, and the next likely underexplored lever is the output path: tying, logit softcap, and output-specific learning dynamics.
- The first output-path probe already moved the frontier a lot. Untying embeddings and the output head was a major gain while still staying under the size cap.
- The second output-path probe also won. The untied frontier improved again with `LOGIT_SOFTCAP=20`, so output calibration is a real lever on top of output expressivity.
- Looser clipping did not win the bracket. `LOGIT_SOFTCAP=40` stayed strong, but it still lost clearly to `20`, so the output-calibration result is directional.
- Slower head LR is not the main win. `HEAD_LR=0.004` stayed competitive, but calibration still matters more than this first head-LR reduction.
- Faster head LR on the untied + softcap20 line is the new best result so far. `HEAD_LR=0.012` beat both the default and slower-head-LR variants, so output-head learning dynamics are a real follow-on lever rather than noise.
- The next compute-worthy question is a narrow one: whether the local output-path optimum sits slightly above, below, or exactly at the current `LOGIT_SOFTCAP=20` and `HEAD_LR=0.012`.
- Tranche G mostly closed that loop. Nearby scalar moves around `LOGIT_SOFTCAP=20` and `HEAD_LR=0.012` did not produce a clear new winner, so the local output neighborhood now looks mostly mapped.
- The next best family is now residual controls and skip topology. These are still distinctive to this script and are now exposed cleanly enough to test with env vars.
- The first residual-control probe already answered something concrete. Removing `resid_mix` regressed badly, so learned input-stream mixing is still materially helping the current frontier.
- The second residual-control probe also lost. Removing learned residual scales regressed more modestly than `H1`, but still clearly enough that plain unit residuals do not look better on this frontier.
- The third residual-control probe was a near tie. `SKIP_MODE=unit` came in only `0.0004` behind the frontier, which strongly suggests skip topology matters more than learning the skip weights.
- The fourth residual-control probe clarified that story. `SKIP_MODE=off` lost much more clearly than `SKIP_MODE=unit`, so the skip topology itself is useful and the learned weighting looks like the softer target.
- The full simplification package failed hard. `H5` did not reveal a hidden interaction win, so there is no sign that residual simplification becomes good when stacked together.
- Tranche H overall says: do not spend the next tranche on generic residual simplification. The only live lesson is that unit skip weights are almost free, while the skip topology itself should stay.

## Open Questions

- Can a more radical attention redesign beat the current `q4/kv2` block instead of merely compressing the existing residual structure?
- Can a hybrid recurrent or delta-style mixer replace some attention layers under the 600s budget?
- Is the next frontier in compressed attention, in hybrid sequence mixing, or in a new golf-native block that combines both?

## Next Planned Runs

- Next tranche candidate I: latent-KV / MLA-lite attention compression
- Next tranche candidate J: hybrid delta-style / state-space mixer layers
- Residual simplification follow-up is not the priority unless it is specifically about fixed unit skip weights

## Go Deeper

- Tranche map: [`tranches.md`](./tranches.md)
- Idea bank: [`ideas.md`](./ideas.md)
- Architecture audit: [`architecture_review.md`](./architecture_review.md)
- Experiment dashboard: [`plots/experiments.html`](./plots/experiments.html)
- Durable findings: [`findings.md`](./findings.md)
- Current budget report: [`budget_report.md`](./budget_report.md)
- Experiment ledger: [`experiments.tsv`](./experiments.tsv)
- Latest narrative log: [`docs/build-logs/2026-03-30-agent-lab.md`](../docs/build-logs/2026-03-30-agent-lab.md)
