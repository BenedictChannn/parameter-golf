# Agent Lab Tranches

This file is the high-level research-program map.

Use it to answer:

- what tranche is active
- what question that tranche is trying to answer
- what is fixed
- what we already learned
- what comes next

For exact run results, use [`experiments.tsv`](./experiments.tsv). For longer reasoning, use the dated build log under [`docs/build-logs/`](../docs/build-logs/).

## T-20260328-A: Local Baseline Calibration

**Status:** completed

**Goal**  
Establish a usable local baseline on the 3090 stack and identify the first levers worth keeping.

**Fixed controls**

- one training shard
- `600s` training cap
- primary metric `final_int8_ttt_lora`

**Main findings**

- `NUM_KV_HEADS 4 -> 2` helped
- `TRAIN_BATCH_TOKENS 524288 -> 262144` helped a lot
- `MATRIX_LR 0.04 -> 0.06` helped modestly

**Key experiments**

- [`AL-20260328-001`](./experiments.tsv)
- [`AL-20260328-002`](./experiments.tsv)
- [`AL-20260328-003`](./experiments.tsv)
- [`AL-20260328-004`](./experiments.tsv)

**Deeper notes**

- [`docs/build-logs/2026-03-28-agent-lab.md`](../docs/build-logs/2026-03-28-agent-lab.md)

## T-20260329-A: Capacity vs Step Frontier

**Status:** active but mostly mapped

**Goal**  
Determine how much extra capacity the current local runtime can support inside a fixed `600s` budget, and whether extra depth only wins when the branch also gets more optimizer steps.

**Main question**  
Is the best local frontier on this stack “more depth plus more steps”, and if so, where does that frontier flatten?

**Fixed controls**

- one training shard
- `600s` training cap
- primary metric `final_int8_ttt_lora`
- tokenizer and validation semantics unchanged

**What we tested**

- refreshed baseline on the current runtime
- `10` layers alone
- `10` layers with smaller batch
- `10` layers with much smaller batch
- `10` layers with cheaper attention via `NUM_KV_HEADS=1`

**What we learned**

- `10` layers alone was wrong because it lost too many steps
- `10` layers plus `196608` batch was a clear win
- `10` layers plus `131072` batch was only a marginal further improvement
- `10` layers plus `NUM_KV_HEADS=1` was worse than the best `kv2` branches

**Current best inside this tranche**

- [`AL-20260329-004`](./experiments.tsv) at `1.3913`

**Key experiments**

- [`AL-20260329-001`](./experiments.tsv)
- [`AL-20260329-002`](./experiments.tsv)
- [`AL-20260329-003`](./experiments.tsv)
- [`AL-20260329-004`](./experiments.tsv)
- [`AL-20260329-005`](./experiments.tsv)

**Stop or pivot rule**  
Stop this tranche when nearby reruns suggest the `196608` vs `131072` difference is mostly noise, or when more frontier pushes cost too much artifact headroom for too little quality.

**Likely next pivot**  
Move from “buy more steps” to “reallocate capacity more intelligently.”

**Deeper notes**

- [`docs/build-logs/2026-03-29-agent-lab.md`](../docs/build-logs/2026-03-29-agent-lab.md)

## T-20260329-B: Architecture Necessity Audit

**Status:** completed

**Goal**  
Break the model into major components and ask, one family at a time, whether each piece is actually earning its bytes, compute, and optimization complexity.

**Main question**  
After the first capacity frontier is partly mapped, is the next gain more likely to come from a better distribution of capacity, or from simplifying or removing overbuilt structure?

**Fixed controls**

- one training shard
- `600s` training cap
- primary metric `final_int8_ttt_lora`
- tokenizer and validation semantics unchanged

**Investigation families**

- MLP width versus depth
- residual controls and skip topology
- output path choices such as tying and logit softcap
- compression-aware architectural tradeoffs

**Working surface**

- [`architecture_review.md`](./architecture_review.md)

**Pivot rule**  
If a family shows only noise-level differences after a few well-chosen runs, move to the next component instead of overfitting one local knob.

### B1: MLP Width vs Depth

**Status:** completed

**Research question**  
With the current `10`-layer line, are we getting more value from extra transformations, or would some of that budget work better as fatter MLPs?

**Why this sub-tranche now**

- depth already proved it can help when step-starvation is fixed
- we still have not asked whether `MLP_MULT=2` is too small, too large, or simply the wrong place to spend capacity

**Controls for this 5-run set**

- use env vars rather than code edits
- keep `NUM_KV_HEADS=2`
- keep `MODEL_DIM=512`
- keep `NUM_HEADS=8`
- keep tied embeddings, tokenizer, and validation unchanged
- use the full `600s` training cap
- use `final_int8_ttt_lora` as the primary metric

**Anchor**

- [`AL-20260329-003`](./experiments.tsv) is the cleanest comparison point because it is strong at `1.3916` and leaves more artifact headroom than [`AL-20260329-004`](./experiments.tsv)

**Planned experiments**

| ID | Shape | Goal | Hypothesis | What it teaches |
|---|---|---|---|---|
| `B1-E1` | `10L / MLP1 / batch 196608 / kv2` | Test a thinner MLP at fixed depth | The current `MLP_MULT=2` may already be wider than this regime needs | Whether reducing MLP width helps by freeing compute without losing too much quality |
| `B1-E2` | `10L / MLP3 / batch 196608 / kv2` | Test a wider MLP at fixed depth | The current model may be under-spending capacity inside each block | Whether pure width helps before we change layer count |
| `B1-E3` | `11L / MLP1 / batch 196608 / kv2` | Reallocate width into more layers | The best use of budget may be deeper but thinner blocks | Whether more transformations beat block-internal width |
| `B1-E4` | `9L / MLP3 / batch 196608 / kv2` | Reallocate depth into more width | Some capacity may be better spent inside each block than on one extra layer | Whether width can replace a layer cleanly |
| `B1-E5` | `9L / MLP3 / batch 131072 / kv2` | Test width with step recovery | Width may also need more optimizer steps, just like depth did | Whether a width loss is fundamental or just another fixed-budget step problem |

**Decision rule for B1**

- if `B1-E1` beats the anchor, the current MLP is likely too wide
- if `B1-E2` beats the anchor, pure width deserves a larger follow-up tranche
- if `B1-E3` wins, the model is probably under-layered relative to its MLP size
- if `B1-E4` or `B1-E5` wins, the next tranche should become width-aware rather than purely depth-aware

**Outcome**

- `B1-E1` (`10L / MLP1`) lost: thinning the MLP at fixed depth bought steps and headroom, but not enough quality
- `B1-E2` (`10L / MLP3`) lost badly and broke the artifact cap
- `B1-E3` (`11L / MLP1`) also lost, so deeper-but-thinner does not beat the current `10L / MLP2` balance
- `B1-E4` (`9L / MLP3 / 196608`) was the first promising width branch, but still slightly over the cap and still behind the valid anchor
- `B1-E5` (`9L / MLP3 / 131072`) produced the best raw score so far at `1.3899`, which strongly suggests width needed more steps, but it is invalid at `17.68 MB`

**Reading**

- width is not dead, but it only became competitive after both reducing depth and recovering more steps
- the best valid frontier is still [`AL-20260329-004`](./experiments.tsv) at `1.3913`
- the most interesting follow-up is no longer “is width good?” but “can the `9L / MLP3` winner be made challenge-valid without losing its score?”

## T-20260329-C: Width Winner Size Recovery

**Status:** completed

**Goal**  
Take the raw width-biased near-miss, [`AL-20260329-010`](./experiments.tsv), and learn which byte cuts preserve the score best while recovering challenge-valid size.

**Main question**  
Can the `9L / MLP3 / 131072 / kv2` branch be pulled under `16 MB`, and which structural cut loses the least performance per byte saved?

**Fixed controls**

- one training shard
- `600s` training cap
- primary metric `final_int8_ttt_lora`
- tokenizer and validation semantics unchanged
- keep `TRAIN_BATCH_TOKENS=131072` unless the experiment explicitly says otherwise
- keep `NUM_KV_HEADS=2`
- keep tied embeddings

**Anchors**

- raw winner: [`AL-20260329-010`](./experiments.tsv) at `1.3899`, but invalid at `17,680,105` bytes
- current best valid comparator: [`AL-20260329-012`](./experiments.tsv) at `1.3838`

**Planned experiments**

| ID | Shape | Goal | Hypothesis | What it teaches |
|---|---|---|---|---|
| `C1-E1` | `9L / MLP3 / DIM480 / batch 131072 / kv2` | Small dimension trim | A mild `MODEL_DIM` cut may recover enough bytes while preserving most of the width gain | Whether width can survive a modest global shrink |
| `C1-E2` | `9L / MLP3 / DIM448 / batch 131072 / kv2` | Stronger dimension trim | A larger `MODEL_DIM` cut may cross under the cap with an acceptable quality hit | How steep the score-vs-dim tradeoff is around the raw winner |
| `C1-E3` | `9L / MLP2 / DIM512 / batch 131072 / kv2` | One-notch MLP shrink | Most of the width gain may survive with `MLP_MULT=2` once steps stay high | Whether the last MLP notch is the main byte offender |
| `C1-E4` | `8L / MLP3 / DIM512 / batch 131072 / kv2` | One-layer trim instead of width trim | The 9th layer may be less valuable than the third MLP notch in this regime | Whether depth or width is the cheaper place to save bytes |
| `C1-E5` | `8L / MLP3 / DIM480 / batch 131072 / kv2` | Two mild trims together | Two small cuts may preserve score better than one aggressive cut | Whether combined light cuts dominate single hard cuts |

**Decision rule for C**

- if `C1-E1` or `C1-E3` is valid and close to `1.3899`, width has a clear path to a challenge-valid frontier
- if only the more aggressive trims become valid, the next question becomes whether optimization can claw back the lost score
- if none of the five get close to the raw winner, the width branch may be too byte-hungry in its current form

**Results so far**

- [`AL-20260329-011`](./experiments.tsv) (`C1-E1`, `9L / MLP3 / DIM480 / 131072 / kv2`) proved a mild global dim trim is enough to recover size validity, but not enough score. It landed at `1.3970` and 15.64 MB.
- [`AL-20260329-012`](./experiments.tsv) (`C1-E3`, `9L / MLP2 / 512 / 131072 / kv2`) produced a much stronger answer: `1.3838`, 14.73 MB, valid, and the new best frontier. This suggests the third MLP notch was the wrong place to spend bytes once the high-step regime was already in place.
- [`AL-20260329-013`](./experiments.tsv) (`C1-E2`, `9L / MLP3 / DIM448 / 131072 / kv2`) showed that stronger global shrinking can recover much more of the width branch than `DIM480` did. It finished at `1.3915` and 14.19 MB. Useful, but still clearly behind `9L / MLP2`.
- [`AL-20260329-014`](./experiments.tsv) (`C1-E4`, `8L / MLP3 / 512 / 131072 / kv2`) answered the “drop a layer instead” question. It finished at `1.3921`, but the artifact was still 16.29 MB, so the one-layer cut was not enough and was less effective than dropping one MLP notch.
- [`AL-20260329-015`](./experiments.tsv) (`C1-E5`, `8L / MLP3 / DIM480 / 131072 / kv2`) showed that two lighter cuts together are better than either fallback cut alone. It landed at `1.3906` and 14.42 MB. Valid, solid, but still not close enough to threaten the `9L / MLP2` winner.

**Current reading**

- structural trims matter more than uniform dim trims
- among global dim trims, `DIM448` is the first one that looks respectable
- one MLP-notch cut is currently dominating the dim-trim approach on both score and size
- dropping one layer alone is not the clean byte-saving move
- the combined-light-cuts backup is worth remembering, but the clear tranche result is that `9L / MLP2 / 131072` is the right survivor to optimize next

## T-20260329-D: Slim Winner Optimization Recovery

**Status:** completed

**Goal**  
Take the two real tranche-C survivors and ask whether optimization or step-recovery can improve them further, with most of the attention on the new `9L / MLP2` winner.

**Main question**  
Is the new valid winner still step-limited or slightly under-tuned, and can the best fallback line be made genuinely competitive?

**Fixed controls**

- one training shard
- `600s` training cap
- primary metric `final_int8_ttt_lora`
- tokenizer and validation semantics unchanged
- focus on the most plausible size-recovered shapes from tranche C

**Why this tranche exists**

- tranche C already identified the structural winner
- B1 and tranche C both suggest strong interactions between shape and step count
- the next informative question is now optimization, not another broad structural sweep

**Anchors**

- primary structural winner: [`AL-20260329-012`](./experiments.tsv) at `1.3838`, 14.73 MB
- fallback survivor: [`AL-20260329-015`](./experiments.tsv) at `1.3906`, 14.42 MB

**Planned experiments**

| ID | Shape | Goal | Hypothesis | What it teaches |
|---|---|---|---|---|
| `D1-E1` | `9L / MLP2 / batch 98304 / kv2` | More steps on the winner | The new winner may still be slightly step-limited inside 600s | Whether the frontier improves by cashing in more updates |
| `D1-E2` | `9L / MLP2 / batch 131072 / kv2 / MATRIX_LR=0.065` | Slightly higher matrix LR on the winner | The winner may want more aggressive matrix motion without changing its step count | Whether the remaining loss is optimizer mismatch rather than capacity |
| `D1-E3` | `9L / MLP2 / batch 98304 / kv2 / MATRIX_LR=0.065` | Interaction test on the winner | Extra steps and slightly higher LR may only work together | Whether the winner still has a two-knob optimization gain available |
| `D1-E4` | `8L / MLP3 / DIM480 / batch 98304 / kv2` | More steps on the best fallback | The smaller backup may look weak only because it has not fully cashed in its saved compute | Whether the fallback line deserves to stay alive |
| `D1-E5` | `8L / MLP3 / DIM480 / batch 98304 / kv2 / MATRIX_LR=0.065` | Interaction test on the fallback | The backup line may need both more steps and stronger updates to become interesting | Whether the backup is only one combo away from relevance |

**Results so far**

- [`AL-20260329-016`](./experiments.tsv) (`D1-E1`, `9L / MLP2 / 98304 / kv2`) landed at `1.3721` and 15.48 MB. This is a major frontier jump and strongly confirms that the tranche-C winner was still step-limited.
- [`AL-20260329-017`](./experiments.tsv) (`D1-E2`, `9L / MLP2 / 131072 / kv2 / MATRIX_LR=0.065`) landed at `1.3909` and 14.92 MB. This is a clear regression and says the big gain did not come from a simple LR mismatch at the old batch size.
- [`AL-20260329-018`](./experiments.tsv) (`D1-E3`, `9L / MLP2 / 98304 / kv2 / MATRIX_LR=0.065`) landed at `1.3786` and 15.55 MB. Still strong, but worse than `98304` alone, so the step win does not want this LR bump on top.
- [`AL-20260329-019`](./experiments.tsv) (`D1-E4`, `8L / MLP3 / DIM480 / 98304 / kv2`) landed at `1.3808` and 15.19 MB. This is a real rescue of the fallback line, but it still does not overtake the main `9L / MLP2 / 98304` frontier.
- [`AL-20260329-020`](./experiments.tsv) (`D1-E5`, `8L / MLP3 / DIM480 / 98304 / kv2 / MATRIX_LR=0.065`) landed at `1.3853` and 15.28 MB. The fallback line also rejected the LR bump.

**Current reading**

- extra steps matter more than we thought on the `9L / MLP2` line
- LR alone does not rescue the old-batch line
- the `98304` winner also does not improve with this LR bump
- the fallback line can be rescued somewhat with more steps, but it is still the backup, not the main frontier
- the LR bump helps neither survivor; the current best-tested story is “more steps yes, simple LR bump no”

**Outcome**

- best result from this tranche: [`AL-20260329-016`](./experiments.tsv) at `1.3721`
- main conclusion: both promising survivors were under-trained at `131072`, but neither wanted `MATRIX_LR=0.065`
- next pivot: return to architecture, compression-aware capacity, or output/residual mechanics, now using `9L / MLP2 / 98304 / kv2` as the operating frontier

**Decision rule for D**

- if one of the optimized slim candidates beats [`AL-20260329-004`](./experiments.tsv) while staying under the cap, it becomes the new valid frontier
- if optimization does not recover the slimmer candidates, the next tranche should pivot from width rescue toward compression-aware structural changes outside the width family

## T-20260329-E: Attention Geometry Audit

**Status:** completed

**Goal**  
Use the new frontier, [`AL-20260329-016`](./experiments.tsv), as the base model and ask whether the next gain comes from attention geometry rather than more optimizer fiddling.

**Main question**  
Is the current `9L / MLP2 / 98304 / q8-kv2 / QK_GAIN_INIT=1.5` setup the right attention shape, or is the frontier now limited by head geometry and attention sharpness?

**Why this tranche exists**


- tranche C already solved the main width-vs-size allocation question
- tranche D already solved the immediate optimization question
- the next compute-worthy pivot should target a different component family
- attention is the cleanest next family because several relevant knobs are already env-exposed and can be tested without speculative code edits

**Base controls**

- anchor shape: `9L / MLP2 / MODEL_DIM=512 / TRAIN_BATCH_TOKENS=98304`
- keep `MAX_WALLCLOCK_SECONDS=600`
- keep primary metric `final_int8_ttt_lora`
- keep tied embeddings and tokenizer/validation semantics unchanged
- keep default optimizer settings from the current winner unless the experiment explicitly changes them

**Anchor**

- [`AL-20260329-016`](./experiments.tsv) at `1.3721`, 15.48 MB

**Planned experiments**

| ID | Shape | Goal | Hypothesis | What it teaches |
|---|---|---|---|---|
| `E1` | `NUM_HEADS=4, NUM_KV_HEADS=2` | Test fewer, wider query heads | The current frontier may be over-fragmenting attention; wider heads could improve a small model's attention quality | Whether the model wants fewer, wider attention heads |
| `E2` | `NUM_HEADS=16, NUM_KV_HEADS=2` | Test more, narrower query heads | The current frontier may be under-headed; more heads could improve routing diversity | Whether the model wants more attention subspaces even at smaller head_dim |
| `E3` | `NUM_HEADS=8, NUM_KV_HEADS=4` | Reduce KV sharing | `kv2` may now be too aggressive on the stronger frontier; giving queries more distinct keys/values may help quality enough to justify the cost | Whether the main line is limited by over-shared KV projections |
| `E4` | `QK_GAIN_INIT=1.0` | Test flatter attention sharpness at init | The current `1.5` gain may make attention too sharp early in training on the step-rich frontier | Whether softer initial attention improves learning dynamics |
| `E5` | `QK_GAIN_INIT=2.0` | Test sharper attention at init | The current `1.5` gain may be too conservative and a stronger signal could help the model focus faster | Whether more aggressive initial attention helps the same frontier |

**Why these five are worth the compute**

- `E1` and `E2` bracket query-head geometry without changing model size dramatically
- `E3` directly tests whether the current `kv2` choice is now the bottleneck rather than the solution
- `E4` and `E5` bracket attention sharpness around the existing setting, so we learn whether the current init is too flat, too sharp, or already near the right point

**Decision rule for E**

- if `E1` or `E2` wins, the next tranche should keep the new head geometry fixed and probe neighboring attention settings
- if `E3` wins, the new frontier may have outgrown `kv2`, and future capacity planning should treat KV sharing as a first-class tradeoff again
- if `E4` or `E5` wins, attention sharpness was mis-set and we should tune around the winning side rather than touching architecture broadly
- if none win clearly, attention geometry is probably not the next bottleneck and the next tranche should pivot to output-path or residual-control simplification

**Results so far**

- [`AL-20260329-021`](./experiments.tsv) (`E1`, `q4/kv2`) landed at `1.3709` and 15.33 MB. This is a real frontier improvement and strongly suggests the current model wants fewer, wider query heads rather than the previous `q8` default.
- [`AL-20260329-022`](./experiments.tsv) (`E2`, `q16/kv2`) landed at `1.3968` and 14.40 MB. This is a clear regression and says the E1 win was directional evidence for wider heads, not a generic reward for changing head count.
- [`AL-20260329-023`](./experiments.tsv) (`E3`, `q8/kv4`) landed at `1.3766` and 15.31 MB. This is respectable and better than the old `q8/kv2` anchor, but it still does not overtake `q4/kv2`.
- [`AL-20260329-024`](./experiments.tsv) (`E4`, `QK_GAIN_INIT=1.0`) landed at `1.3777` and 15.45 MB. Competitive, but still clearly behind `q4/kv2`.
- [`AL-20260329-025`](./experiments.tsv) (`E5`, `QK_GAIN_INIT=2.0`) landed at `1.3743` and 15.48 MB. Better than the softer bracket, but still not enough to beat `q4/kv2`.

**Current reading**

- the frontier appears to be attention-geometry-sensitive after all
- the model appears to prefer fewer, wider query heads rather than more, narrower ones
- less KV sharing helps some, but not enough to beat the wider-head direction
- softer QK init is not enough to beat the wider-head direction
- sharper QK init is better than softer QK init, but still secondary to the head-geometry win

**Outcome**

- best result from this tranche: [`AL-20260329-021`](./experiments.tsv) at `1.3709`
- main conclusion: attention geometry is a real frontier lever, and the strongest gain in this tranche came from fewer, wider query heads (`q4/kv2`)
- secondary conclusion: less KV sharing and QK-gain tuning can help somewhat, but neither beat the `q4/kv2` change
- next pivot: use `9L / MLP2 / 98304 / q4-kv2` as the new anchor and test output-path or residual-control simplification next

## T-20260329-F: Output Path Audit

**Status:** completed

**Goal**  
Use the new frontier, [`AL-20260329-021`](./experiments.tsv), as the base model and ask whether the next gain comes from output-path expressivity or calibration rather than from more attention work.

**Main question**  
Is the current `9L / MLP2 / 98304 / q4-kv2` frontier leaving quality on the table because the output path is too constrained, miscalibrated, or learning at the wrong rate?

**Why this tranche exists**

- tranche C solved the main width-allocation question
- tranche D solved the immediate optimization question
- tranche E solved the first-pass attention question
- the output path is the next worthwhile component family because it is both underexplored and already env-exposed in several meaningful ways

**Base controls**

- anchor shape: `9L / MLP2 / MODEL_DIM=512 / TRAIN_BATCH_TOKENS=98304 / NUM_HEADS=4 / NUM_KV_HEADS=2`
- keep `MAX_WALLCLOCK_SECONDS=600`
- keep primary metric `final_int8_ttt_lora`
- keep tokenizer/validation semantics unchanged
- keep the current best optimizer defaults except when the experiment explicitly changes the output-path learning rate

**Anchor**

- [`AL-20260329-021`](./experiments.tsv) at `1.3709`, 15.33 MB

**Planned experiments**

| ID | Shape | Goal | Hypothesis | What it teaches |
|---|---|---|---|---|
| `F1` | `TIE_EMBEDDINGS=0` | Untie embeddings and output head | The frontier may need a more expressive output head than tying allows, and the extra bytes may still fit the cap | Whether output expressivity is a real bottleneck |
| `F2` | `LOGIT_SOFTCAP=20` | Tighten logit clipping | The current softcap may be too loose, letting logits become poorly calibrated | Whether stronger output regularization helps the frontier |
| `F3` | `LOGIT_SOFTCAP=40` | Relax logit clipping | The current softcap may be too restrictive and suppressing useful confidence | Whether the output path wants less saturation |
| `F4` | `TIE_EMBEDDINGS=0, HEAD_LR=0.004` | Slow down untied output-head updates | The untied output head may be learning too aggressively for the current frontier | Whether the untied output path wants a gentler head learning rate |
| `F5` | `TIE_EMBEDDINGS=0, HEAD_LR=0.012` | Speed up untied output-head updates | The untied output head may be under-updated relative to the rest of the model | Whether the output path wants stronger updates |

**Why these five are worth the compute**

- `F1` tests a qualitatively different hypothesis: expressivity versus byte cost
- `F2` and `F3` bracket output calibration around the current softcap so we can tell if the current setting is too strict, too loose, or already near the right point
- after `F1` won hard, the most compute-worthy use of the last two runs was to bracket the learning dynamics of the untied output head itself instead of staying on the weaker tied branch

**Decision rule for F**

- if `F1` wins, the next tranche should treat untied outputs as a serious frontier direction and optimize around their size budget
- if `F2` or `F3` wins, the next tranche should tune around the winning softcap side before touching architecture again
- if `F4` or `F5` wins, output-path learning dynamics are mis-set and deserve a small local optimization tranche
- if none win clearly, the output path is probably not the next bottleneck and the next pivot should move to residual-control simplification

**Results so far**

- [`AL-20260329-026`](./experiments.tsv) (`F1`, untied outputs) landed at `1.3614` and 15.78 MB. This is a major frontier jump and strongly suggests the current `q4/kv2` line was output-path-limited.
- [`AL-20260329-027`](./experiments.tsv) (`F2`, untied + `LOGIT_SOFTCAP=20`) landed at `1.3582` and 15.77 MB. This is another real improvement and says the untied frontier also wants tighter output clipping.
- [`AL-20260329-028`](./experiments.tsv) (`F3`, untied + `LOGIT_SOFTCAP=40`) landed at `1.3628` and 15.77 MB. Still strong, but clearly behind the tighter softcap.
- [`AL-20260329-029`](./experiments.tsv) (`F4`, untied + `HEAD_LR=0.004`) landed at `1.3596` and 15.74 MB. Competitive, but still not enough to beat the tighter-softcap winner.
- [`AL-20260329-030`](./experiments.tsv) (`F5`, untied + `LOGIT_SOFTCAP=20` + `HEAD_LR=0.012`) landed at `1.3564` and 15.80 MB. This is the best result in the tranche and shows the untied + softcap20 line still wanted a somewhat faster output-head learning rate.

**Current reading**

- output expressivity matters a lot more than the repo had previously explored
- untied outputs should now be treated as the new working assumption for this tranche
- tighter softcap already helps on top of untied outputs
- the softcap bracket is now directional: `20` beat `40`
- head-learning dynamics are also a real lever, and in this first bracket the useful direction was upward rather than downward
- the best-tested output-path recipe is now `untied + LOGIT_SOFTCAP=20 + HEAD_LR=0.012`

**Adaptive follow-up plan**

- after `F1` won clearly, the remaining four runs were upgraded to focus on the untied frontier itself:
- `F2`: `TIE_EMBEDDINGS=0, LOGIT_SOFTCAP=20`
- `F3`: `TIE_EMBEDDINGS=0, LOGIT_SOFTCAP=40`
- `F4`: `TIE_EMBEDDINGS=0, HEAD_LR=0.004`
- `F5`: `TIE_EMBEDDINGS=0, HEAD_LR=0.012`
- this is a better use of compute than continuing to bracket tied-output knobs after untied outputs already showed a large win

**Outcome**

- best result from this tranche: [`AL-20260329-030`](./experiments.tsv) at `1.3564`
- main conclusion: output path is a first-class frontier family on this challenge, not an afterthought
- secondary conclusion: the best-tested configuration now uses untied outputs, tighter clipping, and a somewhat faster output-head learning rate
- next pivot: either run one narrow local tranche around the untied output path, or switch families and test residual-control or skip-topology simplification from the new anchor

## T-20260330-G: Untied Output Local Calibration

**Status:** completed

**Goal**  
Test whether the current best line, [`AL-20260329-030`](./experiments.tsv), still has local output-path headroom nearby before we pivot to a colder component family.

**Main question**  
Is `untied + LOGIT_SOFTCAP=20 + HEAD_LR=0.012` already near the local optimum, or can a tighter nearby calibration beat it?

**Why this tranche exists**

- tranche F ended on an active win, not a flat result
- the most compute-worthy next move is to exploit the still-warm local neighborhood before abandoning the family
- this is also a clean test of the new manifest-driven harness, because the tranche is env-only and tightly scoped

**Base controls**

- anchor shape: `9L / MLP2 / MODEL_DIM=512 / TRAIN_BATCH_TOKENS=98304 / NUM_HEADS=4 / NUM_KV_HEADS=2 / TIE_EMBEDDINGS=0`
- keep `MAX_WALLCLOCK_SECONDS=600`
- keep primary metric `final_int8_ttt_lora`
- keep tokenizer and validation semantics unchanged
- keep all non-output-path optimizer settings at the current best defaults

**Anchor**

- [`AL-20260329-030`](./experiments.tsv) at `1.3564`, 15.80 MB

**Planned experiments**

| ID | Shape | Goal | Hypothesis | What it teaches |
|---|---|---|---|---|
| `G1` | `LOGIT_SOFTCAP=15, HEAD_LR=0.012` | Test stronger clipping | The current winner may still be slightly overconfident, so tighter clipping could help further | Whether the softcap optimum is below `20` |
| `G2` | `LOGIT_SOFTCAP=25, HEAD_LR=0.012` | Test slightly looser clipping | The current winner may already be slightly over-clipped, so relaxing to `25` could help | Whether `20` was too aggressive rather than just better than `40` |
| `G3` | `LOGIT_SOFTCAP=20, HEAD_LR=0.010` | Test a slightly slower head LR | The useful direction may be upward from `0.008`, but the true local optimum could still sit below `0.012` | Whether the head-LR optimum is slightly lower than the current best |
| `G4` | `LOGIT_SOFTCAP=20, HEAD_LR=0.016` | Test a slightly faster head LR | The untied head may still be under-updated | Whether the head-LR optimum is still above the current best |
| `G5` | `LOGIT_SOFTCAP=15, HEAD_LR=0.016` | Test the strongest local combo | If both “more clipping” and “faster head updates” are the right directions, they may stack | Whether the local output-path gains are additive rather than isolated |

**Why these five are worth the compute**

- `G1` and `G2` bracket the current softcap winner tightly instead of wasting compute on distant values we already weakened
- `G3` and `G4` bracket the current head-LR winner tightly for the same reason
- `G5` is the one justified combo test because it stacks the two strongest local “more” directions from tranche F

**Decision rule for G**

- if `G1` or `G2` wins, run one final narrow softcap confirmation before pivoting families
- if `G3` or `G4` wins, run one final head-LR confirmation before pivoting families
- if `G5` wins, the output path is still actively compounding and deserves one more mini-tranche
- if all five lose clearly, close the output-path family for now and pivot to residual-control or skip-topology simplification

**Results**

- [`AL-20260330-001`](./experiments.tsv) (`G1`, `LOGIT_SOFTCAP=15, HEAD_LR=0.012`) landed at `1.3564` and 15.81 MB. This effectively tied the current best at 4 decimals, but it was not clearly better and came in slightly larger.
- [`AL-20260330-002`](./experiments.tsv) (`G2`, `LOGIT_SOFTCAP=25, HEAD_LR=0.012`) landed at `1.3570` and 15.79 MB. Slightly looser clipping lost clear ground.
- [`AL-20260330-003`](./experiments.tsv) (`G3`, `LOGIT_SOFTCAP=20, HEAD_LR=0.010`) landed at `1.3599` and 15.80 MB. A slightly slower head LR was clearly worse.
- [`AL-20260330-004`](./experiments.tsv) (`G4`, `LOGIT_SOFTCAP=20, HEAD_LR=0.016`) landed at `1.3574` and 15.81 MB. A slightly faster head LR was also worse.
- [`AL-20260330-005`](./experiments.tsv) (`G5`, `LOGIT_SOFTCAP=15, HEAD_LR=0.016`) landed at `1.3565` and 15.82 MB. The strongest local combo stayed extremely close, but still did not beat the anchor.

**Current reading**

- the local output-path neighborhood is now much better mapped
- `LOGIT_SOFTCAP=20` still looks like the best local point we have tested
- `HEAD_LR=0.012` also still looks like the best local point we have tested
- there may still be noise-scale room in the family, but not enough to justify another pure scalar micro-sweep right away

**Outcome**

- best result from this tranche: no new winner; the anchor [`AL-20260329-030`](./experiments.tsv) remains best at `1.3564`
- main conclusion: the output family is still strong, but its local scalar neighborhood now looks mostly exhausted
- next pivot: move to residual-control or skip-topology simplification rather than another immediate output micro-sweep

## T-20260330-H: Residual Control Simplification

**Status:** active

**Goal**  
Test whether the current best line is carrying unnecessary residual-control and skip-path complexity, and whether a simpler residual system can match or beat it.

**Main question**  
Are `resid_mix`, learned residual scales, and learned skip weights still helping on the current frontier, or are they now overbuilt baggage?

**Why this tranche exists**

- tranche G mapped the local output-path neighborhood closely enough that another scalar micro-sweep is not the best use of compute
- residual controls and skip topology remain under-tested and are one of the most distinctive architecture families in this script
- this family can now be tested cleanly with env vars instead of a new code edit for every ablation

**Base controls**

- anchor shape: `9L / MLP2 / MODEL_DIM=512 / TRAIN_BATCH_TOKENS=98304 / NUM_HEADS=4 / NUM_KV_HEADS=2 / TIE_EMBEDDINGS=0 / LOGIT_SOFTCAP=20 / HEAD_LR=0.012`
- keep `MAX_WALLCLOCK_SECONDS=600`
- keep primary metric `final_int8_ttt_lora`
- keep tokenizer and validation semantics unchanged

**Anchor**

- [`AL-20260329-030`](./experiments.tsv) at `1.3564`, 15.80 MB

**Planned experiments**

| ID | Shape | Goal | Hypothesis | What it teaches |
|---|---|---|---|---|
| `H1` | `USE_RESID_MIX=0` | Remove learned input-stream mixing | `resid_mix` may be unnecessary now that the frontier is better trained and better calibrated | Whether the initial-embedding blend is still earning its complexity |
| `H2` | `USE_ATTN_SCALE=0, USE_MLP_SCALE=0` | Replace learned residual scales with unit residuals | Per-channel learned residual scales may be optimization baggage rather than useful expressivity | Whether the model wants simpler fixed residual addition |
| `H3` | `SKIP_MODE=unit` | Keep skip topology but remove learned skip weights | The skip structure may help, but the learned skip weights may be unnecessary | Whether skip learning matters more than skip existence |
| `H4` | `SKIP_MODE=off` | Remove skip topology entirely | The encoder/decoder-style skips may no longer be earning their keep | Whether the topology itself is the useful part or just historical carry-over |
| `H5` | `USE_RESID_MIX=0, USE_ATTN_SCALE=0, USE_MLP_SCALE=0, SKIP_MODE=unit` | Test a coherent simplification package | Several individually small simplifications may stack better than one-at-a-time ablations | Whether the residual family wants to simplify as a system, not just by one knob |

**Why these five are worth the compute**

- `H1`, `H2`, and `H3` isolate the three main learned-control families directly
- `H4` asks the stronger topology question instead of only the parameterization question
- `H5` is the one justified combo run because residual controls are likely to interact

**Decision rule for H**

- if `H1` wins, the next residual tranche should focus on input-stream mixing and residual routing
- if `H2` wins, the next residual tranche should simplify control parameterization further
- if `H3` wins but `H4` loses, keep skips but stop learning their weights
- if `H4` wins, skip topology is now a serious simplification target
- if `H5` wins, the residual family likely wants a broader simplification pass
- if all five lose clearly, residual controls are probably not the next bottleneck and we should pivot elsewhere

**Results so far**

- [`AL-20260330-006`](./experiments.tsv) (`H1`, `USE_RESID_MIX=0`) landed at `1.3763` and 15.82 MB. This is a clear regression from the `1.3564` anchor, so learned input-stream mixing is still materially helping on the current frontier.
- [`AL-20260330-007`](./experiments.tsv) (`H2`, `USE_ATTN_SCALE=0, USE_MLP_SCALE=0`) landed at `1.3666` and 15.64 MB. This also regressed clearly, though less than `H1`, so the learned residual scales still appear to be earning their keep.
- [`AL-20260330-008`](./experiments.tsv) (`H3`, `SKIP_MODE=unit`) landed at `1.3568` and 15.80 MB. This nearly tied the `1.3564` anchor, so the skip topology may matter more than learning per-skip weights.
- [`AL-20260330-009`](./experiments.tsv) (`H4`, `SKIP_MODE=off`) landed at `1.3610` and 15.81 MB. This lost much more clearly than `H3`, so the skip topology itself is useful and the learned skip weighting is the more plausible simplification target.
- [`AL-20260330-010`](./experiments.tsv) (`H5`, `USE_RESID_MIX=0, USE_ATTN_SCALE=0, USE_MLP_SCALE=0, SKIP_MODE=unit`) landed at `1.3807` and 15.65 MB. The full simplification package failed badly, so there is no evidence for a hidden “all together is better” interaction.

**Current reading**

- the residual family is not the next frontier family
- `resid_mix` is clearly valuable
- learned residual scales are also useful
- the skip topology itself is useful
- the only near-live simplification result is that unit skip weights nearly match learned skip weights

**Outcome**

- best result from this tranche: no new winner; the anchor [`AL-20260329-030`](./experiments.tsv) remains best at `1.3564`
- main conclusion: generic residual simplification is not the next breakthrough path, but the model likely does not need learned skip weights to get most of the skip benefit
- next pivot: move to a bold architecture tranche, most likely latent-KV attention compression or a hybrid recurrent mixer

## T-20260330-I: Latent-KV Attention Audit

**Status:** completed

**Goal**  
Test whether the current attention block is over-spending on full K/V structure, and whether a latent-KV bottleneck can preserve quality while buying speed or size headroom.

**Main question**  
Can compressed K/V structure keep most of the useful attention behavior under the same `600s` training budget?

**Why this tranche exists**

- tranche H mostly closed the generic residual-simplification path
- the next worthwhile step is to change the block, not prune one more scalar
- latent-KV attention is a bold but still interpretable architecture change

**Base controls**

- anchor shape: `9L / MLP2 / MODEL_DIM=512 / TRAIN_BATCH_TOKENS=98304 / NUM_HEADS=4 / NUM_KV_HEADS=2 / TIE_EMBEDDINGS=0 / LOGIT_SOFTCAP=20 / HEAD_LR=0.012`
- keep `MAX_WALLCLOCK_SECONDS=600`
- keep primary metric `final_int8_ttt_lora`
- keep tokenizer and validation semantics unchanged
- keep `MIXER_LAYERS` disabled for tranche I

**Anchor**

- [`AL-20260329-030`](./experiments.tsv) at `1.3564`, 15.80 MB

**Planned experiments**

| ID | Shape | Goal | Hypothesis | What it teaches |
|---|---|---|---|---|
| `I1` | `LATENT_KV_LAYERS=0,1,2,3,4,5,6,7,8`, `LATENT_KV_DIM=128` | Mild latent-KV compression across all layers | There is real redundancy in the full K/V path, and mild compression may preserve quality almost intact | Whether latent-KV is viable at all |
| `I2` | `LATENT_KV_LAYERS=0,1,2,3,4,5,6,7,8`, `LATENT_KV_DIM=64` | Stronger latent-KV compression across all layers | The frontier may tolerate much more K/V compression than we expect | Whether the idea scales beyond a mild bottleneck |
| `I3` | `LATENT_KV_LAYERS=5,6,7,8`, `LATENT_KV_DIM=64` | Compress only the upper layers | Later layers may tolerate K/V compression better than earlier ones | Whether late compression is the cleanest placement |
| `I4` | `LATENT_KV_LAYERS=0,1,2,3`, `LATENT_KV_DIM=64` | Compress only the lower layers | Early layers may be the cheaper place to simplify attention while preserving later fidelity | Whether early compression is the cleanest placement |
| `I5` | `NUM_LAYERS=10`, `LATENT_KV_LAYERS=0,1,2,3,4,5,6,7,8,9`, `LATENT_KV_DIM=64` | Reinvest the stronger latent-KV savings into more depth | The stronger compression line may only make sense if its saved budget is converted into one more layer | Whether stronger compressed attention plus extra depth beats the plain transformer frontier |

**Why these five are worth the compute**

- every run tests a distinct mechanism-level question, not a scalar retune
- the tranche asks both whether latent-KV works and where it works
- `I5` tests the most important second-order question: what to do with the savings if the idea works

**Results so far**

- [`AL-20260330-011`](./experiments.tsv) (`I1`, all-layer `LATENT_KV_DIM=128`) landed at `1.3718` and 15.83 MB. The run trained cleanly, stayed under the size cap, and kept roughly the same step budget as the anchor, but it still lost clearly on quality. It also took `801s` for TTT eval, so the first latent-KV form did not buy a cleaner evaluation path either.
- [`AL-20260330-012`](./experiments.tsv) (`I2`, all-layer `LATENT_KV_DIM=64`) landed at `1.3865` and 14.73 MB. It bought slightly more steps and much more artifact headroom than `I1`, but it lost further on quality and still took `795s` in TTT eval.
- [`AL-20260330-013`](./experiments.tsv) (`I3`, upper-only `LATENT_KV_DIM=64`) landed at `1.3685` and 15.32 MB. This was the best latent-KV run in the tranche: clearly better than both all-layer variants, but still well behind the `1.3564` frontier.
- [`AL-20260330-014`](./experiments.tsv) (`I4`, lower-only `LATENT_KV_DIM=64`) landed at `1.3737` and 15.30 MB. This improved over all-layer latent64 but trailed the upper-only placement, so the upper stack is the cleaner place to localize compression.
- [`AL-20260330-015`](./experiments.tsv) (`I5`, `10L` + all-layer `LATENT_KV_DIM=64`) landed at `1.3883` and 15.84 MB. Reinvesting the stronger all-layer compression into another layer did not rescue the family; it lost too many steps and still stayed weak on quality.

**Current reading**

- latent-KV is technically viable in this codebase: the model trains, serializes, quantizes, and evaluates without breaking challenge constraints
- but mild all-layer latent-KV is not an immediate drop-in win
- stronger all-layer compression makes the loss worse, not better
- the family is placement-sensitive: upper-only compression is meaningfully better than lower-only or all-layer compression
- reinvesting naive all-layer compression into extra depth does not rescue it

**Outcome**

- no new frontier run; [`AL-20260329-030`](./experiments.tsv) remains best at `1.3564`
- best latent-KV run: [`AL-20260330-013`](./experiments.tsv) at `1.3685`
- main conclusion: latent-KV is not a dead idea, but its first useful regime is localized upper-layer compression, not full-stack compression
- next pivot: move to a new bold architecture family unless we decide the upper-only latent-KV line is worth a second-generation redesign

## T-20260330-J: Hybrid Sequence Mixer Audit

**Status:** active

**Goal**  
Test whether some transformer attention layers can be replaced by a cheaper state-space-inspired sequence mixer without giving back too much quality under the same `600s` budget.

**Main question**  
Does every layer really need full attention, or can a hybrid stack keep most of the quality while using a cheaper mixer in selected layers?

**Why this tranche exists**

- tranche H mostly closed the generic residual-simplification path
- the next worthwhile move is architectural, not scalar
- a hybrid mixer tests a different modeling principle rather than another local transformer retune

**Base controls**

- anchor shape: `9L / MLP2 / MODEL_DIM=512 / TRAIN_BATCH_TOKENS=98304 / NUM_HEADS=4 / NUM_KV_HEADS=2 / TIE_EMBEDDINGS=0 / LOGIT_SOFTCAP=20 / HEAD_LR=0.012`
- keep `MAX_WALLCLOCK_SECONDS=600`
- keep primary metric `final_int8_ttt_lora`
- keep tokenizer and validation semantics unchanged
- keep `MIXER_DIM=256`, `MIXER_KERNEL=4` fixed across the tranche

**Anchor**

- [`AL-20260329-030`](./experiments.tsv) at `1.3564`, 15.80 MB

**Planned experiments**

| ID | Shape | Goal | Hypothesis | What it teaches |
|---|---|---|---|---|
| `J1` | `MIXER_LAYERS=3,4,5` | Replace the middle three attention layers | The middle of the stack may be the safest place to use a cheaper sequence mixer | Whether the hybrid idea is viable at all |
| `J2` | `MIXER_LAYERS=2,3,4,5,6` | Replace the middle five layers | More of the stack may be replaceable than we expect | Whether the hybrid win, if any, scales beyond a light swap |
| `J3` | `MIXER_LAYERS=1,3,5,7` | Alternate attention and mixer layers | Alternating global routing and cheaper local mixing may work better than block replacement | Whether interleaving beats contiguous replacement |
| `J4` | `MIXER_LAYERS=0,1,2,3` | Replace the lower four layers | Early layers may tolerate cheaper mixing while later attention stays full-fidelity | Whether lower-layer replacement is the cleaner placement |
| `J5` | `MIXER_LAYERS=5,6,7,8` | Replace the upper four layers | Later layers may be the better place to save attention cost | Whether upper-layer replacement is the cleaner placement |

**Why these five are worth the compute**

- every run tests a distinct placement theory rather than a scalar retune
- the tranche asks a real architectural question: where, if anywhere, can attention be replaced?
- if all five lose, that still teaches us something strong about the current frontier

## Queued Manifest Roadmap

These are the next six tranche manifests prepared for the lab. All six are now runnable with the current codebase. The sequential program manifest is [`P-20260330-JO`](./program_manifests/20260330-J-to-O.json).

### T-20260330-J - Hybrid Sequence Mixer Audit

- Manifest: [`20260330-J-hybrid-sequence-mixer.json`](./tranche_manifests/20260330-J-hybrid-sequence-mixer.json)
- Status: completed
- Outcome: clear win; [`AL-20260330-104`](./experiments.tsv) is the new best valid frontier at `1.3488`, and [`AL-20260330-103`](./experiments.tsv) also kept the family alive
- Why it is worthy:
- it tests whether some layers can stop using full attention entirely
- it is the cleanest bold replacement family already supported by the repo

### T-20260330-K - Output Head Architecture Audit

- Manifest: [`20260330-K-output-head-architecture.json`](./tranche_manifests/20260330-K-output-head-architecture.json)
- Status: completed
- Outcome: failed as a frontier family on the current backbone; the dense untied head still wins clearly
- Why it is worthy:
- output path has already produced some of the largest gains in the project
- the next good question is output-head architecture, not more scalar tuning

### T-20260330-L - Local-Global Attention Split

- Manifest: [`20260330-L-local-global-attention.json`](./tranche_manifests/20260330-L-local-global-attention.json)
- Status: blocked
- Outcome: [`AL-20260330-111`](./experiments.tsv) crashed before the first metric, so the family is unresolved and needs a debug tranche
- Why it is worthy:
- attention geometry already mattered a lot
- the next structural question is whether global attention is needed in every layer

### T-20260330-M - Skip and Residual Redesign

- Manifest: [`20260330-M-skip-residual-redesign.json`](./tranche_manifests/20260330-M-skip-residual-redesign.json)
- Status: completed
- Outcome: clear secondary win; [`AL-20260330-120`](./experiments.tsv) reached `1.3534` and shared scalar skip gates were also independently useful
- Why it is worthy:
- tranche H showed deletion is mostly the wrong question
- the live path is redesigning routing, not removing it

### T-20260330-N - Mechanism-Specific Learning Rates

- Manifest: [`20260330-N-mechanism-specific-lrs.json`](./tranche_manifests/20260330-N-mechanism-specific-lrs.json)
- Status: completed
- Outcome: supportive but not transformative; several splits nearly tied the old frontier, but none challenged the new hybrid-mixer best
- Why it is worthy:
- some architecture failures may actually be optimization failures
- this gives future bold mechanisms a fairer training regime

### T-20260330-O - Quantization-Aware Warmdown

- Manifest: [`20260330-O-quantization-aware-warmdown.json`](./tranche_manifests/20260330-O-quantization-aware-warmdown.json)
- Status: completed
- Outcome: mostly wrong; broader warmdown and stabilization variants hurt, while head-only cooldown only managed a near-tie to the old pre-program best
- Why it is worthy:
- the submitted model is compressed, not the raw fp model
- challenge-specific end-of-training behavior may matter more than generic schedule quality

## T-20260331-P: Hybrid Mixer Refinement

**Status:** completed

**Goal**  
Refine the current best family, [`AL-20260330-104`](./experiments.tsv), instead of treating the first hybrid-mixer win as fully mapped.

**Main question**  
Is the lower-four-mixer winner best because of the general idea, or is there still clear headroom in the exact mixer placement, width, and kernel?

**Fixed controls**

- start from the current best valid frontier shape
- `9` layers, `MODEL_DIM=512`, `MLP_MULT=2`
- `TRAIN_BATCH_TOKENS=98304`
- `NUM_HEADS=4`, `NUM_KV_HEADS=2`
- untied dense output path, `LOGIT_SOFTCAP=20`, `HEAD_LR=0.012`
- primary metric `final_int8_ttt_lora`

**Anchor**

- [`AL-20260330-104`](./experiments.tsv) at `1.3488`

**Planned experiments**

| ID | Shape | Goal | Hypothesis | Why worthy |
|---|---|---|---|---|
| `P1` | lower `3` mixer layers | Test whether the current winner is over-replacing attention | Three lower mixers may preserve most of the gain while recovering some lost expressivity | This tells us whether the win is broad or very placement-specific |
| `P2` | lower `5` mixer layers | Test whether the frontier still wants even less lower-stack attention | The lower stack may tolerate one more mixer layer than the current winner uses | This is the cleanest way to test whether the winner is still too conservative |
| `P3` | lower `4` mixers, `MIXER_DIM=192` | Test whether the current mixer itself is overbuilt | A smaller mixer may preserve quality while saving bytes and compute | This asks whether the new mechanism is bigger than it needs to be |
| `P4` | lower `4` mixers, `MIXER_DIM=320` | Test whether the current mixer is underpowered | A richer mixer may exploit the lower-stack replacement idea better than the current width | This checks whether the current winner is capacity-limited inside the mixer |
| `P5` | lower `4` mixers, `MIXER_KERNEL=6` | Test whether the lower-stack mixer wants a wider local receptive field | The current kernel may be too narrow, and a slightly wider local window may improve the sequence-mixing trade | This probes mechanism quality, not just placement |

**Decision rule**

- if `P1` wins, the current lower-four placement is likely too aggressive
- if `P2` wins, the frontier wants even less attention in the lower stack
- if `P3` wins, the hybrid idea is even more efficient than the current winner suggests
- if `P4` or `P5` wins, the mixer mechanism itself still has local headroom

**Results**

- [`AL-20260331-001`](./experiments.tsv) (`P1`, lower three mixers) improved the frontier to `1.3453` while staying valid at 15.04 MB.
- [`AL-20260331-002`](./experiments.tsv) (`P2`, lower five mixers) regressed slightly to `1.3502`, despite better speed and size.
- [`AL-20260331-003`](./experiments.tsv) (`P3`, lower four with `MIXER_DIM=192`) stayed exactly flat to the old winner at `1.3488` while saving bytes.
- [`AL-20260331-004`](./experiments.tsv) (`P4`, lower four with `MIXER_DIM=320`) became the new best valid run at `1.3451`.
- [`AL-20260331-005`](./experiments.tsv) (`P5`, lower four with `MIXER_KERNEL=6`) also improved on the old best to `1.3477`.

**Reading**

- the hybrid family is real and still has headroom
- lower-stack replacement is robust rather than brittle
- the best current direction is not “more mixer layers at any cost”; it is a stronger mixer inside the same lower-four placement
- the lower-five run says the frontier still wants some lower-stack attention

**Outcome**

- best result from this tranche: [`AL-20260331-004`](./experiments.tsv) at `1.3451`
- main conclusion: the mixer mechanism itself was still underpowered; widening it helped more than changing the lower-stack placement broadly
- next pivot: test whether the cheap-routing package stacks with the stronger hybrid winner, while keeping AttnRes-lite active as the new bold routing family

## T-20260331-Q: AttnRes-lite Dynamic Depth Routing

**Status:** completed

**Goal**  
Test whether fixed residual and skip routing should give way to a small, input-dependent depth-routing module over a few earlier layer states.

**Main question**  
If skip topology matters and routing redesign matters, is the next step to let the model choose which earlier depth representations matter for the current token?

**Fixed controls**

- use the current best hybrid-mixer frontier as the anchor family
- keep the lower-stack mixer winner fixed unless the candidate explicitly tests a justified combo
- primary metric `final_int8_ttt_lora`

**Anchor**

- [`AL-20260331-004`](./experiments.tsv) is the current best valid frontier

**Planned experiments**

| ID | Shape | Goal | Hypothesis | Why worthy |
|---|---|---|---|---|
| `Q1` | late-layer AttnRes-lite, 3 sources | Test the cheapest viable depth-routing version | Even a tiny dynamic selector can beat fixed skip routing | Lowest-risk test of the whole family |
| `Q2` | late-layer AttnRes-lite, 4 sources | Add one more depth option | One extra earlier-layer candidate may be enough to improve routing quality materially | Tests whether the family is source-limited |
| `Q3` | top-2-layer AttnRes-lite | Ask whether only the final refinement layers need dynamic depth routing | The benefit may come mostly at the very top of the stack | Cheap way to isolate where dynamic routing matters most |
| `Q4` | late-layer AttnRes-lite with shared routing | Use cheaper shared gating instead of richer per-token freedom | Coarse dynamic routing may be enough, and cheaper than a richer selector | Distinguishes mechanism value from implementation heaviness |
| `Q5` | AttnRes-lite + cheap routing combo | Test whether dynamic routing stacks with the best fixed-routing redesign | Cheap fixed routing and dynamic depth routing may be complementary rather than redundant | This is the only justified combo run because routing components likely interact |

**Decision rule**

- if `Q1` or `Q2` wins, dynamic depth routing is immediately a live frontier family
- if only `Q3` wins, the mechanism should stay confined to the top of the stack
- if `Q4` stays close, the family may be useful in a cheaper form
- if only `Q5` wins, dynamic routing likely wants to be part of a package, not a standalone change

**Results**

- [`AL-20260331-006`](./experiments.tsv) (`Q1`, late-layer AttnRes-lite with three sources) crashed before producing a metric because TorchDynamo rejected a Python `id()` call inside the candidate-routing helper.
- [`AL-20260331-007`](./experiments.tsv) (`Q2`, late-layer AttnRes-lite with four sources) failed badly at `1.5043`.
- [`AL-20260331-008`](./experiments.tsv) (`Q3`, AttnRes-lite only on the top two layers) was much cleaner at `1.3499`, but still did not beat the frontier.
- [`AL-20260331-009`](./experiments.tsv) (`Q4`, shared routing) was also catastrophic at `1.5133`.
- [`AL-20260331-010`](./experiments.tsv) (`Q5`, AttnRes-lite plus cheap-routing package) stayed in the same bad regime at `1.5016`.

**Reading**

- broad late-layer dynamic depth routing is the wrong shape on this frontier
- the problem is not just routing cost, because the cheaper shared-routing version also failed badly
- the problem is not solved by combining AttnRes-lite with the best fixed cheap-routing package
- the only remotely live hint is top-of-stack-only routing, which stayed near the old hybrid anchor but still lost to the current best

**Outcome**

- no new winner from this tranche
- main conclusion: AttnRes-lite as implemented is mostly falsified on the current frontier, except for a faint top-of-stack-only hint that is not yet strong enough to prioritize over better live families

## T-20260331-R: Architecture-Specific Schedules

**Status:** completed

**Goal**  
Test whether the strongest new architectures are being judged with the wrong training tail, especially near quantization and export.

**Main question**  
Do the current hybrid winner and similar bold architectures want a different warmdown, stabilization, or final head/backbone cooldown than the old baseline schedule?

**Fixed controls**

- start from the current best hybrid-mixer winner
- keep architecture fixed unless the candidate explicitly tests a schedule interaction
- primary metric `final_int8_ttt_lora`

**Anchor**

- [`AL-20260330-104`](./experiments.tsv) at `1.3488`

**Planned experiments**

| ID | Shape | Goal | Hypothesis | Why worthy |
|---|---|---|---|---|
| `R1` | short mild warmdown | Test whether the hybrid winner wants a slightly gentler tail | A small warmdown may improve the compressed model without freezing learning too early | This is the cleanest first schedule probe on the new winner |
| `R2` | longer warmdown | Test whether the winning architecture wants a much calmer finish | The hybrid backbone may be more quantization-sensitive than the old baseline | This brackets the warmdown question instead of guessing one level |
| `R3` | final stabilization only | Test whether a brief export-focused cooldown helps more than a broad warmdown | The right intervention may be a short final settling phase, not a long decay | This isolates stabilization from full warmdown |
| `R4` | head-focused final cooldown | Test whether the output path remains the fragile part even on the new hybrid frontier | The compressed model may benefit if the head cools differently from the backbone | This directly follows the near-tie behavior from the earlier quantization tranche |
| `R5` | mild warmdown + head-focused cooldown | Test the strongest justified schedule combo | A small architecture-aware tail may need both a gentle global decay and extra head control | This is the one schedule combo worth paying for after the single-factor bracket |

**Decision rule**

- if `R1` or `R2` wins, the new backbone wants a different global tail
- if `R3` wins, final stabilization is the real mechanism
- if `R4` or `R5` wins, the output path remains the main quantization-sensitive surface even on the hybrid winner

**Results**

- [`AL-20260331-011`](./experiments.tsv) (`R1`, mild warmdown) regressed to `1.3591`.
- [`AL-20260331-012`](./experiments.tsv) (`R2`, longer warmdown) regressed further to `1.3649`.
- [`AL-20260331-013`](./experiments.tsv) (`R3`, final stabilization only) stayed respectable at `1.3502` but still lost.
- [`AL-20260331-014`](./experiments.tsv) (`R4`, head-focused cooldown) nearly tied the old hybrid winner at `1.3489`, but did not beat it.
- [`AL-20260331-015`](./experiments.tsv) (`R5`, mild tail plus head cooldown) still lost at `1.3517`.

**Reading**

- the stronger hybrid backbone still does not want a broad colder tail
- the only live schedule hint is output-path-sensitive cooldown
- even that looks like support, not a new frontier lever

**Outcome**

- no new winner from this tranche
- main conclusion: broad warmdown remains the wrong story even on the improved hybrid backbone, and schedule work should stay secondary to architecture

## T-20260331-S: Hybrid Mixer + Cheap Routing Combo

**Status:** completed

**Goal**  
Test whether the stronger hybrid-mixer winner and the cheap-routing redesign are complementary or redundant.

**Anchor**

- [`AL-20260331-004`](./experiments.tsv) at `1.3451`

**Results**

- [`AL-20260331-016`](./experiments.tsv) (`S1`, full cheap-routing package) improved to `1.3434`.
- [`AL-20260331-017`](./experiments.tsv) (`S2`, shared scalar skip gates only) became the new best valid run at `1.3429`.
- [`AL-20260331-018`](./experiments.tsv) (`S3`, scalar `resid_mix` only) also improved to `1.3437`.
- [`AL-20260331-019`](./experiments.tsv) (`S4`, shared residual scales only) lost at `1.3465`.
- [`AL-20260331-020`](./experiments.tsv) (`S5`, full cheap-routing package plus head cooldown) stayed strong at `1.3445`, but did not beat the simpler skip-gate variant.

**Reading**

- the hybrid winner and routing redesign are genuinely complementary
- the main compatible piece is shared skip gating
- scalar `resid_mix` also stacks cleanly, but shared residual scales alone do not
- adding schedule help on top of the combo was unnecessary in this first pass

**Outcome**

- new best valid run: [`AL-20260331-017`](./experiments.tsv) at `1.3429`
- main conclusion: the frontier now prefers the widened lower-four-mixer backbone plus shared scalar skip gates

## T-20260331-T: Local-Global Attention Recovery

**Status:** completed

**Goal**  
After fixing the local-attention backend bug, test whether the remaining attention layers in the hybrid winner still need full global context.

**Anchor**

- [`AL-20260331-004`](./experiments.tsv) at `1.3451`

**Runtime note**

- [`AL-20260331-021`](./experiments.tsv) crashed because the trainer had no valid SDP backend for masked local attention. That was later fixed by enabling a math-SDP fallback when local attention is active.

**Results**

- [`AL-20260331-022`](./experiments.tsv) (`T2`, four upper local layers, window `256`) failed badly at `1.3813`.
- [`AL-20260331-023`](./experiments.tsv) (`T3`, top two local layers, window `512`) was the cleanest repaired run at `1.3545`, but still lost clearly.
- [`AL-20260331-024`](./experiments.tsv) (`T4`, alternating upper local layers, window `256`) landed at `1.3603`.
- [`AL-20260331-025`](./experiments.tsv) (`T5`, top two local layers, window `128`) landed at `1.3582`.

**Reading**

- local attention on the surviving attention stack is mostly the wrong trade on this backbone
- the best repaired run was the widest top-only window, which says the family is not completely random
- but nothing in this tranche threatened the frontier

**Outcome**

- no new winner from this tranche
- main conclusion: plain local-window replacement of the remaining attention layers should be parked unless a new architecture theory emerges

## T-20260331-U: Compression-Native Backbone Audit

**Status:** completed

**Goal**  
Test whether low-rank factorization can act as a first compression-native architecture branch on top of the hybrid winner.

**Anchor**

- [`AL-20260331-004`](./experiments.tsv) at `1.3451`

**Results**

- [`AL-20260331-026`](./experiments.tsv) (`U1`, factorized attention rank `128`) landed at `1.3689`.
- [`AL-20260331-027`](./experiments.tsv) (`U2`, factorized attention rank `64`) landed at `1.3702`.
- [`AL-20260331-028`](./experiments.tsv) (`U3`, factorized MLP rank `256`) collapsed to `1.8307`.
- [`AL-20260331-029`](./experiments.tsv) (`U4`, factorized MLP rank `128`) collapsed further to `2.1075`.
- [`AL-20260331-030`](./experiments.tsv) (`U5`, combined attention+MLP factorization) landed at `1.4604`.

**Reading**

- naive low-rank factorization is not the right compression-native mechanism here
- attention factorization preserves more than MLP factorization, but still loses too much quality
- the MLP path is especially intolerant of this style of factorization

**Outcome**

- no new winner from this tranche
- main conclusion: compression-native design remains important, but the next branch should not be another simple rank sweep

## T-20260331-V: Top-Only Dynamic Depth Routing Revisit

**Status:** completed

**Goal**  
Revisit AttnRes-lite only in the one region where the earlier tranche gave a weak positive signal: the very top of the stack.

**Anchor**

- [`AL-20260331-004`](./experiments.tsv) at `1.3451`

**Results**

- [`AL-20260331-031`](./experiments.tsv) (`V1`, final layer only, two sources) improved to `1.3440`.
- [`AL-20260331-032`](./experiments.tsv) (`V2`, top two layers, two sources) improved to `1.3443`.
- [`AL-20260331-033`](./experiments.tsv) (`V3`, top-two shared-scalar routing) regressed to `1.3526`.
- [`AL-20260331-034`](./experiments.tsv) (`V4`, final layer, three sources) regressed to `1.3465`.
- [`AL-20260331-035`](./experiments.tsv) (`V5`, top-two routing plus shared skip gates) reached `1.3433`.

**Reading**

- top-only dynamic routing is a real secondary family
- the revived family wants token-wise routing and very few source states
- shared skip gates are complementary with the narrow top-only router
- even the best result still trails the simpler skip-gate winner from tranche `S`

**Outcome**

- best result from this tranche: [`AL-20260331-035`](./experiments.tsv) at `1.3433`
- main conclusion: dynamic depth routing is only viable when it is kept extremely narrow and top-heavy

## T-20260331-W: Broad MLP Family Audit

**Status:** completed

**Goal**  
Test whether the current `relu^2` MLP is actually the right broad activation family on the hybrid frontier.

**Anchor**

- [`AL-20260331-004`](./experiments.tsv) at `1.3451`

**Results**

- [`AL-20260331-036`](./experiments.tsv) (`W1`, `relu^2`) reproduced the frontier class at `1.3449`.
- [`AL-20260331-037`](./experiments.tsv) (`W2`, `ReLU`) regressed to `1.3619`.
- [`AL-20260331-038`](./experiments.tsv) (`W3`, `SiLU`) regressed to `1.3605`.
- [`AL-20260331-039`](./experiments.tsv) (`W4`, `GELU`) regressed to `1.3607`.
- [`AL-20260331-040`](./experiments.tsv) (`W5`, gated `SiLU`) stayed competitive at `1.3460`, but broke the size cap at `19.06 MB`.

**Reading**

- the broad MLP family still belongs to `relu^2`
- smooth activation replacements are clearly worse on this backbone
- the only live alternative is gated `SiLU`, but its size cost makes it a later compression/capacity problem rather than an immediate winner

**Outcome**

- no new winner from this tranche
- main conclusion: keep `relu^2` as the broad MLP family anchor and only revisit gating under tighter size control

## T-20260331-X: Polynomial MLP Audit

**Status:** completed

**Goal**  
Probe deeper inside the polynomial MLP family rather than only comparing broad activation families.

**Anchor**

- [`AL-20260331-041`](./experiments.tsv) at `1.3451`

**Results**

- [`AL-20260331-042`](./experiments.tsv) (`X2`, `relu^3`) regressed badly to `1.3630`.
- [`AL-20260331-043`](./experiments.tsv) (`X3`, `relu + 0.5 relu^2`) was the best non-anchor result at `1.3444`.
- [`AL-20260331-044`](./experiments.tsv) (`X4`, `relu^2` plus cubic term) regressed to `1.3532`.
- [`AL-20260331-045`](./experiments.tsv) (`X5`, norm-before-square) stayed near the frontier at `1.3456`, but still lost.

**Reading**

- cubic-heavy polynomial structure is the wrong direction
- the only live alternative inside this family is mixing linear and quadratic behavior
- normalization before squaring is respectable, but not obviously better than the plain baseline

**Outcome**

- no new winner from this tranche
- main conclusion: if the MLP family is revisited later, use `relu + quadratic` as the live polynomial branch and ignore cubic-heavy variants

## Next Bold Question Queue

These are the next architecture tranches after `S` through `X`. They are now **runnable** and live under [`tranche_manifests/`](./tranche_manifests/).

Recommended order:

1. [`T-20260401-Y`](./tranche_manifests/20260401-Y-mlp-structure-minimalism.json) MLP Structure Minimalism
2. [`T-20260401-Z`](./tranche_manifests/20260401-Z-block-uniformity-audit.json) Block Uniformity Audit
3. [`T-20260401-AB`](./tranche_manifests/20260401-AB-compression-native-sharing.json) Compression-Native Sharing Audit
4. [`T-20260401-AA`](./tranche_manifests/20260401-AA-upper-attention-decomposition.json) Upper Attention Decomposition Audit

Top-level chain:

- [`P-20260401-YZABAA`](./program_manifests/20260401-YZABAA.json)

### T-20260401-Y: MLP Structure Minimalism

**Status:** runnable

**Goal**  
Question the dense expand-project FFN assumption directly instead of only comparing activation families.

**Main question**  
Does every block really need a full dense project-up then project-down MLP, or can some of that structure be removed or replaced by a lighter tokenwise computation?

**Why this is next-worthy**

- it attacks one of the biggest unproven assumptions in the whole model
- the MLP is a major parameter and compute sink
- our current findings tell us which nonlinear families are live, but not whether the FFN structure itself is overbuilt

**Planned experiments**

- `Y1`: no-expand in-place quadratic MLP everywhere
- `Y2`: half-width dense MLP everywhere
- `Y3`: structured linear-plus-quadratic no-expand MLP
- `Y4`: light lower-stack MLP, full upper-stack MLP
- `Y5`: full top-stack MLP, lighter MLP elsewhere

**Key interpretation**

- if `Y1` or `Y3` survives, full expand-project FFNs are overbuilt
- if `Y4` or `Y5` survives, MLP richness is a depth-specific resource

### T-20260401-Z: Block Uniformity Audit

**Status:** runnable

**Goal**  
Question the assumption that every layer deserves the same full block recipe.

**Main question**  
Why should every block contain both token mixing and an FFN, and can some stages use mixer-only or MLP-light blocks instead?

**Why this is next-worthy**

- it is one of the cleanest childlike architectural questions available
- the repo already rewards strong depth specialization
- it attacks compute uniformity rather than just another local knob

**Planned experiments**

- `Z1`: lower two mixer layers become mixer-only blocks
- `Z2`: lower four mixer layers become mixer-only blocks
- `Z3`: alternate standard and mixer-only blocks in the lower stack
- `Z4`: lower stack uses MLP-light blocks, upper stack uses full blocks
- `Z5`: periodic heavy blocks every two layers

**Key interpretation**

- if `Z1`, `Z3`, or `Z4` works, the current block template is too uniform
- if `Z5` works, compute should be concentrated rather than evenly distributed

### T-20260401-AB: Compression-Native Sharing Audit

**Status:** runnable

**Goal**  
Try a compression-native architecture direction that is qualitatively different from naive low-rank factorization.

**Main question**  
Can stage-specific structure be shared across similar layers, so the model stops relearning the same transformation with different private weights?

**Why this is next-worthy**

- low-rank factorization already failed clearly
- compression-native design is still one of our biggest open gaps
- the current frontier theory says lower and upper stages each do repeated specialized jobs, which makes sharing plausible

**Planned experiments**

- `AB1`: share weights across the lower two mixer blocks
- `AB2`: share weights across all four lower mixer blocks
- `AB3`: share weights across the top two attention blocks
- `AB4`: share lower mixer weights and reinvest into a wider mixer
- `AB5`: share lower mixer weights and reinvest into one more upper reasoning layer

**Key interpretation**

- if `AB1` or `AB2` works, the lower stage is structurally repetitive and overparameterized
- if `AB4` or `AB5` wins, sharing is useful mainly as reallocation rather than pure savings

### T-20260401-AA: Upper Attention Decomposition Audit

**Status:** runnable

**Goal**  
Question what the remaining upper attention layers are actually doing now that the lower stack is already simplified.

**Main question**  
How much full global attention does the upper stack still need, and is the very top of the network doing a qualitatively different reasoning job?

**Why this is next-worthy**

- local-window replacement lost, but that only killed one simplification story
- we still have not isolated which pieces of the remaining upper attention are essential
- top-only routing being alive hints that the top of the stack may be special

**Planned experiments**

- `AA1`: keep only top three layers as full global attention
- `AA2`: keep only top two layers as full global attention
- `AA3`: top two global-attention layers plus top-only routing
- `AA4`: interleave upper attention and upper lighter refinement blocks
- `AA5`: one final rich global reasoner layer only

**Key interpretation**

- if `AA1` or `AA2` works, upper attention is over-provisioned
- if `AA3` works, top-of-stack routing is part of final reasoning
- if `AA4` works, periodic global refresh beats dense upper attention
