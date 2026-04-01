# Agent Lab Ideas

This is the high-level hypothesis bank. Not every idea should become an experiment immediately. Use this file to track what is active, what is new, and what has already been weakened by evidence.

## Active

### I-20260329-001 - Depth Needs Step Support

- Category: architecture + optimization
- Hypothesis: extra depth helps on this stack only if the branch also recovers enough optimizer steps inside the same `600s` budget.
- Why it might work:
- [`AL-20260329-002`](./experiments.tsv) showed depth alone was step-starved.
- [`AL-20260329-003`](./experiments.tsv) and [`AL-20260329-004`](./experiments.tsv) showed depth becomes competitive or winning when step count rises.
- Status: active
- Related tranche: [`T-20260329-A`](./tranches.md#t-20260329-a---capacity-vs-step-frontier)

### I-20260329-002 - Speed Recovery With Less Batch Noise

- Category: systems + architecture
- Hypothesis: there is a cleaner way to recover speed or artifact headroom than pushing batch ever smaller.
- Why it might work:
- `131072` batch only improved on `196608` by `0.0003`, which is close to noise.
- `NUM_KV_HEADS=1` was not the answer, but the question remains valid.
- Status: active
- Related tranche: [`T-20260329-A`](./tranches.md#t-20260329-a---capacity-vs-step-frontier)

## New

### I-20260329-003 - Compression-Aware Capacity

- Category: compression
- Hypothesis: some forms of capacity growth compress materially better than others, so raw parameter count is not the whole story.
- Why it might work:
- the current best branch is already close to the 16 MB cap
- Status: active
- Related tranche: [`T-20260329-C`](./tranches.md#t-20260329-c-width-winner-size-recovery)
- Evidence so far:
- [`AL-20260329-010`](./experiments.tsv) proved the raw width winner is strong, but too large to submit

### I-20260329-011 - Width Winner Can Be Saved By Mild Byte Cuts

- Category: architecture + compression
- Hypothesis: the raw winner likely does not need a dramatic redesign; a mild trim such as lower `MODEL_DIM`, one less MLP notch, or one less layer may recover validity while keeping most of the gain.
- Why it might work:
- the raw winner is only strong after the width-plus-step interaction clicks, so the safest next move is to shave bytes around that shape instead of abandoning it
- Status: active
- Related tranche: [`T-20260329-C`](./tranches.md#t-20260329-c-width-winner-size-recovery)
- Evidence so far:
- [`AL-20260329-011`](./experiments.tsv) says a mild global dim trim fixes size but gives back too much score
- [`AL-20260329-012`](./experiments.tsv) says a one-notch MLP trim is a much cleaner byte cut; it produced the new best valid frontier at `1.3838`
- [`AL-20260329-013`](./experiments.tsv) says global dim trimming is not hopeless, but even the stronger `DIM448` version still loses to the one-notch MLP trim
- [`AL-20260329-014`](./experiments.tsv) says cutting one whole layer is also a weaker byte-saving mechanism than cutting one MLP notch
- [`AL-20260329-015`](./experiments.tsv) says two lighter cuts together are a respectable backup, but still clearly weaker than the one-notch MLP trim

### I-20260329-012 - Smaller Valid Width Models Need Different Training Dynamics

- Category: optimizer
- Hypothesis: once width-oriented candidates are made smaller, some of the remaining loss versus the raw winner may be recoverable with more steps or a retuned matrix LR.
- Why it might work:
- B1 already showed a strong interaction between architecture and step count
- Status: active
- Related tranche: [`T-20260329-D`](./tranches.md#t-20260329-d-slim-winner-optimization-recovery)
- Evidence so far:
- [`AL-20260329-012`](./experiments.tsv) shows one size-recovered width-oriented survivor is already strong enough to deserve direct optimization follow-ups instead of more blind structural cuts
- [`AL-20260329-015`](./experiments.tsv) keeps a second valid survivor alive, so tranche D can compare “optimize the winner” versus “rescue the backup”
- [`AL-20260329-016`](./experiments.tsv) shows the main winner was still meaningfully step-limited; extra steps, not architecture changes, were the immediate source of the next large gain
- [`AL-20260329-017`](./experiments.tsv) shows a simple LR bump at the old batch is not the answer; any remaining optimizer gain likely has to be evaluated on top of the `98304` line, not instead of it
- [`AL-20260329-018`](./experiments.tsv) shows the `98304` winner also does not want this simple LR bump, so the default LR is currently the best setting among the tested options
- [`AL-20260329-019`](./experiments.tsv) shows the fallback line was also under-trained, but even after step recovery it still remains behind the main frontier
- [`AL-20260329-020`](./experiments.tsv) closes the loop: the fallback line also rejects the LR bump, so the optimizer lesson from tranche D is consistent across both survivors

### I-20260329-004 - Schedule or Optimizer Retune For 10L

- Category: optimizer
- Hypothesis: once `10` layers is no longer step-starved, the next gain may come from retuning learning dynamics rather than from further batch reduction.
- Why it might work:
- the current frontier suggests raw step-count gains are flattening
- Status: new
- Related tranche: none yet

### I-20260329-005 - Structural Capacity Instead of Pure Depth

- Category: architecture
- Hypothesis: some other capacity increase, such as a different projection or MLP structure, may beat the current `10`-layer frontier without exhausting artifact headroom.
- Why it might work:
- depth won, but only after step support, and size is now nearly capped
- Status: new
- Related tranche: none yet

### I-20260329-007 - MLP Width Versus Depth

- Category: architecture
- Hypothesis: moving some capacity budget from depth into MLP width could improve quality or compression efficiency more cleanly than another depth push.
- Why it might work:
- the current frontier suggests depth helps, but we have not yet asked whether `MLP_MULT` is the better place to spend parameters
- Status: active
- Related tranche: [`T-20260329-B`](./tranches.md#t-20260329-b---architecture-necessity-audit)
- Evidence so far:
- [`AL-20260329-007`](./experiments.tsv) says pure width at fixed depth is not promising in this naive form; it was slower, worse, and oversize
- [`AL-20260329-009`](./experiments.tsv) says width becomes more plausible when paired with one fewer layer, but it still trails the anchor and misses the size cap slightly
- [`AL-20260329-010`](./experiments.tsv) says width also needs step recovery; with more steps it became the best raw scorer, but the artifact failure got worse

### I-20260329-010 - The Current MLP May Already Be Too Wide

- Category: architecture
- Hypothesis: a thinner MLP plus one more layer may beat the current `10L / MLP2` balance because the model is over-spending capacity inside each block.
- Why it might work:
- in small models, MLPs can dominate parameter count quickly; more transformations may be a better use of budget than fatter hidden layers
- Status: active
- Related tranche: [`T-20260329-B`](./tranches.md#t-20260329-b---architecture-necessity-audit)
- Evidence so far:
- [`AL-20260329-006`](./experiments.tsv) says `10L / MLP1` alone is not enough; thinner blocks gained steps and headroom but still lost on `val_bpb`
- [`AL-20260329-008`](./experiments.tsv) says even `11L / MLP1` is not enough; moving width into one extra layer did not beat the `10L / MLP2` balance

### I-20260329-008 - Residual Controls And Skip Paths Are Overbuilt

- Category: architecture
- Hypothesis: some of `resid_mix`, `attn_scale`, `mlp_scale`, or `skip_weights` may be unnecessary or overly expensive in complexity relative to the quality they add.
- Why it might work:
- these controls are distinctive to this script and may be carrying historical baggage rather than current necessity
- Status: new
- Related tranche: none yet

### I-20260329-009 - Output Path Is Mismatched To The 10-Layer Regime

- Category: architecture
- Hypothesis: the current tying, initialization, or logit softcap choices may be leaving quality on the table now that the model is deeper and better trained.
- Why it might work:
- output-path choices affect both optimization behavior and compression cost, but have not been tested in the current frontier
- Status: new
- Related tranche: none yet

### I-20260329-013 - The Frontier Is Now Attention-Limited

- Category: attention
- Hypothesis: after tranche C fixed width allocation and tranche D fixed step count, the next meaningful gain may come from changing attention geometry rather than from more optimizer nudging.
- Why it might work:
- the current best line is stable on shape and optimization, but we have barely explored attention geometry beyond `kv1` versus `kv2`
- `NUM_HEADS`, `NUM_KV_HEADS`, and `QK_GAIN_INIT` are already exposed and can test this family cheaply and honestly
- Status: active
- Related tranche: [`T-20260329-E`](./tranches.md#t-20260329-e-attention-geometry-audit)
- Evidence so far:
- [`AL-20260329-021`](./experiments.tsv) says the frontier does respond to attention geometry; `q4/kv2` is better than the previous `q8/kv2` winner
- [`AL-20260329-022`](./experiments.tsv) says the direction is specifically toward fewer wider heads, not toward more narrower ones
- [`AL-20260329-023`](./experiments.tsv) says less KV sharing helps some, but not enough to beat the wider-head direction
- [`AL-20260329-024`](./experiments.tsv) says softer QK init is competitive but still secondary to the head-geometry win
- [`AL-20260329-025`](./experiments.tsv) says sharper QK init is the better side of the bracket, but the head-geometry win is still the dominant signal

### I-20260329-014 - The Frontier Is Now Output-Path-Limited

- Category: output path
- Hypothesis: after width allocation, step count, and first-pass attention geometry are improved, the next meaningful gain may come from the output path: tying, logit calibration, or output-specific learning dynamics.
- Why it might work:
- the output path is still mostly untouched in this repo's search history
- tying and logit softcaps directly affect calibration and expressivity in a tiny-model regime
- all the main knobs are already env-exposed, so this family is cheap to test honestly
- Status: active
- Related tranche: [`T-20260329-F`](./tranches.md#t-20260329-f-output-path-audit)
- Evidence so far:
- [`AL-20260329-026`](./experiments.tsv) says output expressivity is a first-class lever; untied outputs produced a large frontier jump without breaking the size cap
- [`AL-20260329-027`](./experiments.tsv) says output calibration matters on top of that; tighter logit clipping improved the untied frontier again
- [`AL-20260329-028`](./experiments.tsv) says the calibration result is directional; looser clipping lost to the tighter softcap
- [`AL-20260329-029`](./experiments.tsv) says slower head-LR is competitive but secondary to output calibration
- [`AL-20260329-030`](./experiments.tsv) says output-head learning dynamics are also real; a somewhat faster `HEAD_LR=0.012` beat both the default and slower-head-LR versions of the untied + softcap20 frontier

### I-20260330-001 - The Untied Output Frontier Still Has Local Headroom

- Category: output path
- Hypothesis: the current best line is still slightly mis-set locally, and a tighter neighborhood around `LOGIT_SOFTCAP=20` and `HEAD_LR=0.012` can produce one more immediate frontier step.
- Why it might work:
- tranche F ended on an active win rather than a flattening result
- the best-tested softcap and head-LR settings were only bracketed coarsely, not tightly
- Status: active
- Related tranche: [`T-20260330-G`](./tranches.md#t-20260330-g-untied-output-local-calibration)
- Evidence so far:
- [`AL-20260330-001`](./experiments.tsv) says there is still tie-scale sensitivity on the stronger-clipping side, but not a clean new win.
- [`AL-20260330-002`](./experiments.tsv), [`AL-20260330-003`](./experiments.tsv), and [`AL-20260330-004`](./experiments.tsv) say the obvious nearby scalar moves all lose.
- [`AL-20260330-005`](./experiments.tsv) says even the strongest local combo did not beat the anchor.
- Status: parked

### I-20260330-002 - Residual Controls And Skip Paths Are The Next Likely Bottleneck

- Category: architecture
- Hypothesis: after width, steps, attention, and output path are improved, the next meaningful gain may come from simplifying or retuning `resid_mix`, `attn_scale`, `mlp_scale`, or `skip_weights`.
- Why it might work:
- the output-path local neighborhood now looks mostly mapped
- these controls are distinctive to this script and still under-tested
- `architecture_review.md` already flags them as a plausible overbuilt family
- Status: active
- Related tranche: [`T-20260330-H`](./tranches.md#t-20260330-h-residual-control-simplification)

### I-20260330-003 - Not Every Layer Needs Full Attention

- Category: architecture
- Hypothesis: some layers can switch from full attention to a cheaper state-space-inspired sequence mixer without giving back much quality under the same `600s` budget.
- Why it might work:
- tranche H mostly closed the generic residual-simplification path
- the next serious upside is likely in changing the block, not pruning scalars
- a hybrid stack may trade some attention expressivity for more efficient sequence processing
- Status: active
- Related tranche: [`T-20260330-J`](./tranches.md#t-20260330-j-hybrid-sequence-mixer-audit)

### I-20260330-004 - Full K/V Structure May Be Overbuilt

- Category: architecture
- Hypothesis: the current attention block may be over-spending parameters and compute on full K/V projections, and a latent-KV bottleneck can preserve most of the useful attention behavior.
- Why it might work:
- MLA-style ideas suggest K/V structure can be compressed without collapsing usefulness
- the parameter golf setting rewards anything that buys more steps or more capacity under the same wall-clock budget
- Status: active
- Related tranche: [`T-20260330-I`](./tranches.md#t-20260330-i-latent-kv-attention-audit)

## Parked

### I-20260329-006 - KV1 As The Main Frontier Lever

- Category: attention
- Hypothesis: reducing `NUM_KV_HEADS` to `1` is a strong frontier move for the current `10`-layer setup.
- Why parked:
- [`AL-20260329-005`](./experiments.tsv) suggests `kv1` is not competitive with the best `kv2` depth branches on this stack
- Status: parked

## Queued Roadmap

These are the next six tranches worth planning after the active latent-KV tranche closes. They are ordered by expected research value, not by ease.

### Q-20260330-001 - Hybrid Sequence Mixer Layers

- Category: architecture
- Goal: test whether some attention layers can be replaced by a cheaper delta-style or state-space-inspired mixer.
- Main hypothesis: not every layer needs full attention; a hybrid stack can preserve most of the quality while buying more useful training budget.
- Why it is worth the compute:
- it asks a real mechanism question rather than a scalar-tuning one
- it could open a fundamentally different small-model design if it works
- Related tranche: successor to [`T-20260330-J`](./tranches.md#t-20260330-j-hybrid-sequence-mixer-audit)
- Status: queued

### Q-20260330-002 - Output Head Architecture

- Category: output path
- Goal: redesign the untied output head itself instead of tuning only `HEAD_LR` and `LOGIT_SOFTCAP`.
- Main hypothesis: the output path is already proven to matter, but the current dense untied head may not be the best architecture for this challenge.
- Why it is worth the compute:
- output-path changes already produced some of the largest gains in the whole campaign
- moving from scalar tuning to architectural changes is the natural next step
- Example directions:
- low-rank output head
- bottleneck output head
- shared-base plus small residual head
- Status: queued

### Q-20260330-003 - Local-Global Attention Split

- Category: attention
- Goal: test whether the model needs full global attention in every layer, or whether some layers can use a cheaper local/sliding-window pattern.
- Main hypothesis: a local-global split may preserve enough routing power while freeing compute and size budget.
- Why it is worth the compute:
- attention geometry already showed strong leverage
- this is a structural attention question, not a head-count tweak
- Status: queued

## Near-Term Queue

### Q-20260331-001 - Hybrid Mixer Refinement

- Category: architecture
- Goal: refine the current best family instead of abandoning it after the first win.
- Main hypothesis: the lower-stack mixer win is real, but the exact placement, mixer width, and kernel are still probably suboptimal.
- Why it is worth the compute:
- it follows the strongest frontier signal we currently have
- it can reveal whether the mixer win is robust or narrowly tuned
- Related tranche: `T-20260331-P`
- Status: completed
- Evidence so far:
- [`AL-20260331-001`](./experiments.tsv) showed the hybrid family is robust: lower-three mixers still beat the old best.
- [`AL-20260331-004`](./experiments.tsv) became the new best valid run, which means the mixer mechanism itself still had local headroom.
- [`AL-20260331-002`](./experiments.tsv) showed that pushing to five lower mixers goes too far on the current frontier.

### Q-20260331-002 - AttnRes-lite Dynamic Depth Routing

- Category: architecture
- Goal: test whether fixed residual and skip routing should give way to input-dependent routing over a small set of earlier layer states.
- Main hypothesis: skip topology matters, routing redesign matters, and the next bold step is dynamic depth selection rather than another fixed routing tweak.
- Why it is worth the compute:
- it directly builds on the routing lessons from tranches `H` and `M`
- it tests a new principle instead of a local scalar adjustment
- Related tranche: `T-20260331-Q`
- Status: completed
- Evidence so far:
- [`AL-20260331-006`](./experiments.tsv) crashed before producing a metric because a TorchDynamo-incompatible Python `id()` call slipped into the first implementation.
- [`AL-20260331-007`](./experiments.tsv), [`AL-20260331-009`](./experiments.tsv), and [`AL-20260331-010`](./experiments.tsv) all failed badly, which means broad late-layer dynamic depth routing is the wrong shape on the current frontier.
- [`AL-20260331-008`](./experiments.tsv) says top-two-layer routing is the only remotely viable version, but it still did not beat the frontier.

### Q-20260331-003 - Architecture-Specific Schedules

- Category: optimization
- Goal: test whether the strongest new architectures are being judged with the wrong training tail.
- Main hypothesis: hybrid or routing-heavy architectures may want a different warmdown, stabilization, or final head/backbone cooldown than the old baseline.
- Why it is worth the compute:
- it targets compressed-model behavior directly
- it can rescue promising architecture changes that fail under a mismatched schedule
- Related tranche: `T-20260331-R`
- Status: completed
- Evidence so far:
- [`AL-20260331-011`](./experiments.tsv) and [`AL-20260331-012`](./experiments.tsv) show broad warmdown still hurts on the hybrid winner.
- [`AL-20260331-014`](./experiments.tsv) says head-focused cooldown is the only schedule hint that stayed near the frontier, but it still did not win.

## Backlog

### B-20260331-001 - Broad MLP Activation Audit

- Category: architecture
- Goal: compare `relu^2` against broader activation families such as `ReLU`, `SiLU`, `GELU`, and gated variants.
- Main hypothesis: the current MLP may be good, but not necessarily best; the activation family itself may be a real frontier lever.
- Why it is worth keeping:
- the MLP is a large part of the model budget
- this can become a full research branch and should not be rushed
- Status: backlog

### B-20260331-002 - Polynomial MLP Family

- Category: architecture
- Goal: test richer polynomial MLP variants such as `relu^3` and mixed quadratic/cubic forms once the broad activation family is mapped.
- Main hypothesis: the current `relu^2` setup may only be one point in a larger useful polynomial design space.
- Why it is worth keeping:
- it follows naturally from the current nonstandard MLP choice
- it deserves a dedicated follow-up tranche rather than a rushed side experiment
- Status: backlog

### Q-20260330-004 - Skip/Residual Redesign

- Category: architecture
- Goal: redesign the skip and residual system rather than merely ablating it.
- Main hypothesis: the model wants skip topology, but it may not need the current full learned parameterization for skip and residual routing.
- Why it is worth the compute:
- tranche H already showed deletion is mostly the wrong question
- `SKIP_MODE=unit` nearly tied the frontier, so a cheaper or cleaner routing design is still plausible
- Example directions:
- shared skip gates instead of per-channel weights
- fewer but more intentional skip links
- alternative residual routing layouts
- Status: queued

### Q-20260330-005 - Mechanism-Specific Learning Rates

- Category: optimizer
- Goal: stop using only broad parameter-class learning rates and test whether new mechanisms want their own learning dynamics.
- Main hypothesis: promising architecture ideas may be underperforming because they are being trained at the wrong relative speed, not because the mechanism is bad.
- Why it is worth the compute:
- we already split embedding, head, matrix, and scalar LRs successfully
- the same logic may matter for attention vs MLP, early vs late layers, or new blocks like latent-KV and mixers
- Note:
- this is best used as a support tranche for promising architecture families, not as a standalone tuning detour
- Status: queued

### Q-20260330-006 - Quantization-Aware Warmdown

- Category: optimization + compression
- Goal: test whether the tail end of training can be shaped to make the final int8+zlib submission behave better after quantization.
- Main hypothesis: the best fp training schedule is not necessarily the best schedule for the model we actually submit, and a better warmdown could reduce post-quantization damage.
- Why it is worth the compute:
- this is challenge-specific rather than generic tuning
- the score is computed on a compressed model, so end-of-training behavior matters more than usual
- Example directions:
- longer or gentler LR decay at the end
- short stabilization phases before export
- schedule variants judged by post-quantization score, not only training loss
- Status: queued

## 2026-03-31 Program Closeout

### Promote

### I-20260331-001 - Hybrid Mixer Refinement Around The Lower-Stack Winner

- Category: architecture
- Hypothesis: the new best line, [`AL-20260330-104`](./experiments.tsv), still has headroom in mixer placement, width, or kernel shape.
- Why it might work:
- tranche J already proved the family is real
- lower-stack replacement was clearly better than middle-only or upper-only replacement
- the next good question is refinement, not whether the family exists
- Status: active

### I-20260331-002 - Cheap Routing Plus Hybrid Mixer

- Category: architecture
- Hypothesis: the cheap-routing package from [`AL-20260330-120`](./experiments.tsv) may stack with the hybrid-mixer winner rather than competing with it.
- Why it might work:
- both families improved the frontier independently
- one changes the block type while the other changes the routing controls
- this is a justified combo hypothesis, not random stacking
- Status: active

### I-20260331-003 - Local-Global Attention Debug And Completion

- Category: attention
- Hypothesis: the local-global family still deserves a fair test once the lower-stack local-window crash is fixed.
- Why it might work:
- the family is unresolved, not disproven
- the lower-stack motif already looked strong in the hybrid-mixer tranche, so lower-stack locality is still a plausible direction
- Status: active

## Park After Program

### I-20260331-004 - Output Head Architecture As A Near-Term Frontier

- Category: output path
- Hypothesis: changing the head architecture itself is the best next use of compute on the current line.
- Why parked:
- tranche K mostly killed this on the current backbone
- the dense untied head stayed clearly better than low-rank and nonlinear alternatives
- Status: parked

### I-20260331-005 - Quantization-Aware Warmdown As A Near-Term Frontier

- Category: optimization + compression
- Hypothesis: a colder or more carefully staged training tail is the next frontier move.
- Why parked:
- tranche O mostly hurt badly
- only head-only cooldown stayed near the old frontier, and even that did not beat it
- Status: parked

### I-20260331-006 - Mechanism-Specific Learning Rates As A Standalone Tranche

- Category: optimizer
- Hypothesis: LR splits alone are the next major win.
- Why parked:
- tranche N produced useful near-ties but not a real breakthrough
- the better use of this idea is probably as support for future architecture winners
- Status: parked


### I-20260331-007 - Architecture-Specific Schedules On The Hybrid Winner

- Category: optimization
- Hypothesis: the stronger hybrid-mixer backbone may want a different training tail, especially near export.
- Why it might work:
- the earlier quantization-aware schedule tranche was run on an older backbone
- the new backbone could in principle have different fragility near export
- Status: parked
- Evidence so far:
- [`AL-20260331-011`](./experiments.tsv) and [`AL-20260331-012`](./experiments.tsv) show broad warmdown still hurts.
- [`AL-20260331-014`](./experiments.tsv) says head-focused cooldown is the only schedule hint that stayed near the frontier, but it still did not win.

### I-20260331-008 - Top-Only Dynamic Depth Routing

- Category: architecture
- Hypothesis: if dynamic depth routing is useful at all on this backbone, it belongs only in the topmost layers where representations are being selected for output.
- Why it might work:
- broad late-layer AttnRes-lite was clearly too disruptive
- top-two-layer routing was vastly cleaner than any broader placement
- Status: parked
- Evidence so far:
- [`AL-20260331-008`](./experiments.tsv) was the only AttnRes-lite run that stayed near the older hybrid anchor.
- [`AL-20260331-007`](./experiments.tsv), [`AL-20260331-009`](./experiments.tsv), and [`AL-20260331-010`](./experiments.tsv) suggest the main danger is applying dynamic depth routing too broadly.

## 2026-04-01 STUVWX Closeout

### Promote

### I-20260401-001 - Refine The Hybrid Plus Skip-Gate Frontier

- Category: architecture
- Hypothesis: the new best line, [`AL-20260331-017`](./experiments.tsv), still has local headroom in mixer detail or routing strength because the stronger hybrid winner and the skip-gate win both looked robust rather than brittle.
- Why it might work:
- [`AL-20260331-016`](./experiments.tsv), [`AL-20260331-017`](./experiments.tsv), and [`AL-20260331-018`](./experiments.tsv) all improved on the old frontier
- the new best is part of a small winning neighborhood, not a one-off
- Status: active

### I-20260401-002 - Narrow Top-Only Routing As A Secondary Combo Family

- Category: architecture
- Hypothesis: top-only dynamic routing may still have useful headroom, but only as a very narrow mechanism paired with clean fixed routing.
- Why it might work:
- [`AL-20260331-031`](./experiments.tsv), [`AL-20260331-032`](./experiments.tsv), and [`AL-20260331-035`](./experiments.tsv) all show the revived family is real
- the winning shape is now much clearer: top-only, low-source, token-wise
- Status: active

### I-20260401-003 - Second-Generation Compression-Native Branch

- Category: compression + architecture
- Hypothesis: compression-aware architecture is still a gap in the lab, but the next useful version must be structurally different from naive low-rank factorization.
- Why it might work:
- the public frontier still suggests compression-native thinking matters
- tranche `U` closed one mechanism family, not the whole compression-native agenda
- Status: active

### I-20260401-007 - MLP Structure Minimalism

- Category: MLP + block design
- Hypothesis: the current model may not need a full dense project-up then project-down MLP in every block; some of that structure may be inherited rather than earned.
- Why it might work:
- the MLP dominates a large share of parameters and compute
- `relu^2` is strong, but we have only tested activation families, not whether the whole FFN structure is overbuilt
- mixed linear-plus-quadratic behavior already showed one live hint
- Manifest: [`20260401-Y-mlp-structure-minimalism.json`](./tranche_manifests/planned/20260401-Y-mlp-structure-minimalism.json)
- Status: active

### I-20260401-008 - Block Uniformity Audit

- Category: block topology
- Hypothesis: the model may be overpaying for a uniform block recipe where some stages only need token mixing and others carry the expensive per-token refinement burden.
- Why it might work:
- the repo already rewards depth specialization strongly
- lower mixer-heavy layers may not need a full FFN after every mixing step
- this questions one of the strongest inherited transformer assumptions directly
- Manifest: [`20260401-Z-block-uniformity-audit.json`](./tranche_manifests/planned/20260401-Z-block-uniformity-audit.json)
- Status: active

### I-20260401-009 - Compression-Native Sharing

- Category: compression + architecture
- Hypothesis: the next viable compression-native branch is structured sharing or reuse across similar layers, not another low-rank factorization sweep.
- Why it might work:
- naive low-rank factorization already failed cleanly
- the current frontier theory says lower and upper stages each perform repeated specialized jobs, which makes sharing plausible
- Manifest: [`20260401-AB-compression-native-sharing.json`](./tranche_manifests/planned/20260401-AB-compression-native-sharing.json)
- Status: active

### I-20260401-010 - Upper Attention Decomposition

- Category: attention + topology
- Hypothesis: the surviving upper attention stack is still over-provisioned, and only a very small number of true global reasoning layers may actually be necessary.
- Why it might work:
- local-window replacement lost, but that only killed one simplification story
- we still have not isolated which parts of the remaining upper attention are essential
- top-only routing being alive hints that the very top of the stack may be doing a distinct job
- Manifest: [`20260401-AA-upper-attention-decomposition.json`](./tranche_manifests/planned/20260401-AA-upper-attention-decomposition.json)
- Status: active

### Park

### I-20260401-004 - Plain Local-Window Attention On The Hybrid Backbone

- Category: attention
- Hypothesis: replacing the remaining global attention layers with local windows is the next frontier move.
- Why parked:
- the repaired local-attention tranche lost in every measured form
- even the best top-two `512`-window version stayed clearly behind the frontier
- Status: parked

### I-20260401-005 - Broad Smooth-Activation MLP Replacement

- Category: MLP
- Hypothesis: the next MLP gain will come from swapping `relu^2` for a broad smooth activation family like SiLU or GELU.
- Why parked:
- both [`AL-20260331-038`](./experiments.tsv) and [`AL-20260331-039`](./experiments.tsv) lost clearly
- the broad MLP family still belongs to `relu^2`
- Status: parked

### I-20260401-006 - Cubic-Heavy Polynomial MLPs

- Category: MLP
- Hypothesis: stronger cubic emphasis is the right next polynomial direction.
- Why parked:
- [`AL-20260331-042`](./experiments.tsv) and [`AL-20260331-044`](./experiments.tsv) both regressed clearly
- the only live polynomial hint is the mixed linear-plus-quadratic form
- Status: parked
