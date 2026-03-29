# Agent Lab Architecture Review

This file is the model-side questioning surface for autonomous research. Use it to ask, component by component:

- is this needed?
- what job is it doing?
- what is likely limiting it?
- can it become smaller, faster, or better under the same challenge rules?

Read this after [`state.md`](./state.md) and [`tranches.md`](./tranches.md) when the next tranche is architecture-facing rather than only hyperparameter-facing.

## Current Skeleton

The current training script is a compact GPT with:

- token embeddings and usually tied output logits
- `10` layers on the current branch defaults
- grouped-query attention with separate `q`, `k`, `v`, and output projections
- RoPE on `q` and `k`
- `RMSNorm` throughout
- a `relu^2` MLP with `MLP_MULT=2`
- per-block residual controls: `attn_scale`, `mlp_scale`, and `resid_mix`
- an encoder/decoder-style skip structure via `skip_weights`
- Muon for block matrices and Adam for embeddings plus low-dimensional controls
- int8 + zlib roundtrip compression
- LoRA-based test-time training during the main challenge-relevant eval path

## Component Audit

## 1. Embeddings And Logits

- Current design:
- `tok_emb` is shared with the output head when `TIE_EMBEDDINGS=1`
- input embeddings get `RMSNorm` before entering the stack
- logits are soft-capped by `LOGIT_SOFTCAP`
- Questions:
- Is tying embeddings still the right trade once the model is deeper?
- Is the input embedding norm helping stability, or hiding capacity issues elsewhere?
- Is the softcap improving calibration, or flattening useful logit signal?
- Cheap levers:
- `TIE_EMBEDDINGS`
- `TIED_EMBED_LR`
- `TIED_EMBED_INIT_STD`
- `LOGIT_SOFTCAP`
- Why this matters:
- embeddings are a large share of total parameters at this scale
- output path choices affect both compression and final loss quality

## 2. Attention Core

- Current design:
- `8` query heads
- grouped-query attention via `NUM_KV_HEADS`
- RoPE on `q` and `k`
- per-head `q_gain`
- zero-init output projection
- Questions:
- Is current `NUM_KV_HEADS` the right quality-speed trade on this stack?
- Is `q_gain` necessary, or can attention stabilize without it?
- Is the full attention projection layout worth its parameter cost relative to other capacity?
- Would a different RoPE base or head geometry help long-context credit assignment even at fixed seq len?
- Cheap levers:
- `NUM_KV_HEADS`
- `QK_GAIN_INIT`
- `ROPE_BASE`
- `NUM_HEADS` if paired with a compatible `MODEL_DIM`
- Why this matters:
- attention determines both step speed and a significant chunk of model size

## 3. MLP Capacity

- Current design:
- `relu^2` MLP
- hidden size = `MLP_MULT * MODEL_DIM`
- zero-init output projection
- Questions:
- Is `MLP_MULT=2` underpowered, overpowered, or simply the wrong place to spend parameters?
- Would redistributing parameters from depth into MLP width compress better or train faster?
- Is `relu^2` still the right nonlinearity here?
- Cheap levers:
- `MLP_MULT`
- structural MLP edits inside [`train_gpt.py`](./train_gpt.py)
- Why this matters:
- MLP size is one of the cleanest ways to trade parameters against compute

## 4. Residual Controls

- Current design:
- each block has learned `attn_scale`, `mlp_scale`, and `resid_mix`
- `resid_mix` blends the running residual stream with the initial embedding stream
- Questions:
- Are all three controls earning their keep?
- Is `resid_mix` acting as useful recurrence-to-input grounding, or just extra scalar baggage?
- Would a simpler residual rule train just as well and compress more cleanly?
- Cheap levers:
- ablate one control family at a time
- freeze some controls to constants
- compress the control parameterization
- Why this matters:
- these are small parameters individually, but they represent design complexity and may affect optimization behavior a lot

## 5. Skip Topology

- Current design:
- first half of layers stores skip activations
- second half reuses them in reverse order through `skip_weights`
- Questions:
- Are the encoder/decoder-style skips a real contributor, or mostly historical carry-over?
- Does the model need all skip weights, or could a smaller or fixed scheme work?
- Would more ordinary depth without this skip pattern behave better once optimization is retuned?
- Cheap levers:
- ablate `skip_weights`
- replace learned skip weights with constants
- try different split rules between the two halves
- Why this matters:
- this is one of the most distinctive architectural choices in the script

## 6. Optimizer Split

- Current design:
- Muon for matrix parameters in blocks
- Adam for embeddings, untied head, and low-dimensional controls
- Questions:
- Is the optimizer partition still right for the 10-layer winning regime?
- Are scalar controls being over-updated relative to the block matrices?
- Would schedule changes beat architecture changes at this point?
- Cheap levers:
- `MATRIX_LR`
- `SCALAR_LR`
- `MUON_MOMENTUM`
- `MUON_BACKEND_STEPS`
- `WARMUP_STEPS`
- `WARMDOWN_ITERS`
- Why this matters:
- several recent findings suggest some “architecture” gains were actually optimization-budget effects

## 7. Compression And Submission Shape

- Current design:
- per-row / per-tensor int8 quantization
- zlib compression
- strict roundtrip validation
- Questions:
- Which model shapes compress best for the same quality?
- Are some parameter families expensive in bytes but weak in quality contribution?
- Can we free artifact headroom and reinvest it into more useful capacity?
- Cheap levers:
- compare artifact MB directly between sibling runs
- inspect which architectural families inflate payload disproportionately
- Why this matters:
- the current best branch is already close to the `16,000,000` byte cap

## 8. TTT LoRA Evaluation Path

- Current design:
- batched LoRA adapters on LM head plus `q` and `v`
- per-document adaptation during validation
- Questions:
- Which training-time changes improve base loss but fail to survive the TTT path?
- Are some model changes helping the roundtrip metric but not the challenge-relevant TTT metric?
- Cheap levers:
- compare `final_int8_zlib_roundtrip` against `final_int8_ttt_lora`
- use the two metrics deliberately, not interchangeably
- Why this matters:
- the true frontier can look different depending on whether the model adapts well at eval time

## Immediate Hypothesis Families

- Capacity distribution:
- maybe the next gain comes from moving parameters between attention, depth, and MLP width rather than simply adding more layers
- Structural simplification:
- some residual or skip controls may be removable or compressible without hurting quality
- Output-path retune:
- embedding tying, init scale, and logit softcap may be mismatched to the current 10-layer regime
- Optimization-vs-architecture separation:
- several apparent architecture wins may really be training-dynamics wins, so some tranches should hold structure fixed and only retune optimization

## How To Use This

- If the next tranche is architecture-heavy, pick one section above and turn it into a real question in [`tranches.md`](./tranches.md).
- Add the candidate hypothesis to [`ideas.md`](./ideas.md).
- Link the exact experiment ids back here once the branch has evidence.
