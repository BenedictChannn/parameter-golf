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

**Status:** active

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

**Status:** planned next

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
| `B1-E1` | `10L / MLP2 / batch 196608 / kv2` | Replay the clean anchor | The current `10L x MLP2` balance is still the right starting point | Whether later differences are real or just runtime noise |
| `B1-E2` | `11L / MLP1 / batch 196608 / kv2` | Test deeper but thinner | The current model may be overspending on MLP width | Whether more layers beat the current block-internal capacity |
| `B1-E3` | `9L / MLP3 / batch 196608 / kv2` | Test mild width reallocation | Some capacity is better spent inside each block than on one extra layer | Whether width can replace a layer cleanly |
| `B1-E4` | `8L / MLP3 / batch 196608 / kv2` | Push farther toward width | If width is the missing ingredient, shallower-wider should stay competitive | Whether the frontier bends toward width, not depth |
| `B1-E5` | `9L / MLP3 / batch 131072 / kv2` | Test width with more steps | Width may also need step recovery, like depth did | Whether a width loss is fundamental or just step-starvation |

**Decision rule for B1**

- if `B1-E3` and `B1-E4` are both clearly worse than the anchor, width is probably not the next place to spend capacity
- if `B1-E2` wins or stays close, the model may still be under-layered relative to its MLP size
- if `B1-E3` or `B1-E5` wins, the next tranche should become width-aware rather than purely depth-aware
