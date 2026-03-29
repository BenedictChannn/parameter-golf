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
- Status: new
- Related tranche: none yet

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

## Parked

### I-20260329-006 - KV1 As The Main Frontier Lever

- Category: attention
- Hypothesis: reducing `NUM_KV_HEADS` to `1` is a strong frontier move for the current `10`-layer setup.
- Why parked:
- [`AL-20260329-005`](./experiments.tsv) suggests `kv1` is not competitive with the best `kv2` depth branches on this stack
- Status: parked
