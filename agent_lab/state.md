# Agent Lab State

This is the first-read dashboard for autonomous research. Read this file for the high-level picture, then drill down into the linked tranche, idea bank, experiment ledger, and build log.

## Current Best

- Experiment: [`AL-20260329-004`](./experiments.tsv)
- Commit: `41e9478`
- Primary metric: `final_int8_ttt_lora`
- Best `val_bpb`: `1.3913`
- Winning branch shape: `10` layers, `NUM_KV_HEADS=2`, `TRAIN_BATCH_TOKENS=131072`
- Total artifact size: `15,824,353` bytes int8+zlib

## Active Tranche

- Tranche: [`T-20260329-A`](./tranches.md#t-20260329-a---capacity-vs-step-frontier)
- Goal: map the capacity-versus-step frontier under a fixed `600s` training budget
- Status: active

## Working Beliefs

- The mar29 system runtime is stronger than the earlier mar28 local anchor, so new baselines must be compared against the refreshed local frontier, not the older numbers.
- `10` layers are useful only when they are not step-starved.
- Smaller batches recovered enough optimizer steps to turn the `10`-layer branch from a loser into the current winner.
- The frontier is flattening: `196608` and `131072` batch settings are very close.
- Cheaper attention via `NUM_KV_HEADS=1` was not a better trade than keeping `NUM_KV_HEADS=2`.

## Open Questions

- Is the tiny `131072` edge over `196608` real or just noise?
- Can we preserve the `10`-layer gain while reclaiming artifact headroom?
- Is the next tranche better framed around compression-aware capacity, optimizer/schedule tuning, or a different architectural mechanism?

## Next Planned Runs

- Decide whether to confirm the `131072` vs `196608` difference with a focused replay or treat it as effectively tied.
- Start a sharper next tranche instead of continuing blind local search.
- Favor mechanisms that improve the `10`-layer branch without pushing size closer to the cap.

## Go Deeper

- Tranche map: [`tranches.md`](./tranches.md)
- Idea bank: [`ideas.md`](./ideas.md)
- Experiment ledger: [`experiments.tsv`](./experiments.tsv)
- Latest narrative log: [`docs/build-logs/2026-03-29-agent-lab.md`](../docs/build-logs/2026-03-29-agent-lab.md)
