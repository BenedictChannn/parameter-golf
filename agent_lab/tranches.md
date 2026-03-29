# Agent Lab Tranches

This file is the high-level research-program map. Each tranche should have a real question, explicit controls, and a stopping or pivot rule. Link back to exact experiments and detailed logs instead of repeating every metric inline.

## T-20260328-A - Local Baseline Calibration

- Status: completed
- Goal: establish a usable local baseline and identify the first major levers on the 3090 stack.
- Fixed controls:
- one training shard
- `600s` training cap
- primary metric `final_int8_ttt_lora`
- Main findings:
- `NUM_KV_HEADS 4 -> 2` helped
- `TRAIN_BATCH_TOKENS 524288 -> 262144` helped a lot
- `MATRIX_LR 0.04 -> 0.06` helped modestly
- Key experiments:
- [`AL-20260328-001`](./experiments.tsv)
- [`AL-20260328-002`](./experiments.tsv)
- [`AL-20260328-003`](./experiments.tsv)
- [`AL-20260328-004`](./experiments.tsv)
- Deeper notes:
- [`docs/build-logs/2026-03-28-agent-lab.md`](../docs/build-logs/2026-03-28-agent-lab.md)

## T-20260329-A - Capacity vs Step Frontier

- Status: active
- Goal: determine how much extra model capacity the current local runtime can support inside the fixed `600s` budget, and whether extra depth wins only when the branch also gets enough optimizer steps.
- Main question:
- Is the best local frontier on this stack “more depth plus more steps”, and if so, where does that frontier flatten?
- Fixed controls:
- one training shard
- `600s` training cap
- primary metric `final_int8_ttt_lora`
- tokenizer and validation semantics unchanged
- What we tested:
- refreshed baseline on the current runtime
- `10` layers alone
- `10` layers with smaller batch
- `10` layers with much smaller batch
- `10` layers with cheaper attention via `NUM_KV_HEADS=1`
- Current findings:
- `10` layers alone was wrong because it lost too many steps.
- `10` layers plus `196608` batch was a clear win.
- `10` layers plus `131072` batch was only a marginal further improvement, suggesting the frontier is flattening.
- `10` layers plus `NUM_KV_HEADS=1` was worse than the best `kv2` depth branches.
- Current best:
- [`AL-20260329-004`](./experiments.tsv) at `1.3913`
- Key experiments:
- [`AL-20260329-001`](./experiments.tsv)
- [`AL-20260329-002`](./experiments.tsv)
- [`AL-20260329-003`](./experiments.tsv)
- [`AL-20260329-004`](./experiments.tsv)
- [`AL-20260329-005`](./experiments.tsv)
- Stop or pivot rule:
- Stop this tranche when repeated nearby runs suggest the `196608` vs `131072` difference is noise-level, or when new frontier pushes clearly trade too much artifact headroom for too little score.
- Likely next pivot:
- move from “buy more steps” to “free useful capacity or improve optimization without adding much more batch noise.”
- Deeper notes:
- [`docs/build-logs/2026-03-29-agent-lab.md`](../docs/build-logs/2026-03-29-agent-lab.md)
