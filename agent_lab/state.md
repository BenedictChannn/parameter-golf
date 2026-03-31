# Agent Lab State

This is the first-read dashboard for the current research state. Use this for the high-level picture, then drill down into the ledger, tranche map, findings, and build logs.

## Current Best Valid

- Experiment: [`AL-20260331-004`](./experiments.tsv)
- Commit: `7641f2e`
- Primary metric: `final_int8_ttt_lora`
- Best `val_bpb`: `1.3451`
- Winning branch shape: `9` layers, `MLP_MULT=2`, `MODEL_DIM=512`, `NUM_HEADS=4`, `NUM_KV_HEADS=2`, `TRAIN_BATCH_TOKENS=98304`, lower four attention layers replaced by sequence mixers with `MIXER_DIM=320`, `TIE_EMBEDDINGS=0`, `LOGIT_SOFTCAP=20`, `HEAD_LR=0.012`
- Total artifact size: `15,035,357` bytes int8+zlib

## Best Secondary Valid

- Experiment: [`AL-20260331-001`](./experiments.tsv)
- Commit: `7641f2e`
- Primary metric: `final_int8_ttt_lora`
- Best `val_bpb`: `1.3453`
- Branch shape: `9` layers with the lower three attention layers replaced by sequence mixers; otherwise same frontier controls
- Total artifact size: `15,037,610` bytes int8+zlib

## Incomplete Family

- Tranche: [`T-20260330-L`](./tranches.md#t-20260330-l---local-global-attention-split)
- Status: blocked
- Why: [`AL-20260330-111`](./experiments.tsv) crashed before the first local-attention metric was produced
- Next need: debug the local-window attention runtime before treating that family as tested

## Active Tranche

- Tranche: none
- Goal: tranche `Q` is closed; choose the next architecture branch from a cleaner post-`PQR` state
- Status: ready to pivot

## Working Beliefs

- The hybrid mixer family still has headroom. Tranche `P` improved the frontier twice more, which means the original `AL-20260330-104` win was real but not locally exhausted.
- Lower-stack replacement is robust rather than brittle. Lower three mixers, a wider mixer, and a wider mixer kernel all beat the old best; lower five mixers did not.
- The best result so far is a stronger lower-four mixer, not a more aggressive lower-five placement. This suggests the frontier still wants some lower-stack attention, but the mixer mechanism itself was underpowered.
- Dynamic depth routing was mostly the wrong fit in this first form. Late-layer AttnRes-lite failed badly, whether it used richer token-wise routing, cheaper shared routing, or the cheap-routing package underneath.
- The only AttnRes-lite variant that stayed remotely alive was top-two-layer routing. That suggests dynamic depth selection, if useful at all, belongs only at the very top of the stack.
- Output-head architecture was mostly a dead end. The existing dense untied head stayed clearly better than low-rank, bottleneck, and two-stage alternatives.
- The output win from tranche F was about the dense untied head itself, not an obviously more parameter-efficient head architecture.
- Skip/residual redesign is a live secondary family. Shared scalar skip gates helped, and the full cheap-routing package became the second-best new frontier.
- Mechanism-specific learning rates look like support tools, not a frontier by themselves. Several variants nearly tied the old anchor, but none challenged the new hybrid-mixer frontier.
- Architecture-specific schedules were mostly wrong on the hybrid winner. Broad warmdown still hurt, and even the best head-focused cooldown only tied the anchor at the noise level.
- Output-path-sensitive cooldown remains the only live schedule hint, but it is still not a winner.
- Local-global attention is still unresolved because the first local-window run crashed and the family never got a fair comparison.

## Open Questions

- Can the cheap-routing package stack with the new stronger hybrid winner, or are they redundant?
- Is top-of-stack-only dynamic depth routing worth a much lighter second-generation revisit, or is the AttnRes family mostly closed already?
- What exactly is crashing in the local-attention path, and is that family still worth finishing once debugged?
- Is there any architecture-specific schedule win left beyond output-path-sensitive cooldown, or is this family mostly closed?

## Next Planned Runs

- next tranche candidate: hybrid-mixer plus cheap-routing combo around [`AL-20260331-004`](./experiments.tsv)
- next tranche candidate: local-global attention debug and completion
- next tranche candidate: only if justified later, a much lighter top-of-stack AttnRes follow-up
- later backlog: two dedicated MLP exploration tranches, broad-family first and polynomial second

## Go Deeper

- Tranche map: [`tranches.md`](./tranches.md)
- Idea bank: [`ideas.md`](./ideas.md)
- Durable findings: [`findings.md`](./findings.md)
- Experiment ledger: [`experiments.tsv`](./experiments.tsv)
- Runtime ledger: [`results.tsv`](./results.tsv)
- Experiment dashboard: [`plots/experiments.html`](./plots/experiments.html)
- Latest narrative log: [`2026-03-31-agent-lab.md`](../docs/build-logs/2026-03-31-agent-lab.md)
