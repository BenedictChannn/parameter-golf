# Agent Lab State

This is the first-read dashboard for the current research state. Use this for the high-level picture, then drill down into the ledger, tranche map, findings, and build logs.

## Current Best Valid

- Experiment: [`AL-20260330-104`](./experiments.tsv)
- Commit: `c2e25c0`
- Primary metric: `final_int8_ttt_lora`
- Best `val_bpb`: `1.3488`
- Winning branch shape: `9` layers, `MLP_MULT=2`, `MODEL_DIM=512`, `NUM_HEADS=4`, `NUM_KV_HEADS=2`, `TRAIN_BATCH_TOKENS=98304`, lower four attention layers replaced by sequence mixers, `TIE_EMBEDDINGS=0`, `LOGIT_SOFTCAP=20`, `HEAD_LR=0.012`
- Total artifact size: `14,717,182` bytes int8+zlib

## Best Secondary Valid

- Experiment: [`AL-20260330-120`](./experiments.tsv)
- Commit: `c2e25c0`
- Primary metric: `final_int8_ttt_lora`
- Best `val_bpb`: `1.3534`
- Branch shape: cheap routing package with shared scalar skip gates, scalar `resid_mix`, and shared residual scales
- Total artifact size: `15,834,546` bytes int8+zlib

## Incomplete Family

- Tranche: [`T-20260330-L`](./tranches.md#t-20260330-l---local-global-attention-split)
- Status: blocked
- Why: [`AL-20260330-111`](./experiments.tsv) crashed before the first local-attention metric was produced
- Next need: debug the local-window attention runtime before treating that family as tested

## Active Tranche

- Tranche: none
- Goal: synthesize the overnight J-to-O program and choose the next bold family
- Status: ready to pivot

## Working Beliefs

- The strongest new result is architectural, not scalar: replacing the lower four attention layers with sequence mixers is the new frontier.
- The hybrid mixer family is real. Alternating mixer layers also worked, but lower-stack replacement was the cleanest and strongest placement.
- Output-head architecture was mostly a dead end. The existing dense untied head stayed clearly better than low-rank, bottleneck, and two-stage alternatives.
- The output win from tranche F was about the dense untied head itself, not an obviously more parameter-efficient head architecture.
- Skip/residual redesign is a live secondary family. Shared scalar skip gates helped, and the full cheap-routing package became the second-best new frontier.
- Mechanism-specific learning rates look like support tools, not a frontier by themselves. Several variants nearly tied the old anchor, but none challenged the new hybrid-mixer frontier.
- Quantization-aware warmdown was mostly wrong in this first pass. Gentler tails and combined cooldown schedules hurt badly; only head-only cooldown stayed near the old best.
- Local-global attention is still unresolved because the first local-window run crashed and the family never got a fair comparison.

## Open Questions

- Does the hybrid mixer frontier still have headroom if we retune mixer width, kernel, or placement around the lower-stack design?
- Can the cheap-routing package stack with the hybrid mixer winner, or are they redundant?
- Would AttnRes-lite dynamic depth routing beat fixed skip routing if we let later layers choose among a few earlier states?
- What exactly is crashing in the local-attention path, and is that family still worth finishing once debugged?
- Do the strongest new architectures want a different training tail, especially near quantization and export?

## Next Planned Runs

- next tranche: `T-20260331-P` hybrid mixer refinement around [`AL-20260330-104`](./experiments.tsv)
- next tranche: `T-20260331-Q` AttnRes-lite dynamic depth routing (planned, needs code support)
- next tranche: `T-20260331-R` architecture-specific schedules on the hybrid mixer winner
- later backlog: two dedicated MLP exploration tranches, broad-family first and polynomial second

## Go Deeper

- Tranche map: [`tranches.md`](./tranches.md)
- Idea bank: [`ideas.md`](./ideas.md)
- Durable findings: [`findings.md`](./findings.md)
- Experiment ledger: [`experiments.tsv`](./experiments.tsv)
- Runtime ledger: [`results.tsv`](./results.tsv)
- Experiment dashboard: [`plots/experiments.html`](./plots/experiments.html)
- Latest narrative log: [`2026-03-30-agent-lab.md`](../docs/build-logs/2026-03-30-agent-lab.md)
