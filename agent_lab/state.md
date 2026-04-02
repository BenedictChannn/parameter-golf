# Agent Lab State

This is the first-read dashboard for the current research state. Use this for the high-level picture, then drill down into the ledger, tranche map, findings, and build logs.

## Current Best Valid

- Experiment: [`AL-20260331-017`](./experiments.tsv)
- Commit: `1f2c07b`
- Primary metric: `final_int8_ttt_lora`
- Best `val_bpb`: `1.3429`
- Winning branch shape: `9` layers, `MODEL_DIM=512`, `MLP_MULT=2`, `NUM_HEADS=4`, `NUM_KV_HEADS=2`, `TRAIN_BATCH_TOKENS=98304`, lower four attention layers replaced by sequence mixers with `MIXER_DIM=320`, untied dense head, `LOGIT_SOFTCAP=20`, `HEAD_LR=0.012`, plus shared scalar skip gates
- Total artifact size: `15,025,513` bytes int8+zlib

## Best Secondary Valid

- Experiment: [`AL-20260331-035`](./experiments.tsv)
- Commit: `1f2c07b`
- Primary metric: `final_int8_ttt_lora`
- Best `val_bpb`: `1.3433`
- Branch shape: current hybrid winner plus shared scalar skip gates and top-two-layer AttnRes-lite
- Total artifact size: `14,948,987` bytes int8+zlib

## Latest Completed Program

- Queue: `Z -> AC -> AD -> AE`
- Status: partially completed
- Summary:
- `Z` completed cleanly after the mixer-only-block fix and showed that block non-uniformity is real, but only in narrow periodic forms rather than as broad lower-stage FFN removal
- `AC` mostly falsified the first token-selective-compute story; selective heavy attention was especially destructive
- `AD` crashed on its first learned-slot candidate and remains operationally unresolved
- `AE` revived compression-native sharing in a second-generation form: pairwise lower-mixer sharing plus small deltas stayed close, while upper attention and FFN sharing were weaker

## Latest Completed Architecture Families

- `Y` MLP Structure Minimalism: completed
- `Z` Block Uniformity Audit: completed
- `AA` Upper Attention Decomposition: completed
- `AB` Compression-Native Sharing: completed
- `AC` Conditional Heavy / Light Compute: completed
- `AE` Structured Sharing With Layer Deltas: completed
- `AD` Latent Upper Reasoner: blocked by runtime crash on the first candidate

## Working Beliefs

- The frontier now wants a hybrid lower stack plus cleaner routing. Lower-four sequence mixers remain the right backbone, and shared scalar skip gates are the cleanest stacking win on top of it.
- Cheap routing really does stack with the hybrid winner. The new best run is not a brand-new family; it is the stronger hybrid backbone plus the simplest useful routing redesign.
- Local attention on the remaining attention layers looks mostly wrong on this backbone. Even the best repaired local-window run lost clearly, and broader upper-stack locality was much worse.
- Compression-native design is still a real gap, but naive low-rank factorization is not the answer. Attention factorization lost clearly, and MLP factorization collapsed the model.
- Broad late-layer AttnRes-lite was the wrong routing shape, but a very narrow top-only version is now real. Top-one and top-two two-source routing both beat the old hybrid anchor, and the skip-gate combo got close to the new best.
- The MLP family is still anchored on `relu^2`. Plain ReLU, SiLU, and GELU all lost clearly on the current backbone.
- The only broad-family MLP near-miss was gated SiLU, but it broke the size cap. That makes it a later capacity/compression question, not an immediate replacement.
- Inside the polynomial family, cubic structure looks wrong. The only live hint is `relu + 0.5 * relu^2`, which stayed very close to the frontier without beating it.
- Broad MLP-structure minimalism mostly failed. No-expand MLPs were decisively too weak, and even halving dense FFN width lost clearly. The only survivor was lightening the lower MLPs while keeping full upper MLPs, but it still remained clearly behind the frontier.
- Naive stage-wise weight sharing is not the next compression-native win. Lower-two sharing was the least bad sharing run, but broad lower sharing, top-two attention sharing, and share-to-reallocate variants all lost cleanly.
- The upper attention stack is not trivially removable, but it may be reducible by interleaving. The best AA run interleaving upper attention and lighter refinement blocks nearly tied the frontier, while top-two-only and single-final-reasoner plans failed badly.
- Block non-uniformity is now a live architectural clue. Alternating lower standard and mixer-only blocks stayed close, and periodic heavy/light blocks nearly tied the frontier, while broad lower-stage FFN removal still failed.
- Naive token-selective compute is not the next win. Top-only selective FFNs were merely survivable, but token-selective heavy attention and joint heavy-token paths collapsed badly.
- Structured sharing with small deltas is qualitatively better than whole-block sharing. The best lower-mixer pair-sharing run stayed close to the frontier, which suggests lower-stage redundancy exists but only if layer identity is preserved.

## Open Questions

- Does the `AL-20260331-017` winner still have local headroom in mixer width, kernel, or routing strength, or is this branch now mostly polished?
- Is periodic block concentration a real simplification path on top of the hybrid winner, or were [`AL-20260401-053`](./experiments.tsv) and [`AL-20260401-055`](./experiments.tsv) only near-misses?
- Is interleaving upper attention and lighter refinement blocks a real simplification path, or was [`AL-20260401-059`](./experiments.tsv) only a near-miss?
- If compression-native architecture is still important, can lower-stage structured sharing with deltas be strengthened enough to become a real branch, and what should replace the crashed latent-upper-reasoner line?

## Next Runnable Queue

- debug and rerun [`T-20260401-AD`](./tranche_manifests/20260401-AD-latent-upper-reasoner.json) so the latent upper-reasoner branch becomes science rather than only an implementation stub
- decide whether block non-uniformity deserves a focused follow-up around periodic heavy/light concentration and alternating lower FFNs
- decide whether lower-stage pair-sharing with deltas deserves a focused compression-native follow-up, especially as a share-then-reallocate branch
- keep the current frontier [`AL-20260331-017`](./experiments.tsv) as the main comparison anchor

## Later Backlog

- revisit the `AL-20260331-017` winner only after the latent upper-reasoner branch is repaired and the new near-survivors are either promoted or killed
- narrow top-only routing follow-up only if it still looks complementary after the upper-attention story is clearer
- later MLP follow-up around gated SiLU under tighter size control and `relu + quadratic` variants

## Go Deeper

- Frontier memo: [`frontier.md`](./frontier.md)
- Tranche map: [`tranches.md`](./tranches.md)
- Idea bank: [`ideas.md`](./ideas.md)
- Durable findings: [`findings.md`](./findings.md)
- Experiment ledger: [`experiments.tsv`](./experiments.tsv)
- Runtime ledger: [`results.tsv`](./results.tsv)
- Experiment dashboard: [`plots/experiments.html`](./plots/experiments.html)
- Latest narrative log: [`2026-03-31-agent-lab.md`](../docs/build-logs/2026-03-31-agent-lab.md)
