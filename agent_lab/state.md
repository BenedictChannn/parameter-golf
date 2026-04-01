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

- Queue: `S -> T -> U -> V -> W -> X`
- Status: completed
- Summary:
- `S` produced the new frontier and proved cheap routing stacks with the stronger hybrid winner
- `T` ran successfully after the backend fix, but local attention still lost cleanly on the hybrid backbone
- `U` showed that naive low-rank factorization is not yet a viable compression-native architecture
- `V` revived dynamic depth routing in a narrow top-only form
- `W` kept the broad MLP family anchored on `relu^2`
- `X` showed mixed linear-plus-quadratic MLP structure is the only polynomial variant still near the frontier

## Latest Completed Architecture Families

- `Y` MLP Structure Minimalism: completed
- `AB` Compression-Native Sharing: completed
- `AA` Upper Attention Decomposition: completed
- `Z` Block Uniformity Audit: partially run; `Z1` crashed due to dead MLP-scale parameters on mixer-only blocks and needs a clean rerun on the fixed commit

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

## Open Questions

- Does the `AL-20260331-017` winner still have local headroom in mixer width, kernel, or routing strength, or is this branch now mostly polished?
- Is interleaving upper attention and lighter refinement blocks a real simplification path, or was [`AL-20260401-059`](./experiments.tsv) only a near-miss?
- If compression-native architecture is still important, what should replace both naive low-rank factorization and naive cross-layer weight sharing?
- Should the next MLP follow-up focus only on the two live survivors: gated MLPs under tighter size control and mixed linear-plus-quadratic polynomial forms?

## Next Runnable Queue

- rerun [`T-20260401-Z`](./tranche_manifests/20260401-Z-block-uniformity-audit.json) cleanly on the fixed mixer-only-block implementation
- after `Z` is repaired, decide whether `AA4` deserves a focused follow-up branch around interleaved upper attention
- after that, three new smoke-validated bold branches are ready: [`T-20260401-AC`](./tranche_manifests/20260401-AC-conditional-heavy-light-compute.json), [`T-20260401-AD`](./tranche_manifests/20260401-AD-latent-upper-reasoner.json), and [`T-20260401-AE`](./tranche_manifests/20260401-AE-structured-sharing-deltas.json)
- keep the current frontier [`AL-20260331-017`](./experiments.tsv) as the main comparison anchor

## Later Backlog

- revisit the `AL-20260331-017` winner only after the repaired `Z` tranche and any `AA4` follow-up land
- narrow top-only routing follow-up only if it still looks complementary after the upper-attention story is clearer
- later MLP follow-up around gated SiLU under tighter size control and `relu + quadratic` variants
- full bold unknown-branch queue prepared at [`20260401-ACADAE.json`](./program_manifests/20260401-ACADAE.json)

## Go Deeper

- Frontier memo: [`frontier.md`](./frontier.md)
- Tranche map: [`tranches.md`](./tranches.md)
- Idea bank: [`ideas.md`](./ideas.md)
- Durable findings: [`findings.md`](./findings.md)
- Experiment ledger: [`experiments.tsv`](./experiments.tsv)
- Runtime ledger: [`results.tsv`](./results.tsv)
- Experiment dashboard: [`plots/experiments.html`](./plots/experiments.html)
- Latest narrative log: [`2026-03-31-agent-lab.md`](../docs/build-logs/2026-03-31-agent-lab.md)
