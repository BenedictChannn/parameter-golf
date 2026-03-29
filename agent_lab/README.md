# Agent lab harness (Karpathy-style loop)

This folder implements an autonomous experiment loop for Parameter Golf: a coding agent runs training from the **repository root**, compares **`val_bpb`** on a chosen final line, and keeps or reverts git state. Pure sweeps should usually be driven by env vars; code changes should stay confined to **`train_gpt.py` here**. See [`program.md`](program.md) for the full agent playbook.

Design is inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (upstream project name unchanged).

**Runtime note:** On some Linux hosts, `torch` wheels built for CUDA 13.x can fail driver init. This workspace currently runs with the system `python3` / `torchrun` stack, and the wrapper activates `.venv` only if one is present. With the current local stack, full train + **`final_int8_ttt_lora`** eval is roughly **~23–25 min** per run on a 3090, with TTT eval still dominating wall time.

## Human quickstart

1. **Data** (once): from repo root, follow the main [README](../README.md) to download FineWeb + tokenizer (e.g. `python3 data/cached_challenge_fineweb.py --variant sp1024`).

2. **Refresh the training script** (optional): if root `train_gpt.py` changed and you want this copy to match:
   ```bash
   cp train_gpt.py agent_lab/train_gpt.py
   ```

3. **Run a single experiment** (from repo root). Recommended wrapper (activates `.venv`, sets `RUN_ID` if unset):
   ```bash
   cd /path/to/parameter-golf
   RUN_ID=agent_lab_smoke ./scripts/agent_lab/run_exp.sh > agent_lab/run.log 2>&1
   ```
   Or set env vars yourself and call `torchrun` as in the main README.

   **Artifact headroom:** the log line `Total submission size int8+zlib:` is code + compressed weights. The challenge cap is **16,000,000 bytes** total — if you see **~9.9 MB** there, you still have room to grow the model or change compression **as long as** train/eval stay within official limits.

4. **Point your agent at `program.md`** and the Cursor skill **[`.cursor/skills/agent-lab/SKILL.md`](../.cursor/skills/agent-lab/SKILL.md)** (workflow + what to update after each run).

5. **Journal / learning:** see **`docs/build-logs/`** for dated narrative logs (pedagogy + diary + code deltas), e.g. [`docs/build-logs/2026-03-28-agent-lab.md`](../docs/build-logs/2026-03-28-agent-lab.md).

6. **Research memory surfaces:** keep the high-level state and hypothesis map current:
   - [`state.md`](state.md) for the short dashboard
   - [`tranches.md`](tranches.md) for the research-program map
   - [`ideas.md`](ideas.md) for the hypothesis bank
   These are the fast-orientation surfaces. Use them to summarize what is true, then link outward to `experiments.tsv` and the dated build log for details.

## What lives here

| File | Role |
|------|------|
| `program.md` | Instructions for the LLM (setup, loop, grep patterns, constraints). |
| `train_gpt.py` | Editable copy of the baseline training script for structural experiments; **do not edit root `train_gpt.py` for agent-lab runs** unless you intend to change the shared baseline. |
| `experiments.tsv` | **Structured experiment registry** — stable IDs `AL-YYYYMMDD-NNN`, parent commit, hypothesis, **verdict**, metrics (for humans + agents). **Commit this** when you add rows. |
| `state.md` | **Short dashboard** — current best, active tranche, working beliefs, and next planned runs. |
| `tranches.md` | **Research-program map** — tranche goals, controls, findings, and pivot rules. |
| `ideas.md` | **Hypothesis bank** — active, new, parked, and revisit-later ideas with links back to evidence. |
| [`.cursor/skills/agent-lab/SKILL.md`](../.cursor/skills/agent-lab/SKILL.md) | **Commit conventions**, metric line meanings, official time limits, interaction effects, build-log voice. |
| [`scripts/agent_lab/run_exp.sh`](../scripts/agent_lab/run_exp.sh) | Default env + `torchrun` from repo root (optional). |

`results.tsv` and `run.log` are gitignored; create them per `program.md`.

Treat `state.md` + `tranches.md` + `ideas.md` as the autonomous lab's command system: short summaries first, evidence links second.

## Metric choice

`program.md` defaults to the **`final_int8_ttt_lora`** line as the primary score (aligned with the competition-style path in the script). You can switch the documented primary metric in `program.md` if your research targets the zlib roundtrip line instead.

## Rules reminder

Follow the project guardrails: no validation leakage, keep **`val_bpb`** accounting correct if you touch tokenizer/eval paths, respect the 16 MB artifact story for record-track work, and log `RUN_ID` / `SEED` for reproducibility.
