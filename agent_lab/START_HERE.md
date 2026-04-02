# Agent Lab Start Here

Use this file as the single handoff pointer for a new session.

If a future agent asks what to read first, point it here.

## Read Order

1. [AGENTS.md](../AGENTS.md)
2. [README.md](./README.md)
3. [program.md](./program.md)
4. [state.md](./state.md)
5. [frontier.md](./frontier.md)
6. [findings.md](./findings.md)
7. [tranches.md](./tranches.md)
8. [ideas.md](./ideas.md)
9. [experiments.tsv](./experiments.tsv)
10. [.cursor/rules/parameter-golf.mdc](../.cursor/rules/parameter-golf.mdc)

## What Each File Is For

- [AGENTS.md](../AGENTS.md): repo-level doctrine, research behavior, hard constraints, and what to maintain after each run
- [README.md](./README.md): agent-lab harness overview, runtime notes, and where the memory surfaces live
- [program.md](./program.md): exact operating loop, logging rules, metrics, commit format, and after-run checklist
- [state.md](./state.md): current best run, live questions, and what matters right now
- [frontier.md](./frontier.md): one-page research synthesis of what the model seems to want
- [findings.md](./findings.md): durable conclusions and what evidence supports them
- [tranches.md](./tranches.md): why each experiment family exists and what it taught
- [ideas.md](./ideas.md): active, parked, and blocked hypotheses
- [experiments.tsv](./experiments.tsv): exact experiment ledger
- [docs/build-logs/](../docs/build-logs/): long-form narrative trail and debugging history

## Current Default Workflow

1. Read the current theory surfaces first: [state.md](./state.md), [frontier.md](./frontier.md), [findings.md](./findings.md)
2. Read the active branch map: [tranches.md](./tranches.md), [ideas.md](./ideas.md)
3. Read the exact evidence if needed: [experiments.tsv](./experiments.tsv)
4. Read the latest dated build log for narrative context:
   - [2026-04-01-agent-lab.md](../docs/build-logs/2026-04-01-agent-lab.md)
   - [2026-03-31-agent-lab.md](../docs/build-logs/2026-03-31-agent-lab.md)

## After Every Meaningful Run

Do all of these:

1. append [experiments.tsv](./experiments.tsv)
2. update [state.md](./state.md)
3. update [findings.md](./findings.md)
4. update [tranches.md](./tranches.md)
5. update [ideas.md](./ideas.md)
6. update the active dated build log under [docs/build-logs/](../docs/build-logs/)
7. regenerate plots with:

```bash
python3 scripts/agent_lab/plot_experiments.py
```

8. commit with a rich `docs(agent-lab): ...` or `feat(agent-lab): ...` message

## Current Research Situation

- current best valid run: [AL-20260331-017](./experiments.tsv)
- strongest secondary near-survivors:
  - [AL-20260401-055](./experiments.tsv)
  - [AL-20260401-059](./experiments.tsv)
  - [AL-20260401-077](./experiments.tsv)
  - [AL-20260401-080](./experiments.tsv)
- blocked branch that still needs repair:
  - [AL-20260401-071](./experiments.tsv) / latent upper reasoner

## Simple Summary

The current theory is:

- lower-stack simplification is real
- upper global reasoning still matters
- FFNs are still important
- carefully placed asymmetry works better than blunt simplification
- compression-native structure is still open, but lower-stage structured sharing with deltas is the newest live clue
