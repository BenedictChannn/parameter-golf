# Tranche Manifests

These JSON files are the execution surfaces for `scripts/agent_lab/run_tranche.py`.

Use them to define:

- tranche id and title
- research question
- primary metric
- fixed env
- anchor experiment
- candidate experiments
- optional stop or promotion notes for humans

They are not meant to replace [`tranches.md`](../tranches.md). The markdown file remains the human-readable research map. The manifest is the machine-readable execution plan.

Important:

- manifests in this folder are intended to be executable with the current codebase
- manifests under `planned/` are research blueprints for future tranches that still require code support
- do not execute a planned manifest until the required env knobs or code paths exist in `agent_lab/train_gpt.py`
- sequential multi-tranche runs can be chained with `python3 scripts/agent_lab/run_program.py ...`

Optional human-only fields are allowed and ignored by the runner, for example:

- `goal`
- `what_it_teaches`
- `why_worthy`
- `implementation_status`

Minimal shape:

```json
{
  "tranche_id": "T-YYYYMMDD-X",
  "title": "Short title",
  "research_question": "What are we trying to learn?",
  "primary_metric": "final_int8_ttt_lora",
  "run_script": "scripts/agent_lab/run_exp.sh",
  "results_tsv": "agent_lab/results.tsv",
  "experiments_tsv": "agent_lab/experiments.tsv",
  "anchor_exp_id": "AL-YYYYMMDD-NNN",
  "fixed_env": {
    "DATA_PATH": "./data/datasets/fineweb10B_sp1024/",
    "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
    "MAX_WALLCLOCK_SECONDS": "600"
  },
  "candidates": [
    {
      "name": "slug",
      "exp_id": "AL-YYYYMMDD-NNN",
      "hypothesis": "One sentence",
      "description": "Short human summary for results.tsv",
      "env": {
        "NUM_HEADS": "4"
      }
    }
  ]
}
```
