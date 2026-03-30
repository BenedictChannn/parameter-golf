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
