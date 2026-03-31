#!/usr/bin/env python3
"""Parse an agent-lab training log into a reusable structured summary.

The script is intentionally standard-library only so it can run anywhere the
main harness runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class LedgerExperiment:
    exp_id: str
    val_bpb: float
    status: str
    notes: str


@dataclass
class RunSummary:
    run_id: str
    primary_metric: str
    primary_val_bpb: float | None
    roundtrip_val_bpb: float | None
    steps: int | None
    train_time_ms: int | None
    eval_time_ms: int | None
    memory_mib: int | None
    artifact_bytes: int | None
    model_bytes: int | None
    model_params: int | None
    num_heads: int | None
    num_kv_heads: int | None
    tie_embeddings: bool | None
    head_lr: float | None
    matrix_lr: float | None
    train_batch_tokens: int | None
    max_wallclock_seconds: float | None
    seed: int | None


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-log", default=str(repo_root / "agent_lab" / "run.log"))
    parser.add_argument("--experiments-tsv", default=str(repo_root / "agent_lab" / "experiments.tsv"))
    parser.add_argument("--results-tsv", default=str(repo_root / "agent_lab" / "results.tsv"))
    parser.add_argument("--exp-id", default="", help="Optional stable experiment id for display and results.tsv append.")
    parser.add_argument("--anchor-exp-id", default="", help="Optional tranche anchor experiment id for delta reporting.")
    parser.add_argument(
        "--primary-metric",
        default="final_int8_ttt_lora",
        choices=("final_int8_ttt_lora", "final_int8_zlib_roundtrip_exact", "final_int8_zlib_roundtrip"),
    )
    parser.add_argument("--hypothesis", default="", help="Optional hypothesis string for markdown output.")
    parser.add_argument("--description", default="", help="Optional short description for results.tsv output.")
    parser.add_argument("--config-json", default="", help="Optional JSON object of resolved env overrides for this run.")
    parser.add_argument("--status", default="", choices=("", "keep", "discard", "crash"))
    parser.add_argument("--append-results", action="store_true", help="Append a row to agent_lab/results.tsv.")
    parser.add_argument("--format", default="md", choices=("md", "json", "tsv_note"))
    return parser.parse_args()


def load_ledger(path: Path) -> list[LedgerExperiment]:
    rows: list[LedgerExperiment] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                val_bpb = float(row["val_bpb"])
            except (TypeError, ValueError):
                continue
            rows.append(
                LedgerExperiment(
                    exp_id=row["exp_id"],
                    val_bpb=val_bpb,
                    status=row.get("status", ""),
                    notes=row.get("notes", ""),
                )
            )
    return rows


def parse_run_log(path: Path, primary_metric: str) -> RunSummary:
    text = path.read_text(encoding="utf-8")
    primary_match = _search(text, rf"^{re.escape(primary_metric)} val_loss:\S+ val_bpb:(\S+)(?: eval_time:(\d+)ms)?", re.MULTILINE)
    roundtrip_match = _search(text, r"^final_int8_zlib_roundtrip_exact val_loss:\S+ val_bpb:(\S+)", re.MULTILINE)
    latest_step_match = None
    for latest_step_match in re.finditer(
        r"^step:(\d+)/\d+ .*?train_time:(\d+)ms(?: step_avg:\S+)?$",
        text,
        re.MULTILINE,
    ):
        pass
    run_id_match = _search(text, r"^logs/(.+?)\.txt$", re.MULTILINE)
    model_params_match = _search(text, r"^model_params:(\d+)$", re.MULTILINE)
    attention = _search(text, r"^attention_mode:\S+ num_heads:(\d+) num_kv_heads:(\d+)$", re.MULTILINE)
    tie_line = _search(
        text,
        r"^tie_embeddings:(True|False) embed_lr:\S+ head_lr:(\S+) matrix_lr:(\S+) scalar_lr:\S+$",
        re.MULTILINE,
    )
    batch_line = _search(
        text,
        r"^train_batch_tokens:(\d+) train_seq_len:\d+ iterations:\d+ warmup_steps:\d+ max_wallclock_seconds:(\S+)$",
        re.MULTILINE,
    )
    memory_match = _search(text, r"^peak memory allocated:\s*(\d+)\s+MiB", re.MULTILINE)
    artifact_match = _search(text, r"^Total submission size int8\+zlib:\s*(\d+)\s+bytes$", re.MULTILINE)
    model_bytes_match = _search(text, r"^Serialized model int8\+zlib:\s*(\d+)\s+bytes", re.MULTILINE)
    seed_match = _search(text, r"^seed:(\d+)$", re.MULTILINE)
    return RunSummary(
        run_id=run_id_match.group(1) if run_id_match else path.stem,
        primary_metric=primary_metric,
        primary_val_bpb=_parse_float(primary_match.group(1)) if primary_match else None,
        roundtrip_val_bpb=_parse_float(roundtrip_match.group(1)) if roundtrip_match else None,
        steps=_parse_int(latest_step_match.group(1)) if latest_step_match else None,
        train_time_ms=_parse_int(latest_step_match.group(2)) if latest_step_match else None,
        eval_time_ms=_parse_int(primary_match.group(2)) if primary_match and primary_match.group(2) else None,
        memory_mib=_parse_int(memory_match.group(1)) if memory_match else None,
        artifact_bytes=_parse_int(artifact_match.group(1)) if artifact_match else None,
        model_bytes=_parse_int(model_bytes_match.group(1)) if model_bytes_match else None,
        model_params=_parse_int(model_params_match.group(1)) if model_params_match else None,
        num_heads=_parse_int(attention.group(1)) if attention else None,
        num_kv_heads=_parse_int(attention.group(2)) if attention else None,
        tie_embeddings=_parse_bool(tie_line.group(1)) if tie_line else None,
        head_lr=_parse_float(tie_line.group(2)) if tie_line else None,
        matrix_lr=_parse_float(tie_line.group(3)) if tie_line else None,
        train_batch_tokens=_parse_int(batch_line.group(1)) if batch_line else None,
        max_wallclock_seconds=_parse_float(batch_line.group(2)) if batch_line else None,
        seed=_parse_int(seed_match.group(1)) if seed_match else None,
    )


def infer_status(summary: RunSummary, ledger: list[LedgerExperiment], requested: str) -> str:
    if requested:
        return requested
    if summary.primary_val_bpb is None:
        return "crash"
    scored = [row for row in ledger if row.exp_id != ""]
    if not scored:
        return "keep"
    current_best = min(row.val_bpb for row in scored)
    return "keep" if summary.primary_val_bpb < current_best else "discard"


def render_markdown(
    summary: RunSummary,
    ledger: list[LedgerExperiment],
    exp_id: str,
    anchor_exp_id: str,
    hypothesis: str,
    config: dict[str, object],
) -> str:
    current_best = min(ledger, key=lambda row: row.val_bpb) if ledger else None
    anchor = next((row for row in ledger if row.exp_id == anchor_exp_id), None) if anchor_exp_id else None
    lines = []
    title = exp_id or summary.run_id
    lines.append(f"### {title}")
    if hypothesis:
        lines.append("")
        lines.append(f"- hypothesis: {hypothesis}")
    if config:
        lines.append(f"- config: `{render_config(config)}`")
    lines.append(f"- primary metric: `{summary.primary_metric}`")
    if summary.primary_val_bpb is not None:
        lines.append(f"- primary `val_bpb`: `{summary.primary_val_bpb:.4f}`")
    if summary.roundtrip_val_bpb is not None:
        lines.append(f"- roundtrip `val_bpb`: `{summary.roundtrip_val_bpb:.4f}`")
    if summary.steps is not None and summary.train_time_ms is not None:
        lines.append(f"- training: `{summary.steps}` steps in `{summary.train_time_ms}ms`")
    if summary.eval_time_ms is not None:
        lines.append(f"- eval time: `{summary.eval_time_ms}ms`")
    if summary.memory_mib is not None:
        lines.append(f"- peak memory: `{summary.memory_mib} MiB`")
    if summary.artifact_bytes is not None:
        lines.append(f"- artifact size: `{summary.artifact_bytes}` bytes int8+zlib")
    if current_best and summary.primary_val_bpb is not None:
        delta_best = summary.primary_val_bpb - current_best.val_bpb
        pct_best = percent_delta(summary.primary_val_bpb, current_best.val_bpb)
        lines.append(
            f"- delta vs current best `{current_best.exp_id}`: `{delta_best:+.4f}` val_bpb "
            f"({describe_delta(pct_best, 'worse', 'better')})"
        )
    if anchor and summary.primary_val_bpb is not None:
        delta_anchor = summary.primary_val_bpb - anchor.val_bpb
        pct_anchor = percent_delta(summary.primary_val_bpb, anchor.val_bpb)
        lines.append(
            f"- delta vs anchor `{anchor.exp_id}`: `{delta_anchor:+.4f}` val_bpb "
            f"({describe_delta(pct_anchor, 'worse', 'better')})"
        )
    if anchor and summary.steps is not None:
        anchor_steps = extract_steps(anchor.notes)
        if anchor_steps is not None and anchor_steps > 0:
            pct_steps = percent_delta(summary.steps, anchor_steps)
            lines.append(
                f"- steps vs anchor `{anchor.exp_id}`: `{summary.steps - anchor_steps:+d}` "
                f"({describe_delta(pct_steps, 'fewer', 'more')})"
            )
    if anchor and summary.artifact_bytes is not None:
        anchor_bytes = extract_artifact_bytes(anchor.notes)
        if anchor_bytes is not None and anchor_bytes > 0:
            pct_bytes = percent_delta(summary.artifact_bytes, anchor_bytes)
            lines.append(
                f"- artifact vs anchor `{anchor.exp_id}`: `{summary.artifact_bytes - anchor_bytes:+,}` bytes "
                f"({describe_delta(-pct_bytes, 'larger', 'smaller')})"
            )
    if summary.num_heads is not None and summary.num_kv_heads is not None:
        lines.append(f"- attention geometry: `q{summary.num_heads}/kv{summary.num_kv_heads}`")
    if summary.tie_embeddings is not None:
        lines.append(f"- tied outputs: `{summary.tie_embeddings}`")
    return "\n".join(lines) + "\n"


def render_tsv_note(summary: RunSummary, anchor: LedgerExperiment | None, current_best: LedgerExperiment | None) -> str:
    parts = []
    if summary.steps is not None and summary.train_time_ms is not None:
        parts.append(f"{summary.steps} steps in {summary.train_time_ms / 1000:.1f}s")
    if summary.artifact_bytes is not None:
        parts.append(f"artifact {summary.artifact_bytes / 1_000_000:.2f} MB")
    if summary.primary_val_bpb is not None and current_best is not None:
        pct_best = percent_delta(summary.primary_val_bpb, current_best.val_bpb)
        parts.append(
            f"vs best {current_best.exp_id} {summary.primary_val_bpb - current_best.val_bpb:+.4f} "
            f"({describe_delta(pct_best, 'worse', 'better')})"
        )
    if summary.primary_val_bpb is not None and anchor is not None:
        pct_anchor = percent_delta(summary.primary_val_bpb, anchor.val_bpb)
        parts.append(
            f"vs anchor {anchor.exp_id} {summary.primary_val_bpb - anchor.val_bpb:+.4f} "
            f"({describe_delta(pct_anchor, 'worse', 'better')})"
        )
    return "; ".join(parts)


def percent_delta(value: float | int | None, baseline: float | int | None) -> float | None:
    if value is None or baseline in (None, 0):
        return None
    return ((float(value) - float(baseline)) / float(baseline)) * 100.0


def describe_delta(pct: float | None, positive_label: str, negative_label: str) -> str:
    if pct is None:
        return "n/a"
    if abs(pct) < 0.05:
        return "flat"
    label = positive_label if pct > 0 else negative_label
    return f"{abs(pct):.2f}% {label}"


def extract_steps(notes: str) -> int | None:
    match = re.search(r"(\d+)\s+steps\s+in", notes)
    return int(match.group(1)) if match else None


def extract_artifact_bytes(notes: str) -> int | None:
    match_mb = re.search(r"artifact\s+(\d+\.\d+)\s+MB", notes)
    if match_mb:
        return int(float(match_mb.group(1)) * 1_000_000)
    match_bytes = re.search(r"artifact\s+(\d+)\s+bytes", notes)
    if match_bytes:
        return int(match_bytes.group(1))
    return None


def append_results_row(path: Path, exp_id: str, commit: str, summary: RunSummary, status: str, description: str) -> None:
    if not exp_id:
        raise ValueError("--exp-id is required when using --append-results")
    existing_ids = set()
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                existing_ids.add(row.get("exp_id", ""))
    if exp_id in existing_ids:
        return
    header_needed = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if header_needed:
            writer.writerow(["exp_id", "commit", "val_bpb", "memory_gb", "status", "description"])
        memory_gb = f"{(summary.memory_mib or 0) / 1024:.1f}" if summary.memory_mib is not None else "0.0"
        val_bpb = f"{summary.primary_val_bpb:.4f}" if summary.primary_val_bpb is not None else "0.0000"
        writer.writerow([exp_id, commit, val_bpb, memory_gb, status, description])


def render_config(config: dict[str, object]) -> str:
    ordered = [f"{key}={config[key]}" for key in sorted(config)]
    return ", ".join(ordered)


def _search(text: str, pattern: str, flags: int = 0):
    return re.search(pattern, text, flags)


def _parse_int(text: str | None) -> int | None:
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


def _parse_float(text: str | None) -> float | None:
    try:
        return float(text) if text is not None else None
    except ValueError:
        return None


def _parse_bool(text: str | None) -> bool | None:
    if text is None:
        return None
    if text == "True":
        return True
    if text == "False":
        return False
    return None


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    run_log = Path(args.run_log)
    experiments_tsv = Path(args.experiments_tsv)
    results_tsv = Path(args.results_tsv)
    config = json.loads(args.config_json) if args.config_json else {}
    ledger = load_ledger(experiments_tsv)
    summary = parse_run_log(run_log, args.primary_metric)
    status = infer_status(summary, ledger, args.status)
    anchor = next((row for row in ledger if row.exp_id == args.anchor_exp_id), None) if args.anchor_exp_id else None
    current_best = min(ledger, key=lambda row: row.val_bpb) if ledger else None

    if args.append_results:
        commit = (
            subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, text=True, capture_output=True, check=True)
            .stdout.strip()
        )
        description = args.description or render_tsv_note(summary, anchor, current_best)
        append_results_row(results_tsv, args.exp_id, commit, summary, status, description)

    if args.format == "json":
        payload = {
            "summary": asdict(summary),
            "status": status,
            "config": config,
            "anchor_exp_id": args.anchor_exp_id,
            "current_best_exp_id": current_best.exp_id if current_best else "",
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.format == "tsv_note":
        print(render_tsv_note(summary, anchor, current_best))
    else:
        print(render_markdown(summary, ledger, args.exp_id, args.anchor_exp_id, args.hypothesis, config).rstrip())


if __name__ == "__main__":
    main()
