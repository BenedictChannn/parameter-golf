#!/usr/bin/env python3
"""Run a manifest-defined tranche of agent-lab experiments sequentially."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agent_lab.summarize_run import (
    append_results_row,
    infer_status,
    load_ledger,
    parse_run_log,
    render_markdown,
    render_tsv_note,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="Path to a JSON tranche manifest.")
    parser.add_argument("--execute", action="store_true", help="Actually run the tranche; default is dry-run.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing state file if present.")
    parser.add_argument("--max-runs", type=int, default=0, help="Optional cap on how many candidates to execute.")
    parser.add_argument("--state-dir", default=str(REPO_ROOT / "agent_lab" / "tranche_runs"))
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"tranche_id", "title", "primary_metric", "run_script", "fixed_env", "candidates"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"manifest missing required keys: {', '.join(missing)}")
    return data


def load_state(path: Path, manifest: dict, resume: bool) -> dict:
    if resume and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "manifest_path": manifest.get("_path", ""),
        "tranche_id": manifest["tranche_id"],
        "title": manifest["title"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed": {},
    }


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def resolved_env(fixed_env: dict[str, str], candidate_env: dict[str, str], run_id: str) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in fixed_env.items():
        env[str(key)] = str(value)
    for key, value in candidate_env.items():
        env[str(key)] = str(value)
    env["RUN_ID"] = run_id
    return env


def run_candidate(
    repo_root: Path,
    manifest: dict,
    candidate: dict,
    state_dir: Path,
) -> dict:
    exp_id = candidate.get("exp_id") or candidate["name"]
    run_log = repo_root / "agent_lab" / "run.log"
    tranche_dir = state_dir / manifest["tranche_id"]
    tranche_dir.mkdir(parents=True, exist_ok=True)
    run_script = repo_root / manifest["run_script"]
    env = resolved_env(manifest["fixed_env"], candidate.get("env", {}), exp_id)
    with run_log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            [str(run_script)],
            cwd=repo_root,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    ledger = load_ledger(repo_root / manifest.get("experiments_tsv", "agent_lab/experiments.tsv"))
    summary = parse_run_log(run_log, manifest["primary_metric"])
    status = infer_status(summary, ledger, candidate.get("status", ""))
    anchor = candidate.get("anchor_exp_id", manifest.get("anchor_exp_id", ""))
    note = render_tsv_note(summary, next((row for row in ledger if row.exp_id == anchor), None), min(ledger, key=lambda row: row.val_bpb) if ledger else None)
    append_results_row(
        repo_root / manifest.get("results_tsv", "agent_lab/results.tsv"),
        exp_id=exp_id,
        commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, text=True, capture_output=True, check=True).stdout.strip(),
        summary=summary,
        status="crash" if proc.returncode else status,
        description=candidate.get("description", "") or note,
    )
    summary_md = render_markdown(
        summary,
        ledger,
        exp_id,
        anchor,
        candidate.get("hypothesis", ""),
        {**manifest["fixed_env"], **candidate.get("env", {})},
    )
    (tranche_dir / f"{exp_id}.md").write_text(summary_md, encoding="utf-8")
    return {
        "exp_id": exp_id,
        "returncode": proc.returncode,
        "status": "crash" if proc.returncode else status,
        "summary": {
            "primary_val_bpb": summary.primary_val_bpb,
            "artifact_bytes": summary.artifact_bytes,
            "steps": summary.steps,
        },
        "summary_md_path": str((tranche_dir / f"{exp_id}.md").relative_to(repo_root)),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def dry_run_manifest(repo_root: Path, manifest: dict) -> None:
    print(f"Tranche {manifest['tranche_id']}: {manifest['title']}")
    print(f"Primary metric: {manifest['primary_metric']}")
    print("Fixed env:")
    for key in sorted(manifest["fixed_env"]):
        print(f"  {key}={manifest['fixed_env'][key]}")
    print("Candidates:")
    for idx, candidate in enumerate(manifest["candidates"], start=1):
        exp_id = candidate.get("exp_id") or candidate["name"]
        merged = {**manifest["fixed_env"], **candidate.get("env", {})}
        print(f"  {idx}. {exp_id}")
        print(f"     hypothesis: {candidate.get('hypothesis', '')}")
        print(f"     env: {json.dumps(merged, sort_keys=True)}")
    print(f"Run script: {(repo_root / manifest['run_script']).relative_to(repo_root)}")


def main() -> None:
    args = parse_args()
    repo_root = REPO_ROOT
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    manifest = load_manifest(manifest_path)
    manifest["_path"] = str(manifest_path)
    state_dir = Path(args.state_dir)
    if not state_dir.is_absolute():
        state_dir = repo_root / state_dir
    state_path = state_dir / f"{manifest['tranche_id']}.json"
    state = load_state(state_path, manifest, args.resume)

    if not args.execute:
        dry_run_manifest(repo_root, manifest)
        return

    completed = state.get("completed", {})
    remaining = [candidate for candidate in manifest["candidates"] if (candidate.get("exp_id") or candidate["name"]) not in completed]
    if args.max_runs > 0:
        remaining = remaining[: args.max_runs]
    if not remaining:
        print("No remaining candidates to run.")
        return

    for candidate in remaining:
        exp_id = candidate.get("exp_id") or candidate["name"]
        print(f"[run_tranche] starting {exp_id}")
        result = run_candidate(repo_root, manifest, candidate, state_dir)
        completed[exp_id] = result
        state["completed"] = completed
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()
        save_state(state_path, state)
        subprocess.run([sys.executable, str(repo_root / "scripts" / "agent_lab" / "plot_experiments.py")], cwd=repo_root, check=False)
        if result["returncode"] != 0:
            print(f"[run_tranche] {exp_id} crashed with return code {result['returncode']}")
            break
        print(f"[run_tranche] finished {exp_id} -> {result['status']}")


if __name__ == "__main__":
    main()
