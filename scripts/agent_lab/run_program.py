#!/usr/bin/env python3
"""Run multiple tranche manifests sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", help="Path to a JSON program manifest.")
    parser.add_argument("--execute", action="store_true", help="Actually run the program; default is dry-run.")
    parser.add_argument("--resume", action="store_true", help="Pass --resume to each tranche runner.")
    parser.add_argument("--state-dir", default=str(REPO_ROOT / "agent_lab" / "tranche_runs"))
    return parser.parse_args()


def load_program(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"program_id", "title", "tranches"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"program manifest missing required keys: {', '.join(missing)}")
    return data


def main() -> None:
    args = parse_args()
    program_path = Path(args.program)
    if not program_path.is_absolute():
        program_path = REPO_ROOT / program_path
    program = load_program(program_path)
    tranche_paths = program["tranches"]
    if not args.execute:
        print(f"Program {program['program_id']}: {program['title']}")
        for idx, tranche in enumerate(tranche_paths, start=1):
            print(f"  {idx}. {tranche}")
        return

    runner = REPO_ROOT / "scripts" / "agent_lab" / "run_tranche.py"
    for tranche in tranche_paths:
        tranche_path = tranche if Path(tranche).is_absolute() else str((REPO_ROOT / tranche).resolve())
        cmd = [sys.executable, str(runner), tranche_path, "--execute", "--state-dir", args.state_dir]
        if args.resume:
            cmd.append("--resume")
        print(f"[run_program] starting {tranche}")
        proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)
        print(f"[run_program] finished {tranche}")


if __name__ == "__main__":
    main()
