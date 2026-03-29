#!/usr/bin/env python3
"""Render a lightweight experiment dashboard from agent_lab/experiments.tsv.

The script uses only the Python standard library so it can run on minimal hosts.
It emits:

- agent_lab/plots/experiments.svg
- agent_lab/plots/experiments.html
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
from dataclasses import dataclass
from pathlib import Path


SVG_WIDTH = 1320
SVG_HEIGHT = 1040
CARD_W = 290
CARD_H = 78
PANEL_W = 610
PANEL_H = 320

STATUS_COLORS = {
    "keep": "#1f7a4d",
    "discard": "#b03a2e",
    "crash": "#d97706",
    "n/a": "#6b7280",
}


@dataclass
class Experiment:
    index: int
    exp_id: str
    date: str
    commit: str
    hypothesis: str
    verdict: str
    primary_metric: str
    val_bpb: float
    memory_gb: float | None
    status: str
    notes: str
    steps: int | None
    artifact_mb: float | None


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    return argparse.ArgumentParser(description=__doc__).parse_args([])


def load_experiments(path: Path) -> list[Experiment]:
    rows: list[Experiment] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for raw_index, row in enumerate(reader, start=1):
            try:
                val_bpb = float(row["val_bpb"])
            except (TypeError, ValueError):
                continue
            memory_gb = _parse_optional_float(row.get("memory_gb", ""))
            rows.append(
                Experiment(
                    index=len(rows) + 1,
                    exp_id=row["exp_id"],
                    date=row["date"],
                    commit=row["commit"],
                    hypothesis=row["hypothesis"],
                    verdict=row["verdict"],
                    primary_metric=row["primary_metric"],
                    val_bpb=val_bpb,
                    memory_gb=memory_gb,
                    status=row["status"],
                    notes=row["notes"],
                    steps=_extract_int(row["notes"], r"(\d+)\s+steps"),
                    artifact_mb=_extract_float(row["notes"], r"artifact\s+(\d+(?:\.\d+)?)\s+MB"),
                )
            )
    return rows


def _parse_optional_float(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _extract_int(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def _extract_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def render_dashboard(experiments: list[Experiment]) -> str:
    best = min(experiments, key=lambda exp: exp.val_bpb)
    keeps = sum(exp.status == "keep" for exp in experiments)
    discards = sum(exp.status == "discard" for exp in experiments)
    mean_bpb = sum(exp.val_bpb for exp in experiments) / max(len(experiments), 1)
    running_best = []
    current = math.inf
    for exp in experiments:
        current = min(current, exp.val_bpb)
        running_best.append(current)

    parts = [
        _svg_open(),
        _title_block(best, mean_bpb, keeps, discards, len(experiments)),
        _metric_card(40, 110, "Current Best", f"{best.val_bpb:.4f}", f"{best.exp_id} ({best.commit})"),
        _metric_card(350, 110, "Tracked Runs", str(len(experiments)), f"{keeps} keep / {discards} discard"),
        _metric_card(660, 110, "Primary Metric", best.primary_metric, "lower is better"),
        _metric_card(970, 110, "Mean val_bpb", f"{mean_bpb:.4f}", "across scored runs"),
        _line_panel(
            40,
            220,
            PANEL_W,
            PANEL_H,
            "val_bpb by experiment",
            experiments,
            [exp.val_bpb for exp in experiments],
            y_label="val_bpb",
            label_points=True,
        ),
        _line_panel(
            670,
            220,
            PANEL_W,
            PANEL_H,
            "Running best frontier",
            experiments,
            running_best,
            y_label="best so far",
            label_points=False,
        ),
        _scatter_panel(
            40,
            590,
            PANEL_W,
            PANEL_H,
            "Training steps vs val_bpb",
            experiments,
            [exp.steps for exp in experiments],
            x_label="steps in 600s",
            y_label="val_bpb",
        ),
        _scatter_panel(
            670,
            590,
            PANEL_W,
            PANEL_H,
            "Artifact size vs val_bpb",
            experiments,
            [exp.artifact_mb for exp in experiments],
            x_label="artifact MB",
            y_label="val_bpb",
        ),
        _notes_block(40, 940, best, experiments),
        "</svg>\n",
    ]
    return "".join(parts)


def _svg_open() -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" '
        f'viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">'
        '<rect width="100%" height="100%" fill="#f6f4ef"/>'
        '<style>'
        'text{font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#1f2937}'
        '.small{font-size:12px}.body{font-size:14px}.label{font-size:15px;font-weight:600}'
        '.title{font-size:28px;font-weight:700}.cardtitle{font-size:13px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;fill:#475569}'
        '.metric{font-size:30px;font-weight:700}.paneltitle{font-size:18px;font-weight:700}'
        '</style>'
    )


def _title_block(best: Experiment, mean_bpb: float, keeps: int, discards: int, total: int) -> str:
    summary = (
        f"Best run {best.exp_id} reached {best.val_bpb:.4f} val_bpb. "
        f"{total} scored runs are tracked here, with {keeps} kept and {discards} discarded."
    )
    return (
        '<text x="40" y="56" class="title">Agent Lab Experiment Dashboard</text>'
        f'<text x="40" y="84" class="body">{_escape(summary)}</text>'
    )


def _metric_card(x: int, y: int, title: str, metric: str, subtitle: str) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{CARD_W}" height="{CARD_H}" rx="14" fill="#ffffff" stroke="#d6d3d1"/>'
        f'<text x="{x + 20}" y="{y + 24}" class="cardtitle">{_escape(title)}</text>'
        f'<text x="{x + 20}" y="{y + 54}" class="metric">{_escape(metric)}</text>'
        f'<text x="{x + 20}" y="{y + 70}" class="small">{_escape(subtitle)}</text>'
    )


def _panel_frame(x: int, y: int, w: int, h: int, title: str) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="#ffffff" stroke="#d6d3d1"/>'
        f'<text x="{x + 20}" y="{y + 28}" class="paneltitle">{_escape(title)}</text>'
    )


def _line_panel(
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    experiments: list[Experiment],
    values: list[float],
    y_label: str,
    label_points: bool,
) -> str:
    panel = [_panel_frame(x, y, w, h, title)]
    inner = _plot_bounds(x, y, w, h)
    panel.append(_axes(inner, y_label, "experiment order"))
    for marker in _y_ticks(inner, values):
        panel.append(marker)
    points = [_project_point(inner, i, len(values), value, min(values), max(values)) for i, value in enumerate(values)]
    if len(points) > 1:
        panel.append(
            '<polyline fill="none" stroke="#1d4ed8" stroke-width="3" points="'
            + " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
            + '"/>'
        )
    for exp, (px, py) in zip(experiments, points):
        color = STATUS_COLORS.get(exp.status, "#6b7280")
        panel.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="5.5" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')
        if label_points:
            panel.append(
                f'<text x="{px + 8:.2f}" y="{py - 8:.2f}" class="small">{_escape(exp.exp_id[-3:])}</text>'
            )
    return "".join(panel)


def _scatter_panel(
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    experiments: list[Experiment],
    x_values: list[float | int | None],
    x_label: str,
    y_label: str,
) -> str:
    panel = [_panel_frame(x, y, w, h, title)]
    valid = [(exp, xv) for exp, xv in zip(experiments, x_values) if xv is not None]
    if not valid:
        panel.append(f'<text x="{x + 20}" y="{y + 60}" class="body">No data available yet.</text>')
        return "".join(panel)
    y_values = [exp.val_bpb for exp, _ in valid]
    xs = [float(xv) for _, xv in valid]
    inner = _plot_bounds(x, y, w, h)
    panel.append(_axes(inner, y_label, x_label))
    for marker in _y_ticks(inner, y_values):
        panel.append(marker)
    for marker in _x_ticks(inner, xs):
        panel.append(marker)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(y_values), max(y_values)
    for exp, xv in valid:
        px = _scale(float(xv), x_min, x_max, inner["left"], inner["right"])
        py = _scale(exp.val_bpb, y_min, y_max, inner["bottom"], inner["top"])
        color = STATUS_COLORS.get(exp.status, "#6b7280")
        panel.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="6" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')
        panel.append(f'<text x="{px + 8:.2f}" y="{py - 8:.2f}" class="small">{_escape(exp.exp_id[-3:])}</text>')
    return "".join(panel)


def _plot_bounds(x: int, y: int, w: int, h: int) -> dict[str, float]:
    return {
        "left": x + 70,
        "right": x + w - 24,
        "top": y + 48,
        "bottom": y + h - 42,
    }


def _axes(bounds: dict[str, float], y_label: str, x_label: str) -> str:
    left = bounds["left"]
    right = bounds["right"]
    top = bounds["top"]
    bottom = bounds["bottom"]
    return (
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#94a3b8" stroke-width="1.5"/>'
        f'<line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#94a3b8" stroke-width="1.5"/>'
        f'<text x="{(left + right) / 2:.2f}" y="{bottom + 28:.2f}" class="small">{_escape(x_label)}</text>'
        f'<text x="{left - 48:.2f}" y="{top - 10:.2f}" class="small">{_escape(y_label)}</text>'
    )


def _y_ticks(bounds: dict[str, float], values: list[float]) -> list[str]:
    low, high = min(values), max(values)
    if math.isclose(low, high):
        labels = [low]
    else:
        labels = [low + (high - low) * frac for frac in (0.0, 0.5, 1.0)]
    markers: list[str] = []
    for label in labels:
        py = _scale(label, low, high, bounds["bottom"], bounds["top"])
        markers.append(
            f'<line x1="{bounds["left"]}" y1="{py:.2f}" x2="{bounds["right"]}" y2="{py:.2f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        markers.append(
            f'<text x="{bounds["left"] - 54:.2f}" y="{py + 4:.2f}" class="small">{label:.4f}</text>'
        )
    return markers


def _x_ticks(bounds: dict[str, float], values: list[float]) -> list[str]:
    low, high = min(values), max(values)
    if math.isclose(low, high):
        labels = [low]
    else:
        labels = [low + (high - low) * frac for frac in (0.0, 0.5, 1.0)]
    markers: list[str] = []
    for label in labels:
        px = _scale(label, low, high, bounds["left"], bounds["right"])
        markers.append(
            f'<line x1="{px:.2f}" y1="{bounds["bottom"]}" x2="{px:.2f}" y2="{bounds["top"]}" '
            'stroke="#f1f5f9" stroke-width="1"/>'
        )
        markers.append(
            f'<text x="{px - 12:.2f}" y="{bounds["bottom"] + 18:.2f}" class="small">{label:.1f}</text>'
        )
    return markers


def _project_point(
    bounds: dict[str, float],
    index: int,
    total: int,
    value: float,
    low: float,
    high: float,
) -> tuple[float, float]:
    if total <= 1:
        px = (bounds["left"] + bounds["right"]) / 2.0
    else:
        px = bounds["left"] + (bounds["right"] - bounds["left"]) * index / (total - 1)
    py = _scale(value, low, high, bounds["bottom"], bounds["top"])
    return px, py


def _scale(value: float, low: float, high: float, out_low: float, out_high: float) -> float:
    if math.isclose(low, high):
        return (out_low + out_high) / 2.0
    ratio = (value - low) / (high - low)
    return out_low + (out_high - out_low) * ratio


def _notes_block(x: int, y: int, best: Experiment, experiments: list[Experiment]) -> str:
    latest = experiments[-1]
    lines = [
        f"Current best: {best.exp_id} at {best.val_bpb:.4f} val_bpb.",
        f"Latest run: {latest.exp_id} ({latest.status}) tested {latest.hypothesis}.",
        "Read agent_lab/state.md for the live dashboard and agent_lab/tranches.md for the tranche map.",
    ]
    block = [
        f'<rect x="{x}" y="{y}" width="{SVG_WIDTH - 80}" height="72" rx="14" fill="#fffdf8" stroke="#d6d3d1"/>',
    ]
    for i, line in enumerate(lines):
        block.append(f'<text x="{x + 20}" y="{y + 24 + i * 18}" class="body">{_escape(line)}</text>')
    return "".join(block)


def render_html(svg_filename: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Agent Lab Experiment Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{
      margin: 0;
      background: #f3f0e8;
      color: #1f2937;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1360px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    p {{
      margin: 0 0 16px;
      line-height: 1.5;
    }}
    .frame {{
      background: white;
      border: 1px solid #d6d3d1;
      border-radius: 18px;
      overflow: hidden;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Agent Lab Experiment Dashboard</h1>
    <p>This dashboard is generated from <code>agent_lab/experiments.tsv</code>. Re-run
    <code>python3 scripts/agent_lab/plot_experiments.py</code> after logging new experiments.</p>
    <div class="frame">
      <img src="{html.escape(svg_filename)}" alt="Experiment dashboard">
    </div>
  </main>
</body>
</html>
"""


def _escape(text: str) -> str:
    return html.escape(text, quote=False)


def main() -> None:
    _ = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    experiments_path = repo_root / "agent_lab" / "experiments.tsv"
    out_dir = repo_root / "agent_lab" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = load_experiments(experiments_path)
    if not experiments:
        raise SystemExit(f"No scored experiments found in {experiments_path}")

    svg_path = out_dir / "experiments.svg"
    html_path = out_dir / "experiments.html"

    svg_path.write_text(render_dashboard(experiments), encoding="utf-8")
    html_path.write_text(render_html(svg_path.name), encoding="utf-8")
    print(f"Wrote {svg_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
