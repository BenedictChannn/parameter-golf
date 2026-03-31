#!/usr/bin/env python3
"""Break down agent-lab model parameters by component family."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_lab.train_gpt import GPT, Hyperparameters


@dataclass
class BudgetRow:
    name: str
    params: int


def parse_args() -> argparse.Namespace:
    defaults = Hyperparameters()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-layers", type=int, default=defaults.num_layers)
    parser.add_argument("--model-dim", type=int, default=defaults.model_dim)
    parser.add_argument("--num-heads", type=int, default=defaults.num_heads)
    parser.add_argument("--num-kv-heads", type=int, default=defaults.num_kv_heads)
    parser.add_argument("--mlp-mult", type=int, default=defaults.mlp_mult)
    parser.add_argument("--vocab-size", type=int, default=defaults.vocab_size)
    parser.add_argument("--tie-embeddings", type=int, choices=(0, 1), default=1 if defaults.tie_embeddings else 0)
    parser.add_argument("--tied-embed-init-std", type=float, default=defaults.tied_embed_init_std)
    parser.add_argument("--logit-softcap", type=float, default=defaults.logit_softcap)
    parser.add_argument("--rope-base", type=float, default=defaults.rope_base)
    parser.add_argument("--qk-gain-init", type=float, default=defaults.qk_gain_init)
    parser.add_argument("--artifact-total-bytes", type=int, default=0, help="Optional measured total int8+zlib bytes for proportional estimates.")
    parser.add_argument("--format", choices=("md", "json"), default="md")
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> GPT:
    return GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=bool(args.tie_embeddings),
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    )


def classify_param(name: str) -> str:
    if name.startswith("tok_emb."):
        return "token_embedding"
    if name.startswith("lm_head."):
        return "output_head"
    if name.startswith("skip_weights"):
        return "skip_weights"
    if ".attn.c_q." in name:
        return "attention_q"
    if ".attn.c_k." in name:
        return "attention_k"
    if ".attn.c_v." in name:
        return "attention_v"
    if ".attn.proj." in name:
        return "attention_out"
    if ".attn.q_gain" in name:
        return "attention_controls"
    if ".mlp.fc." in name:
        return "mlp_up"
    if ".mlp.proj." in name:
        return "mlp_down"
    if ".attn_scale" in name or ".mlp_scale" in name or ".resid_mix" in name:
        return "block_controls"
    if ".attn_norm" in name or ".mlp_norm" in name or name.startswith("final_norm"):
        return "norms"
    return "other"


def build_rows(model: GPT) -> list[BudgetRow]:
    counts: dict[str, int] = {}
    for name, param in model.named_parameters():
        family = classify_param(name)
        counts[family] = counts.get(family, 0) + int(param.numel())
    return [BudgetRow(name=family, params=params) for family, params in sorted(counts.items())]


def render_markdown(args: argparse.Namespace, rows: list[BudgetRow]) -> str:
    total_params = sum(row.params for row in rows)
    head_dim = args.model_dim // args.num_heads
    q_to_kv_ratio = args.num_heads / args.num_kv_heads
    lines = [
        "# Agent Lab Budget Report",
        "",
        "## Shape",
        "",
        f"- layers: `{args.num_layers}`",
        f"- model_dim: `{args.model_dim}`",
        f"- num_heads: `{args.num_heads}`",
        f"- num_kv_heads: `{args.num_kv_heads}`",
        f"- head_dim: `{head_dim}`",
        f"- query-to-kv ratio: `{q_to_kv_ratio:.1f}`",
        f"- mlp_mult: `{args.mlp_mult}`",
        f"- tie_embeddings: `{bool(args.tie_embeddings)}`",
        f"- total params: `{total_params}`",
        "",
        "## Parameter Budget",
        "",
        "| Family | Params | Share | Raw bf16 MB | Raw int8 MB | Estimated compressed bytes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        share = row.params / total_params if total_params else 0.0
        raw_bf16_mb = row.params * 2 / 1_000_000
        raw_int8_mb = row.params / 1_000_000
        est_compressed = int(args.artifact_total_bytes * share) if args.artifact_total_bytes else 0
        lines.append(
            f"| {row.name} | {row.params} | {share:.1%} | {raw_bf16_mb:.2f} | {raw_int8_mb:.2f} | "
            f"{est_compressed if est_compressed else 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- `Raw bf16 MB` is a simple storage proxy before quantization.",
            "- `Raw int8 MB` is a simple one-byte-per-parameter proxy.",
            "- `Estimated compressed bytes` is proportional to parameter share only when `--artifact-total-bytes` is supplied.",
            "- The compression estimate is directional, not exact, because the real submission payload depends on quantization scales and zlib behavior.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    model = build_model(args)
    rows = build_rows(model)
    if args.format == "json":
        print(
            json.dumps(
                {
                    "shape": {
                        "num_layers": args.num_layers,
                        "model_dim": args.model_dim,
                        "num_heads": args.num_heads,
                        "num_kv_heads": args.num_kv_heads,
                        "mlp_mult": args.mlp_mult,
                        "tie_embeddings": bool(args.tie_embeddings),
                    },
                    "rows": [{"name": row.name, "params": row.params} for row in rows],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    print(render_markdown(args, rows))


if __name__ == "__main__":
    main()
