#!/usr/bin/env python3
"""Cheap smoke tests for tranche families K to O without full training runs."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_lab.train_gpt import (
    BatchedTTTLoRA,
    GPT,
    build_optimizers,
    make_lr_schedule_functions,
)


def base_model_kwargs() -> dict:
    return {
        "vocab_size": 1024,
        "num_layers": 9,
        "model_dim": 512,
        "num_heads": 4,
        "num_kv_heads": 2,
        "mlp_mult": 2,
        "tie_embeddings": False,
        "tied_embed_init_std": 0.005,
        "logit_softcap": 20.0,
        "rope_base": 10000.0,
        "qk_gain_init": 1.5,
        "use_resid_mix": True,
        "use_attn_scale": True,
        "use_mlp_scale": True,
        "resid_mix_mode": "channel",
        "resid_scale_mode": "channel",
        "skip_mode": "learned",
        "skip_link_pattern": "all",
        "latent_kv_layers": set(),
        "latent_kv_dim": 0,
        "local_attn_layers": set(),
        "local_attn_window": 0,
        "mixer_layers": set(),
        "mixer_dim": 256,
        "mixer_kernel": 4,
        "output_head_mode": "dense",
        "output_head_rank": 64,
        "output_head_bottleneck": 256,
    }


def make_args(**overrides):
    defaults = {
        "tie_embeddings": False,
        "embed_lr": 0.6,
        "head_lr": 0.012,
        "tied_embed_lr": 0.05,
        "matrix_lr": 0.06,
        "attn_matrix_lr": 0.0,
        "mlp_matrix_lr": 0.0,
        "scalar_lr": 0.04,
        "early_layer_lr_scale": 1.0,
        "late_layer_lr_scale": 1.0,
        "new_mech_lr_scale": 1.0,
        "muon_momentum": 0.95,
        "muon_backend_steps": 5,
        "beta1": 0.9,
        "beta2": 0.95,
        "adam_eps": 1e-8,
        "warmdown_iters": 1200,
        "warmdown_fraction": 0.0,
        "iterations": 20000,
        "final_lr_scale": 0.0,
        "final_stabilize_steps": 0,
        "final_stabilize_lr_scale": 0.1,
        "final_head_lr_scale": 1.0,
        "final_backbone_lr_scale": 1.0,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def run_forward_smoke(name: str, **overrides) -> None:
    model = GPT(**(base_model_kwargs() | overrides))
    model.eval()
    x = torch.randint(0, 1024, (2, 16))
    y = torch.randint(0, 1024, (2, 16))
    loss = model(x, y)
    assert torch.isfinite(loss), f"{name}: dense forward returned non-finite loss"
    lora = BatchedTTTLoRA(2, model, rank=4)
    lora_loss = model(x, y, lora=lora)
    assert lora_loss.shape == (2, 16), f"{name}: LoRA forward shape mismatch {tuple(lora_loss.shape)}"


def smoke_output_head_modes() -> None:
    run_forward_smoke("k1-low-rank", output_head_mode="low_rank", output_head_rank=128)
    run_forward_smoke("k2-low-rank-small", output_head_mode="low_rank", output_head_rank=64)
    run_forward_smoke("k3-bottleneck", output_head_mode="bottleneck", output_head_bottleneck=256)
    run_forward_smoke("k4-tied-residual", tie_embeddings=True, output_head_mode="tied_residual", output_head_rank=64)
    run_forward_smoke("k5-two-stage", output_head_mode="two_stage", output_head_bottleneck=256)


def smoke_local_and_routing_modes() -> None:
    run_forward_smoke("l-local-lower", local_attn_layers={0, 1, 2, 3}, local_attn_window=256)
    run_forward_smoke(
        "m-cheap-routing",
        skip_mode="shared_scalar",
        skip_link_pattern="alternate",
        resid_mix_mode="scalar",
        resid_scale_mode="shared_scalar",
    )


def smoke_optimizer_splits() -> None:
    model = GPT(
        **(
            base_model_kwargs()
            | {
                "output_head_mode": "low_rank",
                "output_head_rank": 64,
                "latent_kv_layers": {0},
                "latent_kv_dim": 64,
                "mixer_layers": {1},
                "local_attn_layers": {2},
                "local_attn_window": 256,
            }
        )
    )
    args = make_args(
        attn_matrix_lr=0.07,
        mlp_matrix_lr=0.05,
        early_layer_lr_scale=0.9,
        late_layer_lr_scale=1.1,
        new_mech_lr_scale=1.25,
    )
    optimizers, token_lr, output_head_lr = build_optimizers(args, model)
    assert token_lr == args.embed_lr
    assert output_head_lr > args.head_lr
    matrix_base_lrs = sorted(
        {
            group["base_lr"]
            for opt in optimizers
            for group in opt.param_groups
            if group.get("kind") == "matrix"
        }
    )
    assert len(matrix_base_lrs) >= 3, f"expected split matrix learning rates, got {matrix_base_lrs}"
    scalar_kinds = {group.get("kind") for opt in optimizers for group in opt.param_groups}
    assert {"token", "scalar", "matrix", "head"}.issubset(scalar_kinds), scalar_kinds


def smoke_warmdown_schedule() -> None:
    args = make_args(
        warmdown_fraction=0.25,
        final_lr_scale=0.05,
        final_stabilize_steps=128,
        final_stabilize_lr_scale=0.10,
        final_head_lr_scale=0.05,
        final_backbone_lr_scale=0.15,
    )
    lr_mul, in_final_stabilize = make_lr_schedule_functions(args, max_wallclock_ms=600_000.0)
    early = lr_mul(step=100, elapsed_ms=20_000.0)
    late = lr_mul(step=1500, elapsed_ms=590_000.0)
    assert 0.0 < late < early <= 1.0, (early, late)
    assert in_final_stabilize(step=1700, elapsed_ms=599_000.0)
    assert not in_final_stabilize(step=200, elapsed_ms=50_000.0)


def main() -> None:
    smoke_output_head_modes()
    smoke_local_and_routing_modes()
    smoke_optimizer_splits()
    smoke_warmdown_schedule()
    print("K-O smoke tests passed.")


if __name__ == "__main__":
    main()
