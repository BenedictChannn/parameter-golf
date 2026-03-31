# Agent Lab Budget Report

Current reference shape: the best valid line as of [`AL-20260329-030`](./experiments.tsv).

## Shape

- layers: `9`
- model_dim: `512`
- num_heads: `4`
- num_kv_heads: `2`
- head_dim: `128`
- query-to-kv ratio: `2.0`
- mlp_mult: `2`
- tie_embeddings: `False`
- total params: `17584164`
- artifact total: `15797032` bytes int8+zlib

## Parameter Budget

| Family | Params | Share | Raw bf16 MB | Raw int8 MB | Estimated compressed bytes |
|---|---:|---:|---:|---:|---:|
| attention_controls | 36 | 0.0% | 0.00 | 0.00 | 32 |
| attention_k | 1179648 | 6.7% | 2.36 | 1.18 | 1059756 |
| attention_out | 2359296 | 13.4% | 4.72 | 2.36 | 2119513 |
| attention_q | 2359296 | 13.4% | 4.72 | 2.36 | 2119513 |
| attention_v | 1179648 | 6.7% | 2.36 | 1.18 | 1059756 |
| block_controls | 18432 | 0.1% | 0.04 | 0.02 | 16558 |
| mlp_down | 4718592 | 26.8% | 9.44 | 4.72 | 4239027 |
| mlp_up | 4718592 | 26.8% | 9.44 | 4.72 | 4239027 |
| output_head | 524288 | 3.0% | 1.05 | 0.52 | 471003 |
| skip_weights | 2048 | 0.0% | 0.00 | 0.00 | 1839 |
| token_embedding | 524288 | 3.0% | 1.05 | 0.52 | 471003 |

## Reading Guide

- `Raw bf16 MB` is a simple storage proxy before quantization.
- `Raw int8 MB` is a simple one-byte-per-parameter proxy.
- `Estimated compressed bytes` is only a proportional estimate from the total artifact size, not a true per-subsystem compression measurement.
- The main story in the current winner is simple: the MLP dominates the parameter budget, attention matrices are second, and the untied output head is small enough to be affordable.
