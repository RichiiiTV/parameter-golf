# Compliance Matrix

| Idea | Mechanism | Scope | Triton | Class | When |
|---|---|---|---|---|---|
| March 20 top-record reproduction | Recreate `10L Int5-MLP + BigramHash(10240) + SWA(frac=0.4) + WD=0.04` in a standalone record fork | H100-only | Irrelevant | GREEN | Now |
| Sliding eval | Score each token once with richer left context | H100-only | Irrelevant | GREEN | Now |
| `tok_emb.weight` fp16 retention | Preserve the tied embedding/output matrix in fp16 at export | H100-only | Irrelevant | GREEN | Now |
| 10 layers | Increase unique depth within the accepted public stack | H100-only | Irrelevant | GREEN | Now |
| Muon weight decay | Decoupled decay on matrix params | H100-only | Irrelevant | GREEN | Now |
| Overtone tied-embedding init | Reshape tied-embedding singular values | H100-only | Irrelevant | GREEN | Now |
| Phase-transition `resid_mix` init | Layerwise schedule for skip mixing | H100-only | Irrelevant | GREEN | Now |
| Seq2048 / seq4096 scaling | Trade update count against richer context on H100 | H100-only | Irrelevant | GREEN | Now |
| CUDA Graph capture | Reduce launch overhead on the winning GREEN lane | H100-only | Irrelevant | GREEN | Later |
| Pre-projection RMSNorm | Add an extra normalization before projections | H100-only | Irrelevant | YELLOW | After GREEN baseline |
| Recurrent/shared-width transformers | Buy width via parameter sharing | H100-only | Irrelevant | YELLOW | After GREEN baseline |
| Sparse high-precision outlier retention | Spend artifact headroom on selective precision | H100-only | Possibly | YELLOW | After GREEN baseline |
| Tokenizer changes | Retokenize and recalculate tokenizer-agnostic scoring | H100-only | Irrelevant | YELLOW | Later |
| Eval-time adaptation | Update model/state on validation data | Eval-only | Irrelevant | YELLOW | Review first |
| Custom Triton kernels | Repo-local kernels for measured H100 hotspots | H100-only | Central | YELLOW | Only after profiling |
| Auto-launched H100 jobs | Scripted cluster execution | H100-only | Irrelevant | RED | Never |
| External data/network in eval | Non-self-contained evaluation | Eval-only | Irrelevant | RED | Never |
| Oversized artifact | `bytes_total > 16,000,000` | Any | Irrelevant | RED | Never |
| Seed brute force | Spirit-violating search | Any | Irrelevant | RED | Never |

Notes:
- The March 20 record wave is now the operational frontier, led locally by `10L Int5-MLP + BigramHash(10240) + SWA(frac=0.4) + WD=0.04` at `1.14276`.
- The active SOTA pass promotes GREEN work into standalone record folders, not into the root trainer.
- YELLOW ideas stay documented and lane-scoped until the clean GREEN mainline exists.
