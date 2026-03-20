# Compliance Matrix

| Idea | Mechanism | Scope | Triton | Class | When |
|---|---|---|---|---|---|
| Official top-1 reproduction | Recreate `Muon WD + 10 layer` in the reset root trainer | H100-only | Irrelevant | GREEN | Now |
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
- The official public top-1 is `Muon WD + 10 layer` at `1.17475315`.
- The reset pass only promotes GREEN work into the active root path.
- YELLOW ideas stay documented and lane-scoped until the clean GREEN mainline exists.
