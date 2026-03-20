# Compliance Matrix

| Idea | Mechanism | Scope | Triton | Class | When |
|---|---|---|---|---|---|
| Sliding eval | Score each token once with richer left context | Transferable | Irrelevant | GREEN | Now |
| Longer train/eval context | Increase causal context during train or eval | Transferable / H100-only at larger seq | Irrelevant | GREEN | Now |
| `tok_emb.weight` fp16 passthrough | Reduce post-export quantization loss on tied embedding/output head | Transferable | Irrelevant | GREEN | Now |
| Warmdown/LR tuning | Improve convergence under fixed wallclock | Transferable | Irrelevant | GREEN | Now |
| EMA | Smooth endpoint for better export robustness | Transferable | Irrelevant | GREEN | Now |
| SDPA backend sweep | Benchmark `auto|flash|cudnn` on H100 | H100-only | Irrelevant | GREEN | Later |
| Compile sweep | Compare `off|muon_only|fullgraph` | H100-only | Irrelevant | GREEN | Later |
| Tokenizer changes | Retokenize / rebuild export | Transferable | Irrelevant | YELLOW | Later |
| Recurrent/shared blocks | Trade unique depth for width/effective depth | Transferable | Irrelevant | YELLOW | Later |
| Mixed-format export codec | Row subsets / custom tensor formats | Transferable | Possibly | YELLOW | Later |
| Eval-time adaptation | Update model/state on validation set | Eval-only | Irrelevant | YELLOW | Review first |
| Custom Triton kernels | Repo-local kernels for hotspots | H100-only / Transferable | Central | YELLOW | Only after profiling |
| Auto-launched H100 jobs | Scripted remote execution | H100-only | Irrelevant | RED | Never |
| External data/network in eval | Non-self-contained eval | Eval-only | Irrelevant | RED | Never |
| Oversized artifact | `bytes_total > 16,000,000` | Any | Irrelevant | RED | Never |
| Seed brute force | Spirit-violating search | Any | Irrelevant | RED | Never |

Notes:
- The repo’s strongest current public result is `SlidingWindowEval`, even though the top-level README table has not been updated.
- Challenge-safe work in phase 1 is limited to GREEN items plus documentation for YELLOW items.
