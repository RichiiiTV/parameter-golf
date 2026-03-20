# Export Strategy

Observed repo facts:
- Baseline post-export gap is material.
- `tok_emb.weight` is the highest-sensitivity large tensor because it is both input embedding and output head.

Phase 1 export policy:
- Keep the existing int8+zlib exporter.
- Add optional fp16 passthrough for `tok_emb.weight`.
- Expose `INT8_CLIP_PERCENTILE`.
- Use `MLP_HIDDEN` to buy back bytes when fp16 retention is enabled.

Local GTX result:
- Standard seq1024 proxy: `post_quant_val_bpb=4.03490743`, `pre_post_gap=0.0164835`, `bytes_total=5,049,873`
- `tok_emb.weight` fp16 passthrough + `MLP_HIDDEN=992`: `post_quant_val_bpb=4.01945346`, `pre_post_gap=0.00092175`, `bytes_total=5,818,837`
- Interpretation: the fp16 embedding path sharply improves export robustness locally, but costs about `769 KB` more artifact budget

Current local finding:
- On the GTX proxy, `tok_emb.weight` fp16 passthrough cut the pre/post export gap from about `0.01648` to about `0.00092`.
- The same change raised artifact bytes from about `5.05 MB` to about `5.82 MB`.
- Current implication: keep this path GREEN, but pair it with byte-recovery changes such as `MLP_HIDDEN` or other capacity reallocation before promoting it as the default composed candidate.

Rules:
- Measure live and roundtrip metrics under the same eval policy.
- Treat artifact bytes as a first-class metric.
- Do not add a custom binary packer until the composed `seq2048 + fp16 tok_emb + sliding eval` candidate is measured.
