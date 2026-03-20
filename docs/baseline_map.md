# Baseline Map

- Root [`train_gpt.py`](/c:/Users/Ricardo/Desktop/parameter-golf/train_gpt.py) is the research core and record snapshot source.
- Current trainer shape:
  - 9-layer 512-dim causal model
  - 8 attention heads / 4 KV heads
  - U-Net-like skip reuse
  - int8+zlib roundtrip export
- Proven record deltas already in `records/`:
  - `FP16Embed_WD3600`: export-aware tied-embedding handling plus LR/warmdown tuning
  - `LongContextSeq2048`: stronger training-side context scaling
  - `SlidingWindowEval`: strongest eval-side improvement
- Main root gaps before this upgrade:
  - no Pascal-safe runtime path
  - no clean sliding eval mode
  - no export-aware `tok_emb.weight` path
  - no config/registry runner
  - no manual-only H100 workflow layer
