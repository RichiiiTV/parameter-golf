# Export Strategy

## Active Export Mainline
- Keep the existing int8 + zlib exporter from the public-record lineage.
- Keep `tok_emb.weight` in fp16 via `KEEP_FLOAT_EXTRA=tok_emb.weight`.
- Score candidates by post-export `val_bpb`, not by pre-export loss alone.

## Why This Matters
- The current public top-1 already combines sliding eval with fp16 tied-embedding retention.
- Export robustness is part of the mainline, not an optional afterthought.

## GREEN Mainline
- `KEEP_FLOAT_EXTRA=tok_emb.weight`
- no custom codec
- no mixed int8/int6 in the active reset pass

## YELLOW Follow-On
- Sparse or decoupled high-precision outlier retention
- Learned or structured export codecs
- Any rowwise or selective high-precision retention beyond the tied embedding

## Rules
- Compare candidates on post-export `val_bpb`
- Keep artifact bytes under `16,000,000`
- Keep export self-contained and reproducible
- Do not add custom packing complexity until the best GREEN H100 lane is known
