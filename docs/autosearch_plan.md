# Autosearch Plan

Policy:
- Config-only and allowlisted.
- No code mutation.
- No seed search.
- No H100 execution.

Lanes:
- `train_lane`
- `export_lane`
- `eval_lane`
- `h100_lane` as manual queue only

Allowlisted knobs:
- `TRAIN_SEQ_LEN`
- `TIED_EMBED_LR`
- `MATRIX_LR`
- `SCALAR_LR`
- `WARMDOWN_ITERS`
- `MLP_HIDDEN`
- `KEEP_FLOAT_EXTRA`
- `EVAL_MODE`
- `EVAL_STRIDE`
- `EVAL_SEQ_LEN`
- `INT8_CLIP_PERCENTILE`

Objective:
- lowest proxy post-roundtrip `val_bpb`
- then smallest artifact
- then lowest eval time
