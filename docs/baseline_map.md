# Baseline Map

- Root [`train_gpt.py`](/c:/Users/Ricardo/Desktop/parameter-golf/train_gpt.py) has been reset toward the official upstream/public-record trainer shape.
- The active root lineage is:
  - `2026-03-17_NaiveBaseline`
  - `2026-03-18_FP16Embed_WD3600`
  - `2026-03-18_LongContextSeq2048`
  - `2026-03-19_SlidingWindowEval`
  - `2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit`

## Root Trainer Contract
- Keep the official trainer layout and logging style.
- Keep only narrow public-record deltas in root:
  - `EVAL_SEQ_LEN`
  - `EVAL_STRIDE`
  - `KEEP_FLOAT_EXTRA`
  - `NUM_LAYERS`
  - `MUON_WEIGHT_DECAY`
  - `ADAMW_WEIGHT_DECAY`
  - `TIED_EMBED_INIT_MODE`
  - `TIED_EMBED_OVERTONE_POWER`
  - `RESID_MIX_INIT`
  - `RESID_MIX_PHASE_GAIN`
- Do not keep GTX/A100 proxy logic or extra runtime/backend flag surfaces in root.

## Active Workflow
- Active workflow is H100-only and manual-only.
- `scripts/prepare_h100_run.py` is the active handoff generator.
- A100 and GTX material are legacy history only and are not used for ranking in this reset phase.
