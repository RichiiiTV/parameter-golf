# Hopper Plan

Operational frontier for this pass:
- accepted merged record: `#549`
- live operational frontier: `#753`
- donor record path: `pr753:records/track_10min_16mb/2026-03-25_PodracingII_backoff7gram_8xH100`
- reported result: `val_bpb=0.9625`, `bytes_total=15,593,916`
- active root delta: `NGRAM_EVAL_BUCKETS=8388608` only

## Run Order
- Run 1: exact `#753` root repro
- Run 2: `#753` root repro plus `NGRAM_EVAL_BUCKETS=8388608`

## Run 1
- Goal: verify that root matches the live `#753` dense base, GPTQ-calibrated int6+zstd export, and legal score-first 2..7-gram backoff stack on H100.
- Command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_pr753_repro.json`
- Success criteria: stay in-family with the `#753` quality, byte, and timing band.
- Risk: if this misses the donor band, the root port is not trustworthy enough to judge the follow-up lane.

## Run 2
- Goal: test only the larger score-first n-gram cache on top of the exact `#753` root lane.
- Command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_pr753_bucket8m.json`
- Success criteria: improve the final n-gram exact metric without breaking the byte or eval budgets.
- Risk: more buckets may be neutral if the current 4M table is not the dominant remaining collision bottleneck.

## Notes
- Root was hard-reset to the upstream `#753` donor family before this pass.
- The pre-reset `#549` + soft-QAT root is preserved in `snapshots/train_gpt_2026-03-25_pre753_pr549_softqat_root.py`.
- `flash-attn` remains optional and cluster-side.
