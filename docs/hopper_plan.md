# Hopper Plan

Operational frontier for this pass:
- accepted merged record: `#549`
- live operational frontier: `#753`
- donor record path: `pr753:records/track_10min_16mb/2026-03-25_PodracingII_backoff7gram_8xH100`
- reported result: `val_bpb=0.9625`, `bytes_total=15,593,916`
- active root delta: `#809`-style single-pass order-9 n-gram port, first on the dense control and then on the requested-point fixed-block state-space hybrid

## Run Order
- Run 1: dense `#753` control plus `#809`-style chunked order-9 n-gram
- Run 2: current requested-point hybrid plus the same `#809`-style n-gram

## Run 1
- Goal: verify that the current root can carry a record-safer `#809`-style evaluator on the dense `#753` path without changing the model or export path.
- Command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_pr753_repro_ngram809.json`
- Success criteria: improve the old 7-gram control while staying under the train/eval limits and the 16MB artifact cap.
- Risk: if the dense control is slow or unstable, the hybrid lane should not be promoted yet.

## Run 2
- Goal: test whether the requested-point hybrid improves dense-model quality while sharing the same `#809`-style evaluator as the dense control.
- Command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_pr753_hybrid_ssm_ngram809.json`
- Success criteria: improve both `final_int6_sliding_window_exact` and the final n-gram exact metric versus the dense `#809`-style control without breaking the byte or eval budgets.
- Risk: the hybrid may improve dense-model quality but lose enough throughput to miss the 10-minute wallclock.

## Notes
- Root was hard-reset to the upstream `#753` donor family before this pass.
- The pre-hybrid `#753` root is preserved in `snapshots/train_gpt_2026-03-25_pre753_state_space_hybrid_root.py`.
- The pre-reset `#549` + soft-QAT root is preserved in `snapshots/train_gpt_2026-03-25_pre753_pr549_softqat_root.py`.
- `flash-attn` remains optional and cluster-side.
