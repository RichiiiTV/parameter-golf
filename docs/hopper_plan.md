# Hopper Plan

Operational frontier for this pass:
- donor baseline: `pr332:records/track_10min_16mb/2026-03-21_12L_GradQuant_PartialRoPE_EMA`
- reported result: `val_bpb=1.1320`, `bytes_total=15,652,352`
- active reset: dense root plus packed GPTQ export, with TTT-lite as an isolated yellow follow-up

## Run Order
- Run 1: frozen dense snapshot baseline on H100
- Run 2: root dense baseline with `GPTQ_ENABLED=1`
- Run 3: funded dense GPTQ lane with `NUM_LAYERS=14`, `MLP_HIDDEN=1792`, and `BIGRAM_VOCAB_SIZE=4096`
- Run 4: Run 3 plus `TTT_ENABLED=1`

## Run 1
- Goal: establish the saved dense snapshot as the clean H100 truth baseline before judging the new root path.
- Command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_snapshot_b1024_warmup200_xlast2.json`
- Success criteria: clean dense H100 baseline under the challenge byte and time limits.

## Run 2
- Goal: validate packed GPTQ export on the dense 12-layer root without TTT.
- Command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_gptq12_b1024_warmup200_xlast2.json`
- Success criteria: save artifact bytes without materially regressing exact roundtrip.
- Risk: a byte win without a quality win is not enough to promote the mechanism.

## Run 3
- Goal: spend GPTQ headroom on a materially larger dense model instead of more shared-depth complexity.
- Command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_gptq14_mlp1792_b4096_warmup200_xlast2.json`
- Success criteria: beat Runs 1 and 2 on post-export quality while staying under `16,000,000` bytes.
- Risk: step time or post-export loss may erase the byte-funded capacity gain.

## Run 4
- Goal: test doc-isolated TTT-lite only after the funded dense GPTQ lane is healthy.
- Command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_gptq14_mlp1792_b4096_warmup200_xlast2_ttt.json`
- Success criteria: improve the funded dense GPTQ lane on post-export quality while staying inside the eval-time budget.
- Risk: eval-time adaptation may help only marginally or cost too much eval wallclock.

## Notes
- Root `train_gpt.py` was hard-reset from the dense snapshot baseline before these changes.
- The previous shared-depth / q-gain / outlier-rescue root lane is preserved in `snapshots/`, not deleted.
- `flash-attn` remains optional and cluster-side.
