# Hopper Plan

Operational frontier for this pass:
- `pr332:records/track_10min_16mb/2026-03-21_12L_GradQuant_PartialRoPE_EMA`
- Reported result: `val_bpb=1.1320`, `bytes_total=15,652,352`

## Run Order
- Run 1: exact root `#332` reproduction
- Run 2: root `#332` with `TRAIN_BATCH_TOKENS=458752`, `ITERATIONS=11000`, and `BIGRAM_VOCAB_SIZE=4096`
- Run 3: Run 2 plus `EVAL_SEQ_LEN=4096 EVAL_BATCH_SEQS=16` only if Run 2 stays within eval budget

## Run 1
- Goal: verify that root `train_gpt.py` reproduces the `#332` donor command exactly.
- Why it matters: this is the same-script baseline for all later root improvements.
- Risks: flash-attn availability and ordinary reproduction drift.

RUN THIS MANUALLY ON H100
- command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_pr332_repro.json`
- required environment: repo root checkout on the Runpod machine, dataset cache, tokenizer cache
- setup command: `pip install zstandard flash-attn --no-build-isolation`
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: reproduce the `#332` root lane cleanly with an artifact under `16,000,000` bytes

## Run 2
- Goal: combine a conservative step-budget increase with a headroom-funded BigramHash capacity increase.
- Why it matters: `#332` itself credits smaller batch size for part of its gain, and the donor artifact still leaves room for a bigger BigramHash table.
- Risks: the smaller batch may add too much gradient noise, or the 4096-bucket table may compress worse than expected.

RUN THIS MANUALLY ON H100
- command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_pr332_b458k_bigram4096.json`
- required environment: same as Run 1
- setup command: `pip install zstandard flash-attn --no-build-isolation`
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: beat Run 1 on post-export `val_bpb` while staying under `16,000,000` bytes

## Run 3
- Goal: test whether the winning combined branch also benefits from longer eval context.
- Why it matters: sliding-window eval already showed pure-eval gains on this benchmark, so longer eval context is a credible follow-up if it stays within budget.
- Risks: eval may exceed 10 minutes or the longer context may not pay for itself.

RUN THIS MANUALLY ON H100
- command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_pr332_b458k_bigram4096_eval4096.json`
- required environment: same as Run 1
- setup command: `pip install zstandard flash-attn --no-build-isolation`
- expected runtime budget: 10 minutes train, up to 10 minutes eval
- success criteria: stay inside the eval budget and improve or materially close on Run 2
