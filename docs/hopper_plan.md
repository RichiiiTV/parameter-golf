# Hopper Plan

Operational frontier for this pass:
- `pr332:records/track_10min_16mb/2026-03-21_12L_GradQuant_PartialRoPE_EMA`
- Reported result: `val_bpb=1.1320`, `bytes_total=15,652,352`

## Run Order
- Run 1: rerun root `#332` with `TRAIN_BATCH_TOKENS=458752`, `ITERATIONS=11000`, and `BIGRAM_VOCAB_SIZE=2560`
- Run 2: rerun the same branch with `BIGRAM_VOCAB_SIZE=2432` only if Run 1 is still over cap or lands under cap with too little margin
- Run 3: Run 1 plus `EVAL_SEQ_LEN=4096 EVAL_BATCH_SEQS=16` only if Run 1 stays within eval budget and lands comfortably under cap

Current note:
- The newest verified fixed-metric section in `logs/root-pr332-b458k-bigram3584 (2).txt` reports `final_int8_*_roundtrip_exact val_bpb=1.18226244`, but it is still over cap at `16,069,767` bytes and is not H100 truth.

## Run 1
- Goal: keep the current fixed-metric `b458k` branch intact and trim only the auxiliary BigramHash table enough to restore submission compliance.
- Why it matters: the metric path now looks sane again, so the next move should be a surgical size fix rather than another throughput or architecture change.
- Risks: `2560` may still be too close to the cap on H100 or the extra trim may hurt final exact `val_bpb` more than expected.

RUN THIS MANUALLY ON H100
- command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_pr332_b458k_bigram2560.json`
- required environment: repo root checkout on the Runpod machine, dataset cache, tokenizer cache
- setup command: `pip install zstandard flash-attn --no-build-isolation`
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: stay under `16,000,000` bytes with at least a small safety margin and produce a trustworthy `final_int8_*_roundtrip_exact` result with the fixed metric path

## Run 2
- Goal: fall back to a stronger byte-margin trim if the 2560-bucket rescue is still too close to the cap.
- Why it matters: `3584 -> 2432` is the first simple trim that should buy about `10k` to `15k` bytes of margin at the current compression ratio.
- Risks: the extra trim may push the exact post-export score past the acceptable `1.20` region.

RUN THIS MANUALLY ON H100
- command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_pr332_b458k_bigram2432.json`
- required environment: same as Run 1
- setup command: `pip install zstandard flash-attn --no-build-isolation`
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: stay clearly under `16,000,000` bytes and preserve as much of the low `1.18x` exact score band as possible

## Run 3
- Goal: test whether the winning combined branch also benefits from longer eval context.
- Why it matters: sliding-window eval already showed pure-eval gains on this benchmark, so longer eval context is a credible follow-up if it stays within budget.
- Risks: eval may exceed 10 minutes or the longer context may not pay for itself.

RUN THIS MANUALLY ON H100
- command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_pr332_b458k_bigram2560_eval4096.json`
- required environment: same as Run 1
- setup command: `pip install zstandard flash-attn --no-build-isolation`
- expected runtime budget: 10 minutes train, up to 10 minutes eval
- success criteria: stay inside the eval budget and improve or materially close on the fixed 2048-eval 2560-bucket run
