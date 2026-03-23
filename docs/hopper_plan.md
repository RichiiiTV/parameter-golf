# Hopper Plan

Operational frontier for this pass:
- `pr332:records/track_10min_16mb/2026-03-21_12L_GradQuant_PartialRoPE_EMA`
- Reported result: `val_bpb=1.1320`, `bytes_total=15,652,352`

## Run Order
- Run 1: frozen dense snapshot baseline on H100
- Run 2: shared-depth root lane with `UNIQUE_BLOCKS=8`, `MLP_HIDDEN=1664`, and the same best dense training recipe
- Run 2b: less-aggressive shared-depth lane with `UNIQUE_BLOCKS=10`, `MLP_HIDDEN=1920`, and `BIGRAM_VOCAB_SIZE=4096`
- Run 2c: yellow shared-depth adapter lane with `PER_APP_Q_GAIN=1` on top of Run 2b
- Run 2d: yellow export lane with `OUTLIER_ROW_BUDGET=64` on top of Run 2c
- Run 3: `EVAL_SEQ_LEN=4096 EVAL_BATCH_SEQS=16` only on the winning lane if it stays within eval budget

Current note:
- The dense local lane has improved enough to justify one H100 truth run, but the local curve appears saturated and no longer justifies more schedule-only tuning.

## Run 1
- Goal: establish the frozen dense snapshot as the H100 truth baseline before judging the shared lane.
- Why it matters: the dense lane is the best current fallback and gives the cleanest apples-to-apples comparison for the new root architecture.
- Risks: if the dense lane underperforms sharply on H100, the shared lane will need an exact-side comparison before promotion.

RUN THIS MANUALLY ON H100
- command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_snapshot_b1024_warmup200_xlast2.json`
- required environment: repo root checkout on the Runpod machine, dataset cache, tokenizer cache
- setup command: `pip install zstandard flash-attn --no-build-isolation`
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: produce a trustworthy dense H100 baseline under the challenge byte and time limits

## Run 2
- Goal: test whether shared full blocks can buy enough byte headroom to fund a wider MLP and improve sliding `val_bpb`.
- Why it matters: parameter sharing targets the stored-byte cap directly and is a better next lever than more warmup or BigramHash sweeps.
- Risks: if the shared lane slows down too much or hurts exact roundtrip materially, the dense snapshot remains the fallback.

RUN THIS MANUALLY ON H100
- command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_shared8_mlp1664_b1024_warmup200_xlast2.json`
- required environment: same as Run 1
- setup command: `pip install zstandard flash-attn --no-build-isolation`
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: beat the dense baseline on sliding `val_bpb` while staying under `16,000,000` bytes and keeping exact roundtrip in-family

## Run 2b
- Goal: reduce over-sharing after the shared8 width ladder started saturating.
- Why it matters: the local A100 shared8 runs improved strongly with extra width but still left a lot of quality on the table, which suggests under-specialized layers more than underused bytes.
- Risks: the extra unique blocks may erase the byte savings too quickly or slow the step time enough to offset the quality gain.

RUN THIS MANUALLY ON H100
- command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_shared10_mlp1920_b4096_warmup200_xlast2.json`
- required environment: same as Run 1
- setup command: `pip install zstandard flash-attn --no-build-isolation`
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: beat the best shared8 lane on sliding `val_bpb` while staying under `16,000,000` bytes and keeping exact roundtrip in-family

## Run 2c
- Goal: restore per-application head specialization without adding another large matrix family.
- Why it matters: per-application `q_gain` is a tiny yellow adapter on top of the shared10 lane, so it directly tests whether the current bottleneck is shared-attention rigidity rather than missing bulk capacity.
- Risks: it may be too small to matter, or it may help sliding while hurting exact roundtrip.

RUN THIS MANUALLY ON H100
- command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_shared10_mlp1920_b4096_warmup200_xlast2_qgain.json`
- required environment: same as Run 1
- setup command: `pip install zstandard flash-attn --no-build-isolation`
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: beat the shared10 base lane on sliding `val_bpb` without materially regressing exact roundtrip or violating byte and time limits

## Run 2d
- Goal: close the roundtrip gap by preserving a tiny set of high-residual rows in higher precision during export.
- Why it matters: the recent yellow shared-depth run was a near-miss on sliding but still left a noticeable roundtrip gap, which suggests export loss may now matter more than training capacity.
- Risks: the rescued rows may not be the right ones, or the extra fp16 payload may cost too many bytes for too little quality.

RUN THIS MANUALLY ON H100
- command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_shared10_mlp1920_b4096_warmup200_xlast2_qgain_outliers64.json`
- required environment: same as Run 1
- setup command: `pip install zstandard flash-attn --no-build-isolation`
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: improve exact roundtrip `val_bpb` over the shared10+qgain lane while staying under `16,000,000` bytes and keeping sliding performance in-family

## Run 3
- Goal: test whether the winning lane also benefits from longer eval context.
- Why it matters: sliding-window eval already showed pure-eval gains on this benchmark, so longer eval context is a credible follow-up if it stays within budget.
- Risks: eval may exceed 10 minutes or the longer context may not pay for itself.

RUN THIS MANUALLY ON H100
- command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_shared8_mlp1664_b1024_warmup200_xlast2_eval4096.json`
- required environment: same as Run 1
- setup command: `pip install zstandard flash-attn --no-build-isolation`
- expected runtime budget: 10 minutes train, up to 10 minutes eval
- success criteria: stay inside the eval budget and improve or materially close on the winning 2048-eval lane
