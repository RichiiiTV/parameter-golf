# Hopper Plan

Current public best to beat:
- `Muon WD + 10 layer` at `mean_val_bpb=1.17475315`
- Record path: `records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit`

## Active H100 Ladder
- Run 1, GREEN mainline: `configs/h100/root_top1_repro.json`
- Run 2, GREEN throughput: `configs/h100/root_top1_seq2048.json`
- Run 3, GREEN throughput: `configs/h100/root_top1_seq4096.json`
- Run 4, GREEN throughput follow-up: winner of Runs 2 and 3 plus CUDA Graph capture if it fits cleanly
- Run 5, GREEN mechanism: winner plus pre-projection RMSNorm
- Run 6, YELLOW architecture: winner plus recurrent/shared-width transformer blocks
- Run 7, YELLOW compression: winner plus sparse high-precision outlier retention at export

## Run 1
- Goal: prove the reset trainer can express the official public top-1 stack cleanly.
- Why it matters: this is the new root reference point; anything bolder should beat this stack, not the older sliding-only record.
- Risks: reset drift, throughput regressions, export-byte regressions.

RUN THIS MANUALLY ON H100
- command: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_top1_repro.json`
- required environment: repo checkout, dataset cache, tokenizer cache, cluster path where `train_gpt.py` runs from repo root
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: reproduce the public top-1 recipe in the reset root trainer and remain under `16,000,000` bytes

## Run 2
- Goal: test whether seq2048 improves the throughput-quality trade over the 1024-context top-1 recipe.
- Why it matters: longer context is the cleanest H100-only throughput lever already supported by accepted records.
- Risks: fewer optimizer steps inside the same wallclock, worse net quality despite richer context.

RUN THIS MANUALLY ON H100
- command: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_top1_seq2048.json`
- required environment: same as Run 1
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: beat or closely match Run 1 on post-export `val_bpb` while staying within the byte cap

## Run 3
- Goal: test whether seq4096 buys enough context value to offset the smaller token budget.
- Why it matters: the official leaderboard already contains a strong seq4096 result, so this is the next clean throughput branch to test on top of the current top-1 recipe.
- Risks: too few update steps, eval-time cost, weaker quality than seq2048.

RUN THIS MANUALLY ON H100
- command: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_top1_seq4096.json`
- required environment: same as Run 1
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: beat Run 1 and Run 2 on post-export `val_bpb`, or remain close enough to justify the stronger context regime

## Run 4
- Goal: add CUDA Graph capture to the winning GREEN throughput lane if it can be done without bloating the root trainer.
- Why it matters: if launch overhead is still material on 8xH100, graphs are the highest-value systems lever still consistent with a throughput-first strategy.
- Risks: code complexity, graph-break fragility, reduced reproducibility.

RUN THIS MANUALLY ON H100
- command: prepare a dedicated config only after one of Runs 2 or 3 wins cleanly
- required environment: same as Run 1
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: higher train tokens processed than the parent GREEN lane with no score regression

## Run 5
- Goal: test extra RMSNorm before projections as the first bold mechanism lane.
- Why it matters: the research garden currently highlights this as the highest-priority mechanism question.
- Risks: added normalization cost may erase any quality gain.

RUN THIS MANUALLY ON H100
- command: prepare a dedicated config only after the winning GREEN lane is stable
- required environment: same as Run 1
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: clear post-export win over the best GREEN lane with stable artifact size and reproducibility

## Run 6
- Goal: test recurrent/shared-width capacity reallocation against the 16 MB limit.
- Why it matters: aggressive parameter sharing is the clearest route to buying width without exceeding the artifact cap.
- Risks: compliance ambiguity, implementation risk, and lower throughput despite parameter savings.

RUN THIS MANUALLY ON H100
- command: prepare a dedicated config only after the winning GREEN lane is stable
- required environment: same as Run 1
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: clear win over the best GREEN lane and explicit compliance review before promotion

## Run 7
- Goal: convert artifact headroom into quality via sparse or decoupled high-precision outlier retention.
- Why it matters: this is the strongest compression-side bold idea currently surfaced by the research garden.
- Risks: packing complexity, portability risk, and weak byte-efficiency relative to simple fp16 retention.

RUN THIS MANUALLY ON H100
- command: prepare a dedicated config only after the winning GREEN lane is stable
- required environment: same as Run 1
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: better post-export `val_bpb` than the best GREEN lane without breaking artifact accounting or reproducibility
