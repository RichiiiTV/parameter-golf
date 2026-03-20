# Hopper Plan

Current public best to beat:
- `10L Int5-MLP + BigramHash(10240) + SWA(frac=0.4) + WD=0.04` at `mean_val_bpb=1.14276`
- Record path: `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50`

## Active H100 Ladder
- Run 1: `records/track_10min_16mb/2026-03-20_PreProjRMSNorm_Int5MLP_Bigram10240_SWA` with `USE_PREPROJ_RMSNORM=0`
- Run 2: the same folder with `USE_PREPROJ_RMSNORM=1`
- Run 3: batch-token sweep on the winning mechanism setting
- Run 4: three-seed promotion on the winning batch setting
- Run 5: recurrent/shared-width only after the pre-proj fork is settled
- Run 6: sparse outlier retention only after the pre-proj fork is settled

## Run 1
- Goal: reproduce the current March 20 top record stack inside our own fork before enabling the new mechanism.
- Why it matters: this is the true control for the contender.
- Risks: fork drift, missing environment assumptions, or worse throughput than the parent record.

RUN THIS MANUALLY ON H100
- command: from `records/track_10min_16mb/2026-03-20_PreProjRMSNorm_Int5MLP_Bigram10240_SWA`, run the baseline `sbatch --wrap` command from `README.md`
- required environment: repo checkout, dataset cache, tokenizer cache, cluster path where the record folder can run in place
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: match or nearly match the parent March 20 record behavior and remain under `16,000,000` bytes

## Run 2
- Goal: enable pre-projection RMSNorm on the same stack.
- Why it matters: this is the first bold mechanism lane selected for our side.
- Risks: added normalization cost could reduce throughput or hurt quality.

RUN THIS MANUALLY ON H100
- command: from `records/track_10min_16mb/2026-03-20_PreProjRMSNorm_Int5MLP_Bigram10240_SWA`, run the contender `sbatch --wrap` command from `README.md`
- required environment: same as Run 1
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: beat Run 1 on post-export `val_bpb` while staying within the byte cap

## Run 3
- Goal: sweep batch tokens on the better mechanism setting.
- Why it matters: throughput is the cleanest next lever after mechanism choice.
- Risks: higher batch tokens may reduce steps or trigger throughput regressions instead of helping.

RUN THIS MANUALLY ON H100
- command: use the three batch-sweep `sbatch --wrap` commands from the new record folder `README.md`
- required environment: same as Run 1
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: choose the best post-export `val_bpb` among `786432`, `917504`, and `1048576` without violating the byte cap

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
