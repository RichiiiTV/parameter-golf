# Hopper Plan

Current public best to beat:
- `10L Int5-MLP + BigramHash(10240) + SWA(frac=0.4) + WD=0.04` at `mean_val_bpb=1.14276`
- Record path: `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50`

## Active H100 Ladder
- Run 1: `records/track_10min_16mb/2026-03-20_PreProjRMSNorm_Int5MLP_Bigram10240_SWA` with `USE_PREPROJ_RMSNORM=0`
- Run 2: batch-token sweep on `USE_PREPROJ_RMSNORM=0`
- Run 3: three-seed promotion on the winning batch setting
- Run 4: exact parent-folder reproduction only if the fork baseline diverges materially
- Run 5: recurrent/shared-width only after the throughput lane is settled
- Run 6: sparse outlier retention only after the throughput lane is settled
- Run 7: pre-proj RMSNorm revisit only if a later throughput-stable branch justifies it

## Run 1
- Goal: reproduce the current March 20 top record stack inside our own fork on the active path.
- Why it matters: this is the true control for the fork.
- Risks: fork drift, missing environment assumptions, or worse throughput than the parent record.

RUN THIS MANUALLY ON H100
- command: from `records/track_10min_16mb/2026-03-20_PreProjRMSNorm_Int5MLP_Bigram10240_SWA`, run the baseline `sbatch --wrap` command from `README.md`
- required environment: repo checkout, dataset cache, tokenizer cache, cluster path where the record folder can run in place
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: match or nearly match the parent March 20 record behavior and remain under `16,000,000` bytes

## Run 2
- Goal: sweep batch tokens on the active parent-stack path.
- Why it matters: throughput is the cleanest next lever after the negative 4xA100 pre-proj ablation.
- Risks: higher batch tokens may reduce steps or trigger throughput regressions instead of helping.

RUN THIS MANUALLY ON H100
- command: from `records/track_10min_16mb/2026-03-20_PreProjRMSNorm_Int5MLP_Bigram10240_SWA`, run the three batch-sweep `sbatch --wrap` commands from `README.md`
- required environment: same as Run 1
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: choose the best post-export `val_bpb` among `786432`, `917504`, and `1048576` without violating the byte cap

## Run 3
- Goal: promote the winning batch setting to a three-seed check.
- Why it matters: the branch is only useful if the throughput win survives seed variation.
- Risks: weak seed robustness or byte drift.

RUN THIS MANUALLY ON H100
- command: use the three-seed `sbatch --wrap` commands from the new record folder `README.md`, keeping `USE_PREPROJ_RMSNORM=0`
- required environment: same as Run 1
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: mean post-export `val_bpb` improves on the parent-stack reproduction and remains under the byte cap

## Run 4
- Goal: reproduce the exact parent March 20 folder if the fork baseline diverges materially.
- Why it matters: this separates fork drift from real candidate weakness.
- Risks: duplicate compute if the fork baseline already matches the parent closely.

RUN THIS MANUALLY ON H100
- command: run the parent folder `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50` exactly as documented in its `README.md`
- required environment: same as Run 1
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: determine whether the fork is faithful enough to use as the active throughput branch

## Run 5
- Goal: test recurrent/shared-width capacity reallocation against the 16 MB limit.
- Why it matters: aggressive parameter sharing is the clearest route to buying width without exceeding the artifact cap.
- Risks: compliance ambiguity, implementation risk, and lower throughput despite parameter savings.

RUN THIS MANUALLY ON H100
- command: prepare a dedicated config only after the winning GREEN lane is stable
- required environment: same as Run 1
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: clear win over the best GREEN lane and explicit compliance review before promotion

## Run 6
- Goal: convert artifact headroom into quality via sparse or decoupled high-precision outlier retention.
- Why it matters: this is the strongest compression-side bold idea currently surfaced by the research garden.
- Risks: packing complexity, portability risk, and weak byte-efficiency relative to simple fp16 retention.

RUN THIS MANUALLY ON H100
- command: prepare a dedicated config only after the winning GREEN lane is stable
- required environment: same as Run 1
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: better post-export `val_bpb` than the best GREEN lane without breaking artifact accounting or reproducibility

## Run 7
- Goal: revisit pre-projection RMSNorm only after a throughput-stable GREEN lane exists.
- Why it matters: the current same-folder 4xA100 ablation was negative, so this mechanism is deferred rather than active.
- Risks: additional normalization cost may still erase any quality gain on H100.

RUN THIS MANUALLY ON H100
- command: prepare a dedicated config only after the winning GREEN lane is stable
- required environment: same as Run 1
- expected runtime budget: 10 minutes train, 10 minutes eval
- success criteria: clear post-export win over the best GREEN lane with stable artifact size and reproducibility
