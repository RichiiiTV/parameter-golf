# Hopper Plan

Operational frontier for this pass:
- accepted merged record: `#549`
- donor record path: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
- reported result: `val_bpb=1.1194`, `bytes_total=15,990,006`
- active root delta: late soft-round QAT only

## Run Order
- Run 1: exact `#549` root repro
- Run 2: `#549` root repro plus `SOFT_QAT_ENABLED=1`

## Run 1
- Goal: verify that root matches the accepted `#549` banking, export, and legal score-first TTT stack on H100.
- Command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_pr549_repro.json`
- Success criteria: stay in-family with the accepted `#549` quality, byte, and time band.
- Risk: if this misses the accepted band, the root port is not yet trustworthy enough to judge the improvement lane.

## Run 2
- Goal: test only the late soft-round QAT surrogate from the `#589` family on top of the exact `#549` root lane.
- Command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_pr549_softqat.json`
- Success criteria: improve final post-TTT `val_bpb` without exceeding the size or eval budgets.
- Risk: the soft-round surrogate may help pre-TTT exact quality but fail to move the final TTT score enough to matter.

## Notes
- Root was hard-reset to the upstream `#549` donor before this pass.
- The pre-reset dense GPTQ/TTT-lite root is preserved in `snapshots/train_gpt_2026-03-24_pre549_dense_gptq_tttlite_root.py`.
- `flash-attn` remains optional and cluster-side.
