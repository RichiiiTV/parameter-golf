# Top Candidates

## Operational Frontier
- Accepted merged frontier as of March 24, 2026: `#549` / `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` at `val_bpb=1.1194`.
- Upstream-following root donor for this pass: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`.
- Single improvement under test on top of that donor: late soft-round QAT from the `#589` family.

## Active Root Mainline
- Active root runner: `train_gpt.py`
- Frozen pre-reset snapshot: `snapshots/train_gpt_2026-03-24_pre549_dense_gptq_tttlite_root.py`
- Frozen older dense snapshot: `snapshots/train_gpt_2026-03-23_root_pr332_b458k_b1024_warmup200_xlast2.py`

## Active H100 Candidates
- Exact `#549` root repro: `configs/h100/root_pr549_repro.json`
- `#549` + late soft-round QAT: `configs/h100/root_pr549_softqat.json`

## Main Prediction
- The strongest source-backed next delta on the accepted `#549` family is not another exporter rewrite or TTT variant.
- It is the late soft-round QAT surrogate from the `#589` family, because it attacks the remaining post-quantization loss while keeping the same legal score-first TTT frame.
- If the exact `#549` repro does not land in-family on H100, do not trust the soft-round lane yet.

## Ranking Policy
- Rank by lower final post-TTT `val_bpb`.
- Keep `final_int6_roundtrip_exact` as the guardrail metric before TTT.
- Require `bytes_total < 16,000,000`.
- Require H100 truth before promotion.
