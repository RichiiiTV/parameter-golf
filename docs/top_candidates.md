# Top Candidates

## Operational Frontier
- Accepted merged frontier as of March 25, 2026: `#549` / `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` at `val_bpb=1.1194`.
- Live operational frontier for this pass: `#753` / `2026-03-25_PodracingII_backoff7gram_8xH100` at `val_bpb=0.9625` mean over 3 seeds.
- Upstream-following root donor for this pass: `pr753:train_gpt.py`.
- Single improvement under test on top of that donor: double `NGRAM_EVAL_BUCKETS` from `4194304` to `8388608`.

## Active Root Mainline
- Active root runner: `train_gpt.py`
- Frozen pre-reset snapshot: `snapshots/train_gpt_2026-03-25_pre753_pr549_softqat_root.py`
- Frozen older dense snapshot: `snapshots/train_gpt_2026-03-23_root_pr332_b458k_b1024_warmup200_xlast2.py`

## Active H100 Candidates
- Exact `#753` root repro: `configs/h100/root_pr753_repro.json`
- `#753` + `8M` n-gram buckets: `configs/h100/root_pr753_bucket8m.json`

## Main Prediction
- The dominant score lift now comes from legal score-first hashed n-gram backoff, not another training-only tweak.
- The cleanest first extension is a bigger runtime n-gram table, because it attacks collisions directly without spending artifact bytes.
- If the exact `#753` repro does not land in-family on H100, do not trust the `8M` bucket follow-up yet.

## Ranking Policy
- Rank by lower final n-gram exact `val_bpb`.
- Keep `final_int6_sliding_window_exact` as the dense-model guardrail.
- Require `bytes_total < 16,000,000`.
- Require H100 truth before promotion.
