# Top Candidates

## Operational Frontier
- Accepted merged frontier as of March 25, 2026: `#549` / `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` at `val_bpb=1.1194`.
- Live operational frontier for this pass: `#753` / `2026-03-25_PodracingII_backoff7gram_8xH100` at `val_bpb=0.9625` mean over 3 seeds.
- Upstream-following root donor for this pass: `pr753:train_gpt.py`.
- Single requested-point improvement under test on top of that donor: fixed-block early/mid state-space hybrid while porting a record-safer `#809`-style single-pass order-9 n-gram evaluator.

## Active Root Mainline
- Active root runner: `train_gpt.py`
- Frozen pre-hybrid snapshot: `snapshots/train_gpt_2026-03-25_pre753_state_space_hybrid_root.py`
- Frozen pre-reset snapshot: `snapshots/train_gpt_2026-03-25_pre753_pr549_softqat_root.py`
- Frozen older dense snapshot: `snapshots/train_gpt_2026-03-23_root_pr332_b458k_b1024_warmup200_xlast2.py`

## Active H100 Candidates
- Exact `#753` root repro: `configs/h100/root_pr753_repro.json`
- Dense `#753` control + `#809`-style n-gram: `configs/h100/root_pr753_repro_ngram809.json`
- `#753` + state-space hybrid + `#809`-style n-gram: `configs/h100/root_pr753_hybrid_ssm_ngram809.json`

## Main Prediction
- The dominant score lift now comes from legal score-first hashed n-gram backoff, not another training-only tweak.
- The next bounded port is `#809`-style: order-9 chunked score-first backoff with per-order alpha multipliers and chunk-global cache updates.
- The hybrid should still be judged first on `final_int6_sliding_window_exact`, then on the final n-gram exact metric under the same evaluator.
- If the dense `#809`-style control does not look healthy on H100, do not trust the hybrid lane yet.

## Ranking Policy
- Rank by lower final n-gram exact `val_bpb`.
- Keep `final_int6_sliding_window_exact` as the dense-model guardrail.
- Require `bytes_total < 16,000,000`.
- Require H100 truth before promotion.
