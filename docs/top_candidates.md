# Top Candidates

## Valid Mainline
- Accepted merged frontier: `#549` / `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` at `val_bpb=1.1194`.
- Active root runner: `train_gpt.py`
- Completed local truth baseline: `logs/root-pr549-softqat.txt` at `1.11956668`.
- Active root direction: valid funded dense GPTQ base with in-training GPTQ stats only and no TTT follow-up yet.
- Reference baseline snapshot: `snapshots/train_gpt_2026-03-25_pre753_pr549_softqat_root.py`
- Archived GDN snapshot: `snapshots/train_gpt_2026-03-27_prepivot_pr875_gdn_root.py`

## Active H100 Candidates
- Valid funded dense GPTQ base: `configs/h100/root_valid_gptq14_mlp1792_b4096_warmup200_xlast2.json`

## Main Prediction
- The accepted `#549` family is now a completed local reference baseline, not an active compute target.
- The best near-term valid record chance is a larger dense model funded by valid packed GPTQ5 export.
- Eval-cache / n-gram lanes remain archived pending rule clarification from OpenAI.
- The GDN branch is archived after the 1xH100 proxy landed far off-family.
- The funded dense GPTQ lane is the highest-upside non-replay path already scaffolded in this repo, but only if its calibration stays fully inside ordinary training.

## Ranking Policy
- Rank active candidates by lower final post-export `val_bpb` on the active lane's final metric.
- Require `bytes_total < 16,000,000`.
- Require H100 truth before promotion.
- Do not spend more H100 on already-done `#549` / `#589` donor replay lanes.
