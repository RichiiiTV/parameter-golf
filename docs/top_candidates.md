# Top Candidates

## Valid Mainline
- Accepted merged frontier: `#549` / `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` at `val_bpb=1.1194`.
- Latest plausible valid open leader: `#1060` at `1.1122` as of March 29, 2026.
- Active root runner: `train_gpt.py`
- Completed local truth baseline: `logs/root-pr549-softqat.txt` at `1.11956668`.
- Active root direction: `#1060`-derived 11-layer root with coprime multi-shard loading, reserved-time full-Hessian GPTQ6, and no TTT.
- Reference baseline snapshot: `snapshots/train_gpt_2026-03-25_pre753_pr549_softqat_root.py`
- Archived pre-pivot dense-GPTQ snapshot: `snapshots/train_gpt_2026-03-29_pre1060_valid_dense_gptq_root.py`
- Archived GDN snapshot: `snapshots/train_gpt_2026-03-27_prepivot_pr875_gdn_root.py`

## Active H100 Candidates
- Promoted `#1060` follow-up: `configs/h100/root_pr1060_b3072_prune.json`

## Main Prediction
- The accepted `#549` family is now a completed local reference baseline, not an active compute target.
- The best near-term valid record chance is a `#1060`-derived lane with one byte-funded delta: `BigramHash(3072,112)` funded by selective `±1` export pruning.
- Eval-cache / n-gram lanes remain archived pending rule clarification from OpenAI.
- The GDN branch is archived after the 1xH100 proxy landed far off-family.
- `#1047`, `#1056`, and `#875` are not active valid targets for this repo.
- The new root only stays valid if the reserved-time train-data calibration starts before the logged GPTQ reserve boundary and finishes inside the same `600s` wallclock.

## Ranking Policy
- Rank active candidates by lower final post-export `val_bpb` on the active lane's final metric.
- Require `bytes_total < 16,000,000`.
- Require H100 truth before promotion.
- Do not spend more H100 on already-done `#549` / `#589` donor replay lanes or on an exact `#1060` donor replay.
