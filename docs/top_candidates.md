# Top Candidates

## Valid Mainline
- Accepted merged frontier: `#549` / `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` at `val_bpb=1.1194`.
- Active root runner: `train_gpt.py`
- Active root direction: exact `#549` repro plus one `#589`-style late soft-round QAT follow-up.
- Active fallback snapshot: `snapshots/train_gpt_2026-03-25_pre753_pr549_softqat_root.py`
- Archived GDN snapshot: `snapshots/train_gpt_2026-03-27_prepivot_pr875_gdn_root.py`

## Active H100 Candidates
- Exact `#549` repro: `configs/h100/root_pr549_repro.json`
- Exact `#549` + late soft-round QAT: `configs/h100/root_pr549_softqat.json`

## Main Prediction
- The best near-term valid record chance is still the accepted `#549` family, not another fresh architecture.
- The best single follow-up remains late soft-round QAT from the `#589` family.
- Eval-cache / n-gram lanes remain archived pending rule clarification from OpenAI.
- The GDN branch is archived after the 1xH100 proxy landed far off-family.

## Ranking Policy
- Rank active candidates by lower final post-TTT `val_bpb`.
- Require `bytes_total < 16,000,000`.
- Require H100 truth before promotion.
