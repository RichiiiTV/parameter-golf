# Top Candidates

## Active Baseline
- Active root runner: `train_gpt.py`
- Runnable H100 baseline: `configs/h100/root_pr1060_shd_b3072_prune.json`
- Directional proxy: `configs/h100/root_pr1060_shd_proxy_1xh100.json`
- Local sanity: `configs/local/root_pr1060_shd_sanity.json`
- Preserved local reference baseline: `logs/root-pr549-softqat.txt` at `1.11956668`
- Preserved April 22 snapshots: `snapshots/train_gpt_2026-04-22_pre_shd_pivot_root.py` and `snapshots/train_gpt_2026-04-22_pre_shd_only_prune_root.py`

## April 22 Frontier Check
- Accepted merged record remains `#549` / `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` at `1.1194`.
- Latest open upstream leaders checked on April 22, 2026: `#1767` at `1.07209`, `#1765` at `1.07266`, `#1775` at `1.07285`, `#1776` at `1.08083`, and `#1771` at `1.06513` with legality pending.
- Upstream SHD-only reference: `#1774` at `1.09813`.
- Verdict: the current SHD-only SP1024 root is a clean local baseline, not a plausible global SOTA path.

## Current Use
- Use the baseline lane to measure the cleaned root under the existing byte and wallclock rules.
- Do not describe this repo as carrying a promoted frontier lane.
- Do not spend record-intent H100 compute until a separate frontier pivot is planned.
