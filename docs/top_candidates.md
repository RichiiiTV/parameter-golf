# Top Candidates

## Active Frontier Lane
- Active root runner: `train_gpt.py`
- Runnable H100 target: `configs/h100/root_sp8192_pr1667_legal_ttt.json`
- Directional proxy: `configs/h100/root_sp8192_pr1667_legal_ttt_proxy_1xh100.json`
- Local sanity: `configs/local/root_sp8192_pr1667_legal_ttt_sanity.json`
- Required dataset/tokenizer surface: `sp8192`
- Preserved local reference baseline: `logs/root-pr549-softqat.txt` at `1.11956668`
- Preserved April 22 snapshots: `snapshots/train_gpt_2026-04-22_pre_shd_pivot_root.py`, `snapshots/train_gpt_2026-04-22_pre_shd_only_prune_root.py`, and `snapshots/train_gpt_2026-04-22_pre_sp8192_pr1667_pivot_root.py`

## April 22 Frontier Choice
- Latest open upstream leaders checked on April 22, 2026: `#1767` at `1.07209`, `#1765` at `1.07266`, `#1775` at `1.07285`, `#1776` at `1.08083`, and `#1771` at `1.06513` with legality pending.
- Chosen root target: `#1667` at `1.07139`.
- Rejected `#1771` as the primary target because legality is still pending.
- Rejected `#1767` and `#1765` because they depend on the larger `#1626` varlen / phased-TTT stack and fit this compact root poorly.
- Rejected `#1775` and `#1776` because both are lower-upside bridges than the direct `#1667` lane.

## Current Use
- Treat this repo as a compact `#1667`-class SP8192 frontier attempt.
- Use the H100 main config for any record-intent manual run.
- Use the 1xH100 proxy only for cheap directional reads.
