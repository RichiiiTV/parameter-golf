# Top Candidates

## Active Frontier Lane
- Active root runner: `train_gpt.py`
- Active frontier package: `frontier_gdn/`
- Runnable H100 target: `configs/h100/root_sp8192_pr1791_fla_8xh100.json`
- Directional proxy: `configs/h100/root_sp8192_pr1791_fla_proxy_1xh100.json`
- Local sanity: `configs/local/root_sp8192_pr1791_fla_sanity.json`
- Required dataset/tokenizer surface: custom SP8192 export via `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf`
- Preserved local reference baseline: `logs/root-pr549-softqat.txt` at `1.11956668`
- Preserved historical snapshots: `snapshots/train_gpt_2026-04-22_pre_shd_pivot_root.py`, `snapshots/train_gpt_2026-04-22_pre_shd_only_prune_root.py`, `snapshots/train_gpt_2026-04-22_pre_sp8192_pr1667_pivot_root.py`, `snapshots/train_gpt_2026-04-23_pre_public_sp1024_reset_root.py`, and `snapshots/train_gpt_2026-04-23_pre_pr1791_fla_pivot_root.py`

## Upstream Check
- Checked on April 23, 2026: `#1791` leads the open SP8192 pack at `1.0339`.
- Relevant alternatives: `#1787` at `1.06378`, `#1790` at `1.06991`, `#1771` at `1.06513` with legality pending, and `#1767` at `1.07209`.
- Decision: target `#1791` first because it is the strongest visible open lane and avoids the active TTT / CaseOps legality path.

## Current Use
- Use the proxy first to validate the FLA runtime, SP8192 cache, artifact size, and roundtrip stability.
- Promote the 8xH100 lane only after the proxy is stable.
- If the FLA proxy cannot be made evaluation-stable on the pinned runtime, re-plan around `#1787`; do not half-port `#1791`.
