# Hopper Plan

Current runnable baseline for this pass:
- accepted merged record: `#549`
- preserved local reference baseline: `logs/root-pr549-softqat.txt` at `1.11956668`
- preserved April 22 snapshots: `snapshots/train_gpt_2026-04-22_pre_shd_pivot_root.py` and `snapshots/train_gpt_2026-04-22_pre_shd_only_prune_root.py`
- active root delta: `#1060`-derived 11-layer scaffold with reserved-time full-Hessian GPTQ6, a prune-funded `BigramHash(3072,112)` upgrade, and SHD-tied Q/K tails with `SHARED_HEAD_DIM=16`
- strict frontier verdict on April 22, 2026: this root is not a plausible global SOTA path against the latest open SP8192 + TTT leaders

## Run Order
- Run 1: SHD/SP1024 baseline characterization
- Optional proxy only: `configs/h100/root_pr1060_shd_proxy_1xh100.json` on one H100 if you need a cheap directional read; do not treat it as timing, legality, or frontier proof for the 8x lane.

## Run 1
- Goal: measure the cleaned SHD/SP1024 root cleanly on H100 without claiming frontier status.
- Command generation: `python scripts/prepare_h100_run.py configs/h100/root_pr1060_shd_b3072_prune.json`
- Success criteria: stay under `16,000,000` bytes, keep total train-plus-calibration wallclock inside `600s`, and measure the gap versus the completed `1.11956668` local reference baseline on `sliding_window_exact`.
- Risk: full-Hessian reserved-time GPTQ may consume more of the `600s` budget than the retired in-training collector lane.
- Pod sanity check: `rg -n "SHARED_HEAD_DIM|GPTQ_RESERVE_MS|gptq:start reserved_train_data|gptq:calibrated|ttt_" train_gpt.py`
- Timing expectation: unknown until the first H100 truth run; do not spend record-intent H100 compute from this repo before a separate frontier pivot exists.

## Notes
- Latest upstream open leaders checked on April 22, 2026 are `#1767`, `#1765`, `#1775`, `#1776`, and `#1771`; the current repo does not implement those frontier stacks.
- Upstream SHD-only `#1774` validates the portability of SHD as a compression trick, but it still trails the open SP8192 + TTT frontier materially.
- The completed `#549` / `#589` local truth run is reference-only now.
- The active root no longer carries runnable fallback configs; use the preserved April 22 snapshots only for historical inspection.
- `flash-attn` remains optional and cluster-side.
