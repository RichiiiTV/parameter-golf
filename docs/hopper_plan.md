# Hopper Plan

Current runnable frontier target for this pass:
- accepted merged reference: `#549`
- preserved local truth baseline: `logs/root-pr549-softqat.txt` at `1.11956668`
- preserved April 22 root snapshots remain historical references only
- active root delta: compact `#1667`-class SP8192 port with SmearGate, attention-output gate, 3-layer depth recurrence, parallel residuals, int6/int7 export, and legal score-first TTT

## Run Order
- Run 1: `configs/h100/root_sp8192_pr1667_legal_ttt.json`
- Optional proxy only: `configs/h100/root_sp8192_pr1667_legal_ttt_proxy_1xh100.json`

## Run 1
- Goal: test the active SP8192 frontier lane on 8xH100 with the compact root, not the older SHD/SP1024 baseline.
- Command generation: `python scripts/prepare_h100_run.py configs/h100/root_sp8192_pr1667_legal_ttt.json`
- Success criteria: stay under `16,000,000` bytes, keep train and legal score-first TTT eval within their `600s` limits, and materially beat the preserved `#549` truth baseline on `quantized_ttt`.
- Risk: the compact port may lose quality versus the full upstream donor because it omits the heavier `#1626`-family machinery.
- Pod sanity check: `rg -n "VOCAB_SIZE|TTT_ENABLED|SMEAR_GATE|GATE_ATTN_OUT|QK_GAIN_INIT|NUM_LOOPS|PARALLEL_START_LAYER" train_gpt.py`
- Dataset prerequisite: `python data/cached_challenge_fineweb.py --variant sp8192`

## Notes
- The active lane is chosen specifically over `#1771`, `#1767`, `#1765`, `#1775`, and `#1776` for legality or code-fit reasons.
- `flash-attn` remains optional and cluster-side.
- Do not auto-launch H100 runs from this repo.
