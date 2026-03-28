# Hopper Plan

Active valid frontier for this pass:
- accepted merged record: `#549`
- completed local reference baseline: `logs/root-pr549-softqat.txt` at `1.11956668`
- reference baseline snapshot: `snapshots/train_gpt_2026-03-25_pre753_pr549_softqat_root.py`
- archived GDN snapshot: `snapshots/train_gpt_2026-03-27_prepivot_pr875_gdn_root.py`
- active root delta: valid funded dense GPTQ base with no TTT carryover

## Run Order
- Run 1: valid funded dense GPTQ base

## Run 1
- Goal: spend packed GPTQ5 export headroom on a materially larger dense model while staying fully valid.
- Command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_valid_gptq14_mlp1792_b4096_warmup200_xlast2.json`
- Success criteria: beat the completed `#589`-style reference baseline of `1.11956668`, stay under `16,000,000` bytes, and fit the 10-minute train/eval limits.
- Risk: if packed GPTQ no longer buys enough bytes after removing the invalid post-cap calibration pass, the lane may lose its upside.
- Pod sanity check: `rg -n "GPTQStatsCollector|gptq:start in_training|collect_gptq_stats|ttt_" train_gpt.py`
- Timing expectation: unknown until the first truth run; no second H100 lane is active until this base lands cleanly.

## Notes
- `#753` / eval-cache / hybrid work remains archived pending rule clarification and is not part of the active record ladder.
- The `#875` GDN branch is archived after the 1xH100 proxy landed far off-family and did not justify more scarce H100 runs.
- Tokenizer WIP is archived for this pass and should not be mixed into the active ladder.
- The completed `#549` / `#589` local truth run is reference-only now; do not spend more H100 compute replaying accepted donor PRs.
- `flash-attn` remains optional and cluster-side.
