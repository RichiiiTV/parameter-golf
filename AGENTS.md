# Mission
- Maximize final post-export `val_bpb` under the 16,000,000-byte artifact cap and the 10-minute train / 10-minute eval limits on 8xH100 SXM.

# Current challenge interpretation
- Source of truth for accepted records is the official `README.md` plus accepted `records/`.
- Operational SOTA source of truth for this pass is local `pr332`, specifically `records/track_10min_16mb/2026-03-21_12L_GradQuant_PartialRoPE_EMA`, which reports `val_bpb=1.1320`.
- The merged upstream `README.md` still shows the March 20 winner at `1.1428`, so root work in this pass is intentionally targeting the ahead-of-README PR frontier.
- Active work is H100-only, manual-only, and throughput-first.

# Hard constraints
- Never auto-run H100 jobs.
- Never exceed 16,000,000 artifact bytes on a record candidate.
- Never access network or training data during evaluation.
- Never brute-force seeds or run spirit-violating search.
- Keep `train_gpt.py` and `train_gpt_mlx.py` under 1500 lines by `scripts/check_line_budget.py`.

# Local environment assumptions
- Windows PowerShell is the local editing environment.
- Local machines are for code review, static checks, and H100 handoff generation only.
- Local correctness runs are legacy only and are not part of the active ranking workflow.

# H100 target assumptions
- Single-node 8xH100 SXM.
- Human-gated only.
- Standard PyTorch paths are the default optimization surface.
- Root `train_gpt.py` is the canonical runner for this pass and should stay near-verbatim to the `#332` donor script.
- `flash-attn` is optional in code and recommended in the H100 environment, not a mandatory repo dependency.

# GREEN / YELLOW / RED idea board
- `GREEN`: gradient-guided adaptive quantization, 12 layers, `MLP_HIDDEN=1408`, `BIGRAM_VOCAB_SIZE=2048`, `BIGRAM_DIM=128`, `TRAIN_SEQ_LEN=2048`, `TRAIN_BATCH_TOKENS=524288`, `PARTIAL_ROPE_DIMS=16`, `LN_SCALE=1`, `XSA_LAST_N=4`, `EMA_ENABLED=1`, sliding eval at `stride=64`, bigger BigramHash table if it fits measured byte headroom.
- `YELLOW`: `EVAL_SEQ_LEN=4096` follow-up, recurrent/shared-width transformers, sparse high-precision outlier retention, tokenizer changes, eval-time adaptation, custom Triton kernels.
- `RED`: auto-launched H100 jobs, over-budget train/eval, network/data access during eval, oversized artifact, seed brute force.

# Roles and ownership
- Rule Auditor: `docs/compliance_matrix.md`
- Baseline Reverse Engineer: `docs/baseline_map.md`
- Hopper Systems Engineer: `docs/hopper_plan.md`, `configs/h100/*`, `scripts/prepare_h100_run.py`
- Quantization / Export Engineer: `docs/export_strategy.md`
- Architecture / Optimization Researcher: `docs/model_ideas.md`
- Triton Investigator: `docs/triton_plan.md`
- Experiment Manager: `experiments/registry.csv`, `docs/top_candidates.md`
- Legacy Surfaces: `docs/local_harness.md`, `docs/autosearch_plan.md`

# Experiment naming convention
- Executed runs use `YYYYMMDDTHHMMSSZ-<profile-slug>`.
- Planned configs use `h100-<lane>-<major_knob>`.

# Standard metrics to report
- `pre_val_loss`
- `pre_quant_val_bpb`
- `post_val_loss`
- `post_quant_val_bpb`
- `pre_post_gap`
- `bytes_total`
- `eval_time_ms`
- `train_tokens_seen`
- `step_avg_ms`

# Required artifacts for every experiment
- `config.json`
- `env.json`
- `git.json`
- `command.txt`
- `stdout.log`
- `result.json`

# Current hypotheses
- Root `train_gpt.py` should be an exact `#332` port first, not another March 20 derivative.
- The strongest first improvement is the cap-safe combined throughput-and-capacity branch: `TRAIN_BATCH_TOKENS=458752`, `ITERATIONS=11000`, and `BIGRAM_VOCAB_SIZE=2560`.
- The batch reduction is meant to buy more optimization steps inside 600 seconds; the higher iteration cap keeps that gain from being clipped by the static `9000` iteration ceiling.
- The bigger BigramHash table spends measured artifact headroom at near-zero training-step cost.
- The next follow-up should be an eval-only context increase to `EVAL_SEQ_LEN=4096` only if the combined branch stays within the eval budget and lands comfortably under cap.
- The reported `0.88618079` on `8xA100` is invalid for compliance and ranking decisions because it predates the root metric repair.

# Current blockers
- No H100 truth run has been executed from the root `#332` port yet.
- No H100 truth run has been executed from the combined `b458k + BIGRAM_VOCAB_SIZE=2560` root variant yet.
- No H100 truth run has been executed from the `BIGRAM_VOCAB_SIZE=2432` fallback rescue.
- The eval-only `EVAL_SEQ_LEN=4096` follow-up is budget-gated and should not be promoted before the 2560-bucket rescue is healthy.
- The latest fixed-metric `8xA100` rerun of the 3584-bucket branch is still over cap at `16,069,767` bytes even though its exact metric now looks plausible.
- The root metric path was corrupted by the wrong SentencePiece leading-space marker instead of the true `U+2581` marker, so the pre-fix A100 `0.88618079` result is not trustworthy.

# Current best candidates
- root `train_gpt.py` exact `#332` reproduction
- root `train_gpt.py` with `TRAIN_BATCH_TOKENS=458752`, `ITERATIONS=11000`, `BIGRAM_VOCAB_SIZE=2560`
- root `train_gpt.py` with `TRAIN_BATCH_TOKENS=458752`, `ITERATIONS=11000`, `BIGRAM_VOCAB_SIZE=2432` as the cap-margin fallback
- root `train_gpt.py` with `TRAIN_BATCH_TOKENS=458752`, `ITERATIONS=11000`, `BIGRAM_VOCAB_SIZE=2560`, `EVAL_SEQ_LEN=4096`, `EVAL_BATCH_SEQS=16`

# Next 10 experiments
- Run 1: rerun root `#332` with `TRAIN_BATCH_TOKENS=458752`, `ITERATIONS=11000`, and `BIGRAM_VOCAB_SIZE=2560`
- Run 2: rerun the same branch with `BIGRAM_VOCAB_SIZE=2432` only if Run 1 is still over cap or lands under cap with too little margin
- Run 3: Run 1 plus `EVAL_SEQ_LEN=4096 EVAL_BATCH_SEQS=16` if Run 1 stays within eval budget and lands comfortably under cap
- Run 4: three-seed sweep at `SEED=1337` on the winning root `#332` or cap-rescued BigramHash setting
- Run 5: three-seed sweep at `SEED=1338` on the winning root `#332` or cap-rescued BigramHash setting
- Run 6: three-seed sweep at `SEED=1339` on the winning root `#332` or cap-rescued BigramHash setting
- Run 7: exact March 20 accepted top record only if the root `#332` port diverges materially
- Run 8: bigger BigramHash follow-up beyond `4096` only if bytes leave further headroom
- Run 9: YELLOW `EVAL_SEQ_LEN=4096` promotion only after a healthy `BIGRAM_VOCAB_SIZE=2560` run
- Run 10: recurrent/shared-width or sparse-outlier follow-up only after a winning GREEN root lane exists

# Decision log
- Port root `train_gpt.py` from local `pr332:records/track_10min_16mb/2026-03-21_12L_GradQuant_PartialRoPE_EMA/train_gpt.py`.
- Keep the root port semantically identical to `#332`; only trim comments/blank lines for the line cap.
- Keep root default `BIGRAM_VOCAB_SIZE=2048`; express the larger-table upgrades in configs, not by silently changing the donor baseline.
- Promote a smaller-batch branch as the main improvement because `#332` itself attributes part of its gain to more optimization steps from a reduced batch.
- Use `TRAIN_BATCH_TOKENS=458752` because it is a conservative 12.5% reduction from `524288` that preserves clean geometry at `TRAIN_SEQ_LEN=2048`.
- Raise `ITERATIONS` to `11000` on that branch so the run stays wallclock-limited instead of clipping against the donor `9000`-iteration ceiling.
- After the fixed-metric `3584` rerun came in at `16,069,767` total bytes and `1.18226244` exact `val_bpb`, move the active rescue to `BIGRAM_VOCAB_SIZE=2560`.
- At the current ratio `16,004,657 / 28,057,988 ~= 0.5704`, each BigramHash row is worth about `130 * 0.5704 ~= 74.2` compressed bytes.
- `3584 -> 2560` removes `1024` rows and should save about `75,933` compressed bytes at the current ratio.
- Keep `BIGRAM_VOCAB_SIZE=2432` as the fallback because it should buy about `85,425` compressed bytes and therefore stronger cap margin.
- Defer `EVAL_SEQ_LEN=4096` to a run-level follow-up because it is eval-budget-gated.
- Keep `flash-attn` optional in code.
- Keep Triton blocked until a winning post-`#332` GREEN lane exists and a standard-backend bottleneck is measured on H100.
- Restore the correct SentencePiece leading-space marker in root metric code: `U+2581`, not `"?"`.
- Treat `logs/root-pr332-b458k-bigram3584.txt` as pre-fix debugging evidence only.
- Treat the newest verified section of `logs/root-pr332-b458k-bigram3584 (2).txt` as the current planning anchor: plausible metric path but still over cap and not H100 truth.

# Rules for modifying code
- Keep record-critical logic in `train_gpt.py`.
- Keep root changes minimal and close to the donor `#332` structure.
- Do not stack multiple algorithmic deltas in the same root pass.
- Prefer record-folder implementations for aggressive YELLOW ideas.

# Rules for adding dependencies
- No new mandatory runtime dependencies in this pass.
- Standard library only for workflow scripts.
- `flash-attn` remains optional and cluster-side.
- Do not add Triton or other optional packages unless a later H100 benchmark proves they matter.

# Rules for touching metric/eval code
- Root eval changes must preserve challenge semantics and stay self-contained.
- Log sequence length, stride, final roundtrip metrics, and eval time.
- Do not introduce proxy-only evaluation behavior into the active root path.

# Rules for challenge-edge ideas
- Document YELLOW ideas before any implementation.
- Keep YELLOW ideas in explicit lanes with compliance notes.
- Do not merge RED ideas.

# Handoff protocol between agents
- Update `experiments/registry.csv`.
- Update `docs/top_candidates.md`.
- Update the relevant design doc before code or config changes become canonical.
- If agents are spawned, they must use `gpt-5.4-mini` with non-fast reasoning only, then be closed immediately after the audit returns.

# Triton usage policy
- Standard PyTorch SDPA/cuDNN/Inductor first.
- Custom Triton only after measured standard-backend loss on H100.
- Any custom Triton path must be optional and have a non-Triton fallback.

# Local-to-H100 transfer policy
- The active workflow is H100-only.
- Local machines are for editing, static validation, and manual command generation.
- Do not rank active candidates from GTX or A100 behavior in this pass.
