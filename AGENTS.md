# Mission
- Maximize final post-export `val_bpb` under the 16,000,000-byte artifact cap and the 10-minute train / 10-minute eval limits on 8xH100 SXM.

# Current challenge interpretation
- Source of truth for accepted records is the official `README.md` plus accepted `records/`.
- Operational SOTA source of truth for this pass is local `pr332`, specifically `records/track_10min_16mb/2026-03-21_12L_GradQuant_PartialRoPE_EMA`, which reports `val_bpb=1.1320`.
- The merged upstream `README.md` still shows the March 20 winner at `1.1428`, so root work in this pass is intentionally targeting the ahead-of-README PR frontier.
- Active work is H100-only, manual-only, and split between a frozen dense snapshot fallback and a shared-depth root lane.

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
- `GREEN`: frozen dense snapshot `b1024-warmup200-xlast2`, shared-depth root lane with `UNIQUE_BLOCKS=8` and `MLP_HIDDEN=1664`, gradient-guided adaptive quantization, `PARTIAL_ROPE_DIMS=16`, `LN_SCALE=1`, `EMA_ENABLED=1`, sliding eval at `stride=64`.
- `YELLOW`: `EVAL_SEQ_LEN=4096` follow-up, more aggressive shared-depth schedules, sparse high-precision outlier retention, tokenizer changes, eval-time adaptation, custom Triton kernels.
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
- The frozen dense snapshot is the current fallback H100 candidate and must remain runnable by path.
- The next root lane should buy more effective depth-per-byte with `UNIQUE_BLOCKS=8` over 12 applications, not more BigramHash tuning.
- The first funded capacity increase in the shared lane should be `MLP_HIDDEN=1664` while keeping the best dense training recipe otherwise fixed.
- Sliding-window `val_bpb` is the first promotion metric; exact roundtrip remains the guardrail.

# Current blockers
- No H100 truth run has been executed from the frozen dense snapshot lane.
- No H100 truth run has been executed from the shared-depth root lane.
- The shared-depth lane must fit under the root line budget without moving record-critical logic out of `train_gpt.py`.

# Current best candidates
- frozen snapshot `snapshots/train_gpt_2026-03-23_root_pr332_b458k_b1024_warmup200_xlast2.py`
- root `train_gpt.py` with `UNIQUE_BLOCKS=8`, `MLP_HIDDEN=1664`, `TRAIN_BATCH_TOKENS=458752`, `BIGRAM_VOCAB_SIZE=1024`, `WARMUP_STEPS=200`, `SHUFFLE_DATA=1`, `XSA_LAST_N=2`

# Next 10 experiments
- Run 1: frozen dense snapshot on H100 for truth baseline
- Run 2: shared-depth root lane with `UNIQUE_BLOCKS=8` and `MLP_HIDDEN=1664`
- Run 3: eval-only `EVAL_SEQ_LEN=4096 EVAL_BATCH_SEQS=16` on the winner of Runs 1-2 if it stays in budget
- Run 4: adjust shared lane width upward only if Run 2 wins and leaves byte headroom
- Run 5: adjust shared lane width downward only if Run 2 wins on quality but misses the byte cap
- Run 6: three-seed sweep at `SEED=1337` on the winning dense or shared lane
- Run 7: three-seed sweep at `SEED=1338` on the winning dense or shared lane
- Run 8: three-seed sweep at `SEED=1339` on the winning dense or shared lane
- Run 9: more aggressive shared-depth schedule only after an `UNIQUE_BLOCKS=8` truth run
- Run 10: sparse-outlier or other YELLOW follow-up only after a winning GREEN shared or dense lane exists

# Decision log
- Freeze the dense `b1024-warmup200-xlast2` root lane into `snapshots/` before changing root again.
- Keep the dense snapshot runnable by path for later H100 comparison.
- Implement shared depth with a unique-block table plus application schedule, not repeated module aliases in one `ModuleList`.
- Keep per-application `attn_scale`, `mlp_scale`, `resid_mix`, and LN-depth behavior unique; keep `q_gain` shared inside shared attention blocks in v1.
- Use `UNIQUE_BLOCKS=8` only when `NUM_LAYERS=12`; reject other nontrivial sharing schedules in this pass.
- Spend the saved bytes on `MLP_HIDDEN=1664` first, not more BigramHash capacity.
- Keep `flash-attn` optional in code and Triton blocked.

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
