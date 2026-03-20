# Mission
- Maximize final post-export `val_bpb` under the 16,000,000-byte artifact cap and the 10-minute train / 10-minute eval limits on 8xH100 SXM.

# Current challenge interpretation
- Source of truth is the official `README.md` plus accepted `records/`.
- Current public bar is `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50` at `mean_val_bpb=1.14276`.
- Active work is H100-only, manual-only, and throughput-first.
- Evaluation can be aggressive if it is reproducible, self-contained, under 10 minutes on 8xH100, and does not access external data or network.

# Hard constraints
- Never auto-run H100 jobs.
- Never exceed 16,000,000 artifact bytes on a record candidate.
- Never access network or training data during evaluation.
- Never brute-force seeds or run spirit-violating search.
- Keep `train_gpt.py` and `train_gpt_mlx.py` under 1500 lines by `scripts/check_line_budget.py`.

# Local environment assumptions
- Windows PowerShell is the local editing environment.
- Local correctness runs are legacy only and are not part of the active workflow.
- Local machines are for code review, dry checks, and handoff generation, not ranking.

# H100 target assumptions
- Single-node 8xH100 SXM.
- Human-gated only.
- Standard PyTorch/cuDNN/Inductor paths are the default optimization surface.
- Root `train_gpt.py` should look like the official upstream trainer plus narrow record-backed deltas.

# GREEN / YELLOW / RED idea board
- `GREEN`: mixed int5/int6 export, BigramHash, SmearGate, SWA, sliding eval, seq2048 throughput tuning, 10 layers, Muon/AdamW weight decay, zstd-22 fallback compression.
- `YELLOW`: pre-projection RMSNorm, recurrent/shared-width transformers, sparse high-precision outlier retention, tokenizer changes, eval-time adaptation, custom Triton kernels.
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
- The active SOTA path should live in a standalone record folder, not in root `train_gpt.py`.
- The current March 20 top record is the right base for our fork.
- Higher `TRAIN_BATCH_TOKENS` on the parent March 20 stack is the next throughput lever to sweep.
- Pre-projection RMSNorm is currently deferred after a negative same-folder 4xA100 ablation.

# Current blockers
- No H100 truth run has been executed from the new record fork yet.
- Bold YELLOW lanes should stay isolated inside record-style contenders, not root.
- The current record fork still needs H100 truth for the parent-stack throughput lane.

# Current best candidates
- `2026-03-20_10L_Int5MLP_MuonWD04_SWA50`
- `2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA`
- `2026-03-20_PreProjRMSNorm_Int5MLP_Bigram10240_SWA` with `USE_PREPROJ_RMSNORM=0`

# Next 10 experiments
- Run 1: `USE_PREPROJ_RMSNORM=0` in `2026-03-20_PreProjRMSNorm_Int5MLP_Bigram10240_SWA`
- Run 2: `USE_PREPROJ_RMSNORM=0` at `TRAIN_BATCH_TOKENS=917504`
- Run 3: `USE_PREPROJ_RMSNORM=0` at `TRAIN_BATCH_TOKENS=1048576`
- Run 4: three-seed sweep at `SEED=42` on the winning `USE_PREPROJ_RMSNORM=0` batch setting
- Run 5: three-seed sweep at `SEED=1337` on the winning `USE_PREPROJ_RMSNORM=0` batch setting
- Run 6: three-seed sweep at `SEED=2024` on the winning `USE_PREPROJ_RMSNORM=0` batch setting
- Run 7: exact parent March 20 folder reproduction on H100 if the fork baseline diverges materially
- Run 8: pre-proj RMSNorm revisit only if a later throughput-stable branch justifies it
- Run 9: recurrent/shared-width follow-up only after a winning GREEN throughput lane exists
- Run 10: sparse outlier retention follow-up only after a winning GREEN throughput lane exists

# Decision log
- Keep root `train_gpt.py` unchanged in the March 20 SOTA pass.
- Build new SOTA attempts as standalone record folders under `records/track_10min_16mb`.
- Treat the March 20 records as the operational frontier even if the root README is stale.
- Same-folder 4xA100 ablation for pre-projection RMSNorm was negative: `USE_PREPROJ_RMSNORM=0 -> 1.28120795`, `USE_PREPROJ_RMSNORM=1 -> 1.30713253`.
- The active branch in the record fork is now the parent stack plus throughput sweeps with `USE_PREPROJ_RMSNORM=0`.
- Keep Triton blocked until a winning post-March-20 GREEN lane exists and a standard-backend bottleneck is measured on H100.

# Rules for modifying code
- Keep record-critical logic in `train_gpt.py`.
- Keep root changes minimal during SOTA work.
- Prefer record-folder implementations for aggressive ideas.
- Do not mutate root `train_gpt.py` in the March 20 record-fork pass.

# Rules for adding dependencies
- No new mandatory runtime dependencies in this reset pass.
- Standard library only for workflow scripts.
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
- Do not rank active candidates from GTX or A100 behavior in this reset phase.
