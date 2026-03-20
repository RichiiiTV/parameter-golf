# Mission
- Maximize final post-export `val_bpb` under the 16,000,000-byte artifact cap and the 10-minute train / 10-minute eval limits on 8xH100 SXM.

# Current challenge interpretation
- Source of truth is the official `README.md` plus accepted `records/`.
- Current public bar is `records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` at `mean_val_bpb=1.17475315`.
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
- `GREEN`: official top-1 reproduction, sliding eval, fp16 tied-embedding export, 10 layers, Muon weight decay, overtone tied-embedding init, phase-transition `resid_mix`, seq2048 and seq4096 throughput lanes.
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
- The cleanest winning path is to first reproduce the official top-1 stack in the root trainer.
- Throughput is the main lever after that reset, so H100 sequence-length scaling is the next GREEN branch.
- `tok_emb.weight` fp16 retention remains the strongest export-side lever already supported by public evidence.
- Pre-projection RMSNorm is the highest-priority bold mechanism to test after the GREEN mainline is stable.

# Current blockers
- No H100 truth run has been executed from the reset trainer yet.
- The active docs and registry must stay aligned with the H100-only workflow.
- Bold YELLOW lanes should not be promoted until the GREEN mainline is represented cleanly.

# Current best candidates
- `h100-root-top1-repro`
- `h100-root-top1-seq2048`
- `h100-root-top1-seq4096`
- `h100-preproj-rmsnorm` as the first bold mechanism lane after the clean reproduction

# Next 10 experiments
- Run 1: `h100-root-top1-repro`
- Run 2: `h100-root-top1-seq2048`
- Run 3: `h100-root-top1-seq4096`
- Run 4: winner of Runs 2 and 3 with CUDA Graph capture if it fits cleanly
- Run 5: winner plus pre-projection RMSNorm
- Run 6: winner plus recurrent/shared-width blocks
- Run 7: winner plus sparse high-precision outlier retention at export
- Run 8: winning GREEN lane with modest batch-token sweep
- Run 9: winning GREEN lane with eval-length stress test
- Run 10: final promoted H100 contender with full reproducibility package

# Decision log
- Reset the root trainer from the official upstream/public-record lineage instead of extending the local research core.
- Retire GTX and A100 from the active workflow. They remain legacy history only.
- Keep root `train_gpt.py` narrow: only upstream baseline plus public-record model/eval/export deltas.
- Keep bold paper-backed ideas out of the root baseline until the clean H100 mainline is represented.
- Keep Triton blocked until a winning GREEN lane exists and a standard-backend bottleneck is measured on H100.

# Rules for modifying code
- Keep record-critical logic in `train_gpt.py`.
- Prefer record-backed deltas in root and keep bold ideas explicit and isolated.
- Do not add root flags unless they are needed to express a public-record delta or a clearly justified H100 lane.

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
