# Mission
- Maximize final post-export `val_bpb` under the 16,000,000-byte artifact cap and the 10-minute train / 10-minute eval limits on 8xH100 SXM.

# Current challenge interpretation
- Source of truth is `README.md` plus accepted `records/`.
- Current public bar is `records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` at `mean_val_bpb=1.17475315`.
- Use 4xA100 only as a relative preflight surface; use 8xH100 only as the final human-gated truth surface.
- Evaluation can be aggressive if it is reproducible, self-contained, under 10 minutes on 8xH100, and does not access external data or network.

# Hard constraints
- Never auto-run H100 jobs.
- Never exceed 16,000,000 artifact bytes on a record candidate.
- Never access network or training data during evaluation.
- Never brute-force seeds or run spirit-violating search.
- Keep `train_gpt.py` and `train_gpt_mlx.py` under 1500 lines by `scripts/check_line_budget.py`.

# Local environment assumptions
- Windows PowerShell.
- Python 3.11 venv on Windows is the local default; the Python 3.8 shell is legacy only.
- NVIDIA GeForce GTX 1080 Ti with 11GB VRAM.
- Local bootstrap is `scripts/bootstrap_local_windows.ps1`.
- `torch` and dataset cache may be absent until bootstrap completes; use `scripts/check_env.py` first.

# H100 target assumptions
- Single-node 8xH100 SXM.
- Human-gated only.
- Standard PyTorch/cuDNN/Inductor paths are the default optimization surface.

# GREEN / YELLOW / RED idea board
- `GREEN`: sliding eval, fp16 tied-embedding export, 10 layers, Muon weight decay, overtone tied-embedding init, phase-transition `resid_mix` init, longer train/eval context, compile sweeps, batch-token sweeps.
- `YELLOW`: tokenizer changes, recurrent/shared blocks, mixed-format export codecs, eval-time adaptation, custom Triton kernels, bundled eval-time auxiliary state, LoRA-style eval-time updates.
- `RED`: auto-launched H100 jobs, over-budget train/eval, network/data access during eval, oversized artifact, seed brute force.

# Roles and ownership
- Rule Auditor: `docs/compliance_matrix.md`
- Baseline Reverse Engineer: `docs/baseline_map.md`
- Local Harness Engineer: `docs/local_harness.md`, `configs/local/*`
- Hopper Systems Engineer: `docs/hopper_plan.md`, `configs/h100/*`, `configs/a100/*`, `scripts/prepare_a100_run.py`
- Quantization / Export Engineer: `docs/export_strategy.md`
- Architecture / Optimization Researcher: `docs/model_ideas.md`
- Automated Search Integrator: `docs/autosearch_plan.md`, `scripts/autosearch.py`
- Triton Investigator: `docs/triton_plan.md`
- Experiment Manager: `experiments/registry.csv`, `docs/top_candidates.md`

# Experiment naming convention
- Executed runs use `YYYYMMDDTHHMMSSZ-<profile-slug>`.
- Planned configs use `<hardware>-<role>-<major_knob>` slugs.

# Standard metrics to report
- `pre_val_loss`
- `pre_quant_val_bpb`
- `post_val_loss`
- `post_quant_val_bpb`
- `pre_post_gap`
- `bytes_total`
- `eval_time_ms`
- `train_tokens_seen`
- `peak_mem_mib`
- `ms_per_step`

# Required artifacts for every experiment
- `config.json`
- `env.json`
- `git.json`
- `command.txt`
- `stdout.log`
- `result.json`

# Current hypotheses
- Throughput and A100 iteration speed are the priority now.
- Freeze the model-side comparison to two lanes only:
  - Lane A: `seq2048 + fp16 tok_emb export + sliding eval`
  - Lane B: `10L + Muon WD + overtone init + phase-transition resid_mix + sliding eval + fp16 tok_emb export`
- The 10-layer root port is still viable only if systems tuning recovers enough train tokens to justify its slower step time.
- The new 10-layer flags execute cleanly on the local GTX smoke path with finite pre/post-export metrics.

# Current blockers
- Only human-run 4xA100 and H100 experiments can answer final score questions.
- Full sliding eval on 4xA100 is too slow to use on every preflight run.
- Root docs, configs, and registry must stay aligned with the throughput-first preflight policy.

# Current best candidates
- Lane A control: `seq2048 + fp16 tok_emb export + sliding eval`.
- Lane B primary: `10L + Muon WD + overtone init + phase-transition resid_mix + sliding eval + fp16 tok_emb export`.
- Winning H100 candidate is intentionally undecided until the A100 throughput pass finishes.

# Next 10 experiments
- Manual 4xA100 Lane A proxy: `COMPILE_MODE=off`, `TRAIN_BATCH_TOKENS=524288`.
- Manual 4xA100 Lane A proxy: `COMPILE_MODE=fullgraph`, `TRAIN_BATCH_TOKENS=524288`.
- Manual 4xA100 Lane B proxy: `COMPILE_MODE=off`, `TRAIN_BATCH_TOKENS=524288`.
- Manual 4xA100 Lane B proxy: `COMPILE_MODE=fullgraph`, `TRAIN_BATCH_TOKENS=524288`.
- Manual 4xA100 Lane A higher-batch sweep on the better compile mode.
- Manual 4xA100 Lane B higher-batch sweep on the better compile mode.
- Manual 4xA100 Lane A full sliding rerun for the winning proxy point.
- Manual 4xA100 Lane B full sliding rerun for the winning proxy point.
- Manual H100 run only for the lane that wins full sliding on A100.
- H100 systems matrix only after a winning lane is identified.

# Decision log
- March 20 rebase target to `2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` at `1.17475315`.
- Keep `train_gpt_mlx.py` untouched; CUDA work stays in `train_gpt.py`.
- Import only GREEN levers from the accepted top-1 into the root trainer.
- Use 4xA100 as a promotion gate before any H100 run.
- A100 control vs primary results showed the 10-layer lane only barely beat the control while processing materially fewer tokens, so the repo now pivots to a throughput-first pass.
- A100 preflight now uses proxy-then-full sliding eval: fast proxy ranking first, then one full sliding rerun per lane finalist.
- Keep custom Triton out until standard H100 backends lose on a measured hotspot.
- Treat `records/track_10min_16mb/2026-03-19_WarmdownQuantization` as non-canonical until its README/submission mismatch is reviewed.
- Local GTX correctness smoke passed for `NUM_LAYERS=10`, `MUON_WEIGHT_DECAY=0.02`, `ADAMW_WEIGHT_DECAY=0.01`, `TIED_EMBED_INIT_MODE=overtone`, `RESID_MIX_INIT=phase_transition`, and `KEEP_FLOAT_EXTRA=tok_emb.weight`.

# Rules for modifying code
- Keep record-critical logic in `train_gpt.py`.
- Keep new features behind flags.
- Preserve baseline-compatible defaults where practical.
- Do not grow the trainer surface unless the new flag is needed to express a record-backed idea.

# Rules for adding dependencies
- No new mandatory runtime dependencies in phase 1.
- Standard library only for workflow scripts.
- Local bootstrap uses `requirements-local.txt`; do not treat it as counted record code.

# Rules for touching metric/eval code
- Any eval change must log mode, seq length, stride, and eval time.
- Any manual run log must include `script_path`, `trainer_sha256`, `val_scope`, `val_max_seqs`, `train_tokens_seen`, and `ms_per_step`.
- Live and roundtrip metrics must use the same eval policy.
- Do not change scoring semantics while rebasing to the new top-1; only reuse the existing clean sliding-eval path.

# Rules for challenge-edge ideas
- Document YELLOW ideas before any implementation.
- Do not merge RED ideas.
- Keep LoRA TTT, mixed int8/int6 export, EMA, seq4096, and compile-specialized eval paths out of the throughput-first pass.

# Handoff protocol between agents
- Update `experiments/registry.csv`.
- Update `docs/top_candidates.md`.
- Add or amend the relevant doc before code or config changes are treated as canonical.
- If agents are spawned, they must use `gpt-5.4-mini` with non-fast reasoning only, then be closed immediately after the audit returns.

# Triton usage policy
- Standard PyTorch SDPA/cuDNN/Inductor first.
- Custom Triton only after measured standard-backend loss on H100.
- Any custom Triton path must be optional and have a non-Triton fallback.

# Local-to-H100 transfer policy
- Use GTX 1080 Ti only for correctness, proxy ranking, and export-gap triage.
- Use 4xA100 for throughput-first relative comparison with proxy-then-full eval.
- Do not rank H100-only systems knobs from Pascal behavior.
- Promote only the lane that wins full sliding on A100 to H100-ready status.
