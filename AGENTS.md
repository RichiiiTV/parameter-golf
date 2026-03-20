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
- `GREEN`: sliding eval, fp16 tied-embedding export, 10 layers, Muon weight decay, overtone tied-embedding init, phase-transition `resid_mix` init, longer train/eval context, warmdown/LR tuning, EMA, width/MLP/KV reallocation, SDPA backend and compile sweeps.
- `YELLOW`: tokenizer changes, recurrent/shared blocks, mixed-format export codecs, eval-time adaptation, custom Triton kernels, bundled eval-time auxiliary state, LoRA-style eval-time updates.
- `RED`: auto-launched H100 jobs, over-budget train/eval, network/data access during eval, oversized artifact, seed brute force.

# Roles and ownership
- Rule Auditor: `docs/compliance_matrix.md`
- Baseline Reverse Engineer: `docs/baseline_map.md`
- Local Harness Engineer: `docs/local_harness.md`, `configs/local/*`
- Hopper Systems Engineer: `docs/hopper_plan.md`, `configs/h100/*`, `configs/a100/*`
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

# Required artifacts for every experiment
- `config.json`
- `env.json`
- `git.json`
- `command.txt`
- `stdout.log`
- `result.json`

# Current hypotheses
- Porting the top-1 stack into the root trainer is the highest-ROI path now.
- The clean root candidate is `10L + Muon WD + overtone init + phase-transition resid_mix + sliding eval + fp16 tok_emb export`.
- A lower-LR, longer-warmdown 10-layer fallback protects quantization robustness if the primary recipe is too aggressive in the root trainer.
- The old `seq2048 + fp16 tok_emb + sliding eval` path remains a useful 4xA100 control, not the primary target.
- The new 10-layer flags now execute cleanly on the local GTX smoke path with finite pre/post-export metrics.

# Current blockers
- Only human-run 4xA100 and H100 experiments can answer final score questions.
- Root docs, configs, and registry must stay aligned with the merged March 19 records.

# Current best candidates
- Primary: `10L + Muon WD + overtone init + phase-transition resid_mix + sliding eval + fp16 tok_emb export`.
- Fallback: `10L + Muon WD + low LR + long warmdown + sliding eval + fp16 tok_emb export`.
- Legacy control: `seq2048 + fp16 tok_emb export + sliding eval`.

# Next 10 experiments
- Manual 4xA100 control: legacy `seq2048 + fp16 tok_emb + sliding eval`.
- Manual 4xA100 primary: `10L + Muon WD + overtone + phase-transition + sliding eval`.
- Manual 4xA100 fallback: `10L + Muon WD + low LR + long warmdown + sliding eval`.
- Manual H100 primary: 10-layer top-1-style root candidate.
- Manual H100 fallback: quantization-safe 10-layer fallback.
- Manual H100 systems matrix on top of the best 10-layer candidate.
- Manual H100 EMA follow-up only after a winning 10-layer truth run.
- Seq4096 training follow-up only after the 10-layer stack is established.
- Export/codec follow-up only after the 10-layer stack is established.
- Local export-gap sweep only if the 10-layer path needs more byte headroom after A100 truth.

# Decision log
- March 20 rebase target to `2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` at `1.17475315`.
- Keep `train_gpt_mlx.py` untouched; CUDA work stays in `train_gpt.py`.
- Import only GREEN levers from the accepted top-1 into the root trainer.
- Use 4xA100 as a promotion gate before any H100 run.
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
- Live and roundtrip metrics must use the same eval policy.
- Do not change scoring semantics while rebasing to the new top-1; only reuse the existing clean sliding-eval path.

# Rules for challenge-edge ideas
- Document YELLOW ideas before any implementation.
- Do not merge RED ideas.
- Keep LoRA TTT, mixed int8/int6 export, and compile-specialized eval paths out of the first 10-layer root pass.

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
- Do not rank H100-only systems knobs from Pascal behavior.
- Promote only algorithmic and export changes from local proxy to A100 and H100-ready status.
