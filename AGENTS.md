# Mission
- Maximize final post-export `val_bpb` under the 16,000,000-byte artifact cap and the 10-minute train / 10-minute eval limits on 8xH100 SXM.

# Current challenge interpretation
- Accepted `records/track_10min_16mb/*` are the primary competitive source of truth.
- `README.md` remains the rules reference and legacy baseline context, but its leaderboard table is stale.
- `records/track_10min_16mb/2026-03-19_SlidingWindowEval` is the strongest current repo-contained public result at `post-export val_bpb=1.19250007`.
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
- `GREEN`: sliding eval, longer train/eval context, fp16 tied-embedding export, warmdown/LR tuning, EMA, width/MLP/KV reallocation, SDPA backend and compile sweeps.
- `YELLOW`: tokenizer changes, recurrent/shared blocks, mixed-format export codecs, eval-time adaptation, custom Triton kernels, bundled eval-time auxiliary state.
- `RED`: auto-launched H100 jobs, over-budget train/eval, network/data access during eval, oversized artifact, seed brute force.

# Roles and ownership
- Rule Auditor: `docs/compliance_matrix.md`
- Baseline Reverse Engineer: `docs/baseline_map.md`
- Local Harness Engineer: `docs/local_harness.md`, `configs/local/*`
- Hopper Systems Engineer: `docs/hopper_plan.md`, `configs/h100/*`
- Quantization / Export Engineer: `docs/export_strategy.md`
- Architecture / Optimization Researcher: `docs/model_ideas.md`
- Automated Search Integrator: `docs/autosearch_plan.md`, `scripts/autosearch.py`
- Triton Investigator: `docs/triton_plan.md`
- Experiment Manager: `experiments/registry.csv`, `docs/top_candidates.md`

# Experiment naming convention
- `YYYYMMDDTHHMMSSZ-<profile-slug>` for executed runs.
- Config slugs use `hardware-purpose-major_knob`.

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
- The current beat path is a strict composition of `LongContextSeq2048`, `FP16Embed_WD3600`, and `SlidingWindowEval`.
- Clean sliding eval is the highest-ROI eval lever.
- `tok_emb.weight` fp16 passthrough is the highest-ROI export lever.
- Seq2048 remains the strongest proven training-side change.
- EMA may improve post-export robustness cheaply.

# Current blockers
- Local GTX proxy configs need explicit fp32 to stay numerically stable on Pascal.
- H100 benchmark truth remains manual-only.

# Current best candidates
- Current public best: `SlidingWindowEval` with `post-export val_bpb=1.19250007`
- Canonical challenger: `seq2048 + fp16 tok_emb passthrough + sliding eval stride 64`
- Follow-up: canonical challenger plus EMA
- Secondary follow-up: post-challenger `SDPA_BACKEND x COMPILE_MODE` matrix
- Local evidence: `tok_emb.weight` fp16 passthrough sharply reduces the post-export gap on GTX proxy, but increases bytes materially.

# Next 10 experiments
- Manual H100 challenger: `seq2048 + fp16 tok_emb + sliding eval stride 64`.
- Manual H100 follow-up: challenger plus EMA.
- Manual H100 post-challenger systems matrix: `SDPA_BACKEND x COMPILE_MODE`.
- Local proxy1024 + sliding eval stride 128.
- Local proxy1024 + sliding eval stride 64.
- Local proxy1024 + warmdown/LR sweep.
- Local proxy1024 + `MLP_HIDDEN` reallocation.
- Local proxy1024 + KV-head reduction.
- Local export proxy + clip-percentile sweep.
- Local proxy1024 + reduced validation overhead.

# Decision log
- Accepted records are competitive truth; the top-level README leaderboard is legacy context only.
- The primary beat path is a strict composition of `LongContextSeq2048`, `FP16Embed_WD3600`, and `SlidingWindowEval`.
- Phase 1 excludes unused QAT/LoRA/loop code from `SlidingWindowEval`.
- Phase 1 excludes custom Triton by default.
- Root `train_gpt.py` remains self-contained and snapshot-friendly.
- Local GTX bootstrap is pinned to Python 3.11 plus a CUDA wheel, not raw `requirements.txt`.
- Pascal `fp16` was unstable in local validation; checked-in GTX configs now pin `COMPUTE_DTYPE=fp32` and `MUON_DTYPE=fp32`.

# Rules for modifying code
- Keep record-critical logic in `train_gpt.py`.
- Keep new features behind flags.
- Preserve baseline-compatible defaults where practical.

# Rules for adding dependencies
- No new mandatory runtime dependencies in phase 1.
- Standard library only for workflow scripts.
- Local bootstrap uses `requirements-local.txt`; do not treat it as counted record code.

# Rules for touching metric/eval code
- Any eval change must log mode, seq length, stride, and eval time.
- Live and roundtrip metrics must use the same eval policy.

# Rules for challenge-edge ideas
- Document YELLOW ideas before any implementation.
- Do not merge RED ideas.

# Handoff protocol between agents
- Update `experiments/registry.csv`.
- Update `docs/top_candidates.md`.
- Add or amend the relevant doc before code or config changes are treated as canonical.
- Spawned agents must use `gpt-5.4-mini` with normal-or-higher reasoning only; never use Fast mode.
- Use short-lived audit agents only, then close them.

# Triton usage policy
- Standard PyTorch SDPA/cuDNN/Inductor first.
- Custom Triton only after measured standard-backend loss on H100.
- Any custom Triton path must be optional and have a non-Triton fallback.

# Local-to-H100 transfer policy
- Use GTX 1080 Ti only for correctness, proxy ranking, and export-gap triage.
- Do not rank H100-only systems knobs from Pascal behavior.
- Promote only algorithmic and export changes from local proxy to H100-ready status.
