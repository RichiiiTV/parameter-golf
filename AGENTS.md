# Mission
- Maximize final post-export `val_bpb` under the `16,000,000`-byte artifact cap and the `10`-minute train / `10`-minute eval limits on `8xH100 SXM`.

# Frontier
- Accepted record source of truth: upstream `README.md` plus accepted `records/`.
- Current accepted top merged record: `#549` / `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` at `1.1194`.
- Current operational live frontier for this pass: `#753` / `2026-03-25_PodracingII_backoff7gram_8xH100` at `0.9625` mean over 3 seeds.
- Active root path is a compacted `#753` donor port plus one requested-point hybrid: a compact diagonal selective-SSM in fixed early/mid blocks.

# Hard constraints
- Never auto-run H100 jobs.
- Never exceed `16,000,000` artifact bytes on a candidate.
- Never access network or training data during evaluation.
- Never brute-force seeds.
- Keep `train_gpt.py` and `train_gpt_mlx.py` under `1500` lines by `scripts/check_line_budget.py`.

# Workflow
- H100-only, manual-only.
- Root `train_gpt.py` is the canonical runner.
- Frozen snapshots are allowed as fallbacks, but active work happens in root.
- `flash-attn` is optional in code and recommended only in the H100 environment.

# GREEN / YELLOW / RED
- `GREEN`: exact `#753` repro in root.
- `YELLOW`: exact `#753` plus the fixed-block state-space hybrid in root.
- `RED`: auto-launched H100 jobs, over-budget runs, network/data access during eval, oversized artifacts, seed brute force.

# Current hypotheses
- The decisive frontier gain is still legal score-first hashed n-gram backoff with entropy-adaptive interpolation.
- The requested-point contribution in this pass should come from the base model, not another eval-only delta.
- The safest high-upside requested-point branch is a fixed early/mid state-space hybrid that preserves late attention/XSA layers and the current legal n-gram evaluator.

# Current blockers
- No H100 truth run has been executed from the exact `#753` root repro.
- No H100 truth run has been executed from the state-space hybrid root lane.

# Active candidates
- `configs/h100/root_pr753_repro.json`
- `configs/h100/root_pr753_hybrid_ssm.json`
- `snapshots/train_gpt_2026-03-25_pre753_pr549_softqat_root.py`
- `snapshots/train_gpt_2026-03-25_pre753_state_space_hybrid_root.py`

# Run ladder
- Run 1: exact `#753` root repro on H100
- Run 2: exact `#753` plus the fixed-block state-space hybrid on H100

# Rules for code changes
- Keep record-critical logic in root `train_gpt.py`.
- Keep new root changes minimal and self-contained.
- Do not stack multiple unrelated algorithmic deltas in one pass.
- Archive abandoned root lanes in `snapshots/` before resetting root.

# Rules for eval and export
- Preserve challenge semantics.
- Log exact roundtrip metrics, sliding metrics, n-gram exact metrics, artifact bytes, and eval time.
- Keep n-gram scoring score-first and strictly backward-looking.
- Keep GPTQ calibration inside the training phase only.
- Keep the last four blocks attention-backed so XSA and the n-gram-facing late-stack behavior stay aligned.

# Dependencies
- No new mandatory runtime dependencies.
- Standard library only for workflow scripts.
- `flash-attn` remains optional and cluster-side.

# Handoff
- Update `experiments/registry.csv`.
- Update `docs/top_candidates.md`.
- Update `docs/hopper_plan.md`.
- Preserve archived root lanes in `snapshots/` before resetting root.
