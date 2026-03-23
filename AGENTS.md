# Mission
- Maximize final post-export `val_bpb` under the `16,000,000`-byte artifact cap and the `10`-minute train / `10`-minute eval limits on `8xH100 SXM`.

# Frontier
- Accepted record source of truth: upstream `README.md` plus accepted `records/`.
- Operational donor baseline for this root pass: local `pr332:records/track_10min_16mb/2026-03-21_12L_GradQuant_PartialRoPE_EMA` at `1.1320`.
- Current active root path is a dense reset plus packed GPTQ export, with TTT-lite as an isolated yellow follow-up.

# Hard constraints
- Never auto-run H100 jobs.
- Never exceed `16,000,000` artifact bytes on a candidate.
- Never access network or training data during evaluation.
- Never brute-force seeds.
- Keep `train_gpt.py` and `train_gpt_mlx.py` under `1500` lines by `scripts/check_line_budget.py`.

# Workflow
- H100-only, manual-only, throughput-first.
- Root `train_gpt.py` is the canonical runner.
- Frozen snapshots are allowed as fallback baselines, but new active work happens in root.
- `flash-attn` is optional in code and recommended only in the H100 environment.

# GREEN / YELLOW / RED
- `GREEN`: dense snapshot fallback, packed GPTQ export on the dense baseline, funded dense GPTQ lane if exact roundtrip stays healthy.
- `YELLOW`: TTT-lite, longer eval context, tokenizer changes, Triton, sparse high-precision rescue, aggressive export tricks.
- `RED`: auto-launched H100 jobs, over-budget runs, network/data access during eval, oversized artifacts, seed brute force.

# Ownership
- Rule Auditor: `docs/compliance_matrix.md`
- Baseline Reverse Engineer: `docs/baseline_map.md`
- Hopper Systems Engineer: `docs/hopper_plan.md`, `configs/h100/*`, `scripts/prepare_h100_run.py`
- Quantization / Export Engineer: `docs/export_strategy.md`
- Architecture / Optimization Researcher: `docs/model_ideas.md`
- Experiment Manager: `experiments/registry.csv`, `docs/top_candidates.md`

# Current hypotheses
- The shared-depth branch saturated and should not remain the active root mainline.
- Packed low-bit GPTQ-style export is the next mechanism worth spending complexity on because it directly targets stored bytes.
- The first funded dense lane after GPTQ should be `NUM_LAYERS=14`, `MLP_HIDDEN=1792`, `BIGRAM_VOCAB_SIZE=4096`.
- TTT-lite should stay eval-only, doc-isolated, and off by default.

# Current blockers
- No H100 truth run has been executed from the dense snapshot fallback.
- No H100 truth run has been executed from the GPTQ-only dense root lane.
- No H100 truth run has been executed from the funded dense GPTQ lane.
- No H100 truth run has been executed from the funded dense GPTQ lane with TTT-lite.

# Active candidates
- `snapshots/train_gpt_2026-03-23_root_pr332_b458k_b1024_warmup200_xlast2.py`
- `configs/h100/root_snapshot_b1024_warmup200_xlast2.json`
- `configs/h100/root_gptq12_b1024_warmup200_xlast2.json`
- `configs/h100/root_gptq14_mlp1792_b4096_warmup200_xlast2.json`
- `configs/h100/root_gptq14_mlp1792_b4096_warmup200_xlast2_ttt.json`

# Run ladder
- Run 1: dense snapshot baseline on H100
- Run 2: GPTQ-only dense baseline on H100
- Run 3: funded dense GPTQ lane on H100
- Run 4: funded dense GPTQ lane plus TTT-lite on H100

# Rules for code changes
- Keep record-critical logic in root `train_gpt.py`.
- Keep new root changes minimal and self-contained.
- Do not stack multiple unrelated algorithmic deltas in one pass.
- Archive abandoned YELLOW paths in `snapshots/` rather than silently deleting them.

# Rules for eval and export
- Preserve challenge semantics.
- Log exact roundtrip metrics, sliding metrics, packed payload bytes, artifact bytes, and eval time.
- Keep TTT doc-isolated and reset adapters between documents.

# Dependencies
- No new mandatory runtime dependencies.
- Standard library only for workflow scripts.
- `flash-attn` remains optional and cluster-side.

# Handoff
- Update `experiments/registry.csv`.
- Update `docs/top_candidates.md`.
- Update `docs/hopper_plan.md`.
- If a lane is archived, preserve it in `snapshots/` before resetting root.
