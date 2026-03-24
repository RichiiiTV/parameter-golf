# Mission
- Maximize final post-export `val_bpb` under the `16,000,000`-byte artifact cap and the `10`-minute train / `10`-minute eval limits on `8xH100 SXM`.

# Frontier
- Accepted record source of truth: upstream `README.md` plus accepted `records/`.
- Current accepted top merged record: `#549` / `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` at `1.1194`.
- Active root path is a hard reset to the `#549` donor plus one isolated improvement: late soft-round QAT from the `#589` family.

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
- `GREEN`: exact `#549` repro in root.
- `YELLOW`: late soft-round QAT on top of the exact `#549` root lane.
- `RED`: auto-launched H100 jobs, over-budget runs, network/data access during eval, oversized artifacts, seed brute force.

# Current hypotheses
- The previous GPTQ/TTT-lite root lane was the wrong local direction for the merged upstream frontier and is now archived.
- The strongest source-backed next delta on the accepted `#549` family is late soft-round QAT.
- TTT semantics must remain legal score-first and chunk-local.

# Current blockers
- No H100 truth run has been executed from the exact `#549` root repro.
- No H100 truth run has been executed from the `#549` + soft-round-QAT root lane.

# Active candidates
- `configs/h100/root_pr549_repro.json`
- `configs/h100/root_pr549_softqat.json`
- `snapshots/train_gpt_2026-03-24_pre549_dense_gptq_tttlite_root.py`

# Run ladder
- Run 1: exact `#549` root repro on H100
- Run 2: exact `#549` + `SOFT_QAT_ENABLED=1` on H100

# Rules for code changes
- Keep record-critical logic in root `train_gpt.py`.
- Keep new root changes minimal and self-contained.
- Do not stack multiple unrelated algorithmic deltas in one pass.
- Archive abandoned YELLOW paths in `snapshots/` rather than silently deleting them.

# Rules for eval and export
- Preserve challenge semantics.
- Log exact roundtrip metrics, sliding metrics, TTT metrics, artifact bytes, and eval time.
- Keep TTT score-first, chunk-local, and never train on the last scored chunk.

# Dependencies
- No new mandatory runtime dependencies.
- Standard library only for workflow scripts.
- `flash-attn` remains optional and cluster-side.

# Handoff
- Update `experiments/registry.csv`.
- Update `docs/top_candidates.md`.
- Update `docs/hopper_plan.md`.
- Preserve archived root lanes in `snapshots/` before resetting root.
