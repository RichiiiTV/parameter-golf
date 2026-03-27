# Mission
- Maximize final post-export `val_bpb` under the `16,000,000`-byte artifact cap and the `10`-minute train / `10`-minute eval limits on `8xH100 SXM`.

# Frontier
- Accepted record source of truth: upstream `README.md` plus accepted `records/`.
- Current accepted top merged record: `#549` / `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` at `1.1194`.
- Current provisional valid open target: `#875` / `Pure Neural GDN 1.0226`.
- `#753` / eval-cache work remains archived pending rule clarification from issue `#677`.
- Active root path is now a compact `#875`-style Gated DeltaNet mainline with one architecture-first follow-up and one legal TTT follow-up.

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
- `GREEN`: exact pure-neural `#875`-style GDN base in root.
- `YELLOW`: exact GDN base plus one extra GDN block, and the separate legal score-first TTT follow-up.
- `RED`: eval caches, two-pass rescoring, post-cap reopening of `fineweb_train_*`, tokenizer changes without proof, oversized artifacts, seed brute force, auto-launched H100 jobs.

# Current hypotheses
- The best valid near-frontier move is a pure-neural Gated DeltaNet base with exact byte-accounted evaluation, not another cache lane.
- The highest-value few-run follow-up is deeper pure-neural GDN capacity before spending a run on TTT.
- Legal score-first TTT remains available, but it is now the secondary follow-up rather than the primary next run.
- The donor judge math from `#875` should not be reused; root must keep exact tokenizer-LUT byte accounting.

# Current blockers
- No H100 truth run has been executed from the pure-neural GDN base.
- No H100 truth run has been executed from the deeper pure-neural GDN follow-up.
- No H100 truth run has been executed from the GDN + legal TTT follow-up.

# Active candidates
- `configs/h100/root_pr875_gdn_repro.json`
- `configs/h100/root_pr875_gdn_deeper.json`
- `configs/h100/root_pr875_gdn_ttt.json`
- `snapshots/train_gpt_2026-03-25_pre753_pr549_softqat_root.py`
- `snapshots/train_gpt_2026-03-27_pre875_hnet_wip_root.py`

# Run ladder
- Run 1: exact pure-neural GDN base on H100
- Run 2: exact GDN base plus one extra GDN block on H100
- Run 3: exact GDN base plus legal score-first TTT on H100
- If only one follow-up run is affordable after the base, prefer `root_pr875_gdn_deeper` over `root_pr875_gdn_ttt`.

# Rules for code changes
- Keep record-critical logic in root `train_gpt.py`.
- Keep new root changes minimal and self-contained.
- Do not stack multiple unrelated algorithmic deltas in one pass.
- Archive abandoned root lanes in `snapshots/` before resetting root.

# Rules for eval and export
- Preserve challenge semantics.
- Log exact roundtrip metrics, sliding metrics, pre-TTT base metrics, final TTT metrics, TTT gain, artifact bytes, and eval time.
- Do not keep eval-cache or n-gram scoring in the active record path until OpenAI publishes a clarified rule path.
- Do not reopen `fineweb_train_*` after the 600s training phase.
- Keep byte accounting tied to exact tokenizer LUTs, not fixed divisors.

# Dependencies
- No new mandatory runtime dependencies.
- Standard library only for workflow scripts.
- `flash-attn` remains optional and cluster-side.

# Handoff
- Update `experiments/registry.csv`.
- Update `docs/top_candidates.md`.
- Update `docs/hopper_plan.md`.
- Preserve archived root lanes in `snapshots/` before resetting root.
