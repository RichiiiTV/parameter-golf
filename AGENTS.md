# Mission
- Maximize final post-export `val_bpb` under the `16,000,000`-byte artifact cap and the `10`-minute train / `10`-minute eval limits on `8xH100 SXM`.

# Frontier
- Accepted record source of truth: upstream `README.md` plus accepted `records/`.
- Current accepted top merged record: `#549` / `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` at `1.1194`.
- `#753` / eval-cache work remains archived pending rule clarification from issue `#677`.
- `#875` / pure-neural GDN work is archived as a failed low-probability branch after the 1xH100 proxy landed far off-family.
- Active root path is now the archived valid `#549` family with one follow-up: `#589`-style late soft-round QAT.

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
- `YELLOW`: exact `#549` plus late soft-round QAT in root.
- `RED`: eval caches, two-pass rescoring, post-cap reopening of `fineweb_train_*`, tokenizer changes without proof, oversized artifacts, seed brute force, auto-launched H100 jobs.

# Current hypotheses
- The best near-term valid record chance is still the accepted `#549` family.
- The best single follow-up is late soft-round QAT from the `#589` family, because it was rejected for missing the `0.005`-nat bar rather than for invalidity.
- No other architecture or quantization delta should be stacked until the exact `#549` repro lands in-family on H100.

# Current blockers
- No H100 truth run has been executed from the exact `#549` root repro.
- No H100 truth run has been executed from the `#549` + soft-round QAT follow-up.

# Active candidates
- `configs/h100/root_pr549_repro.json`
- `configs/h100/root_pr549_softqat.json`
- `snapshots/train_gpt_2026-03-25_pre753_pr549_softqat_root.py`
- `snapshots/train_gpt_2026-03-27_prepivot_pr875_gdn_root.py`

# Run ladder
- Run 1: exact `#549` repro on H100
- Run 2: exact `#549` plus late soft-round QAT on H100
- If the repro does not land near the accepted `#549` band, stop and debug reproduction fidelity before trying another idea.

# Rules for code changes
- Keep record-critical logic in root `train_gpt.py`.
- Keep new root changes minimal and self-contained.
- Do not stack multiple unrelated algorithmic deltas in one pass.
- Archive abandoned root lanes in `snapshots/` before resetting root.

# Rules for eval and export
- Preserve challenge semantics.
- Log exact roundtrip metrics, sliding metrics, pre-TTT base metrics, final TTT metrics, TTT gain, artifact bytes, and eval time.
- Do not keep eval-cache or n-gram scoring in the active record path until OpenAI publishes a clarified rule path.
- Do not reopen `fineweb_train_*` after the `600s` training phase.
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
