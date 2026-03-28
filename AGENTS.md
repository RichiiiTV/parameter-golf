# Mission
- Maximize final post-export `val_bpb` under the `16,000,000`-byte artifact cap and the `10`-minute train / `10`-minute eval limits on `8xH100 SXM`.

# Frontier
- Accepted record source of truth: upstream `README.md` plus accepted `records/`.
- Current accepted top merged record: `#549` / `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` at `1.1194`.
- `#753` / eval-cache work remains archived pending rule clarification from issue `#677`.
- `#875` / pure-neural GDN work is archived as a failed low-probability branch after the 1xH100 proxy landed far off-family.
- Completed local reference baseline: `#589`-style late soft-round QAT at `1.11956668` in `logs/root-pr549-softqat.txt`.
- Active root path is now the valid funded dense GPTQ base derived from `snapshots/train_gpt_2026-03-24_pre549_dense_gptq_tttlite_root.py`, with post-cap calibration removed and no TTT carryover.

# Hard constraints
- Never auto-run H100 jobs.
- Never exceed `16,000,000` artifact bytes on a candidate.
- Never access network or training data during evaluation.
- Never brute-force seeds.
- Never spend H100 compute on already-done donor PR replays once a completed local truth run exists.
- Keep `train_gpt.py` and `train_gpt_mlx.py` under `1500` lines by `scripts/check_line_budget.py`.

# Workflow
- H100-only, manual-only.
- Root `train_gpt.py` is the canonical runner.
- Frozen snapshots are allowed as fallbacks, but active work happens in root.
- `flash-attn` is optional in code and recommended only in the H100 environment.

# GREEN / YELLOW / RED
- `GREEN`: valid funded dense GPTQ base in root.
- `YELLOW`: completed `#549` / `#589` truth runs are reference-only; no follow-up is active until the dense GPTQ base earns one.
- `RED`: eval caches, two-pass rescoring, post-cap reopening of `fineweb_train_*`, tokenizer changes without proof, donor replay runs after a completed truth run, oversized artifacts, seed brute force, auto-launched H100 jobs.

# Current hypotheses
- The current `#549` family is saturated on bytes locally: the completed soft-QAT run used `15,845,667` bytes and did not beat `#549`.
- The highest expected-value next lane is a larger dense model funded by valid packed GPTQ5 export, not another replay of the accepted donor family.
- GPTQ statistics must come only from already-seen in-training batches collected before the `600s` cap.
- No TTT or other eval delta should be stacked until the funded dense GPTQ base beats the completed `#589`-style reference baseline on its own.

# Current blockers
- No H100 truth run has been executed from the valid funded dense GPTQ base in root.
- The first H100 truth run still needs to confirm whether the valid in-training GPTQ path preserves enough byte headroom and quality to beat the completed local reference baseline.

# Active candidates
- `configs/h100/root_valid_gptq14_mlp1792_b4096_warmup200_xlast2.json`
- `logs/root-pr549-softqat.txt`
- `snapshots/train_gpt_2026-03-24_pre549_dense_gptq_tttlite_root.py`
- `snapshots/train_gpt_2026-03-25_pre753_pr549_softqat_root.py`
- `snapshots/train_gpt_2026-03-27_prepivot_pr875_gdn_root.py`

# Run ladder
- Run 1: valid funded dense GPTQ base on H100
- If the base misses the size or timing gate, stop and adjust the root; do not spend another run on `#549`/`#589` replay lanes.
- Before any H100 run, verify the pod has the new root: `train_gpt.py` should contain `GPTQStatsCollector` and `gptq:start in_training`, and should not contain `collect_gptq_stats` or `ttt_`.

# Rules for code changes
- Keep record-critical logic in root `train_gpt.py`.
- Keep new root changes minimal and self-contained.
- Do not stack multiple unrelated algorithmic deltas in one pass.
- Archive abandoned root lanes in `snapshots/` before resetting root.

# Rules for eval and export
- Preserve challenge semantics.
- Log exact roundtrip metrics, sliding metrics, artifact bytes, and eval time.
- Do not keep eval-cache or n-gram scoring in the active record path until OpenAI publishes a clarified rule path.
- Do not reopen `fineweb_train_*` after the `600s` training phase.
- Keep byte accounting tied to exact tokenizer LUTs, not fixed divisors.
- Keep packed GPTQ calibration entirely inside normal training on already-seen batches.

# Dependencies
- No new mandatory runtime dependencies.
- Standard library only for workflow scripts.
- `flash-attn` remains optional and cluster-side.

# Handoff
- Update `experiments/registry.csv`.
- Update `docs/top_candidates.md`.
- Update `docs/hopper_plan.md`.
- Preserve archived root lanes in `snapshots/` before resetting root.
