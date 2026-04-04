# Mission
- Maximize final post-export `val_bpb` under the `16,000,000`-byte artifact cap and the `10`-minute train / `10`-minute eval limits on `8xH100 SXM`.

# Frontier
- Accepted record source of truth: upstream `README.md` plus accepted `records/`.
- Current accepted top merged record: `#549` / `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` at `1.1194`.
- Latest plausible valid open leader to port: `#1060` at `1.1122` as of March 29, 2026.
- `#753` / eval-cache work remains archived pending rule clarification from issue `#677`.
- `#875` / pure-neural GDN work is archived as a failed low-probability branch after the 1xH100 proxy landed far off-family.
- Completed local reference baseline: `#589`-style late soft-round QAT at `1.11956668` in `logs/root-pr549-softqat.txt`.
- Active root path is now a `#1060`-derived 11-layer `#549`-family scaffold with coprime multi-shard loading, reserved-time full-Hessian GPTQ6, and a Gemma-style hybrid local/global attention schedule.

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
- `GREEN`: `#1060`-derived root with reserved-time full-Hessian GPTQ6, prune-funded `BigramHash(3072,112)`, and Gemma-style hybrid attention `L,L,G,L,L,G,L,L,G,L,G`.
- `YELLOW`: completed `#549` / `#589` truth runs are reference-only; exact all-global `#1060` donor parity is local-sanity-only, not an H100 target.
- `RED`: eval caches, two-pass rescoring, tokenizer changes without proof, donor replay runs after a completed truth run, oversized artifacts, seed brute force, auto-launched H100 jobs, or any calibration that starts after the reserved GPTQ wallclock boundary.

# Current hypotheses
- The completed `#549` family is saturated on bytes locally: the completed soft-QAT run used `15,845,667` bytes and did not beat `#549`.
- The highest expected-value next lane is a `#1060`-derived 11-layer scaffold with one byte-funded delta and a Gemma-style hybrid local/global attention schedule, not another replay of the accepted donor family.
- `#1047`, `#1056`, and `#875` are not active valid targets for this repo.
- Reserved-time train-data calibration is acceptable only if it is explicitly logged and the total train-plus-calibration wallclock remains inside `600s`.
- The promoted lane keeps `BigramHash(3072,112)` funded by selective `±1` export pruning and adds only the Gemma-style hybrid schedule `L,L,G,L,L,G,L,L,G,L,G`; no other eval mechanism is active.

# Current blockers
- No H100 truth run has been executed from the `#1060`-derived root.
- The first H100 truth run still needs to confirm whether the reserved-time full-Hessian GPTQ path preserves enough quality and byte headroom to beat the completed local reference baseline.

# Active candidates
- `configs/h100/root_pr1060_gemma_hybrid_b3072_prune.json`
- `configs/h100/root_pr1060_gemma_hybrid_proxy_1xh100.json`
- `configs/h100/root_pr1060_b3072_prune.json`
- `configs/local/root_pr1060_base_sanity.json`
- `configs/local/root_pr1060_gemma_hybrid_sanity.json`
- `logs/root-pr549-softqat.txt`
- `snapshots/train_gpt_2026-03-29_pre1060_valid_dense_gptq_root.py`
- `snapshots/train_gpt_2026-03-25_pre753_pr549_softqat_root.py`
- `snapshots/train_gpt_2026-03-27_prepivot_pr875_gdn_root.py`

# Run ladder
- Run 1: Gemma-style hybrid `#1060`-derived promoted lane on H100 via `configs/h100/root_pr1060_gemma_hybrid_b3072_prune.json`
- Optional directional proxy only: `configs/h100/root_pr1060_gemma_hybrid_proxy_1xh100.json`
- If the base misses the size or timing gate, stop and adjust the root; do not spend another run on `#549`/`#589` or exact `#1060` replay lanes.
- Before any H100 run, verify the pod has the new root: `train_gpt.py` should contain `ATTN_PATTERN`, `TRAIN_LOADER_MODE`, `GPTQ_RESERVE_MS`, and `gptq:start reserved_train_data`, and should not contain `ttt_`.

# Rules for code changes
- Keep record-critical logic in root `train_gpt.py`.
- Keep new root changes minimal and self-contained.
- Do not stack multiple unrelated algorithmic deltas in one pass.
- Archive abandoned root lanes in `snapshots/` before resetting root.

# Rules for eval and export
- Preserve challenge semantics.
- Log exact roundtrip metrics, sliding metrics, artifact bytes, and eval time.
- Do not keep eval-cache or n-gram scoring in the active record path until OpenAI publishes a clarified rule path.
- Do not access `fineweb_train_*` after the `600s` global budget; reserved-time calibration over train shards is allowed only if it starts before the logged reserve boundary and finishes inside the same total budget.
- Keep byte accounting tied to exact tokenizer LUTs, not fixed divisors.
- Keep packed GPTQ calibration explicit, reserved-time, and full-Hessian; do not fall back to the retired in-training collector path.

# Dependencies
- No new mandatory runtime dependencies.
- Standard library only for workflow scripts.
- `flash-attn` remains optional and cluster-side.

# Handoff
- Update `experiments/registry.csv`.
- Update `docs/top_candidates.md`.
- Update `docs/hopper_plan.md`.
- Preserve archived root lanes in `snapshots/` before resetting root.
