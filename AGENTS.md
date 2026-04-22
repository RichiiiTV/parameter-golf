# Mission
- Maximize final post-export `val_bpb` under the `16,000,000`-byte artifact cap and the `10`-minute train / `10`-minute eval limits on `8xH100 SXM`.

# Frontier
- Accepted record source of truth: upstream `README.md` plus accepted `records/`.
- Current accepted top merged record: `#549` / `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` at `1.1194`.
- Latest upstream open leaders checked on April 22, 2026 are `#1767` at `1.07209`, `#1765` at `1.07266`, `#1775` at `1.07285`, `#1776` at `1.08083`, and `#1771` at `1.06513` with legality pending.
- Upstream SHD-only reference is `#1774` at `1.09813`; that portable delta landed in root, but the current repo remains materially behind the active SP8192 + TTT frontier.
- Completed local reference baseline: `#589`-style late soft-round QAT at `1.11956668` in `logs/root-pr549-softqat.txt`.
- Active root path is a clean SP1024 baseline: a `#1060`-derived 11-layer `#549`-family scaffold with coprime multi-shard loading, reserved-time full-Hessian GPTQ6, and SHD-tied Q/K tails with `SHARED_HEAD_DIM=16`.

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
- Frozen snapshots are preserved history, but active work happens in root.
- `flash-attn` is optional in code and recommended only in the H100 environment.

# GREEN / YELLOW / RED
- `GREEN`: none. This repo does not currently carry a frontier-credible global SOTA lane.
- `YELLOW`: the current SHD/SP1024 root is a clean runnable baseline only; completed `#549` / `#589` truth runs and the preserved April 22 snapshots are reference-only.
- `RED`: eval caches, two-pass rescoring, tokenizer changes without proof, donor replay runs after a completed local truth run, oversized artifacts, seed brute force, auto-launched H100 jobs, or any calibration that starts after the reserved GPTQ wallclock boundary.

# Current hypotheses
- The completed `#549` family is saturated on bytes locally: the completed soft-QAT run used `15,845,667` bytes and did not beat `#549`.
- The current SHD/SP1024 root is useful as a clean baseline and regression target, not as a plausible global SOTA path.
- Any real frontier chase now likely requires a separate SP8192 + legal-TTT pivot that is not present in this repo.
- `#1047`, `#1056`, `#753`, and `#875` are not active targets for this cleaned baseline repo.
- Reserved-time train-data calibration is acceptable only if it is explicitly logged and the total train-plus-calibration wallclock remains inside `600s`.
- The active baseline keeps `BigramHash(3072,112)` funded by selective `+/-1` export pruning and adds only SHD-tied Q/K tails with `SHARED_HEAD_DIM=16`; no other eval mechanism is active.

# Current blockers
- No H100 truth run has been executed from the cleaned SHD/SP1024 root.
- The repo currently has no frontier-credible global SOTA lane; a separate pivot would be needed before spending H100 with record intent.

# Active candidates
- `configs/h100/root_pr1060_shd_b3072_prune.json`
- `configs/h100/root_pr1060_shd_proxy_1xh100.json`
- `configs/local/root_pr1060_shd_sanity.json`

# Historical references
- `logs/root-pr549-softqat.txt`
- `snapshots/train_gpt_2026-04-22_pre_shd_pivot_root.py`
- `snapshots/train_gpt_2026-04-22_pre_shd_only_prune_root.py`

# Run ladder
- Run 1: clean SHD/SP1024 baseline characterization on H100 via `configs/h100/root_pr1060_shd_b3072_prune.json`
- Optional directional proxy only: `configs/h100/root_pr1060_shd_proxy_1xh100.json`
- Do not interpret a successful run as a global SOTA claim; use it only to measure the current root against the preserved local reference baseline.
- Before any H100 run, verify the pod has the new root: `train_gpt.py` should contain `SHARED_HEAD_DIM`, `GPTQ_RESERVE_MS`, and `gptq:start reserved_train_data`, and should not contain any removed fallback knobs.

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
