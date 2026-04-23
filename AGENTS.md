# Mission
- Maximize final post-export `val_bpb` under the `16,000,000`-byte artifact cap and the `10`-minute train / `10`-minute eval limits on `8xH100 SXM`.

# Frontier
- Accepted record source of truth: upstream `README.md` plus accepted `records/`.
- Current accepted top merged record: `#549` / `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` at `1.1194`.
- Latest upstream open leaders checked on April 23, 2026 are `#1791` at `1.0339`, `#1787` at `1.06378`, `#1790` at `1.06991`, `#1771` at `1.06513` with legality pending, and `#1767` at `1.07209`.
- Chosen active root target: `#1791` at `1.0339`, because it is the strongest visible open SP8192 lane and avoids the legality-pending TTT / CaseOps path.
- Active dataset route is a custom HF export, assumed by default to `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf` with `MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=datasets`.
- Preserved local reference baseline: `logs/root-pr549-softqat.txt` at `1.11956668`.
- Active root path is now a thin `train_gpt.py` wrapper over `frontier_gdn/` with the `K_KVShare_Wider` GatedDeltaNet / FLA stack: 10 GDN layers, `model_dim=544`, `8` heads, KV-share stride `2`, `BigramHash(3072,112)`, trigram embeddings, `EMA=0.997`, SWA every `50`, late QAT, and int6 + zstd-22 export.

# Hard constraints
- Never auto-run H100 jobs.
- Never exceed `16,000,000` artifact bytes on a candidate.
- Never access network or training data during evaluation.
- Never brute-force seeds.
- Keep `train_gpt.py` and `train_gpt_mlx.py` under `1500` lines by `scripts/check_line_budget.py`.

# Workflow
- H100-only, manual-only.
- Root `train_gpt.py` is the canonical entrypoint; the active frontier implementation now lives in `frontier_gdn/`.
- Frozen snapshots are preserved history, but active work happens in root plus `frontier_gdn/`.
- The active H100 lane requires the pinned FLA runtime from `requirements-h100-fla.txt`.

# GREEN / YELLOW / RED
- `GREEN`: `configs/h100/root_sp8192_pr1791_fla_8xh100.json`.
- `YELLOW`: `configs/h100/root_sp8192_pr1791_fla_proxy_1xh100.json`, `configs/local/root_sp8192_pr1791_fla_sanity.json`, and preserved historical logs / snapshots.
- `RED`: eval caches, two-pass rescoring, tokenizer edits without proof, legality-pending `#1771` as the primary lane, public-SP1024 instructions presented as frontier guidance, oversized artifacts, seed brute force, or auto-launched H100 jobs.

# Current hypotheses
- The highest-upside repo path is the `#1791` K_KVShare_Wider FLA lane, not further work on the archived transformer/TTT SP1024 root.
- `#1787` and `#1790` remain backup lanes if the active FLA port proves unstable on the required runtime stack.
- `#1771` stays out of the active lane until legality is resolved.

# Current blockers
- The workspace does not currently have the SP8192 dataset/tokenizer cache downloaded locally.
- No H100 truth run has been executed from the new `#1791`-class FLA root yet.

# Active candidates
- `configs/h100/root_sp8192_pr1791_fla_8xh100.json`
- `configs/h100/root_sp8192_pr1791_fla_proxy_1xh100.json`
- `configs/local/root_sp8192_pr1791_fla_sanity.json`

# Historical references
- `logs/root-pr549-softqat.txt`
- `snapshots/train_gpt_2026-04-22_pre_shd_pivot_root.py`
- `snapshots/train_gpt_2026-04-22_pre_shd_only_prune_root.py`
- `snapshots/train_gpt_2026-04-22_pre_sp8192_pr1667_pivot_root.py`
- `snapshots/train_gpt_2026-04-23_pre_public_sp1024_reset_root.py`
- `snapshots/train_gpt_2026-04-23_pre_pr1791_fla_pivot_root.py`

# Run ladder
- Run 1: `configs/h100/root_sp8192_pr1791_fla_8xh100.json`
- Optional proxy only: `configs/h100/root_sp8192_pr1791_fla_proxy_1xh100.json`
- Before any H100 run, install the pinned runtime: `pip install -r requirements-h100-fla.txt`
- Before any H100 run, verify readiness with `python scripts/check_run_ready.py configs/h100/root_sp8192_pr1791_fla_proxy_1xh100.json`.
- Pod sanity check: `rg -n "ARCH_MODE|VOCAB_SIZE|EVAL_COMPILE_ENABLED|K_KVShare_Wider|GatedDeltaNet|token_byte_counts" train_gpt.py frontier_gdn/train_gdn_7k.py frontier_gdn/configs.py frontier_gdn/byte_scoring.py`
- Download the dataset first: `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=datasets python data/cached_challenge_fineweb.py --variant sp8192`

# Rules for code changes
- Keep the canonical entrypoint in root `train_gpt.py`.
- Keep new frontier changes minimal and self-contained inside `frontier_gdn/` plus thin workflow glue.
- Do not stack multiple unrelated algorithmic deltas in one pass.
- Archive abandoned root lanes in `snapshots/` before resetting root.

# Rules for eval and export
- Preserve challenge semantics.
- Log exact roundtrip metrics, sliding metrics, artifact bytes, and eval time.
- Keep byte accounting tied to exact tokenizer LUTs, not fixed divisors.
- Keep the active export path at int6 + zstd-22 with no eval-time network access.
- Keep `FLA_USE_NAIVE=1` as a debug-only local fallback, never the scored H100 path.

# Dependencies
- `requirements-h100-fla.txt` defines the mandatory H100 runtime for the active frontier lane.
- Standard library only for workflow scripts.
- `train_gpt_mlx.py` remains a non-frontier local helper.

# Handoff
- Update `experiments/registry.csv`.
- Update `docs/top_candidates.md`.
- Update `docs/hopper_plan.md`.
- Preserve archived root lanes in `snapshots/` before resetting root.
