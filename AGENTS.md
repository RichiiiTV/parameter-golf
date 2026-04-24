# Mission
- Maximize final post-export `val_bpb` under the `16,000,000`-byte artifact cap and the `10`-minute train / `10`-minute eval limits on `8xH100 SXM`.

# Frontier
- Accepted record source of truth: upstream `README.md` plus accepted `records/`.
- Current accepted top merged record: `#1493` / `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT` at `1.0810`.
- Preserved local reference baseline: `logs/root-pr549-softqat.txt` at `1.11956668`.
- Active root path is now a runnable, cap-enforced copy of the accepted `#1493` record script in root `train_gpt.py`; the upstream reference is preserved under `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py`.
- The `#1791` `K_KVShare_Wider` GatedDeltaNet / FLA lane remains available in `frontier_gdn/`, but is demoted to investigation-only. Upstream review on April 24, 2026 indicates its reported `1.0339` likely came from the older non-canonical byte denominator; canonical rescoring was estimated around `1.2169`.
- Active dataset route remains the custom HF export, assumed by default to `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf` with `MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=datasets`.

# Hard constraints
- Never auto-run H100 jobs.
- Never exceed `16,000,000` total artifact bytes on a candidate.
- Never access network or training data during evaluation.
- Never brute-force seeds.
- Keep root `train_gpt.py` and `train_gpt_mlx.py` under `1500` lines by `scripts/check_line_budget.py`.

# Workflow
- H100-only, manual-only for scoreable runs.
- Root `train_gpt.py` is the canonical entrypoint and contains the accepted `#1493` baseline with local syntax and decimal cap-accounting repairs.
- Frozen snapshots are preserved history. The pre-reset `#1791` root script is archived at `snapshots/train_gpt_2026-04-24_pre_pr1493_accepted_reset_root.py`.
- The demoted FLA investigation lane still requires the pinned FLA runtime from `requirements-h100-fla.txt`.

# GREEN / YELLOW / RED
- `GREEN`: `configs/h100/root_sp8192_pr1493_accepted_8xh100.json`.
- `YELLOW`: preserved historical logs / snapshots and non-promotable local sanity checks.
- `RED`: `configs/h100/root_sp8192_pr1791_fla_8xh100.json`, `configs/h100/root_sp8192_pr1791_fla_proxy_1xh100.json`, eval caches, two-pass rescoring, tokenizer edits without proof, legality-pending CaseOps paths as the primary lane, public-SP1024 instructions presented as frontier guidance, oversized artifacts, seed brute force, or auto-launched H100 jobs.

# Current hypotheses
- The highest-confidence repo path is reproducing accepted `#1493` before spending compute on unverified open PRs.
- `#1791` may remain an architecture research lane, but it cannot be promoted until fresh canonical byte-counted H100 logs exist.
- Open PRs such as `#1735` may be promising but remain reproduction-pending unless accepted upstream or locally proven.

# Current blockers
- The workspace does not currently have the SP8192 dataset/tokenizer cache downloaded locally.
- No local H100 reproduction of the accepted `#1493` baseline has been executed from this root script yet.

# Active candidates
- `configs/h100/root_sp8192_pr1493_accepted_8xh100.json`

# Investigation candidates
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
- `snapshots/train_gpt_2026-04-24_pre_pr1493_accepted_reset_root.py`

# Run ladder
- Run 1: `configs/h100/root_sp8192_pr1493_accepted_8xh100.json`
- Before any H100 run, install the accepted baseline runtime: `pip install brotli sentencepiece` and `pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/`.
- Before any H100 run, verify readiness with `python scripts/check_run_ready.py configs/h100/root_sp8192_pr1493_accepted_8xh100.json`.
- Pod sanity check: `rg -n "QK_GAIN_INIT|TTT_ENABLED|TTT_LR|TTT_EPOCHS|VOCAB_SIZE|MATCHED_FINEWEB" train_gpt.py records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/README.md records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/submission.json`
- Download the dataset first: `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=datasets python data/cached_challenge_fineweb.py --variant sp8192`

# Rules for code changes
- Keep the canonical entrypoint in root `train_gpt.py`.
- Do not stack multiple unrelated algorithmic deltas in one pass.
- Archive abandoned root lanes in `snapshots/` before resetting root.
- Keep `frontier_gdn/` changes self-contained and investigation-only unless promoted by canonical logs.

# Rules for eval and export
- Preserve challenge semantics.
- Log exact roundtrip metrics, sliding metrics, artifact bytes, and eval time.
- Keep byte accounting tied to exact tokenizer LUTs, not fixed divisors.
- Use decimal `16,000,000` total artifact bytes, including counted code bytes plus compressed model bytes.
- Keep no eval-time network access.
- Keep `FLA_USE_NAIVE=1` as a debug-only local fallback, never a scored H100 path.

# Dependencies
- Accepted `#1493` baseline uses `brotli`, `sentencepiece`, and `flash_attn_3` on H100.
- `requirements-h100-fla.txt` remains mandatory only for the demoted FLA investigation lane.
- Standard library only for workflow scripts.
- `train_gpt_mlx.py` remains a non-frontier local helper.

# Handoff
- Update `experiments/registry.csv`.
- Update `docs/top_candidates.md`.
- Update `docs/hopper_plan.md`.
- Preserve archived root lanes in `snapshots/` before resetting root.
