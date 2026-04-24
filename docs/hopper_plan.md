# Hopper Plan

Current frontier target for this pass:
- accepted merged reference: PR `#1493`
- active root baseline: upstream `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT` at `1.0810`
- root `train_gpt.py` is a runnable, cap-enforced copy of the accepted record script
- demoted investigation lane: PR `#1791` `K_KVShare_Wider` GatedDeltaNet / FLA, blocked by canonical BPB evidence from April 24, 2026 upstream review

## Run Order
- Run 1: `configs/h100/root_sp8192_pr1493_accepted_8xh100.json`
- Investigation only: `configs/h100/root_sp8192_pr1791_fla_proxy_1xh100.json`

## Accepted Baseline Gate
- Goal: reproduce the accepted legal SOTA baseline before spending effort on unverified open PRs.
- Command generation: `python scripts/prepare_h100_run.py configs/h100/root_sp8192_pr1493_accepted_8xh100.json`
- Pod payload/import check: `python scripts/check_active_train_payload.py --require-imports configs/h100/root_sp8192_pr1493_accepted_8xh100.json`
- Pod readiness check: `python scripts/check_run_ready.py configs/h100/root_sp8192_pr1493_accepted_8xh100.json`
- Pod sanity check: `python scripts/check_active_train_payload.py --require-imports configs/h100/root_sp8192_pr1493_accepted_8xh100.json`
- Dataset prerequisite: `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=datasets python data/cached_challenge_fineweb.py --variant sp8192`
- Setup prerequisite: `pip install brotli sentencepiece` and `pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/`

## Run 1
- Goal: execute the accepted PR `#1493` SP8192 legal score-first TTT baseline on 8xH100.
- Success criteria: final exact `val_bpb` near `1.0810`, train time under `600s`, eval time under `600s`, total artifact bytes under `16,000,000`, and no eval-time network or unscored-token adaptation.
- Risk: this lane uses a compressed record script with local syntax and artifact-cap repairs. Treat it as a reproduction baseline, not a new root algorithmic delta.

## Notes
- The public `willdepueoai/parameter-golf` dataset repo is not sufficient for this SP8192 baseline.
- PR `#1791` remains locally available under `frontier_gdn/` but is not promotable until a canonical run resolves the byte-accounting mismatch.
- Do not auto-launch H100 runs from this repo.
