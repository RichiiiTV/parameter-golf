# Hopper Plan

Current frontier target for this pass:
- accepted merged reference: `#549`
- preserved local truth baseline: `logs/root-pr549-softqat.txt` at `1.11956668`
- active root delta: `#1791`-class `K_KVShare_Wider` GatedDeltaNet / FLA on SP8192 with KV-share stride `2`, `BigramHash(3072,112)`, trigram embeddings, `EMA=0.997`, SWA every `50`, late QAT, and int6 + zstd-22 export

## Run Order
- Run 1: `configs/h100/root_sp8192_pr1791_fla_8xh100.json`
- Optional proxy only: `configs/h100/root_sp8192_pr1791_fla_proxy_1xh100.json`

## Proxy Gate
- Goal: validate the pinned FLA runtime, the SP8192 custom export, exact roundtrip logging, and artifact-byte safety on one H100 before using the 8xH100 lane.
- Command generation: `python scripts/prepare_h100_run.py configs/h100/root_sp8192_pr1791_fla_proxy_1xh100.json`
- Pod readiness check: `python scripts/check_run_ready.py configs/h100/root_sp8192_pr1791_fla_proxy_1xh100.json`
- Pod sanity grep: `rg -n "ARCH_MODE|VOCAB_SIZE|EVAL_COMPILE_ENABLED|K_KVShare_Wider|GatedDeltaNet|token_byte_counts" train_gpt.py frontier_gdn/train_gdn_7k.py frontier_gdn/configs.py frontier_gdn/byte_scoring.py`
- Dataset prerequisite: `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=datasets python data/cached_challenge_fineweb.py --variant sp8192`
- Setup prerequisite: `pip install -r requirements-h100-fla.txt`

## Run 1
- Goal: execute the active `#1791`-class SP8192 FLA lane on 8xH100 after the proxy proves stable.
- Command generation: `python scripts/prepare_h100_run.py configs/h100/root_sp8192_pr1791_fla_8xh100.json`
- Success criteria: stay under `16,000,000` bytes, finish train/export/eval cleanly, and log the final exact roundtrip metric on the custom SP8192 export without any eval-time network access.
- Risk: this lane depends on the pinned FLA runtime and the custom SP8192 export. If either is unstable, stop and re-plan instead of soft-reverting into the archived transformer lane.

## Notes
- The public `willdepueoai/parameter-golf` dataset repo is not sufficient for this frontier lane.
- `FLA_USE_NAIVE=1` is debug-only and should not appear in the scored H100 command.
- Do not auto-launch H100 runs from this repo.
