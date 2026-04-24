# Top Candidates

## Active Legal Baseline
- Active root runner: `train_gpt.py`
- Active accepted source: `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py`
- Root status: local runnable copy with syntax and decimal artifact-cap enforcement repairs
- Runnable H100 target: `configs/h100/root_sp8192_pr1493_accepted_8xh100.json`
- Required dataset/tokenizer surface: custom SP8192 export via `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf`
- Accepted upstream reference: PR `#1493`, `val_bpb=1.0810`, 3-seed mean, legal score-first TTT, artifacts under `16,000,000` bytes.
- Preserved local reference baseline: `logs/root-pr549-softqat.txt` at `1.11956668`
- Preserved previous root FLA script: `snapshots/train_gpt_2026-04-24_pre_pr1493_accepted_reset_root.py`

## Upstream Check
- Checked on April 24, 2026: PR `#1791` is no longer a promotable target despite its reported `1.0339`.
- Reason: upstream review comments show the submitted `(val_loss, val_bpb)` pairs imply the older non-canonical byte denominator (`~4.3864` bytes/token), not the canonical SP8192 denominator (`~3.7266` bytes/token). Canonical rescoring was estimated around `1.2169`.
- Current clean accepted target is `#1493` at `1.0810`. The PR `#1735` open lane reports `1.0429` with a canonical LUT, but remains reproduction-pending and uses pre-quant TTT; do not treat it as accepted SOTA until merged or locally proven.

## Current Use
- Use `configs/h100/root_sp8192_pr1493_accepted_8xh100.json` for the active legal SOTA baseline.
- Keep `configs/h100/root_sp8192_pr1791_fla_8xh100.json` and `configs/h100/root_sp8192_pr1791_fla_proxy_1xh100.json` as investigation-only.
- Do not schedule the demoted FLA lane as GREEN unless a fresh canonical H100 run produces consistent byte-counted logs and artifact accounting under the decimal cap.
