# Mission
- Maximize final post-export `val_bpb` under the `16,000,000`-byte artifact cap and the `10`-minute train / `10`-minute eval limits on `8xH100 SXM`.

# Frontier
- Accepted record source of truth: upstream `README.md` plus accepted `records/`.
- Current accepted top merged record: `#549` / `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` at `1.1194`.
- Latest upstream open leaders checked on April 22, 2026 are `#1767` at `1.07209`, `#1765` at `1.07266`, `#1775` at `1.07285`, `#1776` at `1.08083`, and `#1771` at `1.06513` with legality pending.
- Chosen active root target: `#1667` at `1.07139`, because it is the strongest concrete likely-legal open lane that fits this compact root better than the larger `#1626` family stack.
- Preserved local reference baseline: `logs/root-pr549-softqat.txt` at `1.11956668`.
- Active root path is now a compact SP8192 frontier port: 11 layers, 512d, 8H / 4KV, LeakyReLU(0.5)^2, partial RoPE 16, SmearGate, attention-output gate, 3-layer depth recurrence, parallel residuals, int6/int7 export, and legal score-first TTT.

# Hard constraints
- Never auto-run H100 jobs.
- Never exceed `16,000,000` artifact bytes on a candidate.
- Never access network or training data during evaluation.
- Never brute-force seeds.
- Keep `train_gpt.py` and `train_gpt_mlx.py` under `1500` lines by `scripts/check_line_budget.py`.

# Workflow
- H100-only, manual-only.
- Root `train_gpt.py` is the canonical runner.
- Frozen snapshots are preserved history, but active work happens in root.
- `flash-attn` is optional in code and recommended only in the H100 environment.

# GREEN / YELLOW / RED
- `GREEN`: `#1667`-class SP8192 legal-TTT root in `train_gpt.py` with `configs/h100/root_sp8192_pr1667_legal_ttt.json`.
- `YELLOW`: `configs/h100/root_sp8192_pr1667_legal_ttt_proxy_1xh100.json`, the local SP8192 sanity config, and preserved April 22 snapshots / `#549` truth logs.
- `RED`: eval caches, two-pass rescoring, tokenizer edits without proof, legality-pending `#1771` as a primary lane, oversized artifacts, seed brute force, or auto-launched H100 jobs.

# Current hypotheses
- The cleaned SHD/SP1024 baseline was useful only as a regression target; the actual frontier chance in this repo is the SP8192 + legal-TTT pivot.
- `#1667` is the best primary target for this compact root because it keeps the strongest likely-legal score among the open lanes that do not require importing the bulkier `#1626` stack.
- `#1771` stays out of the active lane until legality is resolved.
- `#1767`, `#1765`, `#1775`, and `#1776` are reference points, not the active port target.

# Current blockers
- The workspace does not currently have the SP8192 dataset/tokenizer cache downloaded locally.
- No H100 truth run has been executed from the new SP8192 root yet.

# Active candidates
- `configs/h100/root_sp8192_pr1667_legal_ttt.json`
- `configs/h100/root_sp8192_pr1667_legal_ttt_proxy_1xh100.json`
- `configs/local/root_sp8192_pr1667_legal_ttt_sanity.json`

# Historical references
- `logs/root-pr549-softqat.txt`
- `snapshots/train_gpt_2026-04-22_pre_shd_pivot_root.py`
- `snapshots/train_gpt_2026-04-22_pre_shd_only_prune_root.py`
- `snapshots/train_gpt_2026-04-22_pre_sp8192_pr1667_pivot_root.py`

# Run ladder
- Run 1: `configs/h100/root_sp8192_pr1667_legal_ttt.json`
- Optional proxy only: `configs/h100/root_sp8192_pr1667_legal_ttt_proxy_1xh100.json`
- Before any H100 run, verify the pod has the SP8192 frontier root: `train_gpt.py` should contain `VOCAB_SIZE`, `TTT_ENABLED`, `SMEAR_GATE`, `GATE_ATTN_OUT`, `QK_GAIN_INIT`, `NUM_LOOPS`, and `PARALLEL_START_LAYER`.
- Download the dataset first: `python data/cached_challenge_fineweb.py --variant sp8192`

# Rules for code changes
- Keep record-critical logic in root `train_gpt.py`.
- Keep new root changes minimal and self-contained.
- Do not stack multiple unrelated algorithmic deltas in one pass.
- Archive abandoned root lanes in `snapshots/` before resetting root.

# Rules for eval and export
- Preserve challenge semantics.
- Log exact roundtrip metrics, sliding metrics, artifact bytes, and eval time.
- Keep the active TTT path score-first and single-pass.
- Keep byte accounting tied to exact tokenizer LUTs, not fixed divisors.
- Keep the active export path at int6 matrices + int7 embeddings.

# Dependencies
- No new mandatory runtime dependencies.
- Standard library only for workflow scripts.
- `flash-attn` remains optional and cluster-side.

# Handoff
- Update `experiments/registry.csv`.
- Update `docs/top_candidates.md`.
- Update `docs/hopper_plan.md`.
- Preserve archived root lanes in `snapshots/` before resetting root.
