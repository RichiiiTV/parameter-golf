# Top Candidates

## Valid Mainline
- Accepted merged frontier: `#549` / `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` at `val_bpb=1.1194`.
- Provisional valid open target: `#875` / `Pure Neural GDN 1.0226`.
- Active root runner: `train_gpt.py`
- Active root direction: compact `#875`-style Gated DeltaNet with exact byte-accounted eval math.
- Fallback snapshot: `snapshots/train_gpt_2026-03-25_pre753_pr549_softqat_root.py`
- Archived tokenizer WIP snapshot: `snapshots/train_gpt_2026-03-27_pre875_hnet_wip_root.py`

## Active H100 Candidates
- Exact pure-neural GDN base: `configs/h100/root_pr875_gdn_repro.json`
- Pure-neural GDN + one extra GDN block: `configs/h100/root_pr875_gdn_deeper.json`
- Pure-neural GDN + legal TTT: `configs/h100/root_pr875_gdn_ttt.json`

## Main Prediction
- The safest valid high-upside path is a pure-neural GDN port that keeps the root exact byte-accounted judge and simple int8+zlib export.
- The best few-run architecture follow-up is to spend remaining artifact headroom on one extra GDN block before trying more eval machinery.
- Legal score-first TTT stays as the secondary follow-up once the deeper base is judged.
- Eval-cache / n-gram lanes remain archived pending rule clarification from OpenAI.
- Tokenizer experimentation is archived for now and is not part of the active record ladder.

## Ranking Policy
- Rank active candidates by lower final post-TTT `val_bpb`.
- Require `bytes_total < 16,000,000`.
- Require H100 truth before promotion.
