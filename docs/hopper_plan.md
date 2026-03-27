# Hopper Plan

Active valid frontier for this pass:
- accepted merged record: `#549`
- provisional valid open target: `#875`
- active root snapshot fallback: `snapshots/train_gpt_2026-03-25_pre753_pr549_softqat_root.py`
- archived tokenizer snapshot: `snapshots/train_gpt_2026-03-27_pre875_hnet_wip_root.py`
- active root deltas: pure-neural GDN base, one deeper-capacity follow-up, and one legal TTT follow-up

## Run Order
- Run 1: exact pure-neural GDN base
- Run 2: pure-neural GDN plus one extra GDN block
- Run 3: pure-neural GDN plus legal score-first TTT

## Run 1
- Goal: validate the compact `#875`-style GDN base under the root exact byte-accounted metric path.
- Command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_pr875_gdn_repro.json`
- Success criteria: land in the `#875` pure-neural family while staying under the 16,000,000-byte cap and under the 10-minute train/eval limits.
- Risk: the local PyTorch GDN mixer may need tuning if throughput or export quality drifts too far from the donor family.

## Run 2
- Goal: test the cleanest architecture-only improvement with minimal legality risk and minimal attribution ambiguity.
- Command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_pr875_gdn_deeper.json`
- Success criteria: improve final post-export `val_bpb` versus the pure-neural GDN base while staying under the byte cap and within the 10-minute train/eval limits.
- Risk: one extra block may reduce step count enough to erase the capacity gain if throughput falls too much.

## Run 3
- Goal: test whether legal score-first TTT improves the same GDN base without changing training or export semantics.
- Command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_pr875_gdn_ttt.json`
- Success criteria: improve final post-TTT `val_bpb` versus the pure-neural GDN base without breaking byte or time limits.
- Risk: legal TTT may be neutral or too slow to justify promotion even if it remains valid.

## Notes
- `#753` / eval-cache / hybrid work remains archived pending rule clarification and is not part of the active record ladder.
- The donor `#875` judge math is not reused here; root keeps exact SentencePiece LUT byte accounting instead.
- Tokenizer WIP is archived for this pass and should not be mixed into GDN evaluation.
- The deeper-GDN lane is preferred over immediate TurboQuant because it targets model quality directly and preserves clear attribution if only a few H100 runs are available.
- `flash-attn` remains optional and cluster-side.
