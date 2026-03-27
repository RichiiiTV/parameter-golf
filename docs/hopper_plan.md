# Hopper Plan

Active valid frontier for this pass:
- accepted merged record: `#549`
- active root snapshot fallback: `snapshots/train_gpt_2026-03-25_pre753_pr549_softqat_root.py`
- archived GDN snapshot: `snapshots/train_gpt_2026-03-27_prepivot_pr875_gdn_root.py`
- active root deltas: exact `#549` repro, then one late soft-round QAT follow-up

## Run Order
- Run 1: exact `#549` repro
- Run 2: exact `#549` plus late soft-round QAT

## Run 1
- Goal: validate the restored `#549` root line on the true H100 surface.
- Command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_pr549_repro.json`
- Success criteria: land in the accepted `#549` quality, size, and timing band.
- Risk: if the repro misses badly, root fidelity needs debugging before any new idea is judged.

## Run 2
- Goal: test the cleanest known follow-up on the same valid family.
- Command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_pr549_softqat.json`
- Success criteria: improve final post-TTT `val_bpb` versus the exact repro while staying under the byte cap and within the 10-minute train/eval limits.
- Risk: if the repro is already off-family, this lane should not be judged yet.

## Notes
- `#753` / eval-cache / hybrid work remains archived pending rule clarification and is not part of the active record ladder.
- The `#875` GDN branch is archived after the 1xH100 proxy landed far off-family and did not justify more scarce H100 runs.
- Tokenizer WIP is archived for this pass and should not be mixed into the active ladder.
- `flash-attn` remains optional and cluster-side.
