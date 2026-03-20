# Model Ideas

## GREEN Mainline
- Reproduce the official top-1 stack cleanly in root
- Test `TRAIN_SEQ_LEN=2048` on top of that stack
- Test `TRAIN_SEQ_LEN=4096` on top of that stack

## GREEN Mechanism Follow-Up
- Keep pre-projection RMSNorm deferred for now
- Evidence: same-folder 4xA100 ablation was negative (`USE_PREPROJ_RMSNORM=0 -> 1.28120795`, `USE_PREPROJ_RMSNORM=1 -> 1.30713253`)
- Constraint: only revisit after the best GREEN throughput lane is established on H100

## YELLOW Architecture Lanes
- Recurrent/shared-width transformer blocks
- Parameter sharing aimed at buying width under the 16 MB cap
- Motivation comes from Bae et al. (2024), Csordás et al. (2024), and Üyük et al. (2024)

## YELLOW Compression Lanes
- Sparse or decoupled high-precision outlier retention
- Motivation comes from the current research-garden outlier-retention question and Zhang et al. (2026)

## Explicitly Deferred In This Reset
- LoRA TTT
- mixed int8/int6 export in the root path
- tokenizer redesign
- Triton kernels
