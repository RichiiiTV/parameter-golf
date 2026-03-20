# Model Ideas

## GREEN Mainline
- Reproduce the official top-1 stack cleanly in root
- Test `TRAIN_SEQ_LEN=2048` on top of that stack
- Test `TRAIN_SEQ_LEN=4096` on top of that stack

## GREEN Mechanism Follow-Up
- Add pre-projection RMSNorm after the best GREEN lane is established
- Why now: the research garden flags this as the most promising near-term mechanism question and cites Steinmetz et al. (2025)
- Constraint: only promote if the quality gain survives the throughput cost on H100

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
