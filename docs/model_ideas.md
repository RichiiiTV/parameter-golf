# Model Ideas

Highest ROI now:
- Compose seq2048 training with clean sliding eval and tied-embedding fp16 export.
- Add EMA.
- Sweep `MLP_HIDDEN` for byte-aware width/MLP tradeoffs.
- Sweep KV-head reduction if bytes can be reallocated productively.

Later:
- Recurrent/shared blocks with cleaner design than the noisy `SlidingWindowEval` record script.
- Export-faithful late-stage fake quant only if the simple fp16 embedding path is exhausted.

Deprioritized:
- Importing unused QAT/LoRA/loop code from `SlidingWindowEval`
- Custom Triton
- Large architecture rewrites before the composed baseline is measured
