# Top Candidates

## Accepted Record Ladder
- `NaiveBaseline`: baseline architecture and training recipe
- `LongContextSeq2048`: accepted training-side improvement
- `FP16Embed_WD3600`: accepted export-side improvement
- `SlidingWindowEval`: accepted eval-side improvement and current public best
- Canonical challenger: strict composition of those accepted levers

## Current Public Best
- `SlidingWindowEval`: `post-export val_bpb=1.19250007`
- Interpretation: eval-only improvement over the baseline that currently sets the public bar to beat

## Canonical Challenger
- `seq2048 + fp16 tok_emb passthrough + sliding eval stride 64`
- Composition of `LongContextSeq2048`, `FP16Embed_WD3600`, and `SlidingWindowEval`
- Keep the baseline architecture unchanged and do not add EMA, Triton, or new trainer features to the first challenger
- Treat `COMPUTE_DTYPE=auto`, `SDPA_BACKEND=auto`, and `COMPILE_MODE=fullgraph` as H100 execution defaults, not as part of the accepted-lever composition

## Top 10 Low-Hanging Fruits
- Promote the canonical composed challenger as the first manual H100 truth run
- Tighten byte headroom around `tok_emb.weight` fp16 passthrough
- Preserve `WARMDOWN_ITERS=3600` exactly in the seq2048 challenger
- Keep `MLP_HIDDEN=992` as the byte-recovery setting for the challenger
- Run the EMA variant only after the challenger has a truth result
- Use `EVAL_BATCH_SEQS=1024` in the H100 challenger to match the accepted sliding-eval path
- Compare eval seq len `2048` vs `4096` only after the challenger
- Run the `SDPA_BACKEND` sweep only after the challenger
- Run the `COMPILE_MODE` sweep only after the challenger
- Keep Triton out until a standard-backend loss is measured on H100

## Top 5 Hopper-Specific Experiments
- Canonical challenger: `seq2048 + fp16 tok_emb + sliding eval`
- Same challenger plus EMA
- Post-challenger `SDPA_BACKEND` sweep
- Post-challenger `COMPILE_MODE` sweep
- Eval seq len `2048` vs `4096` only if the challenger is under budget

## Top 5 Triton Opportunities Or Reasons To Avoid Triton
- Avoid custom attention kernels
- Avoid custom GEMM
- Prefer standard SDPA/cuDNN/Inductor first
- Consider export packing only after profiling
- Consider sliding-eval scoring only after profiling

## Top 5 Risky But Interesting Challenge-Edge Ideas
- Eval-time adaptation
- Recurrent/shared blocks
- Tokenizer changes
- Mixed-format export codecs
- Bundled auxiliary eval state

## Top 3 Immediate Next Steps
- Treat `SlidingWindowEval` as the current public target, not as an unfinished low-hanging fruit
- Run the canonical composed challenger manually on H100
- Run the EMA and systems follow-ups only after the challenger truth run

## Local Proxy Notes
- `gtx1080ti-smoke`: `pre/post val_bpb = 3.94218527 / 3.95601011`, `bytes_total=5,050,048`
- `gtx1080ti-proxy1024`: `pre/post val_bpb = 4.01842393 / 4.03490743`, `bytes_total=5,049,873`
- `gtx1080ti-export-proxy`: `pre/post val_bpb = 4.01853171 / 4.01945346`, `bytes_total=5,818,837`
- Local conclusion: `KEEP_FLOAT_EXTRA=tok_emb.weight` is a strong export-gap lever on GTX, shrinking the post-export gap by about `18x`, but it costs about `0.77 MB` and still needs H100 truth runs before promotion
