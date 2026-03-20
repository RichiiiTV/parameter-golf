# Local Harness

Goals:
- Run on GTX 1080 Ti without Hopper assumptions.
- Keep export roundtrip mandatory.
- Use deterministic proxy validation before H100 promotion.

Bootstrap:
- Run `powershell -ExecutionPolicy Bypass -File scripts/bootstrap_local_windows.ps1`
- This creates a Python 3.11 `.venv`, installs the CUDA-enabled local wheel, installs `requirements-local.txt`, downloads one training shard, and runs `scripts/check_env.py`.

Trainer auto behavior on Pascal:
- `COMPUTE_DTYPE=auto` resolves to `fp16`
- `MUON_DTYPE=auto` resolves to `fp32`
- `SDPA_BACKEND=auto` resolves to `math`
- `COMPILE_MODE=auto` resolves to `off`
- `USE_FUSED_ADAM=0`

Validated GTX config behavior:
- The checked-in GTX configs pin `COMPUTE_DTYPE=fp32` and `MUON_DTYPE=fp32` for Pascal numerical stability.
- Treat these explicit `fp32` overrides as the canonical local proxy path on this machine, not as an H100 precision recommendation.
- Keep `auto` resolution documented because it describes trainer behavior, but do not treat it as the validated local default.

Validation ladder:
- `.\.venv\Scripts\python.exe scripts\run_experiment.py configs\local\gtx1080ti_smoke.json`
- `.\.venv\Scripts\python.exe scripts\run_experiment.py configs\local\gtx1080ti_proxy1024.json`
- `.\.venv\Scripts\python.exe scripts\run_experiment.py configs\local\gtx1080ti_export_proxy.json`

Validated local results on this GTX 1080 Ti:
- `gtx1080ti-smoke`: completed with finite roundtrip metrics at seq512, `pre/post val_bpb=3.94218527 / 3.95601011`, `bytes_total=5,050,048`
- `gtx1080ti-proxy1024`: completed with finite roundtrip metrics at seq1024, `pre/post val_bpb=4.01842393 / 4.03490743`, `bytes_total=5,049,873`
- `gtx1080ti-export-proxy`: completed with finite roundtrip metrics at seq1024, `pre/post val_bpb=4.01853171 / 4.01945346`, `bytes_total=5,818,837`
- Local finding: Pascal `fp16` was unstable for this trainer; checked-in GTX profiles now use explicit `fp32`
- Local export finding: `KEEP_FLOAT_EXTRA=tok_emb.weight` reduced the seq1024 proxy pre/post gap from about `0.01648` to about `0.00092` at a byte cost of about `0.77 MB`

Profiles:
- `configs/local/gtx1080ti_smoke.json`: correctness, export roundtrip, registry, parser
- `configs/local/gtx1080ti_proxy1024.json`: transfer-focused ranking at seq1024
- `configs/local/gtx1080ti_export_proxy.json`: export-gap triage with tied-embedding fp16 passthrough

Promotion rules:
- Keep only configs with valid roundtrip metrics.
- Reject NaN, oversized artifact, or missing final metric lines.
- Promote only algorithmic and export winners, not Pascal-only throughput wins.
