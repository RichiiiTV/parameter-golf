# Triton Plan

Default phase-1 decision:
- No custom Triton kernels.

GREEN:
- Standard PyTorch SDPA
- cuDNN / Inductor / vendor-backed kernels already reachable through normal flags

YELLOW:
- Export pack/unpack kernels only if H100 profiling shows a real uncovered hotspot
- Sliding-eval suffix scoring kernels only if eval becomes the bottleneck after standard backend tuning

RED:
- Custom attention kernels that duplicate SDPA
- Custom GEMMs that duplicate cuBLAS
- Anything that adds packaging risk without a clear measured win

Gate:
- No custom Triton enters the repo until standard backend, compile, and CUDA-Graph-style opportunities have been benchmarked and lost.
