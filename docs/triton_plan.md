# Triton Plan

Default reset-pass decision:
- No custom Triton kernels.

## GREEN
- Standard PyTorch SDPA
- cuDNN / Inductor / vendor-backed kernels already reachable through the normal stack

## YELLOW
- Export-side layout or outlier-retention kernels only if sparse high-precision retention becomes the leading lane
- Narrow H100-only microkernels only after a measured hotspot survives against standard backends

## RED
- Custom attention kernels that duplicate SDPA
- Custom GEMMs that duplicate cuBLAS
- Any Triton path that becomes required for the mainline

## Gate
- Do not revisit Triton until the best GREEN H100 lane is known and a real standard-backend loss is measured.
