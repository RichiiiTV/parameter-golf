# Local Harness

Status:
- Legacy only.

Reason:
- The active workflow has been reset to H100-only manual runs.
- GTX and A100 history are preserved for context, but they are not used to rank active candidates in this phase.

Active handoff path:
- `py -3.11 scripts/prepare_h100_run.py <config>`
