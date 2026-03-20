from __future__ import annotations

import json
import re
from pathlib import Path


FINAL_PRE_RE = re.compile(r"final_pre_export_exact val_loss:(?P<loss>\S+) val_bpb:(?P<bpb>\S+)")
FINAL_POST_RE = re.compile(r"final_int8_zlib_roundtrip_exact val_loss:(?P<loss>\S+) val_bpb:(?P<bpb>\S+)")
FINAL_POST_FAST_RE = re.compile(r"final_int8_zlib_roundtrip val_loss:\S+ val_bpb:\S+ eval_time:(?P<eval_ms>\d+)ms")
STOP_RE = re.compile(r"stopping_early: wallclock_cap train_time:(?P<train_ms>\d+)ms step:(?P<step>\d+)/(?P<iters>\d+)")
SIZE_RE = re.compile(r"Total submission size int8\+zlib: (?P<bytes>\d+) bytes")
MODEL_SIZE_RE = re.compile(r"Serialized model int8\+zlib: (?P<bytes>\d+) bytes")
CODE_SIZE_RE = re.compile(r"Code size: (?P<bytes>\d+) bytes")
TOKENS_RE = re.compile(r"train_tokens_seen:(?P<tokens>\d+)")
PEAK_MEM_RE = re.compile(r"peak memory allocated: (?P<allocated>\d+) MiB reserved: (?P<reserved>\d+) MiB")


def parse_train_log(path: str | Path) -> dict[str, object]:
    log_path = Path(path)
    text = log_path.read_text(encoding="utf-8")
    result: dict[str, object] = {"log_path": str(log_path)}

    for line in text.splitlines():
        if line.startswith("git_commit:"):
            pieces = dict(part.split(":", 1) for part in line.split() if ":" in part)
            result["git_commit"] = pieces.get("git_commit")
            result["git_dirty"] = pieces.get("git_dirty")
        elif line.startswith("final_metric_summary:"):
            result.update(json.loads(line.split(":", 1)[1]))
        elif match := FINAL_PRE_RE.fullmatch(line):
            result["pre_val_loss"] = float(match.group("loss"))
            result["pre_val_bpb"] = float(match.group("bpb"))
        elif match := FINAL_POST_RE.fullmatch(line):
            result["post_val_loss"] = float(match.group("loss"))
            result["post_val_bpb"] = float(match.group("bpb"))
        elif match := FINAL_POST_FAST_RE.fullmatch(line):
            result["eval_time_ms"] = int(match.group("eval_ms"))
        elif match := STOP_RE.fullmatch(line):
            result["train_time_ms"] = int(match.group("train_ms"))
            result["step_stop"] = int(match.group("step"))
        elif match := SIZE_RE.fullmatch(line):
            result["artifact_bytes"] = int(match.group("bytes"))
            result["bytes_total"] = int(match.group("bytes"))
        elif match := MODEL_SIZE_RE.fullmatch(line):
            result["model_bytes"] = int(match.group("bytes"))
        elif match := CODE_SIZE_RE.fullmatch(line):
            result["code_bytes"] = int(match.group("bytes"))
        elif match := TOKENS_RE.fullmatch(line):
            result["train_tokens_seen"] = int(match.group("tokens"))
        elif match := PEAK_MEM_RE.fullmatch(line):
            result["peak_mem_mib"] = int(match.group("allocated"))
            result["peak_mem_reserved_mib"] = int(match.group("reserved"))
    if "pre_val_bpb" in result and "post_val_bpb" in result:
        result["pre_post_gap"] = float(result["post_val_bpb"]) - float(result["pre_val_bpb"])
    if "pre_val_bpb" in result and "pre_quant_val_bpb" not in result:
        result["pre_quant_val_bpb"] = result["pre_val_bpb"]
    if "post_val_bpb" in result and "post_quant_val_bpb" not in result:
        result["post_quant_val_bpb"] = result["post_val_bpb"]
    if "artifact_bytes_total" in result and "bytes_total" not in result:
        result["bytes_total"] = result["artifact_bytes_total"]
    if "artifact_bytes" in result and "bytes_total" not in result:
        result["bytes_total"] = result["artifact_bytes"]
    return result
