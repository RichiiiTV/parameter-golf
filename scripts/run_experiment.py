from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.parse_log import parse_train_log


REGISTRY_COLUMNS = [
    "experiment_id",
    "timestamp",
    "lane",
    "profile",
    "status",
    "classification",
    "parent_id",
    "seed",
    "hardware",
    "config_path",
    "commit",
    "dirty",
    "train_seq_len",
    "eval_mode",
    "eval_seq_len",
    "eval_stride",
    "train_batch_tokens",
    "grad_accum_steps",
    "compute_dtype",
    "sdpa_backend",
    "compile_mode",
    "mlp_hidden",
    "export_tok_emb_mode",
    "pre_quant_val_bpb",
    "post_quant_val_bpb",
    "pre_post_gap",
    "bytes_total",
    "eval_time_ms",
    "step_stop",
    "promote",
    "notes",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_info(root: Path) -> dict[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    ).stdout.strip() or "unknown"
    dirty = "1" if subprocess.run(
        ["git", "diff-index", "--quiet", "HEAD", "--"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    ).returncode else "0"
    return {"commit": commit, "dirty": dirty}


def trainer_sha256(root: Path) -> str:
    return hashlib.sha256((root / "train_gpt.py").read_bytes()).hexdigest()


def hardware_name() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        ).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def slugify(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def append_registry_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in REGISTRY_COLUMNS})


def export_tok_emb_mode(config: dict[str, Any], env: dict[str, str]) -> str:
    if config.get("export_tok_emb_mode"):
        return str(config["export_tok_emb_mode"])
    keep_float_extra = {part.strip() for part in env.get("KEEP_FLOAT_EXTRA", "").split(",") if part.strip()}
    return "fp16_passthrough" if "tok_emb.weight" in keep_float_extra else "int8_default"


def is_finite_metric(value: object) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local experiment config and record structured artifacts")
    parser.add_argument("config", help="Path to experiment JSON config")
    parser.add_argument("--registry", default="experiments/registry.csv")
    parser.add_argument("--runs-root", default="experiments/runs")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = load_json(config_path)
    if str(config.get("hardware", "")).lower().startswith("8xh100") or config.get("h100_only"):
        raise SystemExit("H100 configs are human-gated. Use scripts/prepare_h100_run.py instead of running them.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    profile = slugify(str(config.get("profile", config_path.stem)))
    experiment_id = f"{timestamp}-{profile}"
    run_dir = ROOT / args.runs_root / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env_overrides = {str(k): str(v) for k, v in dict(config.get("env", {})).items()}
    env_overrides.setdefault("RUN_ID", experiment_id)
    env = os.environ.copy()
    env.update(env_overrides)
    git = git_info(ROOT)

    dump_json(run_dir / "config.json", config)
    dump_json(run_dir / "env.json", env_overrides)
    dump_json(run_dir / "git.json", git)
    (run_dir / "command.txt").write_text(f"{sys.executable} train_gpt.py\n", encoding="utf-8")

    stdout_path = run_dir / "stdout.log"
    with stdout_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            [sys.executable, "train_gpt.py"],
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    trainer_log = ROOT / "logs" / f"{experiment_id}.txt"
    parsed = parse_train_log(trainer_log if trainer_log.exists() else stdout_path)
    bytes_total = parsed.get("artifact_bytes_total", parsed.get("artifact_bytes", ""))
    status = "completed" if proc.returncode == 0 and is_finite_metric(parsed.get("post_val_bpb")) else "failed"
    result = {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "profile": config.get("profile", config_path.stem),
        "status": status,
        "returncode": proc.returncode,
        "config_path": str(config_path.relative_to(ROOT)),
        "hardware": hardware_name(),
        "commit": git["commit"],
        "dirty": git["dirty"],
        "trainer_sha256": trainer_sha256(ROOT),
        "bytes_total": bytes_total,
        "pre_quant_val_bpb": parsed.get("pre_val_bpb"),
        "post_quant_val_bpb": parsed.get("post_val_bpb"),
        **parsed,
    }
    dump_json(run_dir / "result.json", result)

    row = {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "lane": config.get("lane", config.get("stage", "local_proxy")),
        "profile": config.get("profile", config_path.stem),
        "status": status,
        "classification": config.get("classification", "GREEN"),
        "parent_id": config.get("parent_id", config.get("parent_experiment", "")),
        "seed": env_overrides.get("SEED", env.get("SEED", "1337")),
        "hardware": result["hardware"],
        "config_path": str(config_path.relative_to(ROOT)),
        "commit": git["commit"],
        "dirty": git["dirty"],
        "train_seq_len": env_overrides.get("TRAIN_SEQ_LEN", ""),
        "eval_mode": env_overrides.get("EVAL_MODE", "standard"),
        "eval_seq_len": env_overrides.get("EVAL_SEQ_LEN", env_overrides.get("TRAIN_SEQ_LEN", "")),
        "eval_stride": env_overrides.get("EVAL_STRIDE", ""),
        "train_batch_tokens": env_overrides.get("TRAIN_BATCH_TOKENS", ""),
        "grad_accum_steps": env_overrides.get("GRAD_ACCUM_STEPS_OVERRIDE", ""),
        "compute_dtype": env_overrides.get("COMPUTE_DTYPE", "auto"),
        "sdpa_backend": env_overrides.get("SDPA_BACKEND", "auto"),
        "compile_mode": env_overrides.get("COMPILE_MODE", "auto"),
        "mlp_hidden": env_overrides.get("MLP_HIDDEN", "0"),
        "export_tok_emb_mode": export_tok_emb_mode(config, env_overrides),
        "pre_quant_val_bpb": result.get("pre_quant_val_bpb", ""),
        "post_quant_val_bpb": result.get("post_quant_val_bpb", ""),
        "pre_post_gap": result.get("pre_post_gap", ""),
        "bytes_total": bytes_total,
        "eval_time_ms": result.get("eval_time_ms", ""),
        "step_stop": result.get("step_stop", ""),
        "promote": config.get("promote", ""),
        "notes": config.get("notes", ""),
    }
    append_registry_row(ROOT / args.registry, row)
    print(json.dumps({"experiment_id": experiment_id, "status": result["status"], "run_dir": str(run_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
