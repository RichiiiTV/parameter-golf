from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from itertools import product


def slugify(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def wrap_command(*, env: dict[str, str], nproc_per_node: int, run_id: str, script: str) -> str:
    env_items = [f"RUN_ID={shlex.quote(run_id)}"]
    env_items.extend(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items()) if key != "RUN_ID")
    return " ".join(env_items + [f"torchrun --standalone --nproc_per_node={nproc_per_node} {script}"])


def emit_block(
    *,
    root: Path,
    config_path: Path,
    purpose: str,
    budget: int | float,
    env: dict[str, str],
    success: str,
    risks: str,
    point_name: str | None = None,
    slurm: dict[str, object] | None = None,
) -> None:
    slurm = slurm or {}
    point_slug = slugify(point_name or config_path.stem)[:80]
    run_id = slugify(str(slurm.get("run_id", point_slug)))[:100] or "parameter-golf-h100"
    command = wrap_command(env=env, nproc_per_node=8, run_id=run_id, script=str(slurm.get("script", "train_gpt.py")))
    setup_cmd = slurm.get("setup_command", "pip install zstandard flash-attn --no-build-isolation")

    print("RUN THIS MANUALLY ON H100")
    print(f"config: {config_path.relative_to(root)}")
    if point_name:
        print(f"matrix_point: {point_name}")
    print(f"purpose: {purpose}")
    print(f"runtime_budget_minutes: {budget}")
    print(f"setup_command: {setup_cmd}")
    print("command:")
    print(command)
    print("expected_outputs: logs/<RUN_ID>.txt, final_model*.ptz, final roundtrip metrics")
    print(f"risks: {risks}")
    print(f"success_criteria: {success}")
    print()


def emit_single(root: Path, config_path: Path, config: dict[str, object]) -> None:
    env = {str(k): str(v) for k, v in dict(config.get("env", {})).items()}
    emit_block(
        root=root,
        config_path=config_path,
        purpose=str(config.get("purpose", config.get("profile", config_path.stem))),
        budget=config.get("runtime_budget_minutes", 10),
        env=env,
        success=str(config.get("success_criteria", "Improve post-export val_bpb without breaking the 16,000,000-byte cap.")),
        risks=str(config.get("risks", "Compile overhead, eval-time budget, and artifact-byte regressions.")),
        slurm=dict(config.get("slurm", {})),
    )


def emit_matrix(root: Path, config_path: Path, config: dict[str, object]) -> None:
    base_env = {str(k): str(v) for k, v in dict(config.get("env", {})).items()}
    matrix = {str(k): [str(v) for v in values] for k, values in dict(config.get("matrix", {})).items()}
    keys = sorted(matrix)
    if not keys:
        emit_single(root, config_path, config)
        return
    purpose = str(config.get("purpose", config.get("profile", config_path.stem)))
    budget = config.get("runtime_budget_minutes", 10)
    success = str(config.get("success_criteria", "Improve post-export val_bpb without breaking the 16,000,000-byte cap."))
    risks = str(config.get("risks", "Compile overhead, eval-time budget, and artifact-byte regressions."))
    slurm = dict(config.get("slurm", {}))
    for combo in product(*(matrix[key] for key in keys)):
        point_env = dict(base_env)
        point_name = []
        for key, value in zip(keys, combo):
            point_env[key] = value
            point_name.append(f"{key}={value}")
        emit_block(
            root=root,
            config_path=config_path,
            purpose=purpose,
            budget=budget,
            env=point_env,
            success=success,
            risks=risks,
            point_name=", ".join(point_name),
            slurm=slurm,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit manual-only H100 Runpod/SSH commands")
    parser.add_argument("config", help="Path to H100 JSON config")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = (root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not bool(config.get("h100_only", False)):
        raise SystemExit(f"Refusing non-H100 config: {config_path}")
    if "matrix" in config:
        emit_matrix(root, config_path, config)
    else:
        emit_single(root, config_path, config)


if __name__ == "__main__":
    main()
