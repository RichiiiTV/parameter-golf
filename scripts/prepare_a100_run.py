from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path


def format_env(env: dict[str, str]) -> str:
    return "\n".join(f"{key}={value}" for key, value in sorted(env.items()))


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
) -> None:
    print("RUN THIS MANUALLY ON 4xA100")
    print(f"config: {config_path.relative_to(root)}")
    if point_name:
        print(f"matrix_point: {point_name}")
    print(f"purpose: {purpose}")
    print(f"runtime_budget_minutes: {budget}")
    print("environment:")
    print(format_env(env) or "(none)")
    print("command: torchrun --standalone --nproc_per_node=4 train_gpt.py")
    print("expected_outputs: logs/<RUN_ID>.txt, final_model.int8.ptz, final roundtrip metrics")
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
        success=str(
            config.get(
                "success_criteria",
                "Improve post-export val_bpb without breaking the 16,000,000-byte cap.",
            )
        ),
        risks=str(config.get("risks", "Proxy-eval mismatch, compile overhead, or A100 memory limits.")),
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
    success = str(
        config.get(
            "success_criteria",
            "Improve post-export val_bpb without breaking the 16,000,000-byte cap.",
        )
    )
    risks = str(config.get("risks", "Proxy-eval mismatch, compile overhead, or A100 memory limits."))
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
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit manual-only 4xA100 handoff blocks")
    parser.add_argument("config", help="Path to A100 JSON config")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = (root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "matrix" in config:
        emit_matrix(root, config_path, config)
    else:
        emit_single(root, config_path, config)


if __name__ == "__main__":
    main()
