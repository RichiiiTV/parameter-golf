from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot a completed experiment into a record-style folder")
    parser.add_argument("experiment_dir", help="Path to experiments/runs/<experiment_id>")
    parser.add_argument("target_dir", help="Target record folder path")
    parser.add_argument("--author", default="TODO")
    parser.add_argument("--github-id", default="TODO")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    experiment_dir = (root / args.experiment_dir).resolve() if not Path(args.experiment_dir).is_absolute() else Path(args.experiment_dir)
    target_dir = (root / args.target_dir).resolve() if not Path(args.target_dir).is_absolute() else Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    result = json.loads((experiment_dir / "result.json").read_text(encoding="utf-8"))
    config = json.loads((experiment_dir / "config.json").read_text(encoding="utf-8"))
    shutil.copy2(root / "train_gpt.py", target_dir / "train_gpt.py")
    if (experiment_dir / "stdout.log").exists():
        shutil.copy2(experiment_dir / "stdout.log", target_dir / "train.log")
    shutil.copy2(experiment_dir / "result.json", target_dir / "result.json")

    readme = f"""This snapshot was generated from experiment `{result['experiment_id']}`.

Profile: `{config.get('profile', 'unknown')}`
Classification: `{config.get('classification', 'GREEN')}`
Lane: `{config.get('lane', 'local_proxy')}`

Key metrics:
- pre-export val_bpb: `{result.get('pre_quant_val_bpb', result.get('pre_val_bpb', 'unknown'))}`
- post-export val_bpb: `{result.get('post_quant_val_bpb', result.get('post_val_bpb', 'unknown'))}`
- artifact bytes: `{result.get('bytes_total', result.get('artifact_bytes', 'unknown'))}`

Notes:
- Replace this file with a full record README before submission.
"""
    (target_dir / "README.md").write_text(readme, encoding="utf-8")
    submission = {
        "author": args.author,
        "github_id": args.github_id,
        "name": config.get("profile", result["experiment_id"]),
        "blurb": "Generated scaffold from research snapshot; replace before submission.",
        "date": result["experiment_id"].split("-", 1)[0],
        "val_loss": result.get("post_val_loss"),
        "val_bpb": result.get("post_quant_val_bpb", result.get("post_val_bpb")),
        "bytes_total": result.get("bytes_total", result.get("artifact_bytes")),
        "bytes_code": result.get("code_bytes"),
    }
    (target_dir / "submission.json").write_text(json.dumps(submission, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(target_dir)


if __name__ == "__main__":
    main()
