from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from copy import deepcopy
from pathlib import Path


ALLOWLIST = {
    "TRAIN_SEQ_LEN",
    "TIED_EMBED_LR",
    "MATRIX_LR",
    "SCALAR_LR",
    "WARMDOWN_ITERS",
    "MLP_HIDDEN",
    "KEEP_FLOAT_EXTRA",
    "EVAL_MODE",
    "EVAL_STRIDE",
    "EVAL_SEQ_LEN",
    "INT8_CLIP_PERCENTILE",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def score(summary: dict) -> tuple[float, int, float]:
    return (
        float(summary.get("post_val_bpb", 1e9)),
        int(summary.get("bytes_total", summary.get("artifact_bytes", 10**12))),
        float(summary.get("pre_post_gap", 1e12)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Config-only local autosearch for parameter-golf")
    parser.add_argument("config", help="Path to JSON search config")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    search_config = load_json((root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config))
    base_config_path = (root / str(search_config["base_config"])).resolve()
    base_config = load_json(base_config_path)
    search_space = dict(search_config.get("search_space", {}))
    unknown = sorted(set(search_space) - ALLOWLIST)
    if unknown:
        raise SystemExit(f"Search space contains disallowed knobs: {', '.join(unknown)}")

    trials = int(search_config.get("trials", 4))
    top_k = int(search_config.get("top_k", 3))
    seed = int(search_config.get("seed", 1337))
    rng = random.Random(seed)
    search_dir = root / "experiments" / "search_configs"
    search_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []

    for trial_idx in range(trials):
        config = deepcopy(base_config)
        env = dict(config.get("env", {}))
        for key, values in search_space.items():
            env[key] = rng.choice(list(values))
        config["env"] = env
        config["profile"] = f"{config.get('profile', base_config_path.stem)}-search-{trial_idx:03d}"
        config["notes"] = f"autosearch trial {trial_idx}"
        config_path = search_dir / f"{config['profile']}.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "scripts/run_experiment.py", str(config_path.relative_to(root))],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        print(proc.stdout, end="")
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        result = json.loads((Path(payload["run_dir"]) / "result.json").read_text(encoding="utf-8"))
        summaries.append(result)

    best = sorted(summaries, key=score)[:top_k]
    print(json.dumps({"top_k": top_k, "promoted": [x["experiment_id"] for x in best]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
