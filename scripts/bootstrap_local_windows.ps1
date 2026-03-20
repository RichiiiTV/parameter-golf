param(
    [int]$TrainShards = 1
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe data\cached_challenge_fineweb.py --variant sp1024 --train-shards $TrainShards
.\.venv\Scripts\python.exe scripts\check_env.py --config configs\local\gtx1080ti_smoke.json
