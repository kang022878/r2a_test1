#!/usr/bin/env bash
set -euo pipefail

SAMPLE=assets/samples/sample.png
OUT=outputs/smoke

mkdir -p "$OUT"
python - <<'PY'
from pathlib import Path
from image2mesh.pipeline import run_pipeline, PipelineConfig

cfg = PipelineConfig()
output_dir, _ = run_pipeline(image=Path("assets/samples/sample.png"), cfg=cfg, job_id="smoke")
print("Output:", output_dir)
PY
