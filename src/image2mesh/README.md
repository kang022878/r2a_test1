# image2mesh

Pipeline: crop PNG (with alpha) -> 3D reconstruction (TripoSR) -> export (glb/obj).

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# plus TripoSR deps
pip install -r vendor/TripoSR/requirements.txt
```

Run the FastAPI server (optional):
```bash
uvicorn image2mesh.server:app --reload
```

## Notes
- TripoSR is used as a subprocess wrapper via `vendor/TripoSR/run.py` to reduce breakage across versions.
- Outputs are written under `outputs/<job_id>/`.
