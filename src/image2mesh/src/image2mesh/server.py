from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .export import package_obj
from .pipeline import PipelineConfig, run_pipeline

app = FastAPI()


@app.post("/export_3d")
async def export_3d(
    image: UploadFile = File(...),
    mask: Optional[UploadFile] = File(None),
    format: str = "obj",
):
    job_id = uuid.uuid4().hex
    tmp_dir = Path("outputs/tmp") / job_id
    tmp_dir.mkdir(parents=True, exist_ok=True)

    image_path = tmp_dir / image.filename
    with image_path.open("wb") as f:
        shutil.copyfileobj(image.file, f)

    mask_path = None
    if mask is not None:
        mask_path = tmp_dir / mask.filename
        with mask_path.open("wb") as f:
            shutil.copyfileobj(mask.file, f)

    output_dir, _ = run_pipeline(
        image=image_path,
        mask=mask_path,
        job_id=job_id,
        cfg=PipelineConfig(),
    )

    format = format.lower()
    if format == "obj":
        zip_path = package_obj(output_dir)
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=zip_path.name,
        )

    if format == "glb":
        glb_files = list(output_dir.glob("*.glb"))
        if not glb_files:
            raise HTTPException(status_code=404, detail="GLB not found")
        glb_path = glb_files[0]
        return FileResponse(
            glb_path,
            media_type="model/gltf-binary",
            filename=glb_path.name,
        )

    raise HTTPException(status_code=400, detail="Unsupported format")
