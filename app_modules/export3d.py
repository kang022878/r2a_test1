from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse

from image2mesh.pipeline import run_pipeline, PipelineConfig
from image2mesh.export import package_obj

from .image_utils import bgr_to_pil, crop_to_mask, apply_mask_background
from . import tracking


def export_3d_file(selected_id: int, fmt: str = "obj") -> FileResponse:
    track = next((t for t in tracking.TRACKS if t.track_id == selected_id), None)
    if track is None or track.stylized_crop is None:
        raise HTTPException(status_code=404, detail="No stylized crop for selected track")

    styl_bgr = track.stylized_crop
    mask_u8 = track.last_mask_crop
    if mask_u8 is None:
        raise HTTPException(status_code=404, detail="No mask for selected track")

    # Tight-crop to the actual mask so the exported mesh only contains the object.
    styl_bgr, mask_u8 = crop_to_mask(styl_bgr, mask_u8, margin=2)
    styl_bgr = apply_mask_background(styl_bgr, mask_u8)

    cfg = PipelineConfig(bake_texture=False)
    output_dir, _ = run_pipeline(image=bgr_to_pil(styl_bgr), mask=mask_u8, cfg=cfg)

    if fmt == "obj":
        zip_path = package_obj(output_dir)
        return FileResponse(zip_path, media_type="application/zip", filename=zip_path.name)

    glb_files = list(Path(output_dir).rglob("*.glb"))
    if not glb_files:
        raise HTTPException(status_code=404, detail="GLB not found")
    return FileResponse(glb_files[0], media_type="model/gltf-binary", filename=glb_files[0].name)
