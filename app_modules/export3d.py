from pathlib import Path
import shutil
import time
import uuid

import cv2

from fastapi import HTTPException
from fastapi.responses import FileResponse

from image2mesh.pipeline import run_pipeline, PipelineConfig
from image2mesh.export import package_obj

from .image_utils import bgr_to_pil, crop_to_mask, apply_mask_background, refine_mask_for_export, refine_mask_grabcut
from . import tracking


def export_3d_file(selected_id: int, fmt: str = "obj", save_for_preview: bool = False):
    track = next((t for t in tracking.TRACKS if t.track_id == selected_id), None)
    if track is None or track.stylized_crop is None:
        raise HTTPException(status_code=404, detail="No stylized crop for selected track")

    styl_bgr = track.stylized_crop
    mask_u8 = track.last_mask_crop_raw if track.last_mask_crop_raw is not None else track.last_mask_crop
    if mask_u8 is None:
        raise HTTPException(status_code=404, detail="No mask for selected track")

    # Remove reflect padding from the square crop to avoid duplicated content.
    if track.last_crop_info is not None:
        pad_top, pad_left, rh, rw, side, out_size = track.last_crop_info
        if side > 0 and out_size > 0:
            scale = float(out_size) / float(side)
            pt = int(round(pad_top * scale))
            pl = int(round(pad_left * scale))
            rh_s = int(round(rh * scale))
            rw_s = int(round(rw * scale))
            h, w = styl_bgr.shape[:2]
            y1 = max(0, min(h, pt))
            x1 = max(0, min(w, pl))
            y2 = max(0, min(h, y1 + rh_s))
            x2 = max(0, min(w, x1 + rw_s))
            if y2 > y1 and x2 > x1:
                styl_bgr = styl_bgr[y1:y2, x1:x2]
                mask_u8 = mask_u8[y1:y2, x1:x2]

    mask_u8 = refine_mask_for_export(mask_u8, min_area=300, open_px=2, close_px=2)
    mask_u8 = refine_mask_grabcut(styl_bgr, mask_u8, iter_count=2)

    # Tight-crop to the actual mask so the exported mesh only contains the object.
    styl_bgr, mask_u8 = crop_to_mask(styl_bgr, mask_u8, margin=1)
    styl_bgr = apply_mask_background(styl_bgr, mask_u8)

    # Debug: save the exact image fed to the 3D pipeline.
    debug_dir = Path("outputs")
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"export_input_{selected_id}_{int(time.time())}.png"
    cv2.imwrite(str(debug_path), styl_bgr)

    model_fmt = "glb" if fmt == "glb" else "obj"
    try:
        cfg = PipelineConfig(bake_texture=False, foreground_ratio=0.98, model_save_format=model_fmt)
    except TypeError:
        cfg = PipelineConfig(bake_texture=False, foreground_ratio=0.98)
    output_dir, _ = run_pipeline(image=bgr_to_pil(styl_bgr), mask=mask_u8, cfg=cfg)

    if fmt == "obj":
        zip_path = package_obj(output_dir)
        return FileResponse(zip_path, media_type="application/zip", filename=zip_path.name)

    glb_files = list(Path(output_dir).rglob("*.glb"))
    if not glb_files:
        raise HTTPException(status_code=404, detail="GLB not found")
    glb_path = glb_files[0]

    if save_for_preview:
        preview_dir = Path("outputs") / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        counter_file = preview_dir / "next_num.txt"
        try:
            n = int(counter_file.read_text().strip()) if counter_file.exists() else 1
        except (ValueError, OSError):
            n = 1
        download_filename = f"{n}.glb"
        counter_file.write_text(str(n + 1))
        preview_id = uuid.uuid4().hex
        dest = preview_dir / f"{preview_id}.glb"
        shutil.copy2(glb_path, dest)
        return {"preview_id": preview_id, "download_filename": download_filename}

    return FileResponse(glb_path, media_type="model/gltf-binary", filename=glb_path.name)
