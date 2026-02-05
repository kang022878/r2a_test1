import base64
import os
import re
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app_modules import tracking
from app_modules.config import (
    DEVICE,
    CROP_SIZE,
    STEPS,
    STRENGTH,
    CFG,
    SEED,
    CONF_THRESHOLD,
    DETECT_EVERY_N,
    NEGATIVE_BASE,
    PRESERVE_BASE,
    MIN_BOX_SIZE,
    MIN_MASK_PIXELS,
    STYLE_PRESETS,
    STABILIZE_ALPHA,
    TRACK_TTL_SEC,
)
from app_modules.image_utils import (
    decode_upload_to_bgr,
    dilate_and_feather,
    alpha_blend,
    crop_square,
    crop_square_gray,
    paste_back,
    fallback_stylize,
    bgr_to_pil,
)
from app_modules.models import load_models
from app_modules.net import get_local_ip
from app_modules.ui import INDEX_HTML, PREVIEW_HTML
from app_modules.export3d import export_3d_file


def stabilize_stylized(prev_bgr: np.ndarray, curr_bgr: np.ndarray, mask_u8: np.ndarray, alpha_curr: float):
    """Blend prev/curr inside mask. alpha_curr = weight of current frame (e.g. 0.7 = 70% current, 30% prev)."""
    if prev_bgr is None:
        return curr_bgr
    m = (mask_u8.astype(np.float32) / 255.0)[..., None]
    out = curr_bgr.astype(np.float32)
    out = out * (1.0 - m) + (prev_bgr.astype(np.float32) * (1.0 - alpha_curr) + curr_bgr.astype(np.float32) * alpha_curr) * m
    return np.clip(out, 0, 255).astype(np.uint8)

app = FastAPI()
yolo, ghibli_engine = load_models()

WEB_DIST = Path(__file__).resolve().parent / "web" / "dist"
ASSETS_DIR = WEB_DIST / "assets"
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


@app.get("/ip")
def get_ip():
    """같은 Wi‑Fi의 폰에서 접속할 때 쓸 URL (서버 실행 시 --host 0.0.0.0 필요)"""
    port = os.environ.get("PORT", "80")
    ip = get_local_ip()
    return {"ip": ip, "port": port, "url": f"http://{ip}:{port}"}


@app.get("/", response_class=HTMLResponse)
def index():
    index_html = WEB_DIST / "index.html"
    if index_html.exists():
        return FileResponse(index_html)
    return HTMLResponse(
        "<h1>Frontend not built</h1><p>Run: cd web && npm install && npm run build</p>"
    )


@app.get("/camera", response_class=HTMLResponse)
def camera():
    return INDEX_HTML


@app.post("/stylize_auto")
async def stylize_auto(
    image: UploadFile = File(...),
    style: str = Form("ghibli"),
    blend_alpha: float = Form(1.0),
    selected_id: Optional[int] = Form(None),
):
    global yolo, ghibli_engine
    boxes = []
    if style not in STYLE_PRESETS:
        style = "ghibli"

    t_req0 = time.perf_counter()
    img_bytes = await image.read()
    t_read = time.perf_counter()
    bgr = decode_upload_to_bgr(img_bytes)
    t_decode = time.perf_counter()
    now_ts = time.time()
    tracking.prune_tracks(now_ts, keep_id=selected_id)

    tracking.FRAME_COUNT += 1
    run_detect = selected_id is not None or (tracking.FRAME_COUNT % DETECT_EVERY_N) == 1 or not tracking.LAST_DETS
    masks = None
    if run_detect:
        t_det0 = time.perf_counter()
        results = yolo.predict(source=bgr, imgsz=640, conf=CONF_THRESHOLD, verbose=False)
        r = results[0]
        t_det1 = time.perf_counter()
        if r.boxes is None or len(r.boxes) == 0:
            tracking.LAST_DETS = []
            tracking.LAST_IMAGE_WH = (int(bgr.shape[1]), int(bgr.shape[0]))
            return JSONResponse({
                "ok": False,
                "detected": 0,
                "class_name": "",
                "boxes": [],
                "image_w": int(bgr.shape[1]),
                "image_h": int(bgr.shape[0]),
                "timings_ms": {
                    "read": round((t_read - t_req0) * 1000, 1),
                    "decode": round((t_decode - t_read) * 1000, 1),
                    "detect": round((t_det1 - t_det0) * 1000, 1),
                },
            })
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
        boxes = r.boxes

        # Build detection list and keep/update tracks for UX selection.
        selected_track = None
        best_sel_idx = None
        best_sel_iou = 0.0
        if selected_id is not None:
            selected_track = next((t for t in tracking.TRACKS if t.track_id == selected_id), None)
            if selected_track is not None:
                for i in range(len(boxes)):
                    bx1, by1, bx2, by2 = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                    cid = int(boxes.cls[i].cpu().numpy().item())
                    if cid != selected_track.class_id:
                        continue
                    iou = tracking.bbox_iou(selected_track.bbox, (bx1, by1, bx2, by2))
                    if iou > best_sel_iou:
                        best_sel_iou = iou
                        best_sel_idx = i

        det_list = []
        used_tracks = set()
        for i in range(len(boxes)):
            bx1, by1, bx2, by2 = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            cid = int(boxes.cls[i].cpu().numpy().item())
            cname = yolo.names.get(cid, str(cid))
            if selected_track is not None and best_sel_idx is not None and i == best_sel_idx:
                track = selected_track
            else:
                track = tracking.match_track(cid, (bx1, by1, bx2, by2))
            if track is not None and track.track_id in used_tracks:
                track = None
            if track is None:
                track = tracking.Track(
                    track_id=tracking.NEXT_TRACK_ID,
                    class_id=cid,
                    bbox=(bx1, by1, bx2, by2),
                    last_seen=now_ts,
                    stylized_crop=None,
                    last_mask_crop=None,
                    last_mask_crop_raw=None,
                    last_crop_info=None,
                    seed=SEED + tracking.NEXT_TRACK_ID,
                )
                tracking.NEXT_TRACK_ID += 1
                tracking.TRACKS.append(track)
            track.last_seen = now_ts
            track.bbox = (bx1, by1, bx2, by2)
            used_tracks.add(track.track_id)
            det_list.append({
                "x1": bx1,
                "y1": by1,
                "x2": bx2,
                "y2": by2,
                "class_name": cname,
                "track_id": track.track_id,
                "idx": i,
            })
        tracking.LAST_DETS = det_list
        tracking.LAST_IMAGE_WH = (int(bgr.shape[1]), int(bgr.shape[0]))
    else:
        det_list = tracking.LAST_DETS
        if tracking.LAST_IMAGE_WH != (int(bgr.shape[1]), int(bgr.shape[0])):
            tracking.LAST_IMAGE_WH = (int(bgr.shape[1]), int(bgr.shape[0]))

    # Select target detection: either a user-selected track or highest confidence.
    target = None
    for d in det_list:
        if selected_id is not None and d["track_id"] == selected_id:
            target = d
            break
    if selected_id is not None and target is None:
        track_fallback = next((t for t in tracking.TRACKS if t.track_id == selected_id), None)
        if track_fallback is not None and (now_ts - track_fallback.last_seen) <= TRACK_TTL_SEC:
            target = {
                "x1": track_fallback.bbox[0],
                "y1": track_fallback.bbox[1],
                "x2": track_fallback.bbox[2],
                "y2": track_fallback.bbox[3],
                "class_name": yolo.names.get(track_fallback.class_id, str(track_fallback.class_id)),
                "track_id": track_fallback.track_id,
            }
            track_fallback.last_seen = now_ts
        else:
            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            return JSONResponse({
                "ok": False,
                "detected": int(len(det_list)),
                "class_name": "",
                "boxes": det_list,
                "image_w": int(bgr.shape[1]),
                "image_h": int(bgr.shape[0]),
                "image_base64": base64.b64encode(buf.tobytes()).decode("utf-8") if ok else "",
            })
    if selected_id is None:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        t_jpg = time.perf_counter()
        return JSONResponse({
            "ok": False,
            "detected": int(len(det_list)),
            "class_name": "",
            "boxes": det_list,
            "image_w": int(bgr.shape[1]),
            "image_h": int(bgr.shape[0]),
            "image_base64": base64.b64encode(buf.tobytes()).decode("utf-8") if ok else "",
            "timings_ms": {
                "read": round((t_read - t_req0) * 1000, 1),
                "decode": round((t_decode - t_read) * 1000, 1),
                "detect": round((t_det1 - t_det0) * 1000, 1) if run_detect else 0.0,
                "jpeg": round((t_jpg - (t_det1 if run_detect else t_decode)) * 1000, 1),
            },
        })
    if target is None:
        return JSONResponse({
            "ok": False,
            "detected": int(len(det_list)),
            "class_name": "",
            "boxes": det_list,
            "image_w": int(bgr.shape[1]),
            "image_h": int(bgr.shape[0]),
        })

    idx = target["idx"] if "idx" in target else 0
    x1, y1, x2, y2 = target["x1"], target["y1"], target["x2"], target["y2"]
    class_name = target["class_name"]
    track = next((t for t in tracking.TRACKS if t.track_id == target["track_id"]), None)
    class_id = track.class_id if track is not None else None
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < MIN_BOX_SIZE or box_h < MIN_BOX_SIZE:
        return JSONResponse({
            "ok": False,
            "detected": int(len(det_list)),
            "class_name": class_name,
            "boxes": det_list,
            "image_w": int(bgr.shape[1]),
            "image_h": int(bgr.shape[0]),
        })

    if masks is not None:
        mask_full_raw = (masks[idx] > 0.5).astype(np.uint8) * 255
        if mask_full_raw.shape[:2] != bgr.shape[:2]:
            mask_full_raw = cv2.resize(mask_full_raw, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_full = dilate_and_feather(mask_full_raw, dilate_px=2, feather_px=3)
        if int(np.sum(mask_full > 127)) < MIN_MASK_PIXELS:
            mask_full_raw = np.zeros(bgr.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask_full_raw, (x1, y1), (x2, y2), 255, thickness=-1)
            mask_full = mask_full_raw.copy()
    else:
        mask_full_raw = np.zeros(bgr.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask_full_raw, (x1, y1), (x2, y2), 255, thickness=-1)
        mask_full = mask_full_raw.copy()

    crop_bgr, info = crop_square(bgr, x1, y1, x2, y2, out_size=CROP_SIZE)
    if crop_bgr.size == 0:
        return JSONResponse({"ok": False, "detected": int(len(det_list)), "class_name": class_name})
    mask_crop_u8, _ = crop_square_gray(mask_full, x1, y1, x2, y2, out_size=CROP_SIZE)
    mask_crop_raw_u8, _ = crop_square_gray(mask_full_raw, x1, y1, x2, y2, out_size=CROP_SIZE)

    obj_mask = (mask_crop_u8 > 127).astype(np.uint8)
    if np.any(obj_mask):
        obj_pixels = crop_bgr[obj_mask.astype(bool)]
        avg_color = np.mean(obj_pixels, axis=0)
    else:
        avg_color = np.array([127, 127, 127], dtype=np.float32)
    masked_crop = crop_bgr.copy()
    masked_crop[obj_mask == 0] = avg_color

    prompt = STYLE_PRESETS[style]
    preserve = PRESERVE_BASE
    if class_name in ("person", "people"):
        preserve = preserve + ", same face, same hairstyle, same clothing, same body proportions, natural skin tone"
    else:
        preserve = preserve + ", same object, same material, same color palette, same lighting"
    prompt = (preserve + ", " + prompt).strip()
    if not prompt:
        prompt = "cartoon style, preserve details"
    track = next((t for t in tracking.TRACKS if t.track_id == target["track_id"]), None)
    if track is None:
        track = tracking.match_track(class_id, (x1, y1, x2, y2))
    if track is None:
        track = tracking.Track(
            track_id=tracking.NEXT_TRACK_ID,
            class_id=class_id if class_id is not None else -1,
            bbox=(x1, y1, x2, y2),
            last_seen=now_ts,
            stylized_crop=None,
            last_mask_crop=None,
            last_mask_crop_raw=None,
            last_crop_info=None,
            seed=SEED + tracking.NEXT_TRACK_ID,
        )
        tracking.NEXT_TRACK_ID += 1
        tracking.TRACKS.append(track)
    track.last_seen = now_ts
    track.bbox = (x1, y1, x2, y2)

    negative = NEGATIVE_BASE
    if class_name not in ("person", "people"):
        negative = negative + ", human, person, face, body, character"
    else:
        negative = negative + ", different person, different face, different hairstyle, age change, gender change"
    if not negative:
        negative = "text, watermark"

    t_gen0 = time.perf_counter()
    if style in ("pixel", "toon"):
        styl_crop_bgr = fallback_stylize(masked_crop, style)
        t_gen1 = time.perf_counter()
    else:
        try:
            # Ghibli engine: always use engine default prompt/negative_prompt (ghibli style, eyes, etc.)
            jpeg_bytes = ghibli_engine.stylize(
                content_image=bgr_to_pil(masked_crop),
                seed=track.seed,
            )
            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            styl_crop_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if styl_crop_bgr is None:
                raise RuntimeError("Ghibli engine decode failed")
        except RuntimeError:
            styl_crop_bgr = fallback_stylize(masked_crop, style)
        t_gen1 = time.perf_counter()
    styl_crop_bgr = stabilize_stylized(track.stylized_crop, styl_crop_bgr, mask_crop_u8, STABILIZE_ALPHA)
    track.stylized_crop = styl_crop_bgr
    track.last_mask_crop = mask_crop_u8
    track.last_mask_crop_raw = mask_crop_raw_u8
    x1i, y1i, x2i, y2i, pad_top, pad_left, side = info
    track.last_crop_info = (pad_top, pad_left, y2i - y1i, x2i - x1i, side, CROP_SIZE)

    pasted = paste_back(bgr, styl_crop_bgr, info)
    blended = alpha_blend(bgr, pasted, mask_full, alpha=float(blend_alpha))

    ok, buf = cv2.imencode(".jpg", blended, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    t_jpg = time.perf_counter()
    if not ok:
        return JSONResponse({"ok": False, "detected": int(len(det_list)), "class_name": class_name})

    return JSONResponse({
        "ok": True,
        "detected": int(len(det_list)),
        "class_name": class_name,
        "boxes": det_list,
        "image_w": int(bgr.shape[1]),
        "image_h": int(bgr.shape[0]),
        "image_base64": base64.b64encode(buf.tobytes()).decode("utf-8"),
        "timings_ms": {
            "read": round((t_read - t_req0) * 1000, 1),
            "decode": round((t_decode - t_read) * 1000, 1),
            "detect": round((t_det1 - t_det0) * 1000, 1) if run_detect else 0.0,
            "ghibli": round((t_gen1 - t_gen0) * 1000, 1),
            "jpeg": round((t_jpg - t_gen1) * 1000, 1),
        },
    })


@app.post("/export_3d")
async def export_3d(
    selected_id: int = Form(...),
    format: str = Form("obj"),
    preview: Optional[str] = Form(None),
):
    result = export_3d_file(
        selected_id,
        fmt="glb" if preview == "1" else format,
        save_for_preview=(preview == "1"),
    )
    if isinstance(result, dict):
        return JSONResponse(result)
    return result


@app.get("/preview", response_class=HTMLResponse)
def preview_page():
    return PREVIEW_HTML


@app.get("/preview/file/{preview_id}")
def preview_file(preview_id: str):
    if not re.match(r"^[a-f0-9]{32}$", preview_id):
        raise HTTPException(status_code=400, detail="Invalid preview_id")
    path = Path("outputs") / "preview" / f"{preview_id}.glb"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(path, media_type="model/gltf-binary", filename="model.glb")


# Serve built frontend files (e.g., /image1.png, /vite.svg) without shadowing API routes.
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
