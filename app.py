import base64
import io
import os
import socket
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from ultralytics import YOLO
from diffusers import StableDiffusionXLImg2ImgPipeline

YOLO_MODEL = "yolov8n-seg.pt"
SDXL_TURBO_MODEL = "stabilityai/sdxl-turbo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CROP_SIZE = 384
STEPS = 2
STRENGTH = 0.35
CFG = 1.0
SEED = 123
IOU_THRESHOLD = 0.35
TRACK_TTL_SEC = 2.5
STABILIZE_ALPHA = 0.7
CONF_THRESHOLD = 0.15
DETECT_EVERY_N = 4
NEGATIVE_BASE = "text, watermark, logo, extra limbs, deformed"
PRESERVE_BASE = "preserve identity, preserve colors, preserve shape, preserve textures, minimal changes, keep details, same composition"
MIN_BOX_SIZE = 4
MIN_MASK_PIXELS = 30

STYLE_PRESETS = {
    "ghibli": "hand-painted animation style, warm pastel colors, soft shading, detailed illustration",
    "pixel":  "pixel art style, 16-bit game sprite, limited color palette, crisp edges",
    "toon":   "clean toon shading, bold outlines, vibrant colors, comic style",
}


def decode_upload_to_bgr(file_bytes: bytes) -> np.ndarray:
    pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    rgb = np.array(pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def dilate_and_feather(mask_u8: np.ndarray, dilate_px: int = 6, feather_px: int = 5) -> np.ndarray:
    m = (mask_u8 > 127).astype(np.uint8) * 255
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px*2+1, dilate_px*2+1))
        m = cv2.dilate(m, k, iterations=1)
    if feather_px > 0:
        ksize = feather_px*2 + 1
        m = cv2.GaussianBlur(m, (ksize, ksize), 0)
    return m

def alpha_blend(original_bgr: np.ndarray, stylized_bgr: np.ndarray, mask_u8: np.ndarray, alpha: float) -> np.ndarray:
    m = (mask_u8.astype(np.float32) / 255.0)[..., None]
    m = np.clip(m * alpha, 0.0, 1.0)
    out = original_bgr.astype(np.float32) * (1.0 - m) + stylized_bgr.astype(np.float32) * m
    return np.clip(out, 0, 255).astype(np.uint8)

def crop_square(img_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int, out_size: int):
    h, w = img_bgr.shape[:2]
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h, y2))

    roi = img_bgr[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if rh <= 0 or rw <= 0:
        raise ValueError("Invalid crop")

    side = max(rh, rw)
    pad_top = (side - rh)//2
    pad_left = (side - rw)//2
    pad_bottom = side - rh - pad_top
    pad_right = side - rw - pad_left

    roi_pad = cv2.copyMakeBorder(roi, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
    roi_resized = cv2.resize(roi_pad, (out_size, out_size), interpolation=cv2.INTER_LANCZOS4)

    info = (x1, y1, x2, y2, pad_top, pad_left, side)
    return roi_resized, info

def crop_square_gray(img_u8: np.ndarray, x1: int, y1: int, x2: int, y2: int, out_size: int):
    h, w = img_u8.shape[:2]
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h, y2))

    roi = img_u8[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if rh <= 0 or rw <= 0:
        raise ValueError("Invalid crop")

    side = max(rh, rw)
    pad_top = (side - rh)//2
    pad_left = (side - rw)//2
    pad_bottom = side - rh - pad_top
    pad_right = side - rw - pad_left

    roi_pad = cv2.copyMakeBorder(roi, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
    roi_resized = cv2.resize(roi_pad, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

    info = (x1, y1, x2, y2, pad_top, pad_left, side)
    return roi_resized, info

def paste_back(original_bgr: np.ndarray, styl_square_bgr: np.ndarray, info):
    x1, y1, x2, y2, pad_top, pad_left, side = info
    styl_sq = cv2.resize(styl_square_bgr, (side, side), interpolation=cv2.INTER_LANCZOS4)
    rh = y2 - y1
    rw = x2 - x1
    crop = styl_sq[pad_top:pad_top+rh, pad_left:pad_left+rw]
    out = original_bgr.copy()
    out[y1:y2, x1:x2] = crop
    return out

def pixelate_bgr(img_bgr: np.ndarray, scale: float = 0.12) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    sw = max(1, int(w * scale))
    sh = max(1, int(h * scale))
    small = cv2.resize(img_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def toon_bgr(img_bgr: np.ndarray) -> np.ndarray:
    color = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 120)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edges_inv = cv2.bitwise_not(edges)
    edges_inv = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(color, edges_inv)

def fallback_stylize(img_bgr: np.ndarray, style: str) -> np.ndarray:
    if style == "pixel":
        return pixelate_bgr(img_bgr, scale=0.12)
    return toon_bgr(img_bgr)

def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union

def stabilize_stylized(prev_bgr: np.ndarray, curr_bgr: np.ndarray, mask_u8: np.ndarray, alpha_prev: float):
    if prev_bgr is None:
        return curr_bgr
    m = (mask_u8.astype(np.float32) / 255.0)[..., None]
    out = curr_bgr.astype(np.float32)
    out = out * (1.0 - m) + (prev_bgr.astype(np.float32) * alpha_prev + curr_bgr.astype(np.float32) * (1.0 - alpha_prev)) * m
    return np.clip(out, 0, 255).astype(np.uint8)

@dataclass
class Track:
    track_id: int
    class_id: int
    bbox: tuple
    last_seen: float
    stylized_crop: np.ndarray
    seed: int

TRACKS = []
NEXT_TRACK_ID = 1
FRAME_COUNT = 0
LAST_DETS = []
LAST_IMAGE_WH = (0, 0)

def match_track(class_id: int, bbox: tuple):
    best = None
    best_iou = 0.0
    for t in TRACKS:
        if t.class_id != class_id:
            continue
        iou = bbox_iou(t.bbox, bbox)
        if iou > best_iou:
            best_iou = iou
            best = t
    if best_iou >= IOU_THRESHOLD:
        return best
    return None

def prune_tracks(now_ts: float, keep_id: Optional[int] = None):
    global TRACKS
    kept = []
    for t in TRACKS:
        if keep_id is not None and t.track_id == keep_id:
            kept.append(t)
        elif (now_ts - t.last_seen) <= TRACK_TTL_SEC:
            kept.append(t)
    TRACKS = kept


app = FastAPI()

print("[init] YOLOv8-seg loading...")
yolo = YOLO(YOLO_MODEL)

print("[init] SDXL Turbo loading...")
pipe_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
pipe_kwargs = {"torch_dtype": pipe_dtype}
if DEVICE == "cuda":
    pipe_kwargs["variant"] = "fp16"
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    SDXL_TURBO_MODEL,
    **pipe_kwargs,
).to(DEVICE)
pipe.set_progress_bar_config(disable=True)


INDEX_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Live Stylize MVP</title>
  <style>
    body { margin:0; font-family: system-ui, -apple-system, sans-serif; background:#000; color:#fff; }
    #wrap { position: relative; width:100vw; height:100vh; overflow:hidden; }
    video, img { position:absolute; top:0; left:0; width:100%; height:100%; object-fit:cover; }
    #boxes { position:absolute; inset:0; pointer-events:none; }
    .box { position:absolute; border:2px solid rgba(255,255,255,0.8); border-radius:6px; box-shadow: inset 0 0 0 1px rgba(0,0,0,0.5); }
    .box.selected { border-color:#00ffcc; }
    #ui { position:absolute; left:12px; right:12px; bottom:12px; display:flex; gap:10px; align-items:center; }
    button, select { padding:12px 14px; font-size:16px; border-radius:12px; border:none; }
    button { background:#fff; color:#000; font-weight:600; }
    #status { position:absolute; top:12px; left:12px; background:rgba(0,0,0,0.6); padding:8px 10px; border-radius:10px; font-size:13px; }
  </style>
</head>
<body>
<div id="wrap">
  <video id="video" autoplay playsinline></video>
  <img id="overlay" />
  <div id="boxes"></div>
  <div id="status">Idle</div>
  <div id="ui">
    <select id="style">
      <option value="ghibli">Ghibli-ish</option>
      <option value="pixel">Pixel</option>
      <option value="toon">Toon</option>
    </select>
    <button id="start">Start</button>
    <button id="stop">Stop</button>
  </div>
</div>

<script>
const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const statusEl = document.getElementById('status');
const styleSel = document.getElementById('style');

const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

let running = false;
let inflight = false;

const INTERVAL_MS = 33;  // ~30fps (1000/30)
const SEND_W = 512;       // 업로드 해상도(낮출수록 빨라짐)

let selectedId = null;
let lastBoxes = [];
let lastImage = { w: 0, h: 0 };

function setStatus(s){ statusEl.textContent = s; }

async function setupCamera(){
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'environment' },
    audio: false
  });
  video.srcObject = stream;
  await video.play();
  setStatus("Camera ready");
}

function frameToBlob(){
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return null;

  const scale = SEND_W / vw;
  const sw = SEND_W;
  const sh = Math.round(vh * scale);

  canvas.width = sw;
  canvas.height = sh;
  ctx.drawImage(video, 0, 0, sw, sh);

  return new Promise(resolve => {
    canvas.toBlob(b => resolve(b), 'image/jpeg', 0.6);
  });
}

function imageFitMetrics(){
  const rect = overlay.getBoundingClientRect();
  const vw = rect.width, vh = rect.height;
  const iw = lastImage.w, ih = lastImage.h;
  if (!vw || !vh || !iw || !ih) return null;
  const scale = Math.max(vw / iw, vh / ih);
  const dispW = iw * scale;
  const dispH = ih * scale;
  const offsetX = (vw - dispW) / 2;
  const offsetY = (vh - dispH) / 2;
  return { scale, offsetX, offsetY };
}

function renderBoxes(){
  const boxLayer = document.getElementById('boxes');
  boxLayer.innerHTML = '';
  const fit = imageFitMetrics();
  if (!fit) return;
  for (const b of lastBoxes) {
    const x = b.x1 * fit.scale + fit.offsetX;
    const y = b.y1 * fit.scale + fit.offsetY;
    const w = (b.x2 - b.x1) * fit.scale;
    const h = (b.y2 - b.y1) * fit.scale;
    const el = document.createElement('div');
    el.className = 'box' + (b.track_id === selectedId ? ' selected' : '');
    el.style.left = `${x}px`;
    el.style.top = `${y}px`;
    el.style.width = `${w}px`;
    el.style.height = `${h}px`;
    boxLayer.appendChild(el);
  }
}

async function tick(){
  if (!running) return;
  if (inflight) { setTimeout(tick, INTERVAL_MS); return; }

  inflight = true;
  try {
    const blob = await frameToBlob();
    if (!blob) { inflight=false; setTimeout(tick, INTERVAL_MS); return; }

    const fd = new FormData();
    fd.append('image', blob, 'frame.jpg');
    fd.append('style', styleSel.value);
    fd.append('blend_alpha', '1.0');
    if (selectedId !== null) fd.append('selected_id', String(selectedId));

    const t0 = performance.now();
    const res = await fetch('/stylize_auto', { method:'POST', body: fd });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`HTTP ${res.status}: ${txt.slice(0, 120)}`);
    }
    const data = await res.json();
    const dt = Math.round(performance.now() - t0);

    if (data.image_base64) {
      overlay.src = 'data:image/jpeg;base64,' + data.image_base64;
    }
    lastBoxes = data.boxes || [];
    lastImage = { w: data.image_w || 0, h: data.image_h || 0 };
    renderBoxes();

    const t = data.timings_ms || {};
    const timingStr = Object.keys(t).length
      ? ` | 읽기 ${t.read || 0} 디코드 ${t.decode || 0} 검출 ${t.detect || 0} 스타일 ${t.sdxl || 0} 인코딩 ${t.jpeg || 0} (ms)`
      : '';
    if (data.ok) {
      setStatus(`OK | ${dt}ms | det=${data.detected} | ${data.class_name || ''}${timingStr}`);
    } else if (selectedId === null && (data.detected || 0) > 0) {
      setStatus(`Tap a box | ${dt}ms | det=${data.detected}${timingStr}`);
    } else {
      setStatus(`No object | ${dt}ms | ${data.class_name || ''}${timingStr}`);
    }
  } catch (e) {
    console.error(e);
    setStatus('Error: ' + e);
  } finally {
    inflight = false;
    setTimeout(tick, INTERVAL_MS);
  }
}

document.getElementById('start').onclick = () => {
  running = true;
  setStatus("Running...");
  tick();
};
document.getElementById('stop').onclick = () => {
  running = false;
  setStatus("Stopped");
};

setupCamera().catch(e => setStatus("Camera error: " + e));
window.addEventListener('resize', renderBoxes);

overlay.addEventListener('click', (ev) => {
  const fit = imageFitMetrics();
  if (!fit || !lastBoxes.length) return;
  const rect = overlay.getBoundingClientRect();
  const xView = ev.clientX - rect.left;
  const yView = ev.clientY - rect.top;
  const xImg = (xView - fit.offsetX) / fit.scale;
  const yImg = (yView - fit.offsetY) / fit.scale;

  let hit = null;
  let hitArea = Infinity;
  for (const b of lastBoxes) {
    if (xImg >= b.x1 && xImg <= b.x2 && yImg >= b.y1 && yImg <= b.y2) {
      const area = (b.x2 - b.x1) * (b.y2 - b.y1);
      if (area < hitArea) {
        hitArea = area;
        hit = b;
      }
    }
  }
  if (!hit) return;
  if (selectedId === hit.track_id) {
    selectedId = null;
  } else {
    selectedId = hit.track_id;
  }
  renderBoxes();
});
</script>
</body>
</html>
"""

def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@app.get("/ip")
def get_ip():
    """같은 Wi‑Fi의 폰에서 접속할 때 쓸 URL (서버 실행 시 --host 0.0.0.0 필요)"""
    port = os.environ.get("PORT", "80")
    ip = _get_local_ip()
    return {"ip": ip, "port": port, "url": f"http://{ip}:{port}"}


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.post("/stylize_auto")
async def stylize_auto(
    image: UploadFile = File(...),
    style: str = Form("ghibli"),
    blend_alpha: float = Form(1.0),
    selected_id: Optional[int] = Form(None),
):
    global NEXT_TRACK_ID
    global FRAME_COUNT, LAST_DETS, LAST_IMAGE_WH
    boxes = []
    if style not in STYLE_PRESETS:
        style = "ghibli"

    t_req0 = time.perf_counter()
    img_bytes = await image.read()
    t_read = time.perf_counter()
    bgr = decode_upload_to_bgr(img_bytes)
    t_decode = time.perf_counter()
    now_ts = time.time()
    prune_tracks(now_ts, keep_id=selected_id)

    FRAME_COUNT += 1
    run_detect = selected_id is not None or (FRAME_COUNT % DETECT_EVERY_N) == 1 or not LAST_DETS
    masks = None
    if run_detect:
        t_det0 = time.perf_counter()
        results = yolo.predict(source=bgr, imgsz=640, conf=CONF_THRESHOLD, verbose=False)
        r = results[0]
        t_det1 = time.perf_counter()
        if r.boxes is None or len(r.boxes) == 0:
            LAST_DETS = []
            LAST_IMAGE_WH = (int(bgr.shape[1]), int(bgr.shape[0]))
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
            selected_track = next((t for t in TRACKS if t.track_id == selected_id), None)
            if selected_track is not None:
                for i in range(len(boxes)):
                    bx1, by1, bx2, by2 = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                    cid = int(boxes.cls[i].cpu().numpy().item())
                    if cid != selected_track.class_id:
                        continue
                    iou = bbox_iou(selected_track.bbox, (bx1, by1, bx2, by2))
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
                track = match_track(cid, (bx1, by1, bx2, by2))
            if track is not None and track.track_id in used_tracks:
                track = None
            if track is None:
                track = Track(
                    track_id=NEXT_TRACK_ID,
                    class_id=cid,
                    bbox=(bx1, by1, bx2, by2),
                    last_seen=now_ts,
                    stylized_crop=None,
                    seed=SEED + NEXT_TRACK_ID,
                )
                NEXT_TRACK_ID += 1
                TRACKS.append(track)
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
        LAST_DETS = det_list
        LAST_IMAGE_WH = (int(bgr.shape[1]), int(bgr.shape[0]))
    else:
        det_list = LAST_DETS
        if LAST_IMAGE_WH != (int(bgr.shape[1]), int(bgr.shape[0])):
            LAST_IMAGE_WH = (int(bgr.shape[1]), int(bgr.shape[0]))

    # Select target detection: either a user-selected track or highest confidence.
    target = None
    for d in det_list:
        if selected_id is not None and d["track_id"] == selected_id:
            target = d
            break
    if selected_id is not None and target is None:
        track_fallback = next((t for t in TRACKS if t.track_id == selected_id), None)
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
    track = next((t for t in TRACKS if t.track_id == target["track_id"]), None)
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
        mask_full = (masks[idx] > 0.5).astype(np.uint8) * 255
        if mask_full.shape[:2] != bgr.shape[:2]:
            mask_full = cv2.resize(mask_full, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_full = dilate_and_feather(mask_full, dilate_px=2, feather_px=3)
        if int(np.sum(mask_full > 127)) < MIN_MASK_PIXELS:
            mask_full = np.zeros(bgr.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask_full, (x1, y1), (x2, y2), 255, thickness=-1)
    else:
        mask_full = np.zeros(bgr.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask_full, (x1, y1), (x2, y2), 255, thickness=-1)

    crop_bgr, info = crop_square(bgr, x1, y1, x2, y2, out_size=CROP_SIZE)
    if crop_bgr.size == 0:
        return JSONResponse({"ok": False, "detected": int(len(det_list)), "class_name": class_name})
    mask_crop_u8, _ = crop_square_gray(mask_full, x1, y1, x2, y2, out_size=CROP_SIZE)

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
    track = next((t for t in TRACKS if t.track_id == target["track_id"]), None)
    if track is None:
        track = match_track(class_id, (x1, y1, x2, y2))
    if track is None:
        track = Track(
            track_id=NEXT_TRACK_ID,
            class_id=class_id if class_id is not None else -1,
            bbox=(x1, y1, x2, y2),
            last_seen=now_ts,
            stylized_crop=None,
            seed=SEED + NEXT_TRACK_ID,
        )
        NEXT_TRACK_ID += 1
        TRACKS.append(track)
    track.last_seen = now_ts
    track.bbox = (x1, y1, x2, y2)

    g = torch.Generator(device=DEVICE).manual_seed(track.seed)
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
            out_pil = pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=bgr_to_pil(masked_crop),
                strength=float(STRENGTH),
                guidance_scale=float(CFG),
                num_inference_steps=int(STEPS),
                num_images_per_prompt=1,
                generator=g,
            ).images[0]
            styl_crop_bgr = cv2.cvtColor(np.array(out_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        except RuntimeError:
            styl_crop_bgr = fallback_stylize(masked_crop, style)
        t_gen1 = time.perf_counter()
    styl_crop_bgr = stabilize_stylized(track.stylized_crop, styl_crop_bgr, mask_crop_u8, STABILIZE_ALPHA)
    track.stylized_crop = styl_crop_bgr

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
            "sdxl": round((t_gen1 - t_gen0) * 1000, 1),
            "jpeg": round((t_jpg - t_gen1) * 1000, 1),
        },
    })
