import base64
import io
import time

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
STRENGTH = 0.55
CFG = 1.0
SEED = 123

STYLE_PRESETS = {
    "ghibli": "hand-painted animation style, warm pastel colors, soft shading, detailed food illustration",
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

def paste_back(original_bgr: np.ndarray, styl_square_bgr: np.ndarray, info):
    x1, y1, x2, y2, pad_top, pad_left, side = info
    styl_sq = cv2.resize(styl_square_bgr, (side, side), interpolation=cv2.INTER_LANCZOS4)
    rh = y2 - y1
    rw = x2 - x1
    crop = styl_sq[pad_top:pad_top+rh, pad_left:pad_left+rw]
    out = original_bgr.copy()
    out[y1:y2, x1:x2] = crop
    return out


app = FastAPI()

print("[init] YOLOv8-seg loading...")
yolo = YOLO(YOLO_MODEL)

print("[init] SDXL Turbo loading...")
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
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

const INTERVAL_MS = 350;  // 2~4fps
const SEND_W = 640;       // 업로드 해상도(낮출수록 빨라짐)

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
    canvas.toBlob(b => resolve(b), 'image/jpeg', 0.7);
  });
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

    const t0 = performance.now();
    const res = await fetch('/stylize_auto', { method:'POST', body: fd });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`HTTP ${res.status}: ${txt.slice(0, 120)}`);
    }
    const data = await res.json();
    const dt = Math.round(performance.now() - t0);

    if (data.ok) {
      overlay.src = 'data:image/jpeg;base64,' + data.image_base64;
      setStatus(`OK | ${dt}ms | det=${data.detected} | ${data.class_name || ''}`);
    } else {
      setStatus(`No object | ${dt}ms | ${data.class_name || ''}`);
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
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.post("/stylize_auto")
async def stylize_auto(
    image: UploadFile = File(...),
    style: str = Form("ghibli"),
    blend_alpha: float = Form(1.0),
):
    if style not in STYLE_PRESETS:
        style = "ghibli"

    img_bytes = await image.read()
    bgr = decode_upload_to_bgr(img_bytes)

    results = yolo.predict(source=bgr, imgsz=640, conf=0.25, verbose=False)
    r = results[0]
    if r.masks is None or r.boxes is None or len(r.boxes) == 0:
        return JSONResponse({"ok": False, "detected": 0, "class_name": ""})

    masks = r.masks.data.cpu().numpy()
    boxes = r.boxes

    confs = boxes.conf.cpu().numpy()
    idx = int(np.argmax(confs))

    x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy().astype(int).tolist()
    class_id = int(boxes.cls[idx].cpu().numpy().item())
    class_name = yolo.names.get(class_id, str(class_id))
    mask_full = (masks[idx] > 0.5).astype(np.uint8) * 255
    mask_full = dilate_and_feather(mask_full, dilate_px=6, feather_px=5)
    if mask_full.shape[:2] != bgr.shape[:2]:
        mask_full = cv2.resize(mask_full, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    crop_bgr, info = crop_square(bgr, x1, y1, x2, y2, out_size=CROP_SIZE)

    prompt = STYLE_PRESETS[style]
    g = torch.Generator(device=DEVICE).manual_seed(SEED)

    out_pil = pipe(
        prompt=prompt,
        image=bgr_to_pil(crop_bgr),
        strength=float(STRENGTH),
        guidance_scale=float(CFG),
        num_inference_steps=int(STEPS),
        generator=g,
    ).images[0]

    styl_crop_bgr = cv2.cvtColor(np.array(out_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    pasted = paste_back(bgr, styl_crop_bgr, info)
    blended = alpha_blend(bgr, pasted, mask_full, alpha=float(blend_alpha))

    ok, buf = cv2.imencode(".jpg", blended, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return JSONResponse({"ok": False, "detected": int(len(boxes)), "class_name": class_name})

    return JSONResponse({
        "ok": True,
        "detected": int(len(boxes)),
        "class_name": class_name,
        "image_base64": base64.b64encode(buf.tobytes()).decode("utf-8"),
    })
