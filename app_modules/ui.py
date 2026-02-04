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
    <button id="export3d">Export 3D</button>
  </div>
</div>

<script>
const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const statusEl = document.getElementById('status');
const styleSel = document.getElementById('style');
const exportBtn = document.getElementById('export3d');

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
      ? ` | 읽기 ${t.read || 0} 디코드 ${t.decode || 0} 검출 ${t.detect || 0} 스타일 ${t.ghibli || t.sdxl || 0} 인코딩 ${t.jpeg || 0} (ms)`
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

exportBtn.onclick = async () => {
  if (selectedId === null) {
    setStatus("Select an object first");
    return;
  }
  exportBtn.disabled = true;
  setStatus("Exporting 3D...");
  try {
    const fd = new FormData();
    fd.append('selected_id', String(selectedId));
    fd.append('format', 'obj');
    const res = await fetch('/export_3d', { method: 'POST', body: fd });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`HTTP ${res.status}: ${txt.slice(0, 200)}`);
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'model_obj.zip';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    setStatus("Exported 3D");
  } catch (e) {
    console.error(e);
    setStatus("Export error: " + e);
  } finally {
    exportBtn.disabled = false;
  }
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
