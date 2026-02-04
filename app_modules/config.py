import torch

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
