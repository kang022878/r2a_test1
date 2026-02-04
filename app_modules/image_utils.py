import io
import cv2
import numpy as np
from PIL import Image


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
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        m = cv2.dilate(m, k, iterations=1)
    if feather_px > 0:
        ksize = feather_px * 2 + 1
        m = cv2.GaussianBlur(m, (ksize, ksize), 0)
    return m


def alpha_blend(original_bgr: np.ndarray, stylized_bgr: np.ndarray, mask_u8: np.ndarray, alpha: float) -> np.ndarray:
    m = (mask_u8.astype(np.float32) / 255.0)[..., None]
    m = np.clip(m * alpha, 0.0, 1.0)
    out = original_bgr.astype(np.float32) * (1.0 - m) + stylized_bgr.astype(np.float32) * m
    return np.clip(out, 0, 255).astype(np.uint8)


def crop_square(img_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int, out_size: int):
    h, w = img_bgr.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    roi = img_bgr[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if rh <= 0 or rw <= 0:
        raise ValueError("Invalid crop")

    side = max(rh, rw)
    pad_top = (side - rh) // 2
    pad_left = (side - rw) // 2
    pad_bottom = side - rh - pad_top
    pad_right = side - rw - pad_left

    roi_pad = cv2.copyMakeBorder(roi, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
    roi_resized = cv2.resize(roi_pad, (out_size, out_size), interpolation=cv2.INTER_LANCZOS4)

    info = (x1, y1, x2, y2, pad_top, pad_left, side)
    return roi_resized, info


def crop_square_gray(img_u8: np.ndarray, x1: int, y1: int, x2: int, y2: int, out_size: int):
    h, w = img_u8.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    roi = img_u8[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if rh <= 0 or rw <= 0:
        raise ValueError("Invalid crop")

    side = max(rh, rw)
    pad_top = (side - rh) // 2
    pad_left = (side - rw) // 2
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
    crop = styl_sq[pad_top:pad_top + rh, pad_left:pad_left + rw]
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


def crop_to_mask(img_bgr: np.ndarray, mask_u8: np.ndarray, margin: int = 4):
    if mask_u8 is None:
        return img_bgr, mask_u8
    if mask_u8.shape[:2] != img_bgr.shape[:2]:
        mask_u8 = cv2.resize(mask_u8, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(mask_u8 > 127)
    if len(xs) == 0 or len(ys) == 0:
        return img_bgr, mask_u8
    x1 = max(int(xs.min()) - margin, 0)
    y1 = max(int(ys.min()) - margin, 0)
    x2 = min(int(xs.max()) + margin + 1, img_bgr.shape[1])
    y2 = min(int(ys.max()) + margin + 1, img_bgr.shape[0])
    return img_bgr[y1:y2, x1:x2], mask_u8[y1:y2, x1:x2]


def apply_mask_background(img_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    if mask_u8 is None:
        return img_bgr
    m = mask_u8 > 127
    if not np.any(m):
        return img_bgr
    obj_pixels = img_bgr[m]
    avg_color = np.mean(obj_pixels, axis=0).astype(np.uint8)
    out = img_bgr.copy()
    out[~m] = avg_color
    return out
