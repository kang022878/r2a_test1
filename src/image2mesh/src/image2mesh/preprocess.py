from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image


@dataclass
class PreprocessConfig:
    output_path: Path
    size: int = 768
    input_is_bgr: bool = False
    foreground_ratio: float = 0.85
    background: str = "transparent"  # "transparent" or "white"


def _to_pil_image(image: Union[str, Path, Image.Image, np.ndarray], input_is_bgr: bool) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGBA")
    if isinstance(image, Image.Image):
        return image.convert("RGBA")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if input_is_bgr and arr.shape[-1] >= 3:
            arr = arr[..., :3][:, :, ::-1]
        if arr.shape[-1] == 3:
            alpha = np.full(arr.shape[:2] + (1,), 255, dtype=arr.dtype)
            arr = np.concatenate([arr, alpha], axis=-1)
        return Image.fromarray(arr.astype(np.uint8), mode="RGBA")
    raise TypeError("Unsupported image type")


def _to_mask(mask: Union[str, Path, Image.Image, np.ndarray], size: Tuple[int, int]) -> Image.Image:
    if isinstance(mask, (str, Path)):
        m = Image.open(mask).convert("L")
    elif isinstance(mask, Image.Image):
        m = mask.convert("L")
    elif isinstance(mask, np.ndarray):
        arr = mask
        if arr.ndim == 3:
            arr = arr[..., 0]
        m = Image.fromarray(arr.astype(np.uint8), mode="L")
    else:
        raise TypeError("Unsupported mask type")
    if m.size != size:
        m = m.resize(size, Image.LANCZOS)
    return m


def preprocess(
    image: Union[str, Path, Image.Image, np.ndarray],
    mask: Optional[Union[str, Path, Image.Image, np.ndarray]] = None,
    config: Optional[PreprocessConfig] = None,
) -> Path:
    if config is None:
        config = PreprocessConfig(output_path=Path("outputs/tmp/input_rgba.png"))

    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    img = _to_pil_image(image, config.input_is_bgr)

    if mask is not None:
        alpha = _to_mask(mask, img.size)
        img.putalpha(alpha)

    if config.background != "transparent":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        img = bg

    # Resize to square canvas while keeping aspect ratio
    target = config.size
    scale = min(target / img.width, target / img.height)
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (target, target), (0, 0, 0, 0))
    offset = ((target - new_w) // 2, (target - new_h) // 2)
    canvas.paste(img_resized, offset, img_resized)

    canvas.save(config.output_path)
    return config.output_path
