import torch
from ultralytics import YOLO

from .config import YOLO_MODEL, DEVICE


def load_models():
    print("[init] YOLOv8-seg loading...")
    yolo = YOLO(YOLO_MODEL)

    print("[init] Ghibli Diffusion engine loading...")
    from engines.ghibli_diffusion_engine import GhibliDiffusionEngine
    from engines.ghibli_config import GHIBLI_DEFAULT_KWARGS

    ghibli_engine = GhibliDiffusionEngine(**GHIBLI_DEFAULT_KWARGS)
    ghibli_engine.load_models()

    return yolo, ghibli_engine
