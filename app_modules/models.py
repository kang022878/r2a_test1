import torch
from ultralytics import YOLO
from diffusers import StableDiffusionXLImg2ImgPipeline

from .config import YOLO_MODEL, SDXL_TURBO_MODEL, DEVICE


def load_models():
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
    return yolo, pipe
