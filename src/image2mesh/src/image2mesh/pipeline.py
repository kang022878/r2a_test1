from __future__ import annotations

import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

from .preprocess import PreprocessConfig, preprocess


@dataclass
class PipelineConfig:
    output_root: Path = Path(os.getenv("OUTPUT_DIR", "outputs"))
    device: str = os.getenv("TRIPOSR_DEVICE", "cuda")
    mc_resolution: int = int(os.getenv("TRIPOSR_MC_RESOLUTION", "256"))
    bake_texture: bool = os.getenv("TRIPOSR_BAKE_TEXTURE", "true").lower() == "true"
    model_save_format: str = os.getenv("TRIPOSR_MODEL_FORMAT", "obj")
    input_size: int = int(os.getenv("TRIPOSR_INPUT_SIZE", "768"))
    foreground_ratio: Optional[float] = 0.85
    preprocess_background: str = os.getenv("TRIPOSR_PREPROCESS_BG", "white")
    tripo_dir: Path = Path("vendor/TripoSR")


def _run_triposr(
    input_image: Path,
    output_dir: Path,
    cfg: PipelineConfig,
) -> None:
    run_py = cfg.tripo_dir / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(f"TripoSR not found at {run_py}")

    input_image = input_image.resolve()
    output_dir = output_dir.resolve()

    args = [
        sys.executable,
        "run.py",
        str(input_image),
        "--output-dir",
        str(output_dir),
        "--device",
        cfg.device,
        "--mc-resolution",
        str(cfg.mc_resolution),
        "--model-save-format",
        cfg.model_save_format,
    ]

    if cfg.bake_texture:
        args.append("--bake-texture")

    if cfg.foreground_ratio is not None:
        args += ["--foreground-ratio", str(cfg.foreground_ratio)]

    # We already provide an RGBA input, so skip rembg to avoid extra downloads/timeouts.
    args.append("--no-remove-bg")

    subprocess.run(args, check=True, cwd=str(cfg.tripo_dir))


def run_pipeline(
    image: Union[str, Path],
    mask: Optional[Union[str, Path]] = None,
    job_id: Optional[str] = None,
    cfg: Optional[PipelineConfig] = None,
) -> Tuple[Path, Path]:
    if cfg is None:
        cfg = PipelineConfig()

    job_id = job_id or uuid.uuid4().hex
    output_dir = cfg.output_root / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    # TripoSR writes into output_dir/0 for the first image; pre-create to avoid export errors.
    (output_dir / "0").mkdir(parents=True, exist_ok=True)

    rgba_path = preprocess(
        image=image,
        mask=mask,
        config=PreprocessConfig(
            output_path=output_dir / "input_rgba.png",
            size=cfg.input_size,
            background=cfg.preprocess_background,
            crop_to_mask=True,
            crop_margin=6,
            square_pad=True,
        ),
    )

    _run_triposr(rgba_path, output_dir, cfg)
    return output_dir, rgba_path
