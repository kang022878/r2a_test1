from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Optional


OBJ_FILES = ("mesh.obj", "mesh.mtl", "texture.png")


def _resolve_obj_dir(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    # TripoSR saves per-image outputs under output_dir/0, output_dir/1, ...
    if (output_dir / "0").is_dir():
        return output_dir / "0"
    return output_dir


def package_obj(output_dir: Path, zip_name: Optional[str] = None) -> Path:
    output_dir = Path(output_dir)
    obj_dir = _resolve_obj_dir(output_dir)
    zip_path = output_dir / (zip_name or "model_obj.zip")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in OBJ_FILES:
            fpath = obj_dir / fname
            if fpath.exists():
                zf.write(fpath, arcname=fname)
    return zip_path
