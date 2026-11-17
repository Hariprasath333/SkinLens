"""System identity for the validation report and the About page: what exactly was tested, on what."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from functools import lru_cache
from importlib import metadata
from pathlib import Path

from .. import config
from ..version import APP_VERSION, RELEASE_DATE


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


@lru_cache(maxsize=4)
def _model_hash(path: str, mtime: float, size: int) -> str:
    return sha256_file(Path(path))


def model_files() -> list[Path]:
    """The weight files of the model the app loads."""
    kind = config.MODEL_KIND
    if kind == "auto":
        kind = "timm" if (config.EXPORTED_MODEL_DIR / "export.json").exists() else "skinlens"
    if kind == "skinlens":
        return [config.SKINLENS_CHECKPOINT]
    if kind == "timm":
        return sorted(config.EXPORTED_MODEL_DIR.glob("*/best.pt"))
    return []


def model_identity() -> dict:
    files = []
    for p in model_files():
        if p.exists():
            st = p.stat()
            files.append(dict(file=p.name, path=str(p), size_mb=round(st.st_size / 2**20, 1),
                              sha256=_model_hash(str(p), st.st_mtime, st.st_size),
                              modified=datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")))
    settings = json.loads(config.MODEL_SETTINGS.read_text(encoding="utf-8"))
    from ..ml.predictor import get_predictor
    pred = get_predictor()
    return dict(name=pred.display_name, kind=config.MODEL_KIND, classes=pred.classes, files=files,
                prior_alpha=(settings.get("prior_correction") or {}).get("alpha"),
                confidence_levels=settings.get("confidence_levels"),
                settings_sha256=hashlib.sha256(config.MODEL_SETTINGS.read_bytes()).hexdigest(),
                architecture=str(config.SKINLENS_ARCHITECTURE.name))


def _ver(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "not installed"


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True, timeout=20)
            if out.stdout.strip():
                return out.stdout.strip().splitlines()[0]
        except Exception:  # noqa: BLE001
            pass
    return platform.processor() or platform.machine()


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(config.PROJECT_DIR), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "not a git repository"
    except Exception:  # noqa: BLE001
        return "git not available"


def environment() -> dict:
    import psutil
    import torch
    return dict(
        os=f"{platform.system()} {platform.release()} ({platform.version()})",
        machine=platform.machine(), cpu=_cpu_name(), cpu_cores=os.cpu_count(),
        ram_gb=round(psutil.virtual_memory().total / 2**30, 1),
        gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU only)"),
        torch_threads=torch.get_num_threads(),
        python=sys.version.split()[0],
        libraries={k: _ver(k) for k in ["torch", "torchvision", "timm", "numpy", "opencv-python-headless",
                                        "opencv-python", "scikit-learn", "scikit-image", "fastapi", "uvicorn",
                                        "SQLAlchemy", "reportlab", "httpx", "Pillow"]},
    )


def identity() -> dict:
    return dict(app_version=APP_VERSION, app_release=RELEASE_DATE,
                recorded=datetime.now().isoformat(timespec="seconds"),
                git_commit=git_commit(), model=model_identity(), environment=environment())
