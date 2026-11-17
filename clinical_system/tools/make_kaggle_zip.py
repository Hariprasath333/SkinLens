"""
Build kaggle_upload.zip (next to the clinical_system folder) for uploading to Kaggle as a Dataset.

    python tools/make_kaggle_zip.py

Contains clinical_system/ WITHOUT: .venv, the web-app database and uploads (instance/), datasets
(data/), training outputs (runs/, results/, models/), caches, model weights, images and archives.
Shell scripts are stored with Linux line endings and executable permission.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

SYSTEM = Path(__file__).resolve().parents[1]          # clinical_system
TARGET = SYSTEM.parent / "kaggle_upload.zip"
SKIP_DIRS = {".venv", "venv", "instance", "data", "runs", "results", "models", "out",
             "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".git", "backup"}
SKIP_EXT = {".pyc", ".db", ".sqlite", ".sqlite3", ".pt", ".pth", ".npz", ".zip", ".tmp", ".key",
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".csv", ".log"}


def wanted(p: Path) -> bool:
    rel = p.relative_to(SYSTEM)
    return not any(part in SKIP_DIRS for part in rel.parts[:-1]) and p.suffix.lower() not in SKIP_EXT \
        and p.name not in {"secret.key", "app.db"}


def main() -> None:
    files = sorted(p for p in SYSTEM.rglob("*") if p.is_file() and wanted(p))
    with zipfile.ZipFile(TARGET, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            arc = f"clinical_system/{p.relative_to(SYSTEM).as_posix()}"
            data = p.read_bytes()
            info = zipfile.ZipInfo(arc, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            if p.suffix in {".sh", ".py", ".txt", ".md"}:
                data = data.replace(b"\r\n", b"\n")      # Linux line endings (bash fails on CRLF)
            info.external_attr = ((0o755 if p.suffix == ".sh" else 0o644) << 16)
            z.writestr(info, data)
    mb = TARGET.stat().st_size / 1e6
    print(f"Wrote {TARGET} ({len(files)} files, {mb:.2f} MB)")
    for p in files:
        print("  ", p.relative_to(SYSTEM).as_posix())
    if mb > 50:
        sys.exit("Unexpectedly large - check that no data or weights were included.")


if __name__ == "__main__":
    main()
