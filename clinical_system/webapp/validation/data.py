"""Test data: the lesion-grouped HAM10000 test manifest, the images, and the notebook's predictions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

HAM_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
HAM_TO_CODE = {"akiec": "AK", "bcc": "BCC", "bkl": "BKL", "df": "DF", "mel": "MEL", "nv": "NV", "vasc": "VASC"}
CODES = [HAM_TO_CODE[c] for c in HAM_ORDER]


class MissingData(RuntimeError):
    pass


def data_dir() -> Path:
    return config.VALIDATION_DATA_DIR


def image_folders() -> list[Path]:
    """Every folder under validation_data that holds HAM10000 images (ISIC_*.jpg)."""
    root = data_dir()
    return sorted({p.parent for p in root.rglob("ISIC_*.jpg")}) if root.exists() else []


def check() -> dict:
    """What is present and what is missing (shown on the Validation Lab page)."""
    root = data_dir()
    out = dict(root=str(root), manifest=(root / "test_manifest.csv").exists(),
               predictions=(root / "test_predictions.npz").exists(), image_folders=[str(p) for p in image_folders()])
    missing = []
    if not out["manifest"]:
        missing.append("test_manifest.csv")
    if not out["predictions"]:
        missing.append("test_predictions.npz")
    if out["manifest"]:
        m = manifest()
        out["n_test"] = len(m)
        out["n_missing_images"] = int(m["path"].isna().sum())
        if out["n_missing_images"]:
            missing.append(f"{out['n_missing_images']} HAM10000 test images")
    elif not out["image_folders"]:
        missing.append("HAM10000 images")
    out["missing"] = missing
    return out


def manifest() -> pd.DataFrame:
    """test_manifest.csv with a local 'path' column (NaN if the image is not found) and the app's class code."""
    f = data_dir() / "test_manifest.csv"
    if not f.exists():
        raise MissingData(f"{f} not found")
    m = pd.read_csv(f)
    index = {}
    for folder in image_folders():
        for p in folder.glob("ISIC_*.jpg"):
            index[p.stem] = str(p)
    m["path"] = m["image_id"].map(index)
    m["code"] = m["dx"].map(HAM_TO_CODE)
    return m


def meta(row) -> tuple:
    """(age, sex, site) exactly as the app receives them from a patient record / the upload form."""
    age = None if pd.isna(row["age"]) else float(row["age"])
    sex = row["sex"] if row["sex"] in ("female", "male") else None
    site = str(row["localization"]) if isinstance(row["localization"], str) else "unknown"
    return age, sex, site


def notebook_predictions() -> dict:
    f = data_dir() / "test_predictions.npz"
    if not f.exists():
        raise MissingData(f"{f} not found")
    z = np.load(f, allow_pickle=True)
    return {k: z[k] for k in z.files}


def stratified_sample(m: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Proportional stratified sample by class (largest-remainder allocation, at least 1 per class), so its
    class mix matches the full test set and its accuracy estimates the full-set accuracy."""
    counts = m["dx"].value_counts()
    raw = counts / counts.sum() * n
    alloc = np.floor(raw).astype(int).clip(lower=1)
    for c in (raw - np.floor(raw)).sort_values(ascending=False).index:
        if alloc.sum() >= n:
            break
        alloc[c] += 1
    parts = [m[m["dx"] == c].sample(int(k), random_state=seed) for c, k in alloc.items()]
    return pd.concat(parts).sort_index()
