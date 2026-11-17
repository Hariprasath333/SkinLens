"""Small helpers used by kaggle_stage.sh (kept in Python so they behave the same everywhere)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
from pathlib import Path

import pandas as pd


def roots(values: list[str]) -> list[Path]:
    """Folders given on the command line; each value may hold several, separated by commas."""
    out = [Path(p.strip()) for v in values for p in v.split(",") if p.strip()]
    return [p for p in out if p.exists()]


def run_variant(d: Path) -> str:
    for f in ("run_info.json", "model_card.json"):
        if (d / f).exists():
            return json.loads((d / f).read_text()).get("variant", "") or ""
    return ""


def find_run(search: list[Path], model: str, tag: str, variant: str = "") -> Path | None:
    """Most advanced copy of runs/<model>/<tag>: finished (model_card.json) beats paused (last.pt).
    Only runs of the same experiment variant ('' = normal training) are considered."""
    best, best_rank = None, 0
    for root in search:
        for d in root.rglob(tag):
            if not d.is_dir() or d.parent.name != model or run_variant(d) != variant:
                continue
            rank = 2 if (d / "model_card.json").exists() and (d / "predictions.npz").exists() \
                else 1 if (d / "last.pt").exists() else 0
            if rank > best_rank:
                best, best_rank = d, rank
    return best


def cmd_seed_run(a) -> None:
    """Prints done | resume | new, copying a previous session's run into OUT_DIR when useful."""
    dst = Path(a.out_dir) / a.model / a.tag
    if (dst / "model_card.json").exists():
        print("done"); return
    if (dst / "last.pt").exists():
        print("resume"); return
    src = find_run(roots(a.models_dir), a.model, a.tag, a.variant)
    if src is None or src.resolve() == dst.resolve():
        print("new"); return
    finished = (src / "model_card.json").exists()
    skip = {"last.pt"} | ({"best.pt"} if finished and a.light else set())
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.name not in skip and not f.name.endswith(".tmp"):
            shutil.copy2(f, dst / f.name)
    if not finished:
        shutil.copy2(src / "last.pt", dst / "last.pt")
    print("done" if finished else "resume")
    print(f"(copied {src} -> {dst})", file=sys.stderr)


def cmd_best_model(a) -> None:
    search = roots(a.roots)
    for root in search:
        for f in sorted(root.rglob("best_model.json")):
            print(json.loads(f.read_text())["best_single_model"]); return
    best, score = None, -1.0
    for root in search:
        for f in root.rglob("model_card.json"):
            card = json.loads(f.read_text())
            if card.get("run") == "holdout" and not card.get("variant") and \
                    card.get("best_val_macro_f1", -1) > score:
                best, score = card["model"], card["best_val_macro_f1"]
    if best is None:
        sys.exit("No trained holdout models found. Run the train_<model> stages (and ensemble) first, "
                 "or set BEST_MODEL.")
    print(best)


def cmd_epoch_hint(a) -> None:
    """Typical seconds per epoch for this model, from any earlier run (used before starting a new fold)."""
    secs = []
    for root in roots(a.roots):
        for h in root.rglob("history.csv"):
            if h.parent.parent.name == a.model:
                df = pd.read_csv(h)
                if len(df) and "minutes" in df:
                    secs.append(df["minutes"].iloc[-1] * 60 / len(df))
    print(int(max(secs)) if secs else 0)


def cmd_find_data(a) -> None:
    cands = []
    if a.data_dir:
        cands += [Path(a.data_dir)]
    cands += [Path(a.out_dir) / "data_processed"]
    for c in cands:
        if (c / "manifest.csv").exists():
            print(c); return
        hits = sorted(c.rglob("manifest.csv")) if c.exists() else []
        if hits:
            print(hits[0].parent); return
    kin = Path("/kaggle/input")
    if kin.exists():
        hits = sorted(p for p in kin.rglob("manifest.csv") if p.parent.name == "data_processed")
        if hits:
            print(hits[0].parent); return
    sys.exit("Processed data not found. Run the 'data' stage first, then attach its output to this notebook "
             "(Add Input) or set DATA_DIR to the folder that contains manifest.csv.")


def cmd_pack(a) -> None:
    """Kaggle keeps only ~500 files per output: turn every image folder (and any folder holding
    >= min_files files) into one uncompressed .tar next to it (JPG/PNG are already compressed) and
    delete the folder."""
    root = Path(a.dir)
    for d in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
        files = [f for f in d.iterdir() if f.is_file()]
        only_images = files and all(f.suffix.lower() in {".jpg", ".jpeg", ".png"} for f in files)
        if (only_images or len(files) >= a.min_files) and not any(x.is_dir() for x in d.iterdir()):
            tar_path = d.with_suffix(".tar")
            tmp = tar_path.with_suffix(".tar.tmp")
            with tarfile.open(tmp, "w") as t:
                t.add(d, arcname=d.name)
            tmp.replace(tar_path)
            shutil.rmtree(d)
            print(f"packed {len(files):,} files -> {tar_path}", file=sys.stderr)
    n = sum(1 for f in root.rglob("*") if f.is_file())
    print(n)
    if n > 500:
        print(f"WARNING: {root} still has {n} files (Kaggle shows/saves about 500).", file=sys.stderr)


def cmd_unpack(a) -> None:
    """If `src` holds .tar folders (packed output of an earlier session, read-only on Kaggle), copy its
    small files and extract the tars into `dst` (local disk), then print the folder to use."""
    src, dst = Path(a.src), Path(a.dst)
    tars = sorted(src.glob("*.tar"))
    if not tars or all((src / t.stem).is_dir() for t in tars):
        print(src); return
    marker = dst / ".unpacked_from"
    if marker.exists() and marker.read_text() == str(src.resolve()):
        print(dst); return
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".tar":
            shutil.copy2(f, dst / f.name)
        elif f.is_dir():
            shutil.copytree(f, dst / f.name)
    for t in tars:
        print(f"extracting {t.name} ...", file=sys.stderr)
        with tarfile.open(t) as tf:
            try:
                tf.extractall(dst, filter="data")
            except TypeError:  # Python < 3.12
                tf.extractall(dst)
    marker.write_text(str(src.resolve()))
    print(dst)


def cmd_size(a) -> None:
    gb = sum(f.stat().st_size for f in Path(a.dir).rglob("*") if f.is_file()) / 1e9
    print(f"{gb:.2f}")
    if gb > a.max_gb:
        sys.exit(f"{a.dir} is {gb:.2f} GB, more than the {a.max_gb} GB limit. Use a smaller IMG_SIZE "
                 "(e.g. 320) or lower JPEG quality.")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed-run"); s.add_argument("--model"); s.add_argument("--tag")
    s.add_argument("--models-dir", nargs="*", default=[]); s.add_argument("--out-dir")
    s.add_argument("--light", action="store_true", help="do not copy best.pt of finished runs")
    s.add_argument("--variant", default="", help="experiment variant ('' = normal training)")
    s.set_defaults(fn=cmd_seed_run)
    s = sub.add_parser("best-model"); s.add_argument("--roots", nargs="*", default=[]); s.set_defaults(fn=cmd_best_model)
    s = sub.add_parser("epoch-hint"); s.add_argument("--roots", nargs="*", default=[]); s.add_argument("--model")
    s.set_defaults(fn=cmd_epoch_hint)
    s = sub.add_parser("find-data"); s.add_argument("--data-dir", default=""); s.add_argument("--out-dir")
    s.set_defaults(fn=cmd_find_data)
    s = sub.add_parser("size"); s.add_argument("--dir"); s.add_argument("--max-gb", type=float)
    s.set_defaults(fn=cmd_size)
    s = sub.add_parser("pack"); s.add_argument("--dir"); s.add_argument("--min-files", type=int, default=20)
    s.set_defaults(fn=cmd_pack)
    s = sub.add_parser("unpack"); s.add_argument("--src"); s.add_argument("--dst")
    s.set_defaults(fn=cmd_unpack)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
