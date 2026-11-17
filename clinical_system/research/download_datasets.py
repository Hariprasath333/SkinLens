"""
Download the public datasets listed in DATASETS.md.

    python research/download_datasets.py --dry-run       # show what would be downloaded and how big it is
    python research/download_datasets.py                 # download everything (asks first)
    python research/download_datasets.py --only isic2019_gt isic2019_meta ham_meta

Downloads are resumable: if one stops halfway, run the command again.
Default target: data/raw/<dataset>/   (about 10 GB in total)

Manual-only datasets (need a sign-up or a terms page): PAD-UFES-20, PH2, Derm7pt.
See DATASETS.md, then put them in data/raw/pad_ufes20, data/raw/ph2, ...
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

import requests

ISIC = "https://isic-archive.s3.amazonaws.com/challenges"   # checked Nov 2026 (old bucket redirects here)
SOURCES = {
    "isic2019_images": dict(url=f"{ISIC}/2019/ISIC_2019_Training_Input.zip", dir="isic2019", unzip=True),
    "isic2019_gt": dict(url=f"{ISIC}/2019/ISIC_2019_Training_GroundTruth.csv", dir="isic2019"),
    "isic2019_meta": dict(url=f"{ISIC}/2019/ISIC_2019_Training_Metadata.csv", dir="isic2019"),
    # HAM10000 lesion ids (which photos show the same lesion), published by ISIC for the 2018 challenge
    "ham_lesion_groups": dict(url=f"{ISIC}/2018/ISIC2018_Task3_Training_LesionGroupings.csv", dir="ham10000"),
    "isic2018_test_images": dict(url=f"{ISIC}/2018/ISIC2018_Task3_Test_Input.zip", dir="isic2018_test", unzip=True),
    "isic2018_test_gt": dict(url=f"{ISIC}/2018/ISIC2018_Task3_Test_GroundTruth.zip", dir="isic2018_test", unzip=True),
}
UA = {"User-Agent": "research-dataset-downloader/1.0"}


def follow_html_redirect(url: str) -> str:
    """Some ISIC links answer with a tiny HTML page that redirects; return the real file URL."""
    try:
        r = requests.get(url, headers=dict(UA, Range="bytes=0-1023"), timeout=60)
        if "text/html" in r.headers.get("Content-Type", ""):
            m = re.search(r'url=([^"\'>\s]+)', r.text)
            if m:
                return m.group(1)
    except requests.RequestException:
        pass
    return url


def remote_size(url: str) -> int | None:
    try:
        r = requests.head(url, headers=UA, allow_redirects=True, timeout=60)
        if r.ok and r.headers.get("Content-Length"):
            return int(r.headers["Content-Length"])
    except requests.RequestException:
        pass
    return None


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = dict(UA, **({"Range": f"bytes={have}-"} if have else {}))
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        if r.status_code == 416:  # already complete
            part.rename(dest)
            return
        r.raise_for_status()
        mode = "ab" if have and r.status_code == 206 else "wb"
        total = int(r.headers.get("Content-Length", 0)) + (have if mode == "ab" else 0)
        done = have if mode == "ab" else 0
        with open(part, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
    print()
    part.rename(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--only", nargs="*", choices=sorted(SOURCES), default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    ap.add_argument("--delete-zip", action="store_true", help="delete each zip after unzipping (saves disk)")
    args = ap.parse_args()

    names = args.only or list(SOURCES)
    plan = []
    for n in names:
        s = SOURCES[n]
        url = follow_html_redirect(s["url"])
        fname = Path(s["url"]).name
        dest = Path(args.root) / s["dir"] / fname
        plan.append((n, url, dest, s.get("unzip", False), remote_size(url)))

    total = 0
    print(f"{'dataset':24} {'size':>10}  destination")
    for n, url, dest, _, size in plan:
        total += size or 0
        state = "already downloaded" if dest.exists() else ""
        print(f"{n:24} {(f'{size / 1e6:,.0f} MB') if size else '?':>10}  {dest} {state}")
    print(f"Total: about {total / 1e9:.1f} GB")
    if args.dry_run:
        return
    if not args.yes and input("Download these files? [y/N] ").strip().lower() != "y":
        sys.exit("Cancelled.")

    for n, url, dest, unzip, _ in plan:
        if not dest.exists():
            print("Downloading", n)
            download(url, dest)
        if unzip and dest.suffix == ".zip":
            marker = dest.with_suffix(".unzipped")
            if not marker.exists():
                print("Unzipping", dest.name)
                with zipfile.ZipFile(dest) as z:
                    z.extractall(dest.parent)
                marker.write_text("ok")
            if args.delete_zip:
                dest.unlink()
                dest.touch()  # empty placeholder so the file is not downloaded again
    (Path(args.root) / "download_log.json").write_text(
        json.dumps({n: dict(url=u, file=str(d)) for n, u, d, _, _ in plan}, indent=2))
    print("Done. Next: python research/build_dataset.py")


if __name__ == "__main__":
    main()
