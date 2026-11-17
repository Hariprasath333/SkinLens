"""
Merge, clean and split the downloaded datasets into one leakage-free dataset.

    python research/build_dataset.py                      # uses data/raw -> data/merged
    python research/build_dataset.py --cache-size 512     # also writes resized copies (faster training)
    python research/build_dataset.py --extra ph2:data/raw/ph2_sorted:external

Steps (each is counted in data/merged/cleaning_report.md):
 1. load every source and map labels to the 8 classes in common.CLASSES
 2. remove corrupted / unreadable / tiny images
 3. find duplicates: same ISIC id, same file hash (MD5), near-identical picture (perceptual-hash
    candidate CONFIRMED by centre-crop similarity + colour; see audit/audit_report.md)
 4. duplicates with conflicting labels -> resolved (AK vs SCC: finer ISIC 2019 label) or removed
 5. group images by lesion / patient AND by duplicate cluster; a group never crosses splits
 6. 70 / 15 / 15 stratified group split + 5 cross-validation folds on train+val
 7. external test sets are kept apart and cleaned against everything internal

--extra name:folder:role  adds a dataset arranged as folder/<CLASS_CODE>/*.jpg
(role = internal or external). Use it for PH2, Derm7pt, or your own hospital data.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, HAM_MAP, ISIC2019_COLS, PAD_MAP  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SOURCE_PRIORITY = {"isic2019": 0, "ham10000": 1}  # lower = preferred when keeping one copy


# --------------------------------------------------------------------------- loaders
def index_images(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT}


def norm_isic_id(s: str) -> str:
    return str(s).replace("_downsampled", "").strip()


def load_isic2019(raw: Path) -> pd.DataFrame:
    d = raw / "isic2019"
    gt_path = next(d.rglob("ISIC_2019_Training_GroundTruth.csv"), None)
    if gt_path is None:
        return pd.DataFrame()
    gt = pd.read_csv(gt_path)
    meta_path = next(d.rglob("ISIC_2019_Training_Metadata.csv"), None)
    meta = pd.read_csv(meta_path) if meta_path else pd.DataFrame({"image": gt["image"]})
    df = gt.merge(meta, on="image", how="left")
    imgs = index_images(d)
    rows = []
    for _, r in df.iterrows():
        lab = next((c for c in ISIC2019_COLS if r.get(c, 0) == 1), None)  # None = UNK
        rows.append(dict(source="isic2019", src_id=r["image"], path=imgs.get(r["image"]), label=lab,
                         group_key=r.get("lesion_id"), age=r.get("age_approx"), sex=r.get("sex"),
                         site=r.get("anatom_site_general"), role="internal"))
    return pd.DataFrame(rows)


def load_ham(raw: Path, ham_images: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (ham metadata for lesion ids, ham image rows if a HAM image folder is given)."""
    d = raw / "ham10000"
    p = (next(d.rglob("HAM10000_metadata*.csv"), None) or next(d.rglob("*LesionGroupings*.csv"), None)) \
        if d.exists() else None
    if p is None:
        return pd.DataFrame(), pd.DataFrame()
    meta = pd.read_csv(p).rename(columns={"image": "image_id"})  # ISIC LesionGroupings file uses "image"
    if ham_images and "dx" not in meta:
        raise SystemExit("--ham-images needs HAM10000_metadata.csv (with the dx column) in data/raw/ham10000")
    rows = pd.DataFrame()
    if ham_images:
        imgs = index_images(ham_images)
        rows = pd.DataFrame([dict(source="ham10000", src_id=r.image_id, path=imgs.get(r.image_id),
                                  label=HAM_MAP.get(str(r.dx).lower()), group_key=r.lesion_id, age=r.age,
                                  sex=r.sex, site=r.localization, role="internal")
                             for r in meta.itertuples()])
    return meta, rows


def load_isic2018_test(raw: Path) -> pd.DataFrame:
    d = raw / "isic2018_test"
    gt_path = next(d.rglob("ISIC2018_Task3_Test_GroundTruth.csv"), None) if d.exists() else None
    if gt_path is None:
        return pd.DataFrame()
    gt = pd.read_csv(gt_path)
    cols = {"MEL": "MEL", "NV": "NV", "BCC": "BCC", "AKIEC": "AK", "BKL": "BKL", "DF": "DF", "VASC": "VASC"}
    imgs = index_images(d)
    return pd.DataFrame([dict(source="isic2018_test", src_id=r["image"], path=imgs.get(r["image"]),
                              label=next((v for k, v in cols.items() if r.get(k, 0) == 1), None),
                              group_key=None, age=None, sex=None, site=None, role="external")
                         for _, r in gt.iterrows()])


def load_pad(raw: Path, role: str) -> pd.DataFrame:
    d = raw / "pad_ufes20"
    meta_path = next(d.rglob("metadata.csv"), None) if d.exists() else None
    if meta_path is None:
        return pd.DataFrame()
    m = pd.read_csv(meta_path)
    imgs = index_images(d)
    return pd.DataFrame([dict(source="pad_ufes20", src_id=Path(r.img_id).stem, path=imgs.get(Path(r.img_id).stem),
                              label=PAD_MAP.get(str(r.diagnostic)), group_key=f"pad_patient_{r.patient_id}",
                              age=r.age, sex=r.gender, site=r.region, role=role)
                         for r in m.itertuples()])


def load_folder(name: str, folder: Path, role: str) -> pd.DataFrame:
    rows = []
    for cls_dir in folder.iterdir():
        if cls_dir.is_dir() and cls_dir.name.upper() in CLASSES:
            for p in cls_dir.rglob("*"):
                if p.suffix.lower() in IMG_EXT:
                    rows.append(dict(source=name, src_id=p.stem, path=p, label=cls_dir.name.upper(),
                                     group_key=None, age=None, sex=None, site=None, role=role))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- hashing
def phash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8].flatten()
    med = np.median(dct[1:])
    bits = (dct > med).astype(np.uint8)
    return int("".join(map(str, bits)), 2)


THUMB = 32


def centre_crop(img: np.ndarray, frac: float = 0.2) -> np.ndarray:
    h, w = img.shape[:2]
    return img[int(h * frac):h - int(h * frac), int(w * frac):w - int(w * frac)]


def thumbnail(img_bgr: np.ndarray) -> np.ndarray:
    """Grey 32x32 of the CENTRAL 60 % of the image, zero mean, unit length.
    Used to VERIFY near-duplicate candidates. The centre crop ignores the black circular border that
    many dermoscopes leave around the picture - that border dominates the perceptual hash and makes
    unrelated images look alike."""
    g = cv2.resize(cv2.cvtColor(centre_crop(img_bgr), cv2.COLOR_BGR2GRAY), (THUMB, THUMB),
                   interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    g -= g.mean()
    n = np.linalg.norm(g)
    return (g / n if n > 1e-6 else g).astype(np.float16)


def inspect(path) -> dict:
    out = dict(ok=False, md5=None, phash=None, width=0, height=0, error=None, thumb=None, colour=None)
    try:
        data = Path(path).read_bytes()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            out["error"] = "cannot decode"
            return out
        h, w = img.shape[:2]
        if min(h, w) < 64:
            out["error"] = f"too small ({w}x{h})"
            return out
        if float(img.std()) < 2.0:
            out["error"] = "blank image"
            return out
        out.update(ok=True, md5=hashlib.md5(data).hexdigest(), width=w, height=h,
                   phash=phash(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)), thumb=thumbnail(img),
                   colour=centre_crop(img).reshape(-1, 3).mean(0).astype(np.float32))
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)[:200]
    return out


_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(x: np.ndarray) -> np.ndarray:
    """Number of 1-bits per uint64 (np.bitwise_count needs numpy>=2; Kaggle may have 1.x)."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    x = np.ascontiguousarray(x, dtype=np.uint64)
    return _POP8[x.view(np.uint8)].reshape(*x.shape, 8).sum(-1, dtype=np.uint8)


def candidate_pairs(hashes, thumbs, colours, cand_t, old_t, corr_thr, colour_thr=25.0,
                    block=256, chunk=200_000):
    """Returns two arrays of pairs (i, j, hash_distance, similarity):
      verified - hash distance <= cand_t AND centre-thumbnail similarity >= corr_thr AND similar colour
                 (= real duplicates: resized / re-compressed copies of the same photo)
      old      - hash distance <= old_t, NOT verified (what the first version of this script used;
                 kept only for the audit report)"""
    H = hashes.astype(np.uint64)
    T = thumbs.astype(np.float32)
    C = colours.astype(np.float32)
    maxt = max(cand_t, old_t)
    ver, old = [], []
    for i in range(0, len(H), block):
        d = popcount64(H[i:i + block, None] ^ H[None, :])
        ii, jj = np.nonzero(d <= maxt)
        ii = ii + i
        keep = jj > ii
        ii, jj = ii[keep], jj[keep]
        if not len(ii):
            continue
        dd = d[ii - i, jj].astype(np.int16)
        for s in range(0, len(ii), chunk):
            a, b, dist = ii[s:s + chunk], jj[s:s + chunk], dd[s:s + chunk]
            corr = np.einsum("ij,ij->i", T[a], T[b])
            col = np.abs(C[a] - C[b]).max(1)
            is_dup = (dist <= cand_t) & (corr >= corr_thr) & (col <= colour_thr)
            ver.append(np.stack([a[is_dup], b[is_dup], dist[is_dup], corr[is_dup]], 1))
            o = dist <= old_t
            old.append(np.stack([a[o], b[o], dist[o], corr[o]], 1))
    empty = np.zeros((0, 4))
    return (np.concatenate(ver) if ver else empty), (np.concatenate(old) if old else empty)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)

    def labels(self):
        return np.array([self.find(i) for i in range(len(self.p))])


# --------------------------------------------------------------------------- split
def assign_groups(df: pd.DataFrame, fractions: list[float], seed: int) -> np.ndarray:
    """Stratified GROUP assignment: whole groups go to one part; each part receives close to
    `fractions` of every class. Greedy (largest groups first), works even for tiny classes."""
    rng = np.random.default_rng(seed)
    classes = sorted(df["label"].unique())
    cidx = {c: i for i, c in enumerate(classes)}
    totals = df["label"].value_counts().reindex(classes).to_numpy(float)
    target = np.outer(fractions, totals)                         # parts x classes
    have = np.zeros_like(target)
    groups = [(g, idx) for g, idx in df.groupby("group").indices.items()]
    rng.shuffle(groups)
    groups.sort(key=lambda t: -len(t[1]))
    out = np.empty(len(df), dtype=int)
    for _, idx in groups:
        cnt = np.zeros(len(classes))
        for lab in df["label"].to_numpy()[idx]:
            cnt[cidx[lab]] += 1
        need = ((target - have) / np.maximum(target, 1e-9)) @ cnt
        part = int(np.argmax(need + rng.random(len(fractions)) * 1e-6))
        have[part] += cnt
        out[idx] = part
    return out


def grouped_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    part = assign_groups(df, [0.70, 0.15, 0.15], seed)
    df["split"] = np.array(["train", "val", "test"])[part]
    df["cv_fold"] = -1
    tv = df.index[df["split"] != "test"]
    df.loc[tv, "cv_fold"] = assign_groups(df.loc[tv].reset_index(drop=True), [0.2] * 5, seed + 1)
    return df


def naive_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """What many papers do: random IMAGE-level 70/15/15 split (stratified by class only),
    no duplicate removal, no lesion grouping. Used only for the leak_test experiment."""
    rng = np.random.default_rng(seed)
    split = np.empty(len(df), dtype=object)
    for _, idx in df.groupby("label").indices.items():
        idx = rng.permutation(idx)
        n_te, n_va = int(round(0.15 * len(idx))), int(round(0.15 * len(idx)))
        split[idx[:n_te]] = "test"
        split[idx[n_te:n_te + n_va]] = "val"
        split[idx[n_te + n_va:]] = "train"
    df["split"], df["cv_fold"] = split, -1
    return df


# --------------------------------------------------------------------------- audit helpers
def side_by_side(path_a, path_b, caption: str, h: int = 256) -> np.ndarray:
    ims = []
    for p in (path_a, path_b):
        im = cv2.imread(str(p))
        im = cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
        ims.append(im)
    gap = np.full((h, 8, 3), 255, np.uint8)
    row = np.hstack([ims[0], gap, ims[1]])
    bar = np.full((44, row.shape[1], 3), 255, np.uint8)
    for k, line in enumerate(caption.split("\n")[:2]):
        cv2.putText(bar, line, (6, 18 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([row, bar])


def montage(images: list[np.ndarray], cols: int = 4, width: int = 420) -> np.ndarray:
    tiles = [cv2.resize(im, (width, int(im.shape[0] * width / im.shape[1]))) for im in images]
    th = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.full((th - t.shape[0], width, 3), 255, np.uint8)]) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.full((th, width, 3), 255, np.uint8))
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    return np.vstack(rows)


def write_audit(out: Path, df: pd.DataFrame, id_pairs, md5_pairs, old_pairs, ver_pairs, args,
                grouping: dict) -> dict:
    """Explains (1) the HAM10000 lesion grouping and (2) whether label conflicts are genuine,
    comparing the old (hash only) and new (hash + verified similarity) duplicate detection."""
    aud = out / "audit"
    ex_dir = aud / "conflict_examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    n = len(df)
    labels = df["label"].to_numpy()

    def clusters(pair_sets):
        uf = UnionFind(n)
        for ps in pair_sets:
            for a, b in ps[:, :2].astype(int):
                uf.union(a, b)
        return uf.labels()

    def conflict_stats(cl):
        s = pd.DataFrame({"c": cl, "label": labels})
        g = s.groupby("c")
        sizes = g.size()
        nlab = g["label"].nunique()
        bad = nlab[nlab > 1].index
        akscc = [c for c in bad if set(s.loc[s.c == c, "label"]) == {"AK", "SCC"}]
        bad = [c for c in bad if c not in set(akscc)]
        return dict(conflict_clusters=len(bad), conflict_images=int(sizes[bad].sum()) if len(bad) else 0,
                    dup_removed=int((sizes[sizes > 1] - 1).sum()), largest_cluster=int(sizes.max()) if len(sizes) else 0)

    old = conflict_stats(clusters([id_pairs, md5_pairs, old_pairs]))
    new = conflict_stats(clusters([id_pairs, md5_pairs, ver_pairs]))

    # direct pairs with different labels, from the OLD detection, with a verdict
    rows = []
    for reason, ps in (("same ISIC id", id_pairs), ("identical file (MD5)", md5_pairs),
                       (f"perceptual hash <= {args.old_hash_threshold} bits", old_pairs)):
        for a, b, dist, corr in ps:
            a, b = int(a), int(b)
            if labels[a] == labels[b] or {labels[a], labels[b]} == {"AK", "SCC"}:
                continue
            genuine = reason.startswith(("same", "identical")) or \
                (corr >= args.similarity_threshold and dist <= args.near_dup_threshold)
            rows.append(dict(reason=reason, image_a=df.at[a, "src_id"], label_a=labels[a], source_a=df.at[a, "source"],
                             image_b=df.at[b, "src_id"], label_b=labels[b], source_b=df.at[b, "source"],
                             hash_distance=int(dist), similarity=round(float(corr), 3),
                             verdict="genuine duplicate - real label conflict" if genuine
                             else "look-alike, NOT the same photo (false match)",
                             path_a=df.at[a, "path"], path_b=df.at[b, "path"]))
    pairs = pd.DataFrame(rows)
    if len(pairs):  # a pair found by several methods is listed once, with the strongest evidence
        key = pairs[["image_a", "image_b"]].apply(lambda r: tuple(sorted(r)), axis=1)
        pairs = pairs[~key.duplicated()].reset_index(drop=True)
    pairs.to_csv(aud / "conflict_pairs.csv", index=False)

    # 20 examples: the 10 most and 10 least similar conflicting pairs
    examples = []
    if len(pairs):
        ps = pairs.sort_values("similarity")
        pick = pd.concat([ps.head(10), ps.tail(10)]).drop_duplicates(subset=["image_a", "image_b"]).head(20)
        for k, r in enumerate(pick.itertuples()):
            cap = (f"A {r.image_a} [{r.label_a}, {r.source_a}]  |  B {r.image_b} [{r.label_b}, {r.source_b}]\n"
                   f"hash dist {r.hash_distance}, similarity {r.similarity:.2f} -> {r.verdict.split(' -')[0].split(',')[0]}")
            im = side_by_side(r.path_a, r.path_b, cap)
            cv2.imwrite(str(ex_dir / f"pair_{k:02d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
            examples.append(im)
        cv2.imwrite(str(aud / "conflict_examples_montage.jpg"), montage(examples), [cv2.IMWRITE_JPEG_QUALITY, 88])

    g = grouping
    n_gen = int((pairs["verdict"].str.startswith("genuine")).sum()) if len(pairs) else 0
    rep = ["# Data audit", "",
           "## 1. HAM10000 lesion grouping", "",
           f"- ISIC 2019 images with a `lesion_id` already in the ISIC 2019 metadata: **{g['isic_with_lesion']:,}** "
           f"of {g['isic_total']:,} (by prefix: {g['prefixes']})",
           f"- rows in the HAM10000 lesion-groupings file: {g['ham_rows']:,}; of these found in ISIC 2019: "
           f"{g['ham_matched']:,}",
           f"- of the matched images, already had a lesion id: {g['already']:,} "
           f"(same id as the HAM file: {g['agree']:,}, different: {g['disagree']:,})",
           f"- lesion ids filled in from the HAM file: **{g['filled']:,}**", "",
           "**Explanation:** " + g["explanation"], "",
           "## 2. Label conflicts between 'duplicates'", "",
           "| Duplicate detection | Images in conflicting clusters (removed) | Conflict clusters | "
           "Duplicate copies collapsed | Largest cluster |", "|---|---|---|---|---|",
           f"| old: perceptual hash <= {args.old_hash_threshold} bits only | {old['conflict_images']:,} | "
           f"{old['conflict_clusters']:,} | {old['dup_removed']:,} | {old['largest_cluster']:,} |",
           f"| new: hash <= {args.near_dup_threshold} bits AND centre similarity >= {args.similarity_threshold} "
           f"AND same colour | {new['conflict_images']:,} | {new['conflict_clusters']:,} | {new['dup_removed']:,} | "
           f"{new['largest_cluster']:,} |", "",
           f"Directly matched pairs with different labels (old method): **{len(pairs):,}**, of which "
           f"**{n_gen:,}** are genuine duplicates (same ISIC id, identical file, or verified near-identical) and "
           f"**{len(pairs) - n_gen:,}** are look-alikes (different photos that only share a similar overall layout, "
           "e.g. the dark circular dermoscope border).",
           "", "The old method also *chains* matches (A~B, B~C => A,B,C one cluster), so a few false matches can merge "
           "hundreds of unrelated images; all of them were removed when the labels disagreed, and only one copy was kept "
           "when they agreed. The new method only links verified near-identical copies.",
           "", "Files: `conflict_pairs.csv` (every conflicting pair with its verdict), "
           "`conflict_examples_montage.jpg` and `conflict_examples.tar` (20 side-by-side examples: "
           "10 least similar, 10 most similar)."]
    (aud / "audit_report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    return dict(old=old, new=new, pairs=len(pairs), genuine=n_gen)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/merged")
    ap.add_argument("--ham-images", default=None, help="folder with HAM10000 images (optional)")
    ap.add_argument("--pad-role", choices=["external", "internal"], default="external")
    ap.add_argument("--extra", action="append", default=[], help="name:folder:role")
    ap.add_argument("--near-dup-threshold", type=int, default=8,
                    help="max perceptual-hash bit difference for a near-duplicate CANDIDATE")
    ap.add_argument("--similarity-threshold", type=float, default=0.92,
                    help="min centre-thumbnail similarity to confirm a near-duplicate")
    ap.add_argument("--old-hash-threshold", type=int, default=4, help="old method, audit only")
    ap.add_argument("--cache-size", type=int, default=0, help="write resized copies with this short side")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--no-naive", action="store_true", help="do not write manifest_naive.csv")
    ap.add_argument("--audit-only", action="store_true", help="only write audit/ (no split, no images)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = ["# Dataset cleaning report", ""]

    ham_meta, ham_rows = load_ham(raw, Path(args.ham_images) if args.ham_images else None)
    parts = [load_isic2019(raw), ham_rows, load_isic2018_test(raw), load_pad(raw, args.pad_role)]
    for spec in args.extra:
        name, folder, role = spec.split(":") if spec.count(":") == 2 else (*spec.rsplit(":", 1), "internal")
        parts.append(load_folder(name, Path(folder), role))
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    if df.empty:
        sys.exit(f"No datasets found under {raw}. Run research/download_datasets.py first.")

    # ---- HAM10000 lesion ids (ISIC 2019 metadata usually already contains them)
    isic = df["source"] == "isic2019"
    gk = df.loc[isic, "group_key"]
    prefixes = gk.dropna().astype(str).str.extract(r"^([A-Za-z]+)_")[0].fillna("other").value_counts().to_dict()
    grouping = dict(isic_total=int(isic.sum()), isic_with_lesion=int(gk.notna().sum()),
                    prefixes=", ".join(f"{k}: {v:,}" for k, v in prefixes.items()) or "none",
                    ham_rows=len(ham_meta), ham_matched=0, already=0, agree=0, disagree=0, filled=0)
    if len(ham_meta):
        lesion = dict(zip(ham_meta["image_id"].astype(str), ham_meta["lesion_id"].astype(str)))
        key = df["src_id"].map(norm_isic_id)
        matched = key.isin(lesion)
        have = matched & df["group_key"].notna()
        same = have & (df["group_key"].astype(str) == key.map(lesion))
        fill = matched & df["group_key"].isna()
        df.loc[fill, "group_key"] = key[fill].map(lesion)
        grouping.update(ham_matched=int(matched.sum()), already=int(have.sum()), agree=int(same.sum()),
                        disagree=int((have & ~same).sum()), filled=int(fill.sum()))
    if grouping["ham_rows"] and grouping["filled"] == 0 and grouping["already"] == grouping["ham_matched"]:
        grouping["explanation"] = (
            "all HAM10000 images inside ISIC 2019 already carry their HAM10000 lesion id in the ISIC 2019 "
            "metadata, so there was nothing left to fill in. '0 attached' is therefore expected and harmless: "
            "the lesion grouping was applied through the ISIC 2019 metadata"
            + (" and agrees with the HAM file." if grouping["disagree"] == 0 else
               f", but {grouping['disagree']:,} ids differ from the HAM file - check those."))
    elif grouping["ham_rows"] == 0:
        grouping["explanation"] = "the HAM10000 lesion-groupings file was not found in data/raw/ham10000."
    elif grouping["ham_matched"] == 0:
        grouping["explanation"] = ("no image id from the HAM file matched an ISIC 2019 image id - check the file "
                                   "format (expected columns image/image_id and lesion_id).")
    else:
        grouping["explanation"] = (f"{grouping['filled']:,} images had no lesion id in the ISIC 2019 metadata and "
                                   "received it from the HAM file.")
    rep += [f"- Lesion ids: {grouping['isic_with_lesion']:,} ISIC 2019 images already had one; "
            f"{grouping['filled']:,} filled in from the HAM10000 file ({grouping['explanation']})"]

    rep += ["", "## 1. Loaded", "", "| Source | Role | Images |", "|---|---|---|"]
    for (s, r), n in df.groupby(["source", "role"]).size().items():
        rep.append(f"| {s} | {r} | {n:,} |")
    rep.append(f"| **total** | | **{len(df):,}** |")

    removed = []  # (reason, count)

    def drop(mask, reason, log_name=None):
        nonlocal df
        n = int(mask.sum())
        if n:
            if log_name:
                df[mask].drop(columns=["thumb", "colour"], errors="ignore").to_csv(out / log_name, index=False)
            df = df[~mask].reset_index(drop=True)
        removed.append((reason, n))

    drop(df["path"].isna(), "image file listed but missing", "removed_missing.csv")
    drop(df["label"].isna(), "no usable label (ISIC 'UNK' / unmapped class)", "removed_unlabelled.csv")

    print(f"Checking {len(df):,} images (decode + hashes) ...", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        info = pd.DataFrame(list(ex.map(inspect, df["path"])))
    df = pd.concat([df, info], axis=1)
    drop(~df["ok"], "corrupted / unreadable / blank / too small", "removed_corrupted.csv")
    n = len(df)
    thumbs = np.stack(df["thumb"].to_numpy())
    colours = np.stack(df["colour"].to_numpy())

    # ---- duplicate evidence
    key = df["src_id"].map(norm_isic_id)
    id_pairs, md5_pairs = [], []
    is_isic = key.str.startswith("ISIC_")
    sub = df[is_isic]
    for _, pos in sub.groupby(key[is_isic]).indices.items():
        idx = sub.index[pos]
        id_pairs += [(idx[0], j, 0, 1.0) for j in idx[1:]]
    for _, idx in df.groupby("md5").indices.items():
        md5_pairs += [(idx[0], j, 0, 1.0) for j in idx[1:]]
    id_pairs = np.array(id_pairs, float).reshape(-1, 4)
    md5_pairs = np.array(md5_pairs, float).reshape(-1, 4)
    print("Searching near-duplicates ...", flush=True)
    ver_pairs, old_pairs = candidate_pairs(df["phash"].to_numpy(np.uint64), thumbs, colours,
                                           args.near_dup_threshold, args.old_hash_threshold,
                                           args.similarity_threshold)
    pd.DataFrame([(df.at[int(i), "src_id"], df.at[int(j), "src_id"], int(d), round(float(c), 3))
                  for i, j, d, c in ver_pairs],
                 columns=["image_a", "image_b", "hash_distance", "similarity"]).to_csv(
        out / "near_duplicate_pairs.csv", index=False)

    dup = UnionFind(n)
    for ps in (id_pairs, md5_pairs, ver_pairs):
        for a, b in ps[:, :2].astype(int):
            dup.union(a, b)
    df["dup_cluster"] = dup.labels()

    # groups over ALL images (before removal): lesion/patient id + duplicate cluster
    grp = UnionFind(n)
    for col in ("group_key", "dup_cluster"):
        for _, idx in df.dropna(subset=[col]).groupby(col).indices.items():
            for j in idx[1:]:
                grp.union(idx[0], j)
    df["group"] = [f"g{c}" for c in grp.labels()]
    df["image_uid"] = [f"img{i:06d}" for i in range(n)]

    audit = write_audit(out, df, id_pairs, md5_pairs, old_pairs, ver_pairs, args, grouping)
    if args.audit_only:
        print((out / "audit" / "audit_report.md").read_text(encoding="utf-8"))
        return

    full = df  # every readable, labelled image (used for the naive split)

    # ---- clean set: one copy per duplicate cluster; conflicting clusters removed
    keep = np.zeros(n, dtype=bool)
    conflicts, n_dup_removed, n_ext_overlap = [], 0, 0
    for _, idx in df.groupby("dup_cluster").indices.items():
        sub = df.loc[idx]
        labels = set(sub["label"])
        if len(labels) > 1:
            if labels == {"AK", "SCC"} and (sub["source"] == "isic2019").any():
                df.loc[idx, "label"] = sub.loc[sub["source"] == "isic2019", "label"].iloc[0]
            else:
                conflicts.append(sub.assign(conflict_labels="/".join(sorted(labels))))
                continue
        if len(idx) > 1:
            internal = sub[sub["role"] == "internal"]
            pool = internal if len(internal) else sub
            if len(internal) and (sub["role"] == "external").any():
                n_ext_overlap += int((sub["role"] == "external").sum())
            best = pool.assign(pr=pool["source"].map(SOURCE_PRIORITY).fillna(9),
                               px=-(pool["width"] * pool["height"])).sort_values(["pr", "px"]).index[0]
            keep[best] = True
            n_dup_removed += len(idx) - 1
        else:
            keep[idx[0]] = True
    if conflicts:
        pd.concat(conflicts).drop(columns=["thumb", "colour"]).to_csv(out / "removed_label_conflicts.csv", index=False)
    removed.append(("label conflict between verified duplicate copies (all copies removed)",
                    int(sum(len(c) for c in conflicts))))
    removed.append(("duplicate / near-duplicate copies (one copy kept)", n_dup_removed - n_ext_overlap))
    removed.append(("external-test images that duplicate internal images", n_ext_overlap))
    df = df[keep].reset_index(drop=True)
    internal_groups = set(df.loc[df["role"] == "internal", "group"])
    drop((df["role"] == "external") & df["group"].isin(internal_groups),
         "external images of a lesion that is also in the internal data")

    # ---- split
    internal = grouped_split(df[df["role"] == "internal"].reset_index(drop=True), args.seed)
    external = df[df["role"] == "external"].copy()
    external["split"] = "external_" + external["source"]
    external["cv_fold"] = -1
    df = pd.concat([internal, external], ignore_index=True)

    # ---- leakage checks
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    va = df[df["split"] == "val"]
    assert set(tr["group"]).isdisjoint(te["group"]) and set(tr["group"]).isdisjoint(va["group"]) \
        and set(va["group"]).isdisjoint(te["group"]), "group leakage"
    assert set(tr["md5"]).isdisjoint(te["md5"]), "identical file in train and test"
    uid_row = {u: i for i, u in enumerate(full["image_uid"])}
    if len(te) and len(tr):
        Tt = thumbs[[uid_row[u] for u in te["image_uid"]]].astype(np.float32)
        Tr = thumbs[[uid_row[u] for u in tr["image_uid"]]].astype(np.float32)
        max_sim = max(float((Tt[i:i + 512] @ Tr.T).max()) for i in range(0, len(Tt), 512))
    else:
        max_sim = float("nan")

    # ---- naive split (leak_test experiment): every internal image, random image-level split
    naive = None
    if not args.no_naive:
        naive = naive_split(full[full["role"] == "internal"].reset_index(drop=True).copy(), args.seed)
        naive = pd.concat([naive, external], ignore_index=True)

    # ---- optional resized cache (one file per image uid, shared by both manifests)
    need = set(df["image_uid"]) | (set(naive["image_uid"]) if naive is not None else set())
    if args.cache_size:
        cache = out / f"images_{args.cache_size}"
        cache.mkdir(exist_ok=True)
        src = dict(zip(full["image_uid"], full["path"]))

        def resize(uid):
            dst = cache / f"{uid}.jpg"
            img = cv2.imread(str(src[uid]))
            s = args.cache_size / min(img.shape[:2])
            if s < 1:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            # relative to the manifest folder, so the processed folder can be moved (e.g. Kaggle dataset)
            return uid, f"{cache.name}/{dst.name}"
        with ThreadPoolExecutor(args.workers) as ex:
            rel = dict(ex.map(resize, sorted(need)))
        for d in (df, naive) if naive is not None else (df,):
            d["original_path"] = d["path"].astype(str)
            d["path"] = d["image_uid"].map(rel)

    cols = ["image_uid", "path", "label", "split", "cv_fold", "group", "dup_cluster", "source", "src_id", "role",
            "age", "sex", "site", "width", "height", "md5", "phash"] + \
        (["original_path"] if "original_path" in df else [])
    for d in (df, naive) if naive is not None else (df,):
        d["path"] = d["path"].astype(str)
        d["phash"] = d["phash"].astype("uint64").astype(str)
    df[cols].to_csv(out / "manifest.csv", index=False)
    for split, part in df[cols].groupby("split"):  # one CSV per split, convenient for the paper
        part.to_csv(out / f"{split}.csv", index=False)
    leak_info = ""
    if naive is not None:
        naive[cols].to_csv(out / "manifest_naive.csv", index=False)
        ntr, nte = naive[naive["split"] == "train"], naive[naive["split"] == "test"]
        leaked = nte["group"].isin(set(ntr["group"]))
        leak_info = (f"Naive split (manifest_naive.csv, leak_test only): {len(naive[naive.role == 'internal']):,} "
                     f"images, no de-duplication; {100 * leaked.mean():.1f}% of its test images have a duplicate "
                     f"or another photo of the same lesion in its training set.")
    size_gb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e9

    # ---- report
    rep += ["", "## 2. Removed", "", "| Reason | Images |", "|---|---|"]
    rep += [f"| {r} | {c:,} |" for r, c in removed]
    rep += ["", f"Duplicate detection: perceptual-hash candidates (<= {args.near_dup_threshold} bits) are only "
            f"accepted when the centre of the two images is near-identical (similarity >= "
            f"{args.similarity_threshold}) and has the same colour. Old hash-only method would have removed "
            f"{audit['old']['conflict_images']:,} images as label conflicts; this method removes "
            f"{audit['new']['conflict_images']:,}. Details: audit/audit_report.md"]
    rep += ["", "## 3. Final dataset (images per class and split)", ""]
    table = pd.crosstab(df["label"], df["split"]).reindex(CLASSES, fill_value=0)
    table["total"] = table.sum(1)
    rep.append("| Class | " + " | ".join(table.columns) + " |")
    rep.append("|---|" + "---|" * len(table.columns))
    for c, r in table.iterrows():
        rep.append(f"| {c} | " + " | ".join(f"{v:,}" for v in r) + " |")
    inner = df[df["role"] == "internal"]
    counts = inner["label"].value_counts()
    rep += ["", f"Internal images: {len(inner):,} in {inner['group'].nunique():,} lesion/patient groups.",
            "Split fractions (images): " + ", ".join(
                f"{s} {100 * (inner['split'] == s).mean():.1f}%" for s in ("train", "val", "test")),
            "", "## 4. Class imbalance", "",
            f"Largest / smallest class: {counts.max():,} / {counts.min():,} "
            f"(ratio {counts.max() / max(counts.min(), 1):.1f} : 1).",
            "It is handled during training with class-weighted loss (default), an optional "
            "balanced sampler (`--imbalance sampler`) and data augmentation. The test set is "
            "**not** re-balanced, so the reported numbers reflect the real class mix.",
            "", "## 5. Leakage checks (all passed)", "",
            "- no lesion/patient group appears in more than one of train / val / test",
            "- no identical file (MD5) in both train and test",
            f"- highest centre-similarity between any test and any train image: {max_sim:.3f} "
            f"(duplicates are >= {args.similarity_threshold} and also share the hash)",
            "", leak_info,
            "", f"Processed folder size: {size_gb:.2f} GB"
            + (f" (images resized to {args.cache_size} px short side)" if args.cache_size else "")]
    (out / "cleaning_report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = table.drop(columns="total").plot.bar(figsize=(8, 4), logy=True)
        ax.set_ylabel("images (log scale)"); ax.set_title("Class distribution per split")
        plt.tight_layout(); plt.savefig(out / "class_distribution.png", dpi=200); plt.close()
    except ImportError:
        pass
    print("\n".join(rep))
    print(f"\nSaved {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
