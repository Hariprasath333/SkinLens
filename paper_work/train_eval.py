"""
SkinLens paper experiments: train / evaluate baselines and ablations on HAM10000.

Run this on Kaggle (GPU, with the "skin-cancer-mnist-ham10000" dataset attached).
It re-creates the SAME lesion-grouped train/val/test split used in
references/skinclass.ipynb (fold 0), so every number is comparable with the
85.45% / 76.03% macro-F1 result already reported.

Every run writes to  <out-dir>/<preset>_seed<seed>_fold<fold>/ :
    results.npz       val/test logits + labels + lesion ids (input for analyze.py)
    history.csv       per-epoch train loss / val macro-F1 / val accuracy
    run_config.json   every setting used, parameter count, inference time
    best.pt           best checkpoint (by validation macro-F1)
    last.pt           resume checkpoint (lets you continue after a 12 h Kaggle stop)

Examples
--------
# 1) Dump val+test logits of your EXISTING trained model (no training, ~10 min):
python train_eval.py --preset proposed --eval-only \
    --checkpoint /kaggle/input/<your-model>/best_skinlens_hybrid.pt --tta

# 2) Train a baseline / ablation:
python train_eval.py --preset effnet_only --seed 42
python train_eval.py --preset proposed --seed 7          # extra seed of the main model
python train_eval.py --preset proposed --sampler none    # imbalance ablation

# 3) 5-fold cross-validation of one preset:
for f in 0 1 2 3 4; do python train_eval.py --preset proposed --fold $f; done

# 4) Quick check that the script works (tiny subset, 1 epoch):
python train_eval.py --preset effnet_only --smoke
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import albumentations as A
import timm
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

# ----------------------------------------------------------------------------
# Constants (identical to the training notebook and backend/app/metadata.py)
# ----------------------------------------------------------------------------
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
SITES = [
    "abdomen", "back", "chest", "ear", "face", "foot", "genital",
    "hand", "lower extremity", "neck", "scalp", "trunk",
    "upper extremity", "acral", "oral/genital", "unknown",
]
AGE_SCALE = 90.0
META_DIM = 2 + 3 + len(SITES)  # 21
SPLIT_SEED = 42  # the data split never changes with --seed

EFFNET = "tf_efficientnetv2_s.in21k_ft_in1k"
SWIN = "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k"

# Each preset = one row in the paper's baseline / ablation table.
PRESETS = {
    # proposed model (same as the notebook)
    "proposed":        dict(backbones=[EFFNET, SWIN], use_meta=True, aux_weight=0.15, fusion="concat"),
    # ablations of the proposed model
    "no_aux":          dict(backbones=[EFFNET, SWIN], use_meta=True, aux_weight=0.0, fusion="concat"),
    "dual_image_only": dict(backbones=[EFFNET, SWIN], use_meta=False, aux_weight=0.0, fusion="concat"),
    "effnet_meta":     dict(backbones=[EFFNET], use_meta=True, aux_weight=0.15, fusion="concat"),
    "swin_meta":       dict(backbones=[SWIN], use_meta=True, aux_weight=0.15, fusion="concat"),
    # single-backbone baselines
    "effnet_only":     dict(backbones=[EFFNET], use_meta=False, aux_weight=0.0, fusion="concat"),
    "swin_only":       dict(backbones=[SWIN], use_meta=False, aux_weight=0.0, fusion="concat"),
    "resnet50":        dict(backbones=["resnet50.a1_in1k"], use_meta=False, aux_weight=0.0, fusion="concat"),
    "densenet121":     dict(backbones=["densenet121.tv_in1k"], use_meta=False, aux_weight=0.0, fusion="concat"),
    "vit_b16":         dict(backbones=["vit_base_patch16_384.augreg_in21k_ft_in1k"], use_meta=False,
                            aux_weight=0.0, fusion="concat"),
    # metadata-only baseline (how much can age/sex/site alone predict?)
    "meta_only":       dict(backbones=[], use_meta=True, aux_weight=0.0, fusion="concat"),
    # NEW fusion idea: metadata-conditioned feature modulation (FiLM-style)
    "proposed_film":   dict(backbones=[EFFNET, SWIN], use_meta=True, aux_weight=0.15, fusion="film"),
}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def encode_metadata(age, sex, localization):
    v = np.zeros(META_DIM, dtype=np.float32)
    age_missing = age is None or (isinstance(age, float) and math.isnan(age)) or pd.isna(age)
    v[0] = 0.5 if age_missing else float(np.clip(float(age), 0, AGE_SCALE) / AGE_SCALE)
    v[1] = float(age_missing)
    v[{"female": 2, "male": 3}.get(str(sex).lower(), 4)] = 1.0
    site = str(localization).lower()
    if site not in SITES:
        site = "unknown"
    v[5 + SITES.index(site)] = 1.0
    return v


MISSING_META = encode_metadata(None, "unknown", "unknown")  # "patient data not given"


def find_data_root(user_root):
    if user_root:
        return Path(user_root)
    for base in [Path("/kaggle/input"), Path("/content"), Path(".")]:
        if base.exists():
            hits = list(base.rglob("HAM10000_metadata.csv"))
            if hits:
                return hits[0].parent
    raise FileNotFoundError("HAM10000_metadata.csv not found; pass --data-root")


def load_dataframe(root):
    root = Path(root)
    df = pd.read_csv(next(root.rglob("HAM10000_metadata.csv")))
    index = {}
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for fp in root.rglob(ext):
            index[fp.stem] = str(fp.resolve())
    df["image_path"] = df["image_id"].astype(str).map(index)
    df = df.dropna(subset=["image_path"]).copy()
    df["dx"] = df["dx"].astype(str).str.lower().str.strip()
    df = df[df["dx"].isin(CLASS_NAMES)].copy()
    df["label"] = df["dx"].map({n: i for i, n in enumerate(CLASS_NAMES)}).astype(int)
    df["lesion_id"] = df["lesion_id"].fillna(df["image_id"]).astype(str)
    df["age"] = pd.to_numeric(df.get("age", np.nan), errors="coerce")
    df["sex"] = df.get("sex", "unknown").fillna("unknown").astype(str).str.lower().str.strip()
    df["localization"] = (df.get("localization", "unknown").fillna("unknown")
                          .astype(str).str.lower().str.strip())
    df.loc[~df["sex"].isin(["female", "male"]), "sex"] = "unknown"
    df.loc[~df["localization"].isin(SITES), "localization"] = "unknown"
    return df


def make_split(df, fold):
    """Same procedure as the notebook. fold=0 reproduces the notebook split exactly."""
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
    trainval_idx, test_idx = list(outer.split(df, y=df["label"], groups=df["lesion_id"]))[fold]
    trainval = df.iloc[trainval_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SPLIT_SEED + 1)
    tr_idx, va_idx = next(inner.split(trainval, y=trainval["label"], groups=trainval["lesion_id"]))
    train, val = trainval.iloc[tr_idx].reset_index(drop=True), trainval.iloc[va_idx].reset_index(drop=True)
    g = [set(x.lesion_id) for x in (train, val, test)]
    assert g[0].isdisjoint(g[1]) and g[0].isdisjoint(g[2]) and g[1].isdisjoint(g[2]), "lesion leakage!"
    return train, val, test


def get_transforms(size):
    train_tf = A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=30, p=0.4),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.35),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    eval_tf = A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_tf, eval_tf


class HAMDataset(Dataset):
    def __init__(self, frame, transform, meta_dropout=0.0, force_missing_meta=False):
        self.f = frame.reset_index(drop=True)
        self.t = transform
        self.meta_dropout = meta_dropout
        self.force_missing = force_missing_meta

    def __len__(self):
        return len(self.f)

    def __getitem__(self, i):
        r = self.f.iloc[i]
        img = cv2.cvtColor(cv2.imread(r["image_path"]), cv2.COLOR_BGR2RGB)
        img = self.t(image=img)["image"]
        if self.force_missing or (self.meta_dropout > 0 and random.random() < self.meta_dropout):
            meta = MISSING_META.copy()
        else:
            meta = encode_metadata(r["age"], r["sex"], r["localization"])
        return img, torch.from_numpy(meta), torch.tensor(int(r["label"]))


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class FusionNet(nn.Module):
    """Generalised SkinLensHybrid. With backbones=[EFFNET, SWIN], use_meta=True and
    fusion='concat' it is exactly the notebook / backend model."""

    def __init__(self, backbones, use_meta, fusion="concat", pretrained=True, num_classes=7):
        super().__init__()
        self.backbones = nn.ModuleList(
            [timm.create_model(n, pretrained=pretrained, num_classes=0) for n in backbones])
        self.img_dim = sum(b.num_features for b in self.backbones)
        self.use_meta = use_meta
        self.fusion = fusion
        meta_out = 32 if use_meta else 0
        if use_meta:
            self.meta_mlp = nn.Sequential(
                nn.Linear(META_DIM, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.25),
                nn.Linear(64, 32), nn.GELU())
            self.meta_aux_head = nn.Linear(32, num_classes)
            if fusion == "film" and self.img_dim > 0:
                # metadata predicts a per-channel scale and shift of the image features;
                # zero-init so training starts from the plain concat model
                self.film = nn.Linear(32, 2 * self.img_dim)
                nn.init.zeros_(self.film.weight)
                nn.init.zeros_(self.film.bias)
        self.classifier = nn.Sequential(
            nn.Linear(self.img_dim + meta_out, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(0.4), nn.Linear(512, num_classes))

    def forward(self, image, metadata, return_aux=False):
        parts = [b(image) for b in self.backbones]
        img = torch.cat(parts, dim=1) if parts else None
        aux = None
        if self.use_meta:
            m = self.meta_mlp(metadata)
            aux = self.meta_aux_head(m)
            if img is not None and self.fusion == "film":
                gamma, beta = self.film(m).chunk(2, dim=1)
                img = img * (1 + gamma) + beta
            fused = m if img is None else torch.cat([img, m], dim=1)
        else:
            fused = img
        logits = self.classifier(fused)
        return (logits, aux) if return_aux else logits


def load_any_checkpoint(model, path, device):
    """Loads a FusionNet checkpoint OR the original SkinLensHybrid checkpoint
    (keys effnet.* / swin.*), so your existing trained model can be evaluated."""
    sd = torch.load(path, map_location=device, weights_only=True)
    if isinstance(sd, dict) and "model" in sd and "optimizer" in sd:
        sd = sd["model"]
    if any(k.startswith("effnet.") for k in sd):
        sd = {("backbones.0." + k[7:] if k.startswith("effnet.") else
               "backbones.1." + k[5:] if k.startswith("swin.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)


# ----------------------------------------------------------------------------
# Train / evaluate
# ----------------------------------------------------------------------------
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


TTA_OPS = [  # 8 dihedral views of the image tensor (B,C,H,W)
    lambda x: x, lambda x: x.flip(3), lambda x: x.flip(2), lambda x: x.flip(2).flip(3),
    lambda x: x.transpose(2, 3), lambda x: x.transpose(2, 3).flip(3),
    lambda x: x.transpose(2, 3).flip(2), lambda x: x.transpose(2, 3).flip(2).flip(3),
]


@torch.no_grad()
def collect_logits(model, loader, device, amp, tta=False):
    model.eval()
    out, ys = [], []
    for img, meta, y in loader:
        img, meta = img.to(device, non_blocking=True), meta.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            if tta:
                # average probabilities over views, return log-probs as "logits"
                p = torch.stack([F.softmax(model(op(img), meta).float(), 1) for op in TTA_OPS]).mean(0)
                lg = torch.log(p.clamp_min(1e-12))
            else:
                lg = model(img, meta).float()
        out.append(lg.cpu()); ys.append(y)
    return torch.cat(out).numpy(), torch.cat(ys).numpy()


@torch.no_grad()
def time_inference(model, device, size, n=20):
    model.eval()
    x = torch.randn(1, 3, size, size, device=device)
    m = torch.from_numpy(MISSING_META).unsqueeze(0).to(device)
    for _ in range(3):
        model(x, m)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        model(x, m)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--seed", type=int, default=42, help="training seed (split is fixed)")
    ap.add_argument("--fold", type=int, default=0, choices=range(5), help="0 = notebook split")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-dir", default="/kaggle/working/paper_runs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--sampler", choices=["weighted", "none"], default="weighted",
                    help="weighted = notebook's inverse-frequency sampler")
    ap.add_argument("--class-weights", choices=["sqrt", "inverse", "none"], default="sqrt",
                    help="sqrt = notebook's 1/sqrt(count) loss weights")
    ap.add_argument("--meta-dropout", type=float, default=0.0,
                    help="probability of hiding metadata during training (robustness idea)")
    ap.add_argument("--aux-weight", type=float, default=None, help="override preset aux weight")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--checkpoint", default=None, help="checkpoint for --eval-only")
    ap.add_argument("--tta", action="store_true", help="also save 8-view test-time-augmentation logits")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny subset + 1 epoch, to test the script")
    ap.add_argument("--tag", default="", help="extra text for the run folder name")
    args = ap.parse_args()

    cfg = dict(PRESETS[args.preset])
    if args.aux_weight is not None:
        cfg["aux_weight"] = args.aux_weight
    if args.smoke:
        args.epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    set_seed(args.seed)

    name = f"{args.preset}{('_' + args.tag) if args.tag else ''}_seed{args.seed}_fold{args.fold}"
    if args.sampler != "weighted" or args.class_weights != "sqrt" or args.meta_dropout:
        name += f"_samp-{args.sampler}_cw-{args.class_weights}_md-{args.meta_dropout}"
    run_dir = Path(args.out_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    print("Run:", run_dir, "| device:", device)

    # ---- data
    df = load_dataframe(find_data_root(args.data_root))
    train_df, val_df, test_df = make_split(df, args.fold)
    for n, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{n}: {len(d)} images, {d.lesion_id.nunique()} lesions")
        d[["image_id", "lesion_id", "dx"]].to_csv(run_dir / f"{n}_manifest.csv", index=False)
    if args.fold == 0 and not args.smoke and (len(train_df), len(val_df), len(test_df)) != (6007, 1994, 2014):
        print("WARNING: split sizes differ from the notebook (6007/1994/2014). "
              "Check your sklearn/pandas versions before comparing numbers.")
    if args.smoke:
        train_df = train_df.groupby("label", group_keys=False).head(4)
        val_df = val_df.groupby("label", group_keys=False).head(2)
        test_df = test_df.groupby("label", group_keys=False).head(2)

    train_tf, eval_tf = get_transforms(args.img_size)
    mk = lambda d, tf, **kw: DataLoader(HAMDataset(d, tf, **kw), batch_size=args.batch_size,
                                       shuffle=False, num_workers=args.workers, pin_memory=amp)
    val_loader = mk(val_df, eval_tf)
    test_loader = mk(test_df, eval_tf)
    test_loader_nometa = mk(test_df, eval_tf, force_missing_meta=True)

    counts = np.bincount(train_df["label"].to_numpy(), minlength=len(CLASS_NAMES))

    # ---- model
    model = FusionNet(cfg["backbones"], cfg["use_meta"], cfg["fusion"],
                      pretrained=not (args.no_pretrained or args.eval_only)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")

    history = []
    if not args.eval_only:
        g = torch.Generator().manual_seed(args.seed)
        if args.sampler == "weighted":
            w = (1.0 / np.maximum(counts, 1))[train_df["label"].to_numpy()]
            sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w),
                                            replacement=True, generator=g)
            train_loader = DataLoader(HAMDataset(train_df, train_tf, meta_dropout=args.meta_dropout),
                                      batch_size=args.batch_size, sampler=sampler,
                                      num_workers=args.workers, pin_memory=amp)
        else:
            train_loader = DataLoader(HAMDataset(train_df, train_tf, meta_dropout=args.meta_dropout),
                                      batch_size=args.batch_size, shuffle=True, generator=g,
                                      num_workers=args.workers, pin_memory=amp)

        if args.class_weights == "none":
            cw = None
        else:
            raw = 1.0 / (np.sqrt(np.maximum(counts, 1)) if args.class_weights == "sqrt" else np.maximum(counts, 1))
            cw = torch.tensor(raw / raw.mean(), dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)

        head_params = [p for n_, p in model.named_parameters() if not n_.startswith("backbones.")]
        groups = [{"params": head_params, "lr": args.lr_head}]
        if len(model.backbones):
            groups.insert(0, {"params": model.backbones.parameters(), "lr": args.lr_backbone})
        optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)

        start_epoch, best_f1 = 1, -1.0
        last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
        if last_path.exists():  # resume after a Kaggle time-out
            ck = torch.load(last_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
            start_epoch, best_f1, history = ck["epoch"] + 1, ck["best_f1"], ck["history"]
            print(f"Resumed from epoch {ck['epoch']}")

        t0 = time.time()
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            running, steps = 0.0, len(train_loader)
            optimizer.zero_grad(set_to_none=True)
            for step, (img, meta, y) in enumerate(train_loader, start=1):
                img, meta, y = img.to(device), meta.to(device), y.to(device)
                with torch.autocast(device_type=device.type, enabled=amp):
                    logits, aux = model(img, meta, return_aux=True)
                    loss = criterion(logits, y)
                    if aux is not None and cfg["aux_weight"] > 0:
                        loss = loss + cfg["aux_weight"] * criterion(aux, y)
                    loss = loss / args.accum
                scaler.scale(loss).backward()
                running += loss.item() * args.accum
                if step % args.accum == 0 or step == steps:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            vl, vy = collect_logits(model, val_loader, device, amp)
            vp = vl.argmax(1)
            rec = dict(epoch=epoch, train_loss=running / max(steps, 1),
                       val_macro_f1=f1_score(vy, vp, average="macro", zero_division=0),
                       val_accuracy=accuracy_score(vy, vp), minutes=(time.time() - t0) / 60)
            history.append(rec)
            print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
            if rec["val_macro_f1"] > best_f1:
                best_f1 = rec["val_macro_f1"]
                torch.save(model.state_dict(), best_path)
                print("  new best checkpoint")
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                            epoch=epoch, best_f1=best_f1, history=history), last_path)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        load_any_checkpoint(model, best_path, device)
    else:
        if not args.checkpoint:
            raise SystemExit("--eval-only needs --checkpoint")
        load_any_checkpoint(model, args.checkpoint, device)

    # ---- final evaluation (val logits are needed for calibration / conformal in analyze.py)
    print("Collecting validation and test logits ...")
    val_logits, val_labels = collect_logits(model, val_loader, device, amp)
    test_logits, test_labels = collect_logits(model, test_loader, device, amp)
    extra = {}
    if cfg["use_meta"]:
        extra["test_logits_nometa"], _ = collect_logits(model, test_loader_nometa, device, amp)
    if args.tta:
        extra["val_logits_tta"], _ = collect_logits(model, val_loader, device, amp, tta=True)
        extra["test_logits_tta"], _ = collect_logits(model, test_loader, device, amp, tta=True)
    np.savez_compressed(
        run_dir / "results.npz",
        val_logits=val_logits, val_labels=val_labels,
        test_logits=test_logits, test_labels=test_labels,
        test_lesion_ids=test_df["lesion_id"].to_numpy().astype(str),
        test_image_ids=test_df["image_id"].to_numpy().astype(str),
        train_counts=counts, class_names=np.array(CLASS_NAMES), **extra)

    ms = time_inference(model, device, args.img_size)
    tp = test_logits.argmax(1)
    summary = dict(
        preset=args.preset, config=cfg, args=vars(args), params=n_params,
        device=str(device), inference_ms_per_image=ms,
        test_accuracy=float(accuracy_score(test_labels, tp)),
        test_macro_f1=float(f1_score(test_labels, tp, average="macro", zero_division=0)),
        best_val_macro_f1=max([h["val_macro_f1"] for h in history], default=None),
        best_epoch=(max(history, key=lambda h: h["val_macro_f1"])["epoch"] if history else None),
    )
    (run_dir / "run_config.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ["test_accuracy", "test_macro_f1", "params",
                                              "inference_ms_per_image", "best_epoch"]}, indent=2))
    print("Saved:", run_dir / "results.npz")


if __name__ == "__main__":
    main()
