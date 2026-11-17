"""
Train one model (transfer learning + fine-tuning) on the merged dataset.

    python research/train.py --model efficientnet_b3                 # train on train, early-stop on val
    python research/train.py --model efficientnet_b3 --cv-fold 2     # one fold of 5-fold cross-validation
    python research/train.py --model resnet50 --smoke                # 2-minute check that everything runs

Training recipe (same for every model, so the comparison is fair):
  * ImageNet-pretrained backbone from timm
  * stage 1: backbone frozen, only the new classifier head is trained (--freeze-epochs)
  * stage 2: whole network fine-tuned; backbone LR = head LR / 10
  * AdamW + linear warm-up + cosine decay, mixed precision on GPU
  * augmentation: flips, 90-degree rotations, affine, colour jitter, coarse dropout
  * class imbalance: class-weighted cross-entropy (default) or a balanced sampler
  * early stopping on validation macro-F1 (--patience)
  * temperature scaling fitted on the validation set (calibrated probabilities)

Outputs in runs/<model>/<holdout|cvK>/: best.pt, model_card.json, history.csv, predictions.npz
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, IMAGENET_MEAN, IMAGENET_STD, MODELS, build_model, get_transforms  # noqa: E402


class ImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.paths = frame["path"].tolist()
        self.labels = frame["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
        self.t = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(self.paths[i]), cv2.COLOR_BGR2RGB)
        return self.t(image=img)["image"], int(self.labels[i])


class MaskedImageDataset(ImageDataset):
    """Training images + lesion masks (for the lesion-focus experiment). Images without a mask get an
    all-zero mask and has_mask=0, so they are trained normally.
    bg_aug: with this probability the skin AROUND the lesion is blurred or recoloured, so background
    cues (rulers, ink, skin tone, vignette) stop being useful for the prediction."""

    def __init__(self, frame, transform, masks: dict[str, str], bg_aug: float = 0.0):
        super().__init__(frame, transform)
        self.uids = frame["image_uid"].tolist()
        self.masks, self.bg_aug = masks, bg_aug

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(self.paths[i]), cv2.COLOR_BGR2RGB)
        mp = self.masks.get(self.uids[i])
        if mp:
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            m = (cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
        else:
            m = np.zeros(img.shape[:2], np.uint8)
        if mp and m.any() and random.random() < self.bg_aug:
            k = max(3, int(0.06 * min(m.shape)) | 1)
            keep = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))[..., None].astype(np.float32)
            if random.random() < 0.5:
                bg = cv2.GaussianBlur(img, (0, 0), 15)
            else:
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
                hsv[..., 0] = (hsv[..., 0] + random.randint(-12, 12)) % 180
                hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.7, 1.2), 0, 255)
                bg = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            img = (img * keep + bg * (1 - keep)).astype(np.uint8)
        out = self.t(image=img, mask=m)
        return out["image"], int(self.labels[i]), out["mask"].float(), float(mp is not None)


def cam_inside_loss(model, feats, y, masks, has_mask):
    """Mask-guided attention loss: 1 - (share of the target-class activation map inside the lesion).
    The activation map is computed from the last feature map and the classifier weights (CAM), so it
    costs one einsum - no second backward pass. Only for CNNs (4-D feature maps)."""
    sel = has_mask > 0
    if feats.dim() != 4 or not sel.any():
        return feats.new_zeros(())
    w = model.get_classifier().weight                                   # (K, C)
    cam = torch.relu(torch.einsum("bchw,bc->bhw", feats.float(), w[y].float()))
    m = F.adaptive_avg_pool2d(masks.unsqueeze(1).float(), cam.shape[-2:]).squeeze(1)
    inside = (cam * m).sum((1, 2)) / (cam.sum((1, 2)) + 1e-6)
    return (1 - inside)[sel].mean()


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def macro_f1(y, p, k=len(CLASSES)) -> float:
    f1s = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        if tp + fn == 0:
            continue
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1s))


@torch.no_grad()
def predict(model, loader, device, amp) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, ys = [], []
    for x, y, *_ in loader:
        with torch.autocast(device_type=device.type, enabled=amp):
            logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        ys.append(y)
    return torch.cat(logits).numpy(), torch.cat(ys).numpy()


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    z, t = torch.tensor(logits), torch.tensor(y)
    grid = np.exp(np.linspace(np.log(0.05), np.log(10), 300))
    losses = [F.cross_entropy(z / g, t).item() for g in grid]
    return float(grid[int(np.argmin(losses))])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--manifest", default="data/merged/manifest.csv")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--cv-fold", type=int, default=-1, help="-1 = normal train/val split; 0-4 = CV fold")
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--freeze-epochs", type=int, default=2)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4, help="head LR; backbone uses lr/10 after unfreezing")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--imbalance", choices=["weights", "sampler", "none"], default="weights")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 2 epochs, no pretrained download")
    ap.add_argument("--deadline", type=float, default=0.0,
                    help="unix time; stop cleanly (exit code 3) before an epoch that would end after it")
    ap.add_argument("--epoch-hint", type=float, default=0.0,
                    help="expected seconds per epoch (lets --deadline protect the very first epoch too)")
    ap.add_argument("--keep-last", action="store_true", help="keep last.pt after training finishes")
    ap.add_argument("--force", action="store_true", help="retrain even if this run already finished")
    ap.add_argument("--variant", default="", help="experiment name (e.g. naive_split, lesion_focus); "
                    "variant runs are kept apart from the normal model comparison")
    ap.add_argument("--masks", default="", help="folder with masks_index.csv (lesion-focus training)")
    ap.add_argument("--attn-weight", type=float, default=0.0, help="weight of the mask-guided attention loss")
    ap.add_argument("--bg-aug", type=float, default=0.0, help="probability of background blur/recolour")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    size = args.img_size or MODELS[args.model]["size"]
    if args.smoke:
        # ViT has a fixed input size; the others can use a small image for speed
        args.epochs, args.freeze_epochs = 2, 1
        size = size if args.model.startswith("vit") else min(size, 128)

    df = pd.read_csv(args.manifest)
    # paths may be stored relative to the manifest folder (portable processed dataset, e.g. on Kaggle)
    base = Path(args.manifest).resolve().parent
    df["path"] = [p if Path(p).is_absolute() else str(base / p) for p in df["path"].astype(str)]
    internal =df[df["split"].isin(["train", "val", "test"])]
    if args.cv_fold >= 0:
        trval = internal[internal["split"] != "test"]
        train_df = trval[trval["cv_fold"] != args.cv_fold]
        val_df = trval[trval["cv_fold"] == args.cv_fold]
        tag = f"cv{args.cv_fold}"
    else:
        train_df, val_df, tag = internal[internal["split"] == "train"], internal[internal["split"] == "val"], "holdout"
    test_df = internal[internal["split"] == "test"]
    external = {s: g for s, g in df.groupby("split") if s.startswith("external_")}
    if args.smoke:
        take = lambda d, n: d.groupby("label", group_keys=False).head(n)  # noqa: E731
        train_df, val_df, test_df = take(train_df, 6), take(val_df, 3), take(test_df, 3)
        external = {k: take(v, 2) for k, v in external.items()}

    run_dir = Path(args.out) / args.model / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "model_card.json").exists() and not args.force:
        print(f"{run_dir} is already finished (model_card.json exists) - skipping. Use --force to retrain.")
        return
    (run_dir / "run_info.json").write_text(json.dumps(dict(model=args.model, run=tag, variant=args.variant,
                                                           manifest=str(args.manifest)), indent=2))
    print(f"{args.model} | {tag} | train {len(train_df)} | val {len(val_df)} | test {len(test_df)} | {device}"
          + (f" | variant {args.variant}" if args.variant else ""))

    train_tf, eval_tf = get_transforms(size)
    masks = {}
    if args.masks:
        idx = pd.read_csv(Path(args.masks) / "masks_index.csv")
        masks = {u: str(Path(args.masks) / p) for u, p in zip(idx["image_uid"], idx["mask_path"])}
        n_m = int(train_df["image_uid"].isin(masks).sum())
        print(f"Lesion masks for {n_m:,} of {len(train_df):,} training images "
              f"(attention loss weight {args.attn_weight}, background augmentation p={args.bg_aug})")

    def train_ds():
        if masks:
            return MaskedImageDataset(train_df, train_tf, masks, args.bg_aug)
        return ImageDataset(train_df, train_tf)
    counts = train_df["label"].value_counts().reindex(CLASSES, fill_value=0).to_numpy()
    y_train = train_df["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    g = torch.Generator().manual_seed(args.seed)
    if args.imbalance == "sampler":
        w = (1.0 / np.maximum(counts, 1))[y_train]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), generator=g)
        train_loader = DataLoader(train_ds(), args.batch_size, sampler=sampler,
                                  num_workers=args.workers, pin_memory=amp, drop_last=True)
    else:
        train_loader = DataLoader(train_ds(), args.batch_size, shuffle=True, generator=g,
                                  num_workers=args.workers, pin_memory=amp, drop_last=len(train_df) > args.batch_size)
    mk = lambda d: DataLoader(ImageDataset(d, eval_tf), args.batch_size * 2,  # noqa: E731
                              num_workers=args.workers, pin_memory=amp)
    val_loader, test_loader = mk(val_df), mk(test_df)

    if args.imbalance == "weights":
        cw = 1.0 / np.sqrt(np.maximum(counts, 1))
        weight = torch.tensor(cw / cw.mean(), dtype=torch.float32, device=device)
    else:
        weight = None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)

    model = build_model(args.model, len(CLASSES), pretrained=not args.smoke).to(device)
    use_attn = bool(masks) and args.attn_weight > 0
    if use_attn and args.model.startswith("vit"):
        print("Note: the attention loss needs a CNN feature map; for ViT only background augmentation is used.")
        use_attn = False
    head = set(id(p) for p in model.get_classifier().parameters())
    backbone_params = [p for p in model.parameters() if id(p) not in head]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr / 10},
        {"params": list(model.get_classifier().parameters()), "lr": args.lr},
    ], weight_decay=args.weight_decay)
    steps_per_epoch = max(len(train_loader), 1)
    total, warm = args.epochs * steps_per_epoch, steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    history, best_f1, bad, start_epoch, done_before = [], -1.0, 0, 1, 0.0
    last_path = run_dir / "last.pt"
    if last_path.exists():  # resume after a stop (Kaggle time limit, crash, ...)
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        history, best_f1, bad = ck["history"], ck["best_f1"], ck["bad"]
        start_epoch, done_before = ck["epoch"] + 1, ck.get("minutes", 0.0)
        random.setstate(ck["rng_python"]); np.random.set_state(ck["rng_numpy"]); torch.set_rng_state(ck["rng_torch"])
        if torch.cuda.is_available() and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
        print(f"Resuming from {last_path} after epoch {ck['epoch']} (best val macro-F1 so far {best_f1:.4f})")
        if ck.get("stopped_early"):
            start_epoch = args.epochs + 1  # early stopping had already triggered

    t0 = time.time()
    last_epoch_secs = done_before * 60 / len(history) if history else args.epoch_hint
    for epoch in range(start_epoch, args.epochs + 1):
        if args.deadline and last_epoch_secs and time.time() + 1.1 * last_epoch_secs > args.deadline:
            print(f"TIME LIMIT: stopping before epoch {epoch}. Progress is saved in {last_path}; "
                  "run the same stage again to continue.", flush=True)
            sys.exit(3)
        e0 = time.time()
        g.manual_seed(args.seed * 1000 + epoch)  # same shuffling whether or not training was resumed
        frozen = epoch <= args.freeze_epochs
        for p in backbone_params:
            p.requires_grad = not frozen
        model.train()
        run_loss, run_ok, seen = 0.0, 0, 0
        run_attn = 0.0
        for batch in train_loader:
            x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                if use_attn:
                    feats = model.forward_features(x)
                    out = model.forward_head(feats)
                    attn = cam_inside_loss(model, feats, y, batch[2].to(device), batch[3].to(device))
                    loss = criterion(out, y) + args.attn_weight * attn
                    run_attn += float(attn) * len(y)
                else:
                    out = model(x)
                    loss = criterion(out, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            run_loss += loss.item() * len(y); run_ok += (out.argmax(1) == y).sum().item(); seen += len(y)
        vl, vy = predict(model, val_loader, device, amp)
        val_loss = F.cross_entropy(torch.tensor(vl), torch.tensor(vy)).item()
        rec = dict(epoch=epoch, stage="head-only" if frozen else "fine-tune",
                   train_loss=run_loss / max(seen, 1), train_acc=run_ok / max(seen, 1),
                   val_loss=val_loss, val_acc=float((vl.argmax(1) == vy).mean()),
                   val_macro_f1=macro_f1(vy, vl.argmax(1)), minutes=done_before + (time.time() - t0) / 60)
        if use_attn:
            rec["train_attn_loss"] = run_attn / max(seen, 1)
        history.append(rec)
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
        stop = False
        if rec["val_macro_f1"] > best_f1:
            best_f1, bad = rec["val_macro_f1"], 0
            atomic_save(model.state_dict(), run_dir / "best.pt")
        else:
            bad += 1
            stop = bad >= args.patience and not frozen
        # checkpoint after EVERY epoch (written to a temp file first, so a kill never leaves a broken file)
        atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), epoch=epoch,
                         history=history, best_f1=best_f1, bad=bad, minutes=rec["minutes"], stopped_early=stop,
                         rng_python=random.getstate(), rng_numpy=np.random.get_state(),
                         rng_torch=torch.get_rng_state(),
                         rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                    last_path)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        last_epoch_secs = time.time() - e0
        if stop:
            print(f"Early stopping at epoch {epoch}")
            break
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    vl, vy = predict(model, val_loader, device, amp)
    temperature = fit_temperature(vl, vy)
    tl, ty = predict(model, test_loader, device, amp)
    preds = dict(val_logits=vl, val_labels=vy, test_logits=tl, test_labels=ty,
                 val_uids=val_df["image_uid"].to_numpy().astype(str),
                 test_uids=test_df["image_uid"].to_numpy().astype(str))
    for name, d in external.items():
        el, ey = predict(model, mk(d), device, amp)
        preds[f"{name}_logits"], preds[f"{name}_labels"] = el, ey
    np.savez_compressed(run_dir / "predictions.npz", **preds)

    # inference speed (one image, batch 1)
    model.eval()
    x = torch.randn(1, 3, size, size, device=device)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        t = time.perf_counter()
        for _ in range(10):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    card = dict(model=args.model, timm_name=MODELS[args.model]["timm"], img_size=size, classes=CLASSES,
                mean=IMAGENET_MEAN, std=IMAGENET_STD, temperature=temperature, cam_layer=MODELS[args.model]["cam_layer"],
                weights="best.pt", run=tag, best_val_macro_f1=best_f1,
                best_epoch=int(max(history, key=lambda h: h["val_macro_f1"])["epoch"]),
                epochs_run=len(history), params=sum(p.numel() for p in model.parameters()),
                inference_ms=(time.perf_counter() - t) / 10 * 1000, device=str(device),
                train_images=len(train_df), variant=args.variant, manifest=Path(args.manifest).name,
                args=vars(args))
    tmp = run_dir / "model_card.json.tmp"
    tmp.write_text(json.dumps(card, indent=2))
    tmp.replace(run_dir / "model_card.json")  # model_card.json = "this run is finished"
    if not args.keep_last and last_path.exists():
        last_path.unlink()  # optimizer state is large and no longer needed
    print(f"Saved {run_dir} | best val macro-F1 {best_f1:.4f} | T={temperature:.3f}")


if __name__ == "__main__":
    main()
