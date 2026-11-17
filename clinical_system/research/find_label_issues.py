"""
Find images that are probably MISLABELLED, using out-of-fold predictions from
5-fold cross-validation (a simple version of "confident learning", Northcutt et al. 2021).

    python research/find_label_issues.py --runs runs --model efficientnet_b3

An image is flagged when, in the fold where it was NOT used for training,
 * the model gives its given label a probability below that class's average
   self-confidence threshold, and
 * the model is confident (above that class's threshold) that it is another class.

Flagged images are written to data/merged/label_issues.csv for a human to review.
They are NOT deleted automatically: most labels in these datasets come from
histopathology, and a model disagreeing is not proof of a wrong label.
To exclude reviewed images, list their image_uid in data/merged/exclude.txt and
rerun training with --manifest data/merged/manifest_clean.csv (written by this script).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES  # noqa: E402


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["runs"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", default="data/merged/manifest.csv")
    ap.add_argument("--out", default=None, help="output folder (default: next to the manifest)")
    args = ap.parse_args()

    from evaluate import discover_runs
    _, cv = discover_runs([Path(r) for r in args.runs])
    probs, labels, uids = [], [], []
    for _, d in sorted(cv.get(args.model, {}).items()):
        z = np.load(d / "predictions.npz")
        T = json.loads((d / "model_card.json").read_text()).get("temperature", 1.0)
        probs.append(softmax(z["val_logits"] / T)); labels.append(z["val_labels"]); uids.append(z["val_uids"])
    if not probs:
        sys.exit("No CV runs found. Train with --cv-fold 0..4 first.")
    P, y, uid = np.concatenate(probs), np.concatenate(labels), np.concatenate(uids)

    thr = np.array([P[y == c, c].mean() if (y == c).any() else 1.0 for c in range(len(CLASSES))])
    given = P[np.arange(len(y)), y]
    other = P.copy()
    other[np.arange(len(y)), y] = -1
    alt = other.argmax(1)
    flag = (given < thr[y]) & (other[np.arange(len(y)), alt] >= thr[alt])

    man = pd.read_csv(args.manifest)
    issues = pd.DataFrame(dict(image_uid=uid[flag], given_label=[CLASSES[i] for i in y[flag]],
                               suggested_label=[CLASSES[i] for i in alt[flag]],
                               prob_given=given[flag], prob_suggested=other[flag, alt[flag]]))
    issues = issues.merge(man[["image_uid", "path", "source"]], on="image_uid", how="left")
    issues = issues.sort_values("prob_given")
    out = Path(args.out) if args.out else Path(args.manifest).parent
    out.mkdir(parents=True, exist_ok=True)
    issues.to_csv(out / "label_issues.csv", index=False)
    excl = out / "exclude.txt"
    excluded = set(excl.read_text().split()) if excl.exists() else set()
    base = Path(args.manifest).resolve().parent
    if out.resolve() != base:  # keep image paths valid when the clean manifest lives elsewhere
        man["path"] = [p if Path(p).is_absolute() else str(base / p) for p in man["path"].astype(str)]
    man[~man["image_uid"].isin(excluded)].to_csv(out / "manifest_clean.csv", index=False)
    print(f"Checked {len(y):,} out-of-fold predictions; flagged {flag.sum():,} "
          f"({100 * flag.mean():.2f}%) possible label issues -> {out / 'label_issues.csv'}")
    print(issues.groupby(["given_label", "suggested_label"]).size().sort_values(ascending=False).head(10))


if __name__ == "__main__":
    main()
