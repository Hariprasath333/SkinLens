"""
Tests the research pipeline on a small FAKE dataset that contains planted problems:
exact duplicates, a resized near-duplicate, a corrupted file, an 'UNK' label, a duplicate
with a conflicting label, several photos of one lesion, and an external-test image that
is also in the training data. Then trains every model for a few steps and evaluates.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]


def blob(rng, i):
    """A distinct random 'lesion' picture (different shapes/colours so hashes differ)."""
    img = np.full((180, 220, 3), rng.integers(150, 230, 3), np.uint8)
    for _ in range(6):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.ellipse(img, tuple(int(v) for v in rng.integers(30, 180, 2)), tuple(int(v) for v in rng.integers(10, 60, 2)),
                    float(rng.integers(0, 180)), 0, 360, c, -1)
    cv2.putText(img, str(i), (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    return img


def run(*args):
    r = subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipe")
    make_fake_raw(tmp / "raw")
    return tmp


def vignetted(seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    img = np.zeros((450, 600, 3), np.uint8)
    cv2.circle(img, (300, 225), 215, (150, 160, 200), -1)          # skin disc inside a black border
    for _ in range(3):
        cv2.ellipse(img, (int(r.integers(240, 360)), int(r.integers(170, 280))),
                    (int(r.integers(15, 45)), int(r.integers(15, 45))), float(r.integers(0, 180)), 0, 360,
                    tuple(int(v) for v in r.integers(20, 120, 3)), -1)
    return img


def lookalike_pair():
    sys.path.insert(0, str(ROOT / "research"))
    import build_dataset as b
    ims = [vignetted(s) for s in range(40)]
    H = np.array([b.phash(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)) for i in ims], dtype=np.uint64)
    T = np.stack([b.thumbnail(i) for i in ims]).astype(np.float32)
    for i in range(len(ims)):
        for j in range(i + 1, len(ims)):
            if int(b.popcount64(np.array([H[i] ^ H[j]]))[0]) <= 4 and float(T[i] @ T[j]) < 0.8:
                return ims[i], ims[j]
    raise RuntimeError("no look-alike pair found")


def make_fake_masks(raw: Path, out: Path) -> Path:
    """Fake '<ISIC id>_segmentation.png' masks (central ellipse) for about two thirds of the images."""
    out.mkdir(parents=True, exist_ok=True)
    for p in sorted((raw / "isic2019").rglob("ISIC_*.jpg"))[::3] + sorted((raw / "isic2019").rglob("ISIC_*.jpg"))[1::3]:
        im = cv2.imread(str(p))
        if im is None:
            continue
        m = np.zeros(im.shape[:2], np.uint8)
        cv2.ellipse(m, (im.shape[1] // 2, im.shape[0] // 2), (im.shape[1] // 4, im.shape[0] // 4), 0, 0, 360, 255, -1)
        cv2.imwrite(str(out / f"{p.stem}_segmentation.png"), m)
    return out


def make_fake_raw(raw: Path) -> None:
    """Writes a tiny fake copy of the downloaded datasets (same file layout as the real ones)."""
    imgdir = raw / "isic2019" / "ISIC_2019_Training_Input"
    imgdir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    gt, meta, ham = [], [], []
    classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    n = 0
    for cls in classes:
        for k in range(14):
            name = f"ISIC_{n:07d}"
            cv2.imwrite(str(imgdir / f"{name}.jpg"), blob(rng, n))
            gt.append({"image": name, **{c: float(c == cls) for c in COLS}})
            lesion = f"HAM_{n // 2:07d}" if k < 4 else None   # pairs of photos of the same lesion
            meta.append(dict(image=name, age_approx=50, anatom_site_general="back", lesion_id=lesion, sex="male"))
            n += 1
    # planted problems
    first = imgdir / "ISIC_0000000.jpg"
    (imgdir / "ISIC_9000001.jpg").write_bytes(first.read_bytes())                     # exact duplicate
    gt.append({"image": "ISIC_9000001", **{c: float(c == "MEL") for c in COLS}})
    im = cv2.imread(str(imgdir / "ISIC_0000020.jpg"))                                   # NV image
    cv2.imwrite(str(imgdir / "ISIC_9000002.jpg"), cv2.resize(im, (330, 270)), [cv2.IMWRITE_JPEG_QUALITY, 80])
    gt.append({"image": "ISIC_9000002", **{c: float(c == "NV") for c in COLS}})        # near duplicate
    (imgdir / "ISIC_9000003.jpg").write_bytes(b"\xff\xd8broken")                       # corrupted
    gt.append({"image": "ISIC_9000003", **{c: float(c == "BCC") for c in COLS}})
    cv2.imwrite(str(imgdir / "ISIC_9000004.jpg"), blob(rng, 999))
    gt.append({"image": "ISIC_9000004", **{c: float(c == "UNK") for c in COLS}})       # unknown label
    im = cv2.imread(str(imgdir / "ISIC_0000030.jpg"))                                   # BCC image copied...
    cv2.imwrite(str(imgdir / "ISIC_9000005.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 70])
    gt.append({"image": "ISIC_9000005", **{c: float(c == "DF") for c in COLS}})        # ...labelled DF
    pd.DataFrame(gt).to_csv(raw / "isic2019" / "ISIC_2019_Training_GroundTruth.csv", index=False)
    pd.DataFrame(meta).to_csv(raw / "isic2019" / "ISIC_2019_Training_Metadata.csv", index=False)
    # external test set with one image that is also in the training data
    ext = raw / "isic2018_test" / "ISIC2018_Task3_Test_Input"
    ext.mkdir(parents=True)
    rows = []
    for j in range(6):
        nm = f"ISIC_8{j:06d}"
        cv2.imwrite(str(ext / f"{nm}.jpg"), blob(rng, 5000 + j))
        rows.append(dict(image=nm, MEL=float(j % 2 == 0), NV=float(j % 2 == 1), BCC=0, AKIEC=0, BKL=0, DF=0, VASC=0))
    (ext / "ISIC_8999999.jpg").write_bytes((imgdir / "ISIC_0000050.jpg").read_bytes())
    rows.append(dict(image="ISIC_8999999", MEL=0, NV=0, BCC=0, AKIEC=0, BKL=1, DF=0, VASC=0))
    gtdir = raw / "isic2018_test" / "ISIC2018_Task3_Test_GroundTruth"
    gtdir.mkdir()
    pd.DataFrame(rows).to_csv(gtdir / "ISIC2018_Task3_Test_GroundTruth.csv", index=False)
    # look-alike pair: two DIFFERENT lesions inside the same black dermoscope border, different labels.
    # The old hash-only method called them duplicates (label conflict); the verified method must not.
    a, b = lookalike_pair()
    cv2.imwrite(str(imgdir / "ISIC_9100001.jpg"), a)
    cv2.imwrite(str(imgdir / "ISIC_9100002.jpg"), b)
    gt_extra = [{"image": "ISIC_9100001", **{c: float(c == "MEL") for c in COLS}},
                {"image": "ISIC_9100002", **{c: float(c == "NV") for c in COLS}}]
    pd.concat([pd.read_csv(raw / "isic2019" / "ISIC_2019_Training_GroundTruth.csv"), pd.DataFrame(gt_extra)]).to_csv(
        raw / "isic2019" / "ISIC_2019_Training_GroundTruth.csv", index=False)
    # HAM10000 lesion groupings file (as published by ISIC): two photos of one lesion
    (raw / "ham10000").mkdir()
    pd.DataFrame(dict(image=["ISIC_0000006", "ISIC_0000007"], lesion_id=["HAM_9999999", "HAM_9999999"],
                      diagnosis_confirm_type=["histopathology"] * 2)).to_csv(
        raw / "ham10000" / "ISIC2018_Task3_Training_LesionGroupings.csv", index=False)


def test_build_dataset(fake):
    out = fake / "merged"
    run("research/build_dataset.py", "--raw", str(fake / "raw"), "--out", str(out), "--workers", "4")
    man = pd.read_csv(out / "manifest.csv")
    report = (out / "cleaning_report.md").read_text()
    ids = set(man["src_id"])
    assert "ISIC_9000003" not in ids                                  # corrupted removed
    assert "ISIC_9000004" not in ids                                  # UNK removed
    assert not ({"ISIC_0000030", "ISIC_9000005"} & ids)               # conflicting duplicate removed
    assert len({"ISIC_0000000", "ISIC_9000001"} & ids) == 1           # exact duplicate collapsed
    assert len({"ISIC_0000020", "ISIC_9000002"} & ids) == 1           # near duplicate collapsed
    assert "ISIC_8999999" not in ids                                  # external overlap removed
    internal = man[man["split"].isin(["train", "val", "test"])]
    for a, b in [("train", "test"), ("train", "val"), ("val", "test")]:
        assert set(internal.loc[internal.split == a, "group"]).isdisjoint(internal.loc[internal.split == b, "group"])
    frac = internal["split"].value_counts(normalize=True)
    assert 0.55 < frac["train"] < 0.85
    assert (man.loc[man.split == "external_isic2018_test"].shape[0]) == 6
    assert set(internal.loc[internal.split != "test", "cv_fold"]) == {0, 1, 2, 3, 4}
    assert "Leakage checks (all passed)" in report
    g = man.set_index("src_id")["group"]
    assert g["ISIC_0000006"] == g["ISIC_0000007"]                    # lesion groupings file was used
    # look-alikes (same dermoscope border, different lesion) are NOT duplicates any more
    assert {"ISIC_9100001", "ISIC_9100002"} <= ids and g["ISIC_9100001"] != g["ISIC_9100002"]
    audit = (out / "audit" / "audit_report.md").read_text()
    pairs = pd.read_csv(out / "audit" / "conflict_pairs.csv")
    la = pairs[(pairs.image_a.isin(["ISIC_9100001", "ISIC_9100002"])) & (pairs.image_b.isin(["ISIC_9100001", "ISIC_9100002"]))]
    assert len(la) == 1 and la.verdict.iloc[0].startswith("look-alike")
    assert pairs[pairs.reason.str.contains("MD5|hash")].verdict.str.startswith("genuine").any()  # BCC/DF copy
    assert (out / "audit" / "conflict_examples_montage.jpg").exists()
    assert len(list((out / "audit" / "conflict_examples").glob("*.jpg"))) == len(pairs.head(20))
    assert "Explanation" in audit and "Images in conflicting clusters" in audit
    naive = pd.read_csv(out / "manifest_naive.csv")
    assert {"ISIC_9000001", "ISIC_0000000", "ISIC_0000030", "ISIC_9000005"} <= set(naive.src_id)  # no de-dup
    assert set(naive.loc[naive.role == "internal", "split"]) == {"train", "val", "test"}


@pytest.mark.parametrize("model", ["resnet50", "efficientnet_b3", "densenet121", "convnext_tiny", "vit_b16"])
def test_train_smoke(fake, model):
    run("research/train.py", "--model", model, "--manifest", str(fake / "merged" / "manifest.csv"),
        "--out", str(fake / "runs"), "--smoke", "--workers", "0", "--batch-size", "8")
    d = fake / "runs" / model / "holdout"
    card = json.loads((d / "model_card.json").read_text())
    z = np.load(d / "predictions.npz")
    assert card["img_size"] > 0 and "external_isic2018_test_logits" in z.files
    assert (d / "history.csv").exists()


def test_cv_and_evaluate_and_export(fake):
    man = str(fake / "merged" / "manifest.csv")
    for k in range(2):  # two folds are enough to test the code path
        run("research/train.py", "--model", "resnet50", "--manifest", man, "--out", str(fake / "runs"),
            "--smoke", "--workers", "0", "--batch-size", "8", "--cv-fold", str(k))
    res, exp = fake / "results", fake / "export"
    run("research/evaluate.py", "--runs", str(fake / "runs"), "--results", str(res), "--export", str(exp))
    table = pd.read_csv(res / "results.csv")
    for col in ["accuracy", "precision_macro", "recall_sensitivity_macro", "specificity_macro", "f1_macro",
                "auc_roc_macro"]:
        assert col in table.columns
    assert table["model"].str.startswith("ensemble").any() and len(table) == 6
    for f in ["results_per_class.csv", "results_cv.csv", "results_external.csv", "summary.md",
              "roc_resnet50.png", "confusion_resnet50.png", "curves_resnet50.png", "roc_all_models.png"]:
        assert (res / f).exists(), f
    run("research/find_label_issues.py", "--runs", str(fake / "runs"), "--model", "resnet50", "--manifest", man)
    assert (fake / "merged" / "label_issues.csv").exists()

    # the web app can load whatever was exported (single model or ensemble) and explain it
    info = json.loads((exp / "export.json").read_text())
    code = (
        "import sys, json, numpy as np, cv2; sys.path.insert(0, '.');"
        "from pathlib import Path; from webapp.ml.predictor import TimmPredictor; from webapp.ml import xai;"
        f"p = TimmPredictor(Path(r'{exp}'));"
        "img = cv2.cvtColor(cv2.imread(r'" + str(next((fake / 'raw').rglob('ISIC_0000001.jpg'))) + "'), cv2.COLOR_BGR2RGB);"
        "r = p.predict(img); cam = xai.gradcam(p, img, p.classes.index(r['top_class']));"
        "assert cam.shape == (384, 384) and np.isfinite(cam).all();"
        "l = xai.lime(p, img, 0, 12, 15); print('OK', p.name, r['top_class'])"
    )
    assert "OK" in run("-c", code)
    assert info["members"]


def test_gradcam_all_architectures():
    """Grad-CAM must work for every candidate architecture (incl. ViT token reshaping)."""
    code = (
        "import sys, numpy as np, torch; sys.path.insert(0, '.'); sys.path.insert(0, 'research');"
        "from common import MODELS, build_model; from webapp.ml.predictor import Predictor, Member, _cam_layer;"
        "from webapp.ml import xai\n"
        "for name, spec in MODELS.items():\n"
        "    m = build_model(name, 8, pretrained=False).eval(); layer, tok = _cam_layer(m, spec['cam_layer'])\n"
        "    p = Predictor(); p.members = [Member(m, spec['size'], 1.0, layer, tok)]; p.classes = list('ABCDEFGH')\n"
        "    img = (np.random.default_rng(0).random((300, 300, 3)) * 255).astype('uint8')\n"
        "    cam = xai.gradcam(p, img, 3); assert cam.shape == (384, 384) and np.isfinite(cam).all(), name\n"
        "    print(name, 'ok')\n"
    )
    out = run("-c", code)
    assert out.count("ok") == 5
