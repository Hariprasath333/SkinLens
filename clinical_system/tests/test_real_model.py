"""
Checks the REAL trained model (original SkinLens checkpoint from backup/original_model).
Slow on CPU (about 1-3 minutes), so it only runs when asked:

    set RUN_REAL_MODEL=1   (PowerShell: $env:RUN_REAL_MODEL=1)
    python -m pytest tests/test_real_model.py -q
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.getenv("RUN_REAL_MODEL") != "1", reason="set RUN_REAL_MODEL=1")
ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT.parent / "references" / "test_images" / "sample.jpg"


def test_skinlens_model_matches_backend_and_explains():
    from webapp.ml import xai
    from webapp.ml.predictor import PriorCorrection, SkinLensPredictor, load_model_settings
    p = SkinLensPredictor()
    p.prior, _ = PriorCorrection.for_classes(load_model_settings(), p.classes)  # as get_predictor() does
    rgb = cv2.cvtColor(cv2.imread(str(SAMPLE)), cv2.COLOR_BGR2RGB)
    with p.session(age=45, sex="male", site="upper extremity"):
        res = p.predict(rgb)
        cam = xai.gradcam(p, rgb, p.classes.index(res["top_class"]))
        lime = xai.lime(p, rgb, p.classes.index(res["top_class"]), 24, 25)
    assert res["top_class"] == "NV"  # sample.jpg is a HAM10000 nevus

    # same numbers as the original backend inference code
    sys.path.insert(0, str(ROOT.parent / "backend"))
    from app.inference import InferencePipeline
    ref = InferencePipeline().predict(str(SAMPLE), age=45, sex="male", localization="upper extremity")
    # the shown probabilities are prior-corrected; the model's own softmax must equal the backend's
    assert abs(ref["raw_probabilities"]["nv"] - res["raw_probs"]["NV"]) < 1e-3
    assert res["prior_corrected"] and abs(sum(res["probs"].values()) - 1) < 1e-6
    assert cam.shape == (384, 384) and cam.max() > 0.99
    assert lime["image"].shape == (256, 256, 3) and len(lime["top"]) >= 1
    text = xai.written_explanation("NV", xai.measurements(rgb, cam))
    print("\n", res["ranked"][:3], "\n", text)
