"""
Acceptance criteria, fixed BEFORE any test is run.

They are copied into run.json (with a SHA-256 of this text) when a run is created, and the report reads
them from there, so a criterion cannot be changed after the results are seen. T1-T5 and T7 are the
project team's criteria; T6 had no criterion in the request and was set here before the first run.
"""
from __future__ import annotations

import hashlib
import json

CRITERIA_VERSION = "2026-11-15"

REFERENCE = dict(accuracy=0.8580, accuracy_before_correction=0.8545, macro_f1=0.763, mel_recall=0.788, ece=0.070,
                 source="training notebook, full lesion-grouped test set (2,014 images), prior correction alpha = 0.3")

CRITERIA = {
    "T1": dict(name="Clinical accuracy (laboratory)",
               criterion="Full test set (2,014 images): accuracy within 1.5 percentage points of 85.80%. "
                         "300-image sample: the 95% confidence interval is reported instead (no pass/fail).",
               tolerance=0.015),
    "T2": dict(name="Integration agreement",
               criterion="At least 99% of test images get the identical predicted class from the app and from "
                         "test_predictions.npz (y_pred_corrected); the largest probability difference is reported.",
               min_agreement=0.99),
    "T3": dict(name="Robustness to photo conditions",
               criterion="On 200 test images, the average share of predictions unchanged by the 8 conditions is "
                         "at least 85% (every condition is reported separately).",
               min_stability=0.85),
    "T4": dict(name="Speed",
               criterion="95th percentile of prediction + Grad-CAM time under 30 seconds on this laptop "
                         "(50 images; all stages reported).",
               max_p95_s=30.0),
    "T5": dict(name="Reliability and load",
               criterion="100 analyses in a row and 5 simultaneous users through the real HTTP API: 0 crashes and "
                         "0 failed requests.",
               max_failures=0),
    "T6": dict(name="Photo quality check",
               criterion="At least 90% of the degraded images in each group (blurred, dark, tiny, non-skin) get the "
                         "matching warning. Set by the team before the first run; warnings on unchanged test "
                         "images are reported for information.",
               min_detection=0.90),
    "T7": dict(name="Functional and security tests",
               criterion="The complete automated test suite passes (0 failures, 0 errors).",
               max_failures=0),
}

# fixed test settings (also frozen into run.json)
SETTINGS = dict(
    seed=2026,
    t1_sample_size=300,
    t3_images=200,
    t3_conditions=["blur", "darker", "brighter", "rotate90", "rotate180", "jpeg50", "phone_resize", "crop90"],
    t3_blur_sigma=1.0, t3_dark_factor=0.7, t3_bright_factor=1.3, t3_jpeg_quality=50, t3_phone_side=1024,
    t3_crop=0.9,
    t4_images=50,
    t5_sequential=100, t5_users=5, t5_requests_per_user=4,
    t6_images=50, t6_blur_sigma=3.0, t6_dark_factor=0.25, t6_tiny_side=200,
    ece_bins=15,
)


def frozen() -> dict:
    body = dict(version=CRITERIA_VERSION, reference=REFERENCE, criteria=CRITERIA, settings=SETTINGS)
    text = json.dumps(body, sort_keys=True)
    return dict(body, sha256=hashlib.sha256(text.encode()).hexdigest())
