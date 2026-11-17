import math
import sys
from pathlib import Path

# Add backend directory to path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.preprocessing import preprocess_image
from app.metadata import encode_metadata
from app.inference import InferencePipeline, CLASS_NAMES, CLASS_FULL

def run_inference_tests():
    print("=" * 70)
    print("SkinLens Single-Image Inference Pipeline Verification (Phase 3)")
    print("=" * 70)

    # 1. Image path
    sample_img_path = BACKEND_DIR.parent / "references" / "test_images" / "sample.jpg"
    print(f"Sample image path: {sample_img_path}")
    if not sample_img_path.exists():
        print(f"ERROR: Sample image not found at {sample_img_path}")
        sys.exit(1)

    print(f"Sample image size on disk: {sample_img_path.stat().st_size} bytes")

    # 2. Test Preprocessing
    print("\n[Step 1] Verifying Image Preprocessing...")
    img_tensor = preprocess_image(sample_img_path)
    print(f"  Processed image tensor shape: {tuple(img_tensor.shape)}")
    print(f"  Dtype: {img_tensor.dtype}")
    print(f"  Value range: [{img_tensor.min().item():.4f}, {img_tensor.max().item():.4f}]")
    assert img_tensor.shape == (1, 3, 384, 384), f"Expected (1, 3, 384, 384), got {img_tensor.shape}"
    assert not torch.isnan(img_tensor).any(), "NaN found in image tensor!"
    assert not torch.isinf(img_tensor).any(), "Inf found in image tensor!"
    print("  Image preprocessing verification: PASSED")

    # 3. Test Metadata Encoding
    print("\n[Step 2] Verifying Metadata Encoding Variations...")
    
    # Case A: Normal specified metadata
    meta_a = encode_metadata(age=45, sex="male", localization="upper extremity")
    print(f"  Case A (Age=45, Sex=male, Loc=upper extremity) shape: {tuple(meta_a.shape)}")
    assert meta_a.shape == (1, 21), f"Expected (1, 21), got {meta_a.shape}"
    v_a = meta_a[0].numpy()
    assert abs(v_a[0] - (45.0 / 90.0)) < 1e-5, f"Expected age 0.5, got {v_a[0]}"
    assert v_a[1] == 0.0, "Age missing flag should be 0.0"
    assert v_a[3] == 1.0, "Sex male (idx 3) should be 1.0"
    assert v_a[17] == 1.0, "Localization upper extremity (idx 17) should be 1.0"
    print(f"  Case A vector verified: age={v_a[0]:.2f}, male_flag={v_a[3]}, loc_upper_ext={v_a[17]}")

    # Case B: Missing age
    meta_b = encode_metadata(age=None, sex="female", localization="back")
    v_b = meta_b[0].numpy()
    assert v_b[0] == 0.5 and v_b[1] == 1.0, "Missing age flag check failed"
    assert v_b[2] == 1.0, "Female flag check failed"
    print("  Case B (Missing age) verified: age=0.5, age_missing=1.0, female=1.0")

    # Case C: Completely missing metadata
    meta_c = encode_metadata(age=None, sex=None, localization=None)
    v_c = meta_c[0].numpy()
    assert v_c[0] == 0.5 and v_c[1] == 1.0, "Missing age check failed"
    assert v_c[4] == 1.0, "Sex unknown check failed"
    assert v_c[20] == 1.0, "Site unknown check failed"
    print("  Case C (All missing) verified: age=0.5, age_missing=1.0, sex_unknown=1.0, site_unknown=1.0")
    print("  Metadata encoding verification: PASSED")

    # 4. Pipeline Initialization & Inference
    print("\n[Step 3] Initializing InferencePipeline and running inference...")
    pipeline = InferencePipeline.get_instance()
    print(f"  Device: {pipeline.device}")
    print(f"  Fitted Temperature loaded: {pipeline.temperature:.6f}")

    # Primary Test with user's provided real sample & metadata
    # Age: 45, Sex: male, Anatomical localization: upper extremity
    result = pipeline.predict(
        image_source=sample_img_path,
        age=45,
        sex="male",
        localization="upper extremity"
    )

    print("\n[Step 4] Inference Results on Real HAM10000 Sample:")
    print(f"  Predicted Class: {result['predicted_class']} ({result['predicted_class_name']})")
    print(f"  Raw Confidence: {result['raw_confidence'] * 100:.2f}%")
    print(f"  Calibrated Confidence: {result['calibrated_confidence'] * 100:.2f}%")
    print(f"  Temperature Applied: {result['temperature_applied']:.6f}")

    print("\n  Class Probability Distribution:")
    print("  ----------------------------------------------------------------------")
    print(f"  {'Class':<8} {'Full Name':<42} {'Raw Prob':<12} {'Calibrated Prob':<15}")
    print("  ----------------------------------------------------------------------")
    for cls in CLASS_NAMES:
        raw_p = result["raw_probabilities"][cls]
        cal_p = result["calibrated_probabilities"][cls]
        print(f"  {cls:<8} {CLASS_FULL[cls]:<42} {raw_p * 100:>6.2f}%     {cal_p * 100:>6.2f}%")
    print("  ----------------------------------------------------------------------")

    # 5. Numerical Validity Checks
    raw_sum = sum(result["raw_probabilities"].values())
    cal_sum = sum(result["calibrated_probabilities"].values())
    print(f"\n  Sum of raw probabilities: {raw_sum:.6f}")
    print(f"  Sum of calibrated probabilities: {cal_sum:.6f}")

    assert abs(raw_sum - 1.0) < 1e-4, f"Raw probabilities sum ({raw_sum}) deviates from 1.0"
    assert abs(cal_sum - 1.0) < 1e-4, f"Calibrated probabilities sum ({cal_sum}) deviates from 1.0"

    for cls in CLASS_NAMES:
        rp = result["raw_probabilities"][cls]
        cp = result["calibrated_probabilities"][cls]
        assert not (math.isnan(rp) or math.isinf(rp)), f"Non-finite raw prob for {cls}"
        assert not (math.isnan(cp) or math.isinf(cp)), f"Non-finite cal prob for {cls}"
        assert 0.0 <= rp <= 1.0, f"Raw probability out of bounds: {rp}"
        assert 0.0 <= cp <= 1.0, f"Calibrated probability out of bounds: {cp}"

    print("  Numerical validity checks: PASSED")

    # 6. Post-Inference Ground-Truth Comparison
    GROUND_TRUTH = "nv"
    print("\n[Step 5] Ground-Truth Comparison (Post-Inference):")
    print(f"  Ground Truth: {GROUND_TRUTH} ({CLASS_FULL[GROUND_TRUTH]})")
    print(f"  Model Prediction: {result['predicted_class']} ({result['predicted_class_name']})")
    if result["predicted_class"] == GROUND_TRUTH:
        print("  Comparison: Model prediction matches the ground truth label!")
    else:
        print(f"  Comparison: Model predicted '{result['predicted_class']}' while ground truth is '{GROUND_TRUTH}'.")
        print(f"  Probability assigned to true class '{GROUND_TRUTH}': {result['calibrated_probabilities'][GROUND_TRUTH]*100:.2f}%")

    print("\n" + "=" * 70)
    print("PHASE 3: INFERENCE PIPELINE VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    import torch
    run_inference_tests()
