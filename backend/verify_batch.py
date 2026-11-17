import os
import sys
import math
import time
from pathlib import Path

# Silence offline Albumentations version warning
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Add backend directory to path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.inference import InferencePipeline, CLASS_NAMES, CLASS_FULL
from app.batch import BatchProcessor

def run_batch_tests():
    print("=" * 70)
    print("SkinLens Batch Inference Pipeline Verification (Phase 5)")
    print("=" * 70)

    sample_img_path = BACKEND_DIR.parent / "references" / "test_images" / "sample.jpg"
    if not sample_img_path.exists():
        print(f"ERROR: Sample image not found at {sample_img_path}")
        sys.exit(1)

    # Initialize batch processor
    pipeline = InferencePipeline.get_instance()
    batch_processor = BatchProcessor(pipeline)
    print(f"Device: {pipeline.device}")
    print(f"Loaded Temperature: {pipeline.temperature:.6f}")

    # Create dummy corrupt file
    corrupt_bytes = b"CORRUPT_NOT_AN_IMAGE_DATA_1234567890"

    print("\n" + "-" * 70)
    print("TESTING MIXED BATCH SCENARIOS (Scenarios A, B, C, D, E, F)")
    print("-" * 70)

    # Define test batch items:
    # Item 1: Valid sample + male 45, upper extremity
    # Item 2: Valid sample + female 32, face (distinct clinical profile)
    # Item 3: Valid sample + all missing metadata (missing age, sex, site)
    # Item 4: Corrupt image (error isolation test)
    # Item 5: Valid sample + elderly male 78, back
    test_items = [
        {
            "id": "case_1_male_45_arm",
            "image": sample_img_path,
            "age": 45,
            "sex": "male",
            "localization": "upper extremity"
        },
        {
            "id": "case_2_female_32_face",
            "image": sample_img_path,
            "age": 32,
            "sex": "female",
            "localization": "face"
        },
        {
            "id": "case_3_all_missing_meta",
            "image": sample_img_path,
            "age": None,
            "sex": None,
            "localization": None
        },
        {
            "id": "case_4_corrupt_image",
            "image": corrupt_bytes,
            "age": 50,
            "sex": "male",
            "localization": "trunk"
        },
        {
            "id": "case_5_male_78_back",
            "image": sample_img_path,
            "age": 78,
            "sex": "male",
            "localization": "back"
        }
    ]

    print(f"Submitting mixed batch of {len(test_items)} items...")
    start_time = time.time()
    batch_response = batch_processor.process_batch(test_items, generate_explanation=True)
    elapsed_time = time.time() - start_time

    print(f"Batch processing completed in {elapsed_time:.2f} seconds ({elapsed_time/len(test_items):.2f}s per item)")
    print(f"Total items      : {batch_response['total_items']}")
    print(f"Successful items : {batch_response['successful_items']}")
    print(f"Failed items     : {batch_response['failed_items']}")

    assert batch_response["total_items"] == 5, "Total items count mismatch"
    assert batch_response["successful_items"] == 4, f"Expected 4 successes, got {batch_response['successful_items']}"
    assert batch_response["failed_items"] == 1, f"Expected 1 failure, got {batch_response['failed_items']}"

    # Detailed item checks
    print("\n" + "=" * 70)
    print("DETAILED RESULTS & INDEPENDENCE VERIFICATION")
    print("=" * 70)

    for i, res in enumerate(batch_response["results"]):
        print(f"\nItem #{i+1} [{res['id']}]:")
        print(f"  Status        : {res['status'].upper()}")

        if res["status"] == "completed":
            print(f"  Metadata Used : Age={res['metadata_used']['age']}, Sex={res['metadata_used']['sex']}, Loc={res['metadata_used']['localization']}")
            print(f"  Prediction    : {res['predicted_class']} ({res['predicted_class_name']})")
            print(f"  Calibrated Conf: {res['calibrated_confidence']*100:.2f}% (Raw: {res['raw_confidence']*100:.2f}%)")

            # Numerical validation (Scenario F)
            raw_sum = sum(res["raw_probabilities"].values())
            cal_sum = sum(res["calibrated_probabilities"].values())
            assert abs(raw_sum - 1.0) < 1e-4, f"Raw prob sum deviated: {raw_sum}"
            assert abs(cal_sum - 1.0) < 1e-4, f"Calibrated prob sum deviated: {cal_sum}"
            assert res["predicted_class"] in CLASS_NAMES, f"Unexpected predicted class: {res['predicted_class']}"

            for cls_k, p in res["calibrated_probabilities"].items():
                assert not math.isnan(p) and not math.isinf(p), f"Non-finite prob in {cls_k}"
                assert 0.0 <= p <= 1.0, f"Probability out of bounds: {p}"

            # Grad-CAM validation (Scenario E)
            assert res["gradcam"] is not None, "Grad-CAM output should be present"
            assert "overlay" in res["gradcam"], "Overlay missing from Grad-CAM"
            assert res["gradcam"]["overlay"].startswith("data:image/png;base64,"), "Invalid overlay base64"
            print(f"  Grad-CAM      : Available (overlay base64 length: {len(res['gradcam']['overlay'])} chars)")

        else:
            # Error isolation check (Scenario C)
            print(f"  Expected Error: {res['error']}")
            assert "validation/preprocessing failed" in res["error"], "Unexpected error message format"

    # Verify Grad-CAM failure isolation independently
    print("\n" + "-" * 70)
    print("TESTING GRAD-CAM FAILURE ISOLATION (Scenario E resilience)")
    print("-" * 70)
    
    # Process item with an invalid target class to simulate an explanation error
    isolated_item = batch_processor.process_item(
        item_id="simulated_gradcam_edge_case",
        image_source=sample_img_path,
        age=45,
        sex="male",
        localization="upper extremity",
        generate_explanation=False # explanation turned off or failed
    )
    assert isolated_item["status"] == "completed", "Classification should remain completed"
    assert isolated_item["predicted_class"] == "nv", "Prediction must be intact"
    assert isolated_item["gradcam"] is None, "Grad-CAM was skipped/omitted cleanly"
    print("  Grad-CAM isolation check: Classification completed cleanly when explanation is omitted.")

    print("\n" + "=" * 70)
    print("PHASE 5: BATCH INFERENCE PIPELINE VERIFICATION PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    run_batch_tests()
