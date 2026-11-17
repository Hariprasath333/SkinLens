import os
import sys
import time
import json
from pathlib import Path

# Silence offline Albumentations version warning
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from app.main import app, MAX_FILE_SIZE_BYTES
from app.inference import CLASS_NAMES

def run_tests():
    print("=" * 75)
    print("SkinLens FastAPI Backend Automated Verification Suite")
    print("=" * 75)

    sample_img_path = BACKEND_DIR.parent / "references" / "test_images" / "sample.jpg"
    if not sample_img_path.exists():
        print(f"ERROR: Sample image not found at {sample_img_path}")
        sys.exit(1)

    with open(sample_img_path, "rb") as f:
        real_image_bytes = f.read()

    test_results = []

    def record_result(name: str, passed: bool, duration: float, detail: str = ""):
        test_results.append({
            "name": name,
            "status": "PASS" if passed else "FAIL",
            "duration": duration,
            "detail": detail
        })
        status_str = "[PASS]" if passed else "[FAIL]"
        print(f"  --> {status_str} in {duration:.2f}s | {detail}")

    print("\n[Phase A] Initializing TestClient and FastAPI Lifespan (Single Model Load)...", flush=True)
    t_start_client = time.time()
    
    # TestClient manages application lifespan context
    with TestClient(app) as client:
        t_client_ready = time.time() - t_start_client
        print(f"  Lifespan completed and model ready in {t_client_ready:.2f}s.\n")

        # ------------------------------------------------------------------
        # Group 1: Lightweight Metadata & Validation Tests (Fast)
        # ------------------------------------------------------------------
        print("-" * 75)
        print("GROUP 1: Lightweight Endpoints & Request Validation (No Heavy Inference)")
        print("-" * 75)

        # Test 1: GET /api/health
        print("\nTest 1: GET /api/health", flush=True)
        t0 = time.time()
        try:
            res = client.get("/api/health")
            duration = time.time() - t0
            data = res.json()
            assert res.status_code == 200, f"Expected 200, got {res.status_code}"
            assert data["status"] == "healthy"
            assert data["model_ready"] is True
            assert abs(data["temperature"] - 1.064053) < 1e-4
            record_result("GET /api/health", True, duration, f"status='{data['status']}', temp={data['temperature']:.6f}")
        except Exception as e:
            record_result("GET /api/health", False, time.time() - t0, str(e))

        # Test 2: GET /api/classes
        print("\nTest 2: GET /api/classes", flush=True)
        t0 = time.time()
        try:
            res = client.get("/api/classes")
            duration = time.time() - t0
            data = res.json()
            assert res.status_code == 200, f"Expected 200, got {res.status_code}"
            assert data["total"] == 7
            keys = [c["key"] for c in data["classes"]]
            assert keys == CLASS_NAMES, f"Class list mismatch: {keys}"
            record_result("GET /api/classes", True, duration, f"7 classes verified in canonical order: {keys}")
        except Exception as e:
            record_result("GET /api/classes", False, time.time() - t0, str(e))

        # Test 3: Unsupported file format rejection
        print("\nTest 3: Reject unsupported format (.txt)", flush=True)
        t0 = time.time()
        try:
            res = client.post("/api/predict", files={"image": ("notes.txt", b"plain text data", "text/plain")})
            duration = time.time() - t0
            assert res.status_code == 400, f"Expected 400, got {res.status_code}"
            record_result("Validation: Unsupported File Format", True, duration, f"HTTP {res.status_code} - Rejected non-image input")
        except Exception as e:
            record_result("Validation: Unsupported File Format", False, time.time() - t0, str(e))

        # Test 4: Oversized upload rejection
        print("\nTest 4: Reject oversized file (> 15MB limit)", flush=True)
        t0 = time.time()
        try:
            oversized = b"\xff\xd8\xff" + b"X" * (MAX_FILE_SIZE_BYTES + 512)
            res = client.post("/api/predict", files={"image": ("large.jpg", oversized, "image/jpeg")})
            duration = time.time() - t0
            assert res.status_code == 413, f"Expected 413, got {res.status_code}"
            record_result("Validation: Oversized Upload (>15MB)", True, duration, f"HTTP {res.status_code} - Payload Too Large")
        except Exception as e:
            record_result("Validation: Oversized Upload (>15MB)", False, time.time() - t0, str(e))

        # Test 5: Missing required image field
        print("\nTest 5: Reject request missing required image field", flush=True)
        t0 = time.time()
        try:
            res = client.post("/api/predict", data={"age": "45"})
            duration = time.time() - t0
            assert res.status_code == 422, f"Expected 422, got {res.status_code}"
            record_result("Validation: Missing Image Field", True, duration, f"HTTP {res.status_code} - Unprocessable Entity")
        except Exception as e:
            record_result("Validation: Missing Image Field", False, time.time() - t0, str(e))

        # Test 6: Malformed metadata_json in batch
        print("\nTest 6: Reject malformed metadata_json in batch", flush=True)
        t0 = time.time()
        try:
            files = [("images", ("test.jpg", real_image_bytes, "image/jpeg"))]
            res = client.post("/api/predict/batch", files=files, data={"metadata_json": "MALFORMED_JSON_STRING"})
            duration = time.time() - t0
            assert res.status_code == 400, f"Expected 400, got {res.status_code}"
            record_result("Validation: Malformed Metadata JSON", True, duration, f"HTTP {res.status_code} - Bad Request")
        except Exception as e:
            record_result("Validation: Malformed Metadata JSON", False, time.time() - t0, str(e))

        # ------------------------------------------------------------------
        # Group 2: Model Inference & Interpretability Tests
        # ------------------------------------------------------------------
        print("\n" + "-" * 75)
        print("GROUP 2: Core Model Inference & Interpretability (Real HAM10000 Sample)")
        print("-" * 75)

        # Test 7: POST /api/predict (Real sample, full metadata + Grad-CAM)
        print("\nTest 7: POST /api/predict (Real sample: Age=45, Sex=male, Loc=upper extremity)", flush=True)
        t0 = time.time()
        try:
            files = {"image": ("sample.jpg", real_image_bytes, "image/jpeg")}
            form = {"age": "45", "sex": "male", "localization": "upper extremity", "include_gradcam": "true"}
            res = client.post("/api/predict", files=files, data=form)
            duration = time.time() - t0
            assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
            data = res.json()
            assert data["predicted_class"] == "nv"
            assert data["gradcam"] is not None and data["gradcam"]["available"] is True
            cal_conf = data["calibrated_confidence"] * 100
            raw_conf = data["raw_confidence"] * 100
            record_result("POST /api/predict (Full Case)", True, duration, f"Pred: {data['predicted_class']} | Calibrated: {cal_conf:.2f}% (Raw: {raw_conf:.2f}%) | Grad-CAM: OK")
        except Exception as e:
            record_result("POST /api/predict (Full Case)", False, time.time() - t0, str(e))

        # Test 8: POST /api/predict with completely omitted metadata
        print("\nTest 8: POST /api/predict (Missing metadata test)", flush=True)
        t0 = time.time()
        try:
            files = {"image": ("sample.jpg", real_image_bytes, "image/jpeg")}
            form = {"include_gradcam": "false"}  # no metadata passed
            res = client.post("/api/predict", files=files, data=form)
            duration = time.time() - t0
            assert res.status_code == 200, f"Expected 200, got {res.status_code}"
            data = res.json()
            assert data["predicted_class"] == "nv"
            assert data["metadata_used"]["age"] is None
            cal_conf = data["calibrated_confidence"] * 100
            record_result("POST /api/predict (Missing Metadata)", True, duration, f"Pred: {data['predicted_class']} | Calibrated: {cal_conf:.2f}% | Defaulted correctly")
        except Exception as e:
            record_result("POST /api/predict (Missing Metadata)", False, time.time() - t0, str(e))

        # Test 9: POST /api/predict/batch with mixed batch (1 valid + 1 corrupt image)
        print("\nTest 9: POST /api/predict/batch (Mixed batch: 1 valid + 1 corrupt image)", flush=True)
        t0 = time.time()
        try:
            mixed_files = [
                ("images", ("case_valid.jpg", real_image_bytes, "image/jpeg")),
                ("images", ("case_corrupt.jpg", b"CORRUPTED_NON_IMAGE_BYTES_0123", "image/jpeg")),
            ]
            meta = json.dumps([
                {"age": 45, "sex": "male", "localization": "upper extremity"},
                {"age": 28, "sex": "female", "localization": "back"}
            ])
            res = client.post("/api/predict/batch", files=mixed_files, data={"metadata_json": meta, "include_gradcam": "true"})
            duration = time.time() - t0
            assert res.status_code == 200, f"Expected 200, got {res.status_code}"
            data = res.json()
            assert data["total_items"] == 2
            assert data["successful_items"] == 1
            assert data["failed_items"] == 1
            assert data["results"][0]["status"] == "completed"
            assert data["results"][0]["predicted_class"] == "nv"
            assert data["results"][1]["status"] == "failed"
            record_result("POST /api/predict/batch (Mixed Batch)", True, duration, f"Total: 2, Success: 1, Failed: 1 (Isolated cleanly)")
        except Exception as e:
            record_result("POST /api/predict/batch (Mixed Batch)", False, time.time() - t0, str(e))

        # Test 10: POST /api/explain (Grad-CAM dedicated endpoint)
        print("\nTest 10: POST /api/explain (Grad-CAM endpoint)", flush=True)
        t0 = time.time()
        try:
            files = {"image": ("sample.jpg", real_image_bytes, "image/jpeg")}
            form = {"age": "45", "sex": "male", "localization": "upper extremity", "target_class": "nv"}
            res = client.post("/api/explain", files=files, data=form)
            duration = time.time() - t0
            assert res.status_code == 200, f"Expected 200, got {res.status_code}"
            data = res.json()
            assert data["target_class"] == "nv"
            assert data["target_layer"] == "model.effnet.conv_head"
            assert data["overlay"].startswith("data:image/png;base64,")
            record_result("POST /api/explain", True, duration, f"Target: {data['target_class']}, Layer: {data['target_layer']}")
        except Exception as e:
            record_result("POST /api/explain", False, time.time() - t0, str(e))

    # Print Summary Table
    print("\n" + "=" * 75)
    print("AUTOMATED VERIFICATION SUMMARY REPORT")
    print("=" * 75)
    print(f"{'#':<3} {'Test Name':<38} {'Status':<8} {'Time (s)':<10} {'Notes'}")
    print("-" * 75)
    all_passed = True
    total_time = 0.0
    for idx, r in enumerate(test_results, 1):
        total_time += r["duration"]
        if r["status"] != "PASS":
            all_passed = False
        print(f"{idx:<3} {r['name']:<38} {r['status']:<8} {r['duration']:<10.2f} {r['detail']}")
    print("-" * 75)
    print(f"Total API test execution time: {total_time:.2f}s | Overall Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 75)

    if not all_passed:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
