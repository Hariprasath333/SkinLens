import os
import sys
from pathlib import Path
import numpy as np

# Set environment variable to silence offline Albumentations version check warning
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Add backend directory to path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.inference import InferencePipeline, CLASS_NAMES, CLASS_FULL
from app.gradcam import GradCAMEngine, SCIENTIFIC_LIMITATION_DISCLAIMER

def verify_gradcam():
    print("=" * 70)
    print("SkinLens Grad-CAM Engine Verification (Phase 4)")
    print("=" * 70)

    # 1. Image path
    sample_img_path = BACKEND_DIR.parent / "references" / "test_images" / "sample.jpg"
    print(f"Sample image path: {sample_img_path}")
    if not sample_img_path.exists():
        print(f"ERROR: Sample image not found at {sample_img_path}")
        sys.exit(1)

    # 2. Initialize Inference & Grad-CAM engines
    print("\n[Step 1] Initializing InferencePipeline and GradCAMEngine...")
    pipeline = InferencePipeline.get_instance()
    gradcam_engine = GradCAMEngine(pipeline)
    print(f"  Device: {pipeline.device}")
    print(f"  Target layer: {gradcam_engine.target_layer}")

    # 3. Predict & Generate Grad-CAM
    # Metadata: Age: 45, Sex: male, Localization: upper extremity
    print("\n[Step 2] Generating Grad-CAM on real sample:")
    print("  Metadata: Age=45, Sex=male, Localization=upper extremity")
    
    result = gradcam_engine.explain(
        image_source=sample_img_path,
        age=45,
        sex="male",
        localization="upper extremity"
    )

    pred_cls = result["predicted_class"]
    target_cls = result["target_class"]
    print(f"  Predicted class : {pred_cls} ({result['predicted_class_name']})")
    print(f"  Target class    : {target_cls} ({result['target_class_name']})")
    print(f"  Calibrated Prob : {result['target_class_cal_prob'] * 100:.2f}%")
    print(f"  Raw Prob        : {result['target_class_raw_prob'] * 100:.2f}%")

    # 4. Verify Dimensions and Numerical Properties
    print("\n[Step 3] Verifying Dimensions and Numerical Properties...")
    cam_norm = result["cam_norm"]
    heatmap_np = result["heatmap_np"]
    overlay_np = result["overlay_np"]
    orig_np = result["original_image_np"]

    print(f"  Heatmap array shape : {cam_norm.shape} (dtype: {cam_norm.dtype})")
    print(f"  Heatmap RGB shape   : {heatmap_np.shape} (dtype: {heatmap_np.dtype})")
    print(f"  Overlay RGB shape   : {overlay_np.shape} (dtype: {overlay_np.dtype})")
    print(f"  Original RGB shape  : {orig_np.shape} (dtype: {orig_np.dtype})")

    assert cam_norm.shape == (384, 384), f"Expected (384, 384), got {cam_norm.shape}"
    assert heatmap_np.shape == (384, 384, 3), f"Expected (384, 384, 3), got {heatmap_np.shape}"
    assert overlay_np.shape == (384, 384, 3), f"Expected (384, 384, 3), got {overlay_np.shape}"
    assert orig_np.shape == (384, 384, 3), f"Expected (384, 384, 3), got {orig_np.shape}"

    # Check finite numbers (no NaN or Inf)
    assert not np.isnan(cam_norm).any(), "NaN found in Grad-CAM heatmap!"
    assert not np.isinf(cam_norm).any(), "Inf found in Grad-CAM heatmap!"
    assert not np.isnan(overlay_np).any(), "NaN found in overlay image!"
    assert not np.isinf(overlay_np).any(), "Inf found in overlay image!"

    print(f"  Heatmap value range : [{cam_norm.min():.4f}, {cam_norm.max():.4f}]")
    assert 0.0 <= cam_norm.min() and cam_norm.max() <= 1.0 + 1e-6, "Heatmap out of [0, 1] range!"
    print("  Dimensions and numerical checks: ALL PASSED")

    # 5. Check Base64 Outputs
    print("\n[Step 4] Verifying Base64 Image Outputs...")
    assert result["original_image_base64"].startswith("data:image/png;base64,"), "Invalid original image base64"
    assert result["heatmap_base64"].startswith("data:image/png;base64,"), "Invalid heatmap base64"
    assert result["overlay_base64"].startswith("data:image/png;base64,"), "Invalid overlay base64"
    print(f"  Original image base64 length : {len(result['original_image_base64'])} chars")
    print(f"  Heatmap base64 length        : {len(result['heatmap_base64'])} chars")
    print(f"  Overlay base64 length        : {len(result['overlay_base64'])} chars")
    print("  Base64 image encoding: ALL PASSED")

    # 6. Save Composite Visualization Image
    print("\n[Step 5] Saving 3-Panel Comparison Figure...")
    output_path = BACKEND_DIR / "outputs" / "gradcam_visualization.png"

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=300)

        # Panel 1: Original Image
        axes[0].imshow(orig_np)
        axes[0].set_title("1. Original Dermoscopic Image\n(Sample HAM10000 | Ground Truth: nv)", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Grad-CAM Heatmap
        im2 = axes[1].imshow(cam_norm, cmap="jet")
        axes[1].set_title(f"2. Grad-CAM Heatmap\nBranch: {result['target_layer']}", fontsize=11, fontweight="bold")
        axes[1].axis("off")
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: Overlay
        axes[2].imshow(overlay_np)
        axes[2].set_title(
            f"3. Heatmap Overlay (0.6x Original + 0.4x Heatmap)\n"
            f"Pred: {pred_cls} | Calibrated Conf: {result['target_class_cal_prob']*100:.1f}%",
            fontsize=11,
            fontweight="bold"
        )
        axes[2].axis("off")

        plt.suptitle("SkinLens Grad-CAM Visual Explanation (EfficientNetV2-S Branch Only)", fontsize=13, y=0.98)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  Note: matplotlib visualization fallback to PIL rendering ({e})")
        from PIL import Image, ImageDraw

        panel_w, panel_h = 384, 384
        header_h = 70
        total_w = panel_w * 3 + 40
        total_h = panel_h + header_h + 30

        canvas = Image.new("RGB", (total_w, total_h), color=(248, 249, 250))
        draw = ImageDraw.Draw(canvas)

        # Paste images
        p1 = Image.fromarray(orig_np)
        p2 = Image.fromarray(heatmap_np)
        p3 = Image.fromarray(overlay_np)

        canvas.paste(p1, (10, header_h))
        canvas.paste(p2, (20 + panel_w, header_h))
        canvas.paste(p3, (30 + panel_w * 2, header_h))

        # Add text headers
        draw.text((15, 15), "SkinLens Grad-CAM Explanation (EfficientNetV2-S Branch Only)", fill=(20, 20, 20))
        draw.text((10, 48), "1. Original Image (Ground Truth: nv)", fill=(50, 50, 50))
        draw.text((20 + panel_w, 48), "2. Grad-CAM Heatmap (conv_head)", fill=(50, 50, 50))
        draw.text((30 + panel_w * 2, 48), f"3. Overlay (Pred: {pred_cls} | Calibrated: {result['target_class_cal_prob']*100:.1f}%)", fill=(50, 50, 50))

        canvas.save(output_path, format="PNG")

    print(f"  Visualization successfully saved to: {output_path}")
    print(f"  Output file size: {output_path.stat().st_size} bytes")

    # 7. Print Scientific Limitation
    print("\n[Step 6] Scientific Interpretability Scope:")
    print(f"  Scope: {result['explanation_scope']}")
    print(f"  Notice: {result['limitation_disclaimer']}")

    print("\n" + "=" * 70)
    print("PHASE 4: GRAD-CAM VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    verify_gradcam()
