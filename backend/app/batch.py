import gc
import logging
from typing import List, Dict, Any, Optional
import torch

from app.preprocessing import preprocess_image
from app.metadata import encode_metadata
from app.inference import InferencePipeline, CLASS_NAMES, CLASS_FULL
from app.gradcam import GradCAMEngine, SCIENTIFIC_LIMITATION_DISCLAIMER

logger = logging.getLogger("skinlens.batch")

class BatchProcessor:
    """
    Sequential batch inference processor for SkinLens.
    
    Processes multiple dermoscopic images independently:
    - Each image receives its own independent metadata, prediction, and Grad-CAM explanation.
    - Errors in individual items are isolated without impacting other items in the batch.
    - Sequential execution avoids CPU memory accumulation and thread contention.
    """
    def __init__(self, inference_pipeline: Optional[InferencePipeline] = None):
        self.pipeline = inference_pipeline or InferencePipeline.get_instance()
        self.gradcam_engine = GradCAMEngine(self.pipeline)

    def process_item(
        self,
        item_id: str,
        image_source: Any,
        age: Optional[Any] = None,
        sex: Optional[str] = None,
        localization: Optional[str] = None,
        generate_explanation: bool = True
    ) -> Dict[str, Any]:
        """
        Processes a single item within the batch with complete error isolation.
        """
        result: Dict[str, Any] = {
            "id": str(item_id),
            "status": "failed",
            "predicted_class": None,
            "predicted_class_name": None,
            "raw_confidence": None,
            "calibrated_confidence": None,
            "raw_probabilities": None,
            "calibrated_probabilities": None,
            "metadata_used": {
                "age": age,
                "sex": sex,
                "localization": localization
            },
            "gradcam": None,
            "error": None
        }

        # Step 1: Preprocess Image
        try:
            img_tensor = preprocess_image(image_source).to(self.pipeline.device)
        except Exception as e:
            result["error"] = f"Image validation/preprocessing failed: {str(e)}"
            return result

        # Step 2: Encode Metadata (always succeeds because missing/invalid values fallback to canonical encoding)
        meta_tensor = encode_metadata(age=age, sex=sex, localization=localization).to(self.pipeline.device)

        # Step 3: Run Model Forward Pass & Temperature Scaling
        try:
            with torch.inference_mode():
                logits = self.pipeline.model(img_tensor, meta_tensor)
                raw_probs = torch.softmax(logits, dim=1).squeeze(0)
                cal_probs = torch.softmax(logits / self.pipeline.temperature, dim=1).squeeze(0)

            pred_idx = int(cal_probs.argmax().item())
            pred_cls = CLASS_NAMES[pred_idx]

            raw_dist = {cls_name: float(raw_probs[i].item()) for i, cls_name in enumerate(CLASS_NAMES)}
            cal_dist = {cls_name: float(cal_probs[i].item()) for i, cls_name in enumerate(CLASS_NAMES)}

            result.update({
                "status": "completed",
                "predicted_class": pred_cls,
                "predicted_class_name": CLASS_FULL[pred_cls],
                "raw_confidence": float(raw_probs[pred_idx].item()),
                "calibrated_confidence": float(cal_probs[pred_idx].item()),
                "raw_probabilities": raw_dist,
                "calibrated_probabilities": cal_dist,
            })
        except Exception as e:
            result["error"] = f"Model inference failed: {str(e)}"
            return result
        finally:
            del img_tensor, meta_tensor

        # Step 4: Optional Grad-CAM Explanation (isolated from classification success)
        if generate_explanation and result["status"] == "completed":
            try:
                cam_res = self.gradcam_engine.explain(
                    image_source=image_source,
                    age=age,
                    sex=sex,
                    localization=localization,
                    target_class=result["predicted_class"]
                )
                result["gradcam"] = {
                    "original_image": cam_res["original_image_base64"],
                    "heatmap": cam_res["heatmap_base64"],
                    "overlay": cam_res["overlay_base64"],
                    "target_layer": cam_res["target_layer"],
                    "scope": cam_res["explanation_scope"],
                    "disclaimer": cam_res["limitation_disclaimer"]
                }
            except Exception as e:
                logger.warning(f"Grad-CAM explanation failed for item {item_id}: {e}")
                result["gradcam"] = {
                    "available": False,
                    "error": f"Explanation generation failed: {str(e)}",
                    "disclaimer": SCIENTIFIC_LIMITATION_DISCLAIMER
                }

        # Clean memory after each item
        gc.collect()
        return result

    def process_batch(
        self,
        items: List[Dict[str, Any]],
        default_metadata: Optional[Dict[str, Any]] = None,
        generate_explanation: bool = True
    ) -> Dict[str, Any]:
        """
        Executes batch inference over a list of items sequentially.
        
        Args:
            items: List of dicts, each containing:
                   - 'id': unique identifier
                   - 'image': file path, bytes, or PIL Image
                   - 'age': optional
                   - 'sex': optional
                   - 'localization': optional
            default_metadata: optional shared metadata fallback for fields omitted in item.
            generate_explanation: whether to compute Grad-CAM for each image.
        """
        results = []
        successful_count = 0
        failed_count = 0

        for idx, item in enumerate(items):
            item_id = item.get("id", f"case_{idx + 1}")
            image_source = item.get("image")

            # Determine metadata with individual priority over shared default
            def_meta = default_metadata or {}
            age = item.get("age") if item.get("age") is not None else def_meta.get("age")
            sex = item.get("sex") if item.get("sex") is not None else def_meta.get("sex")
            loc = item.get("localization") if item.get("localization") is not None else def_meta.get("localization")

            # Execute single item with isolated exception catching
            item_result = self.process_item(
                item_id=item_id,
                image_source=image_source,
                age=age,
                sex=sex,
                localization=loc,
                generate_explanation=generate_explanation
            )

            if item_result["status"] == "completed":
                successful_count += 1
            else:
                failed_count += 1

            results.append(item_result)

        return {
            "total_items": len(items),
            "successful_items": successful_count,
            "failed_items": failed_count,
            "results": results
        }
