import base64
from io import BytesIO
from typing import Dict, Any, Optional, Union
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from app.preprocessing import load_image_as_rgb, preprocess_image, IMG_SIZE
from app.metadata import encode_metadata
from app.inference import InferencePipeline, CLASS_NAMES, CLASS_FULL

SCIENTIFIC_LIMITATION_DISCLAIMER = (
    "Grad-CAM visualizes spatial regions in the input image that most strongly influenced "
    "the EfficientNetV2-S convolutional feature branch. It does NOT explain the Swin Transformer V2 "
    "attention maps or the clinical metadata fusion branch. This visualization is an interpretability "
    "aid for research and education, not proof of clinical reasoning or diagnostic validation."
)

class GradCAMEngine:
    """
    Hook-based Grad-CAM explanation engine targeting the EfficientNetV2-S convolutional head:
    model.effnet.conv_head.
    
    Ensures safe hook lifecycle management (registration and cleanup in try/finally blocks).
    """
    def __init__(self, inference_pipeline: Optional[InferencePipeline] = None):
        self.pipeline = inference_pipeline or InferencePipeline.get_instance()
        self.model = self.pipeline.model
        self.device = self.pipeline.device
        self.target_layer = self.model.effnet.conv_head
        self.activations = None
        self.gradients = None

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def _array_to_base64_png(self, img_array_uint8: np.ndarray) -> str:
        """Encodes an RGB uint8 numpy array to a base64 PNG data URL."""
        pil_img = Image.fromarray(img_array_uint8)
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def explain(
        self,
        image_source: Any,
        age: Optional[Union[float, int, str]] = None,
        sex: Optional[str] = None,
        localization: Optional[str] = None,
        target_class: Optional[Union[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Generates Grad-CAM visual explanation for the EfficientNetV2-S branch.
        """
        # Load raw RGB image and resize to model dimensions (384, 384)
        raw_rgb = load_image_as_rgb(image_source)
        orig_resized_rgb = cv2.resize(raw_rgb, (IMG_SIZE, IMG_SIZE))
        orig_norm_float = orig_resized_rgb.astype(np.float32) / 255.0

        # Preprocess image and metadata tensors
        img_tensor = preprocess_image(image_source).to(self.device)
        meta_tensor = encode_metadata(age=age, sex=sex, localization=localization).to(self.device)

        # Register forward and backward hooks
        self.activations = None
        self.gradients = None
        f_handle = self.target_layer.register_forward_hook(self._save_activation)
        b_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

        try:
            # Ensure model is in eval mode and gradients are enabled for Grad-CAM
            self.model.eval()
            self.model.zero_grad(set_to_none=True)

            with torch.enable_grad():
                # Forward pass (without inference_mode/no_grad so graph is tracked)
                logits = self.model(img_tensor, meta_tensor)
                
                # Determine target class
                if target_class is None:
                    target_class_idx = int(logits.argmax(dim=1).item())
                elif isinstance(target_class, str):
                    target_class_clean = target_class.strip().lower()
                    if target_class_clean not in CLASS_NAMES:
                        raise ValueError(f"Unknown target class '{target_class}'. Must be one of {CLASS_NAMES}")
                    target_class_idx = CLASS_NAMES.index(target_class_clean)
                elif isinstance(target_class, int):
                    if not (0 <= target_class < len(CLASS_NAMES)):
                        raise ValueError(f"Target class index {target_class} out of range [0, {len(CLASS_NAMES)-1}]")
                    target_class_idx = target_class
                else:
                    raise TypeError(f"Invalid target_class type: {type(target_class)}")

                # Backward pass on the target class logit
                target_score = logits[0, target_class_idx]
                target_score.backward()

            # Ensure activations and gradients were captured
            if self.activations is None or self.gradients is None:
                raise RuntimeError("Failed to capture activations or gradients from model.effnet.conv_head")

            # Activations shape: [B, C, H, W], Gradients shape: [B, C, H, W]
            acts = self.activations[0]  # [C, H, W]
            grads = self.gradients[0]   # [C, H, W]

            # Global average pooling of gradients over spatial dimensions (H, W) -> [C, 1, 1]
            weights = grads.mean(dim=(1, 2), keepdim=True)

            # Weighted combination of activation maps
            cam = (weights * acts).sum(dim=0)  # [H, W]
            cam = torch.relu(cam).detach().cpu().numpy()

            # Resize to input dimensions (384, 384)
            cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

            # Normalize to [0, 1]
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 1e-8:
                cam_norm = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam_norm = np.zeros_like(cam)

            # Apply OpenCV JET colormap
            heatmap_uint8 = np.uint8(255 * cam_norm)
            heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
            heatmap_float = heatmap_rgb.astype(np.float32) / 255.0

            # Generate overlay: 0.6 * original + 0.4 * heatmap
            overlay_float = np.clip(0.6 * orig_norm_float + 0.4 * heatmap_float, 0.0, 1.0)
            overlay_uint8 = np.uint8(255 * overlay_float)

            # Calculate raw and calibrated probabilities
            with torch.no_grad():
                raw_probs = F.softmax(logits, dim=1).squeeze(0)
                cal_probs = F.softmax(logits / self.pipeline.temperature, dim=1).squeeze(0)

            pred_idx = int(cal_probs.argmax().item())
            target_cls_name = CLASS_NAMES[target_class_idx]

            return {
                "predicted_class": CLASS_NAMES[pred_idx],
                "predicted_class_name": CLASS_FULL[CLASS_NAMES[pred_idx]],
                "target_class": target_cls_name,
                "target_class_name": CLASS_FULL[target_cls_name],
                "target_class_raw_prob": float(raw_probs[target_class_idx].item()),
                "target_class_cal_prob": float(cal_probs[target_class_idx].item()),
                "heatmap_min": float(cam_norm.min()),
                "heatmap_max": float(cam_norm.max()),
                "original_image_base64": self._array_to_base64_png(orig_resized_rgb),
                "heatmap_base64": self._array_to_base64_png(heatmap_rgb),
                "overlay_base64": self._array_to_base64_png(overlay_uint8),
                "original_image_np": orig_resized_rgb,
                "heatmap_np": heatmap_rgb,
                "overlay_np": overlay_uint8,
                "cam_norm": cam_norm,
                "target_layer": "model.effnet.conv_head",
                "explanation_scope": "EfficientNetV2-S convolutional branch only",
                "limitation_disclaimer": SCIENTIFIC_LIMITATION_DISCLAIMER
            }

        finally:
            # Clean up hooks and cached tensors
            f_handle.remove()
            b_handle.remove()
            self.activations = None
            self.gradients = None
            self.model.zero_grad(set_to_none=True)
