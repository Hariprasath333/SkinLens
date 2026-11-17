import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
import torch.nn.functional as F

from model.architecture import SkinLensHybrid
from app.preprocessing import preprocess_image
from app.metadata import encode_metadata

# Exact HAM10000 7-class ordering from notebook
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_FULL = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

class InferencePipeline:
    """
    Singleton inference engine for SkinLens.
    Loads checkpoint and calibration temperature once at initialization.
    """
    _instance: Optional["InferencePipeline"] = None

    def __init__(self, checkpoint_path: Optional[Path] = None, temp_path: Optional[Path] = None):
        base_dir = Path(__file__).resolve().parent.parent
        self.checkpoint_path = checkpoint_path or (base_dir / "models" / "best_skinlens_hybrid.pt")
        self.temp_path = temp_path or (base_dir / "models" / "temperature.json")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = self._load_temperature()
        self.model = self._load_model()

    @classmethod
    def get_instance(cls) -> "InferencePipeline":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_temperature(self) -> float:
        if not self.temp_path.exists():
            raise FileNotFoundError(f"Temperature file not found at: {self.temp_path}")
        with open(self.temp_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        temp_val = float(data.get("temperature", 1.0))
        return temp_val

    def _load_model(self) -> SkinLensHybrid:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {self.checkpoint_path}")

        model = SkinLensHybrid(num_classes=len(CLASS_NAMES), meta_dim=21, pretrained=False)
        state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        return model

    def predict(
        self,
        image_source: Any,
        age: Optional[Union[float, int, str]] = None,
        sex: Optional[str] = None,
        localization: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs single-image inference:
        1. Preprocesses image -> (1, 3, 384, 384)
        2. Encodes metadata -> (1, 21)
        3. Runs forward pass through SkinLensHybrid
        4. Scales logits by fitted temperature
        5. Computes raw and calibrated probabilities
        """
        img_tensor = preprocess_image(image_source).to(self.device)
        meta_tensor = encode_metadata(age=age, sex=sex, localization=localization).to(self.device)

        with torch.inference_mode():
            logits = self.model(img_tensor, meta_tensor)  # shape: (1, 7)
            raw_probs = F.softmax(logits, dim=1).squeeze(0)  # shape: (7,)
            calibrated_probs = F.softmax(logits / self.temperature, dim=1).squeeze(0)  # shape: (7,)

        pred_idx = int(calibrated_probs.argmax().item())
        pred_class = CLASS_NAMES[pred_idx]
        pred_class_name = CLASS_FULL[pred_class]

        raw_probs_list = [float(p) for p in raw_probs.cpu().tolist()]
        cal_probs_list = [float(p) for p in calibrated_probs.cpu().tolist()]
        logits_list = [float(l) for l in logits.squeeze(0).cpu().tolist()]

        raw_dist = {cls_name: raw_probs_list[i] for i, cls_name in enumerate(CLASS_NAMES)}
        cal_dist = {cls_name: cal_probs_list[i] for i, cls_name in enumerate(CLASS_NAMES)}

        return {
            "predicted_class": pred_class,
            "predicted_class_name": pred_class_name,
            "calibrated_confidence": cal_probs_list[pred_idx],
            "raw_confidence": raw_probs_list[pred_idx],
            "calibrated_probabilities": cal_dist,
            "raw_probabilities": raw_dist,
            "logits": logits_list,
            "temperature_applied": self.temperature,
            "device_used": str(self.device),
            "image_shape": list(img_tensor.shape),
            "metadata_shape": list(meta_tensor.shape),
        }
