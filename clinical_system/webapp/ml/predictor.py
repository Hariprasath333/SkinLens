"""
Model loading and prediction for the web app.

Three kinds of model are supported (chosen by MODEL_KIND in config.py):
  * "timm"     - the best model / ensemble exported by research/evaluate.py (8 classes)
  * "skinlens" - the original SkinLens hybrid model (backup/original_model, 7 HAM10000 classes,
                 uses age / sex / body site)
  * "dummy"    - a tiny untrained network, ONLY for automated tests
"auto" picks "timm" if an exported model exists, otherwise "skinlens".
"""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config

sys.path.insert(0, str(config.SYSTEM_DIR / "research"))
from common import CLASS_FULL, CLASSES, IMAGENET_MEAN, IMAGENET_STD, MODELS, RISK  # noqa: E402

HAM_TO_CODE = {"akiec": "AK", "bcc": "BCC", "bkl": "BKL", "df": "DF", "mel": "MEL", "nv": "NV", "vasc": "VASC"}
SKINLENS_SITES = {"abdomen", "back", "chest", "ear", "face", "foot", "genital", "hand", "lower extremity",
                  "neck", "scalp", "trunk", "upper extremity", "acral", "oral/genital", "unknown"}


@dataclass
class Member:
    """One network plus what Grad-CAM needs to know about it."""
    model: nn.Module
    size: int
    temperature: float
    cam_layer: nn.Module
    tokens: int = 0                       # >0 for ViT: number of prefix tokens to drop for Grad-CAM
    forward: Callable | None = None       # forward(x) -> logits (used for models needing metadata)


class Predictor:
    """Common interface: predict(), predict_batch() for LIME, members for Grad-CAM."""

    name: str = ""
    classes: list[str] = []
    members: list[Member] = []
    uses_metadata: bool = False
    is_dummy: bool = False
    prior: "PriorCorrection | None" = None   # set by get_predictor() from model_settings.json

    def __init__(self):
        self.lock = threading.RLock()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._meta = None

    @contextmanager
    def session(self, age=None, sex=None, site=None):
        """Holds the model for one patient: metadata is set for every forward pass inside."""
        with self.lock:
            self._meta = (age, sex, site)
            try:
                yield self
            finally:
                self._meta = None

    # -- preprocessing
    @staticmethod
    def to_tensor(rgb: np.ndarray, size: int) -> torch.Tensor:
        # INTER_LINEAR = the resize used in training (v1.5.1; v1.5.0 used INTER_AREA, which the Validation
        # Lab's T2 found differed from the training notebook on about 3% of test images)
        x = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
        return torch.from_numpy(x.transpose(2, 0, 1))

    def _forward(self, m: Member, x: torch.Tensor) -> torch.Tensor:
        return m.forward(x) if m.forward else m.model(x)

    @torch.no_grad()
    def predict_batch(self, images: list[np.ndarray], calibrated: bool = True) -> np.ndarray:
        """Class probabilities (ensemble = mean over members). calibrated=False gives the plain
        softmax of the model (no temperature scaling), which is what prior correction expects.
        Used directly by LIME, so LIME explains the model output exactly as before."""
        out = []
        for m in self.members:
            x = torch.stack([self.to_tensor(im, m.size) for im in images]).to(self.device)
            t = m.temperature if calibrated else 1.0
            out.append(F.softmax(self._forward(m, x).float() / t, 1).cpu().numpy())
        p = np.mean(out, 0)
        t = getattr(self, "ensemble_temperature", None)
        if calibrated and t and len(out) > 1:  # calibration of the averaged ensemble (calibrate stage)
            z = np.log(np.clip(p, 1e-12, 1)) / t
            z = np.exp(z - z.max(1, keepdims=True))
            p = z / z.sum(1, keepdims=True)
        return p

    def predict(self, rgb: np.ndarray) -> dict:
        """The probabilities shown to the user. With prior correction active they are the CORRECTED
        probabilities; 'raw_probs' keeps the model's own output for reference."""
        with self.lock:
            if self.prior is not None:
                raw = self.predict_batch([rgb], calibrated=False)[0]
                p = self.prior.apply(raw)
            else:
                raw = p = self.predict_batch([rgb])[0]
        order = np.argsort(-p)
        return {
            "probs": {self.classes[i]: float(p[i]) for i in range(len(p))},
            "raw_probs": {self.classes[i]: float(raw[i]) for i in range(len(raw))},
            "ranked": [(self.classes[i], float(p[i])) for i in order],
            "top_class": self.classes[order[0]],
            "prior_corrected": self.prior is not None,
        }

    @property
    def display_name(self) -> str:
        """Model name as stored with each scan, including the post-processing used."""
        return self.name + (f" + prior correction (alpha={self.prior.alpha:g})" if self.prior is not None else "")

    def describe(self) -> dict:
        return dict(name=self.name, classes=self.classes, uses_metadata=self.uses_metadata,
                    is_dummy=self.is_dummy, device=str(self.device),
                    prior_correction=self.prior.alpha if self.prior is not None else None,
                    prior_note=getattr(self, "prior_note", ""))


# ---------------------------------------------------------------------------- prior correction
def load_model_settings() -> dict:
    return json.loads(config.MODEL_SETTINGS.read_text(encoding="utf-8"))


class PriorCorrection:
    """score = log(p + eps) + alpha * log_prior;  corrected = softmax(score);  prediction = argmax(score).
    log_prior is given in the settings' class order and re-ordered to the model's class order."""

    def __init__(self, alpha: float, log_prior: np.ndarray, eps: float = 1e-8):
        self.alpha, self.log_prior, self.eps = float(alpha), np.asarray(log_prior, np.float64), float(eps)

    def scores(self, p: np.ndarray) -> np.ndarray:
        return np.log(np.asarray(p, np.float64) + self.eps) + self.alpha * self.log_prior

    def apply(self, p: np.ndarray) -> np.ndarray:
        s = self.scores(p)
        z = np.exp(s - s.max(-1, keepdims=True))
        return z / z.sum(-1, keepdims=True)

    @classmethod
    def for_classes(cls, settings: dict, model_classes: list[str]) -> tuple["PriorCorrection | None", str]:
        pc = settings.get("prior_correction") or {}
        if not pc.get("enabled"):
            return None, "prior correction is disabled in model_settings.json"
        order = [HAM_TO_CODE.get(str(c).lower(), str(c).upper()) for c in pc["class_order"]]
        if len(pc["log_prior"]) != len(order):
            raise ValueError("model_settings.json: log_prior and class_order have different lengths")
        if sorted(order) != sorted(model_classes):
            return None, (f"prior correction NOT applied: model_settings.json covers {order}, "
                          f"but the model predicts {list(model_classes)}")
        lp = dict(zip(order, pc["log_prior"]))
        return cls(pc["alpha"], [lp[c] for c in model_classes], pc.get("epsilon", 1e-8)), "applied"


# ---------------------------------------------------------------------------- exported timm models
def _cam_layer(model: nn.Module, spec: str) -> tuple[nn.Module, int]:
    if spec == "stages":
        return model.stages[-1], 0
    if spec == "blocks":
        return model.blocks[-1], getattr(model, "num_prefix_tokens", 1)
    return getattr(model, spec), 0


class TimmPredictor(Predictor):
    def __init__(self, export_dir: Path):
        super().__init__()
        import timm
        info = json.loads((export_dir / "export.json").read_text())
        self.members = []
        for name in info["members"]:
            card = json.loads((export_dir / name / "model_card.json").read_text())
            model = timm.create_model(card["timm_name"], pretrained=False, num_classes=len(card["classes"]))
            model.load_state_dict(torch.load(export_dir / name / "best.pt", map_location=self.device,
                                             weights_only=True))
            model.to(self.device).eval()
            layer, tokens = _cam_layer(model, card.get("cam_layer", MODELS[card["model"]]["cam_layer"]))
            self.members.append(Member(model, card["img_size"], card.get("temperature", 1.0), layer, tokens))
            self.classes = card["classes"]
        self.name = ("Ensemble of " if len(self.members) > 1 else "") + " + ".join(info["members"])
        self.ensemble_temperature = info.get("ensemble_temperature")


# ---------------------------------------------------------------------------- original SkinLens model
class SkinLensPredictor(Predictor):
    uses_metadata = True

    def __init__(self):
        super().__init__()
        spec = importlib.util.spec_from_file_location("skinlens_architecture", config.SKINLENS_ARCHITECTURE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.SkinLensHybrid(num_classes=7, meta_dim=21, pretrained=False)
        model.load_state_dict(torch.load(config.SKINLENS_CHECKPOINT, map_location=self.device, weights_only=True))
        model.to(self.device).eval()
        t = json.loads(config.SKINLENS_TEMPERATURE.read_text()).get("temperature", 1.0) \
            if config.SKINLENS_TEMPERATURE.exists() else 1.0
        self.classes = [HAM_TO_CODE[c] for c in ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]]
        self.members = [Member(model, 384, t, model.effnet.conv_head, forward=self._with_meta(model))]
        self.name = "SkinLens hybrid (EfficientNetV2-S + Swin V2 + metadata)"

    def _meta_vector(self) -> torch.Tensor:
        age, sex, site = self._meta or (None, None, None)
        v = np.zeros(21, np.float32)
        try:
            a = float(age)
            v[0], v[1] = np.clip(a, 0, 90) / 90.0, 0.0
        except (TypeError, ValueError):
            v[0], v[1] = 0.5, 1.0
        v[{"female": 2, "male": 3}.get(str(sex).lower(), 4)] = 1.0
        sites = ["abdomen", "back", "chest", "ear", "face", "foot", "genital", "hand", "lower extremity",
                 "neck", "scalp", "trunk", "upper extremity", "acral", "oral/genital", "unknown"]
        s = str(site).lower() if str(site).lower() in sites else "unknown"
        v[5 + sites.index(s)] = 1.0
        return torch.from_numpy(v).unsqueeze(0).to(self.device)

    def _with_meta(self, model):
        def fwd(x):
            return model(x, self._meta_vector().expand(x.shape[0], -1))
        return fwd


# ---------------------------------------------------------------------------- dummy (tests only)
class DummyPredictor(Predictor):
    is_dummy = True

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 16, 3, stride=2, padding=1),
                            nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, len(CLASSES)))
        net.eval()
        self.classes = CLASSES
        self.members = [Member(net, 64, 1.0, net[2])]
        self.name = "DUMMY test model (untrained - not for real use)"


_predictor: Predictor | None = None
_init_lock = threading.Lock()


def get_predictor() -> Predictor:
    global _predictor
    with _init_lock:
        if _predictor is None:
            kind = config.MODEL_KIND
            if kind == "auto":
                kind = "timm" if (config.EXPORTED_MODEL_DIR / "export.json").exists() else "skinlens"
            p = {"timm": lambda: TimmPredictor(config.EXPORTED_MODEL_DIR),
                 "skinlens": SkinLensPredictor, "dummy": DummyPredictor}[kind]()
            p.prior, p.prior_note = PriorCorrection.for_classes(load_model_settings(), p.classes)
            if p.prior is None:
                logging.getLogger("skinlens").warning(p.prior_note)
            _predictor = p
    return _predictor


def class_info(code: str) -> dict:
    return dict(code=code, name=CLASS_FULL.get(code, code), risk=RISK.get(code, "benign"))
