"""Shared definitions for the data pipeline, training, evaluation and the web app."""
from __future__ import annotations

CLASSES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]

CLASS_FULL = {
    "AK": "Actinic keratosis (pre-cancerous)",
    "BCC": "Basal cell carcinoma",
    "BKL": "Benign keratosis",
    "DF": "Dermatofibroma",
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus (mole)",
    "SCC": "Squamous cell carcinoma",
    "VASC": "Vascular lesion",
}

# "malignant" and "premalignant" are shown as "disease detected" (red) in the app
RISK = {
    "MEL": "malignant", "BCC": "malignant", "SCC": "malignant",
    "AK": "premalignant",
    "NV": "benign", "BKL": "benign", "DF": "benign", "VASC": "benign",
}

# label maps from each source into CLASSES
HAM_MAP = {"akiec": "AK", "bcc": "BCC", "bkl": "BKL", "df": "DF", "mel": "MEL", "nv": "NV", "vasc": "VASC"}
PAD_MAP = {"ACK": "AK", "BCC": "BCC", "MEL": "MEL", "NEV": "NV", "SCC": "SCC", "SEK": "BKL"}
ISIC2019_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]  # UNK is dropped

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Candidate models for the comparison (timm names, input size, Grad-CAM layer)
MODELS = {
    "efficientnet_b3": dict(timm="efficientnet_b3.ra2_in1k", size=300, cam_layer="conv_head"),
    "resnet50": dict(timm="resnet50.a1_in1k", size=224, cam_layer="layer4"),
    "densenet121": dict(timm="densenet121.tv_in1k", size=224, cam_layer="features"),
    "convnext_tiny": dict(timm="convnext_tiny.fb_in22k_ft_in1k", size=224, cam_layer="stages"),
    "vit_b16": dict(timm="vit_base_patch16_224.augreg2_in21k_ft_in1k", size=224, cam_layer="blocks"),
}


def build_model(name: str, num_classes: int, pretrained: bool = True):
    import timm
    spec = MODELS[name]
    return timm.create_model(spec["timm"], pretrained=pretrained, num_classes=num_classes)


def get_transforms(size: int):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    train = A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-30, 30), p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03, p=0.4),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.12),
                        hole_width_range=(0.05, 0.12), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    evaluate = A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return train, evaluate
