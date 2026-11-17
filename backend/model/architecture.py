import torch
import torch.nn as nn
import timm

class SkinLensHybrid(nn.Module):
    """
    Multimodal Skin Lesion Classification Hybrid Architecture.
    
    Fuses:
    1. EfficientNetV2-S (tf_efficientnetv2_s.in21k_ft_in1k) -> 1280 features
    2. Swin Transformer V2 (swinv2_base_window12to24_192to384.ms_in22k_ft_in1k) -> 1024 features
    3. Clinical Metadata MLP -> 32 features
    Total Fused Dimension: 1280 + 1024 + 32 = 2336
    
    Classifier Head:
    Linear(2336, 512) -> LayerNorm(512) -> GELU() -> Dropout(0.4) -> Linear(512, num_classes)
    
    Auxiliary Metadata Head (from training):
    Linear(32, num_classes)
    """
    def __init__(self, num_classes: int = 7, meta_dim: int = 21, pretrained: bool = False):
        super().__init__()
        # Feature extractors return pooled feature vectors when num_classes=0
        self.effnet = timm.create_model(
            "tf_efficientnetv2_s.in21k_ft_in1k",
            pretrained=pretrained,
            num_classes=0
        )
        self.swin = timm.create_model(
            "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
            pretrained=pretrained,
            num_classes=0
        )
        self.eff_dim = self.effnet.num_features
        self.swin_dim = self.swin.num_features

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.GELU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.eff_dim + self.swin_dim + 32, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

        # Auxiliary head encourages the metadata branch to learn a useful representation
        self.meta_aux_head = nn.Linear(32, num_classes)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor, return_aux: bool = False):
        eff_features = self.effnet(image)
        swin_features = self.swin(image)
        meta_features = self.meta_mlp(metadata)
        fused = torch.cat([eff_features, swin_features, meta_features], dim=1)
        logits = self.classifier(fused)
        if return_aux:
            return logits, self.meta_aux_head(meta_features)
        return logits
