import math
from typing import Optional, Union
import numpy as np
import torch

# HAM10000 Canonical Metadata Sites and Dimensions from training notebook
SITES = [
    "abdomen", "back", "chest", "ear", "face", "foot", "genital",
    "hand", "lower extremity", "neck", "scalp", "trunk",
    "upper extremity", "acral", "oral/genital", "unknown"
]

AGE_SCALE = 90.0
META_DIM = 2 + 3 + len(SITES)  # 2 + 3 + 16 = 21

SEX_MAP = {
    "female": 2,
    "male": 3,
    "unknown": 4,
}

def encode_metadata(
    age: Optional[Union[float, int, str]] = None,
    sex: Optional[str] = None,
    localization: Optional[str] = None
) -> torch.Tensor:
    """
    Encodes clinical metadata into the exact 21-dimensional vector used during training:
    - Index 0: Normalized age (clip(age, 0, 90) / 90.0) or 0.5 if missing
    - Index 1: Age missing indicator flag (1.0 if missing, else 0.0)
    - Indices 2-4: Sex one-hot [female=2, male=3, unknown=4]
    - Indices 5-20: Anatomical site one-hot (16 categories, unknown defaults to index 20)
    
    Returns:
        torch.Tensor of shape (1, 21) with dtype float32.
    """
    v = np.zeros(META_DIM, dtype=np.float32)

    # 1. Age & Age Missing Flag
    age_missing = True
    if age is not None:
        try:
            # Handle string conversions or numeric conversions
            age_float = float(age)
            if not math.isnan(age_float):
                age_missing = False
                v[0] = float(np.clip(age_float, 0.0, AGE_SCALE) / AGE_SCALE)
                v[1] = 0.0
        except (ValueError, TypeError):
            age_missing = True

    if age_missing:
        v[0] = 0.5
        v[1] = 1.0

    # 2. Sex One-Hot
    sex_str = str(sex).strip().lower() if sex is not None else "unknown"
    sex_idx = SEX_MAP.get(sex_str, 4)
    v[sex_idx] = 1.0

    # 3. Anatomical Localization One-Hot
    site_str = str(localization).strip().lower() if localization is not None else "unknown"
    if site_str not in SITES:
        site_str = "unknown"
    site_start = 5
    v[site_start + SITES.index(site_str)] = 1.0

    # Return shape (1, 21)
    return torch.tensor(v, dtype=torch.float32).unsqueeze(0)
