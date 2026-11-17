import os
import sys
from pathlib import Path
import torch

# Ensure the backend directory is in the Python search path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from model.architecture import SkinLensHybrid

def verify_checkpoint():
    checkpoint_path = BACKEND_DIR / "models" / "best_skinlens_hybrid.pt"
    
    print("=" * 60)
    print("SkinLens Model Checkpoint Verification")
    print("=" * 60)
    print(f"Checkpoint target path: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint file not found at {checkpoint_path}")
        sys.exit(1)
        
    file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    print(f"Checkpoint file size: {file_size_mb:.2f} MB")
    
    # 1. Instantiate the exact architecture without downloading pre-trained weights
    print("\n1. Instantiating SkinLensHybrid architecture (pretrained=False)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Target compute device: {device}")
    
    model = SkinLensHybrid(num_classes=7, meta_dim=21, pretrained=False)
    
    # 2. Load the state dictionary
    print("\n2. Loading state dictionary from checkpoint...")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    print(f"   Total tensors in state_dict: {len(state_dict)}")
    
    # 3. Load weights with strict=True
    print("\n3. Loading state_dict into model with strict=True...")
    try:
        load_result = model.load_state_dict(state_dict, strict=True)
        print("   SUCCESS: Model weights loaded with strict=True!")
        print(f"   Missing keys: {len(load_result.missing_keys)}")
        print(f"   Unexpected keys: {len(load_result.unexpected_keys)}")
    except Exception as e:
        print(f"   FAILED: strict=True loading failed with error:\n{e}")
        sys.exit(1)
        
    model.to(device)
    model.eval()
    
    # 4. Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n4. Model parameter verification:")
    print(f"   Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"   EfficientNetV2-S feature dim: {model.eff_dim}")
    print(f"   Swin Transformer V2 feature dim: {model.swin_dim}")
    print(f"   Fused classifier input dim: {model.eff_dim + model.swin_dim + 32}")
    
    # 5. Dummy forward pass
    print("\n5. Running dummy forward pass:")
    dummy_img = torch.randn(1, 3, 384, 384, device=device)
    dummy_meta = torch.randn(1, 21, device=device)
    print(f"   Input image tensor shape: {tuple(dummy_img.shape)}")
    print(f"   Input metadata tensor shape: {tuple(dummy_meta.shape)}")
    
    with torch.no_grad():
        logits = model(dummy_img, dummy_meta)
        
    print(f"   Output logits shape: {tuple(logits.shape)}")
    print(f"   Output logits values: {logits.cpu().numpy()[0].round(4).tolist()}")
    
    assert logits.shape == (1, 7), f"Expected shape (1, 7), got {logits.shape}"
    print("\n" + "=" * 60)
    print("ALL VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    verify_checkpoint()
