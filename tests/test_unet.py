import os, sys
import torch

# Ensure the repository root is on the path so we can import `models`
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models import LargeUNet


def test_large_unet_forward():
    B, H, W = 2, 32, 32
    model = LargeUNet(in_ch=1, out_ch=3, base=16)
    x = torch.randn(B, 1, H, W)
    out = model(x, 0)
    assert out.shape == (B, 3, H, W)

