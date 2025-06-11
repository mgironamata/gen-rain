import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models import JointRainDiffuser


def test_unet_input_channels():
    model = JointRainDiffuser(T=2, device='cpu')
    # unet_mask should take 1 input channel
    assert model.unet_mask.inc[0].in_channels == 1
    # unet_rain should take 2 input channels
    assert model.unet_rain.inc[0].in_channels == 2
