import os, sys
import torch
from torch.utils.data import DataLoader

# Add repository root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data import GaussianRainfieldGenerator
from models import JointRainDiffuser, train, sample


def test_occurrence_only_pipeline():
    gen = GaussianRainfieldGenerator(grid_height=32, grid_width=32)
    ds = gen.make_dataset(n_samples=4, occurrence_only=True)
    loader = DataLoader(ds, batch_size=2)
    model = JointRainDiffuser(T=2, device='cpu')
    train(model, loader, epochs=1, device='cpu', occurrence_only=True)
    out = sample(model, n=1, occurrence_only=True)
    assert out.shape == (1, 1, 32, 32)
    assert out.max() <= 1 and out.min() >= 0
