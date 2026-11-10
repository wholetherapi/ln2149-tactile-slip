import torch

from src.dataset import *  # noqa: F401,F403


def test_smoke_shape():
    x = torch.randn(2, 4, 8)
    assert x.shape[0] == 2
