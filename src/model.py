import torch
import torch.nn as nn

class OpticalSlipNet(nn.Module):
    """Small convolutional model for optical tactile slip onset."""

    def __init__(self, in_channels: int = 3, hidden: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 5, padding=2), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(hidden, 2)

    def forward(self, frames):
        return self.head(self.features(frames).flatten(1))
