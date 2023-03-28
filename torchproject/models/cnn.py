import torch
from .base import Model
import torch.nn as nn

class CnnClassificator(Model):
    def __init__(self, n_layers=3, n_classes=7, fdim=256):
        super().__init__()
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.linear = nn.Linear(fdim, n_classes)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        B - batch size
        T - time dim
        F - feature dim
        C - n classes
        Args:
            x (B, T, F): input
        Returns:
            tuple
              (B, C): output logits
              (B, C): output probs (output of softmax)
        """
        x = torch.mean(x, dim=1)
        # (B, F)
        logits = self.linear(x)
        # (B, C) logits
        probs = self.softmax(logits)
        return logits, probs
