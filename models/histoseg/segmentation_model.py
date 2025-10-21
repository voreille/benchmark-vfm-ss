# model.py
import torch
import torch.nn as nn
from typing import Dict


class SegModel(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.encoder(
            x)  # dict with keys {"s4","s8","s16","s32"} (+ optional "s1")
        return self.decoder(x, feats)
