# seg_model.py
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- small blocks ----------
class Conv2DBlock(nn.Module):

    def __init__(self, in_ch, out_ch, k=3, dropout=0.0):
        super().__init__()
        pad = (k - 1) // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class UpsampleConv(nn.Module):
    """bilinear x2 → 3×3 conv (no transposed convs)"""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = Conv2DBlock(in_ch, out_ch, 3, dropout)

    def forward(self, x):
        x = F.interpolate(x,
                          scale_factor=2,
                          mode="bilinear",
                          align_corners=False)
        return self.conv(x)


# assume Conv2DBlock and UpsampleConv are defined as before


class SegFPN(nn.Module):
    """
    Decoder that fuses {'s4','s8','s16','s32'} and (optionally) an 's1' image skip.
    Expects encoder features to already have the per-scale channels given by `pyramid_channels`.
    """

    def __init__(
            self,
            num_classes: int,
            pyramid_channels: Optional[Dict[str, int]] = None,
            dropout: float = 0.0,
            deep_supervision: bool = True,
            img_skip_ch: int = 64,
            assert_shapes: bool = True,  # sanity-check incoming channels
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.assert_shapes = assert_shapes

        self.img_to_s1 = nn.Sequential(
            Conv2DBlock(3, 32, dropout=dropout),
            Conv2DBlock(32, img_skip_ch, dropout=dropout),
        )

        if pyramid_channels is None:
            pyramid_channels = {"s4": 64, "s8": 128, "s16": 256, "s32": 256}

        self.pyramid_channels = pyramid_channels

        # Top-down path uses the per-scale channel sizes directly
        C4, C8, C16, C32 = self.pyramid_channels["s4"], self.pyramid_channels[
            "s8"], self.pyramid_channels["s16"], self.pyramid_channels["s32"]

        # upsample + smooth blocks keep channels per level
        self.up32_to_16 = UpsampleConv(C32, C16, dropout=dropout)  # s32 -> s16
        self.smooth16 = Conv2DBlock(C16, C16, dropout=dropout)

        self.up16_to_8 = UpsampleConv(C16, C8, dropout=dropout)  # s16 -> s8
        self.smooth8 = Conv2DBlock(C8, C8, dropout=dropout)

        self.up8_to_4 = UpsampleConv(C8, C4, dropout=dropout)  # s8 -> s4
        self.smooth4 = Conv2DBlock(C4, C4, dropout=dropout)

        # Head: fuse p4 with s1 (downsampled to 1/4) then predict
        self.head = nn.Sequential(
            Conv2DBlock(C4 + img_skip_ch, max(C4 // 2, 32), dropout=dropout),
            nn.Conv2d(max(C4 // 2, 32), num_classes, kernel_size=1),
        )

        if deep_supervision:
            self.aux8 = nn.Conv2d(C8, num_classes, kernel_size=1)
            self.aux16 = nn.Conv2d(C16, num_classes, kernel_size=1)

    def _check_shapes(self, feats: Dict[str, torch.Tensor]):
        assert feats["s4"].shape[1] == self.pyramid_channels[
            "s4"], f"s4 channels {feats['s4'].shape[1]} != {self.pyramid_channels['s4']}"
        assert feats["s8"].shape[1] == self.pyramid_channels[
            "s8"], f"s8 channels {feats['s8'].shape[1]} != {self.pyramid_channels['s8']}"
        assert feats["s16"].shape[1] == self.pyramid_channels[
            "s16"], f"s16 channels {feats['s16'].shape[1]} != {self.pyramid_channels['s16']}"
        assert feats["s32"].shape[1] == self.pyramid_channels[
            "s32"], f"s32 channels {feats['s32'].shape[1]} != {self.pyramid_channels['s32']}"

    def forward(self, x: torch.Tensor,
                feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B, _, H, W = x.shape
        if self.assert_shapes:
            self._check_shapes(feats)

        s1 = self.img_to_s1(x)

        # Use features as-is (already in the specified channels and strides)
        s4, s8, s16, s32 = feats["s4"], feats["s8"], feats["s16"], feats["s32"]

        # Top-down fusion to p4
        p16 = self.smooth16(s16 + self.up32_to_16(s32))  # 1/16
        p8 = self.smooth8(s8 + self.up16_to_8(p16))  # 1/8
        p4 = self.smooth4(s4 + self.up8_to_4(p8))  # 1/4

        # Fuse image skip at 1/4 and predict
        s1_1_4 = F.interpolate(s1,
                               size=p4.shape[-2:],
                               mode="bilinear",
                               align_corners=False)
        logits_1_4 = self.head(torch.cat([p4, s1_1_4], dim=1))
        main = F.interpolate(logits_1_4,
                             size=(H, W),
                             mode="bilinear",
                             align_corners=False)

        out = {"main": main}
        if self.deep_supervision:
            aux8 = F.interpolate(self.aux8(p8),
                                 size=(H, W),
                                 mode="bilinear",
                                 align_corners=False)
            aux16 = F.interpolate(self.aux16(p16),
                                  size=(H, W),
                                  mode="bilinear",
                                  align_corners=False)
            out.update({"aux8": aux8, "aux16": aux16})
        return out
