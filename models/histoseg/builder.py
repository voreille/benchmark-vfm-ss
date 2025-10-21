# models/histoseg/builder.py
from typing import Dict, Tuple, Optional, Iterable, List
import torch
import torch.nn as nn
import timm

from .encoder import ViTEncoderPyramidHooks
from .segfpn import SegFPN
from .segmentation_model import SegModel


class Uni2Seg(nn.Module):
    """
    Lightning-friendly wrapper that builds:
      ViT (UNI2) -> ViTEncoderPyramidHooks -> SegFPN -> SegModel

    YAML:
      network:
        class_path: models.histoseg.builder.Uni2Seg
        init_args:
          encoder_name: "hf-hub:MahmoodLab/UNI2-h"
          img_size: [448, 448]
          num_classes: 6
    """

    def __init__(
        self,
        num_classes: int = 7,
        dropout: float = 0.2,
        deep_supervision: bool = True,
        img_skip_ch: int = 64,
        extract_layers: Iterable[int] = (6, 12, 18, 24),
        embed_dim: Optional[int] = None,  # auto if None
        pyramid_channels: Optional[Dict[
            str, int]] = None,  # {"s4":64,"s8":128,"s16":256,"s32":256}
    ):
        super().__init__()

        self.save_hyperparameters = getattr(
            nn.Module, "save_hyperparameters",
            lambda *a, **k: None)  # no-op if not LightningModule

        # 1) Backbone
        timm_kwargs = {
            'img_size': 224,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667 * 2,
            'num_classes': 0,
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8,
            'dynamic_img_size': True
        }
        vit = timm.create_model("hf-hub:MahmoodLab/UNI2-h",
                                pretrained=True,
                                **timm_kwargs)

        # 2) Infer dims + normalization
        if embed_dim is None:
            embed_dim = getattr(vit, "embed_dim", None) or getattr(
                vit, "num_features", None)
            if embed_dim is None:
                raise ValueError(
                    "Could not infer ViT embed_dim; pass embed_dim explicitly."
                )

        pixel_mean = torch.tensor(vit.default_cfg["mean"]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor(vit.default_cfg["std"]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

        # 3) Channel schedule (per-scale)
        if pyramid_channels is None:
            pyramid_channels = {"s4": 64, "s8": 128, "s16": 256, "s32": 256}

        # 4) Encoder adapter (hooks)
        encoder = ViTEncoderPyramidHooks(
            vit=vit,
            extract_layers=extract_layers,
            pyramid_channels=pyramid_channels,
            embed_dim=embed_dim,
        )

        # 5) Decoder (assumes same per-scale channels)
        decoder = SegFPN(
            num_classes=num_classes,
            pyramid_channels=pyramid_channels,
            img_skip_ch=img_skip_ch,
            dropout=dropout,
            deep_supervision=deep_supervision,
            assert_shapes=True,
        )

        # 6) Compose, naming convention for the benchmark
        self.model = SegModel(encoder=encoder, decoder=decoder)
        # so the lightning module can find it and freeze if needed
        self.encoder = vit

    def forward(self, x: torch.Tensor):
        x = (x - self.pixel_mean) / self.pixel_std
        output = self.model(x)
        return output["main"]
