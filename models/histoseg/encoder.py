import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import (
    resample_abs_pos_embed,
    resample_abs_pos_embed_nhwc,
    resample_patch_embed,
)
from torchvision.models import ResNet50_Weights, resnet50


class ViTEncoderPyramid(
        nn.Module):  # TODO: add the rescaling of the pixel mean and std !!
    """
    Fixed-size ViT/UNI2 adapter:
      - one-time patch/pos-embed surgery in __init__ for (img_size, patch_size)
      - forward collects chosen layers, turns tokens -> [B,D,Ht,Wt]
      - projects D -> C_k per scale and bilinear-resizes to {s4,s8,s16,s32}
    """

    def __init__(
        self,
        vit: nn.Module,
        extract_layers: Iterable[int] = (6, 12, 18, 24),
        img_size: Tuple[int, int] = (448, 448),
        patch_size: int = 14,
        embed_dim: int = 1536,
        has_cls: bool = True,
        n_register_tokens: int = 8,
        pyramid_channels: Optional[Dict[str, int]] = None,
        sub_norm: bool = False,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        self.vit = vit
        self.extract_layers = tuple(extract_layers)
        self.D = embed_dim
        self.ps = patch_size
        self.has_cls = has_cls
        self.n_reg = n_register_tokens
        self.img_size = img_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)

        if pyramid_channels is None:
            # tapered schedule (change as you like)
            pyramid_channels = {"s4": 64, "s8": 128, "s16": 256, "s32": 256}

        # Per-scale 1x1 projections (project BEFORE resize)
        self.proj = nn.ModuleDict({
            k: nn.Conv2d(self.D, c, 1, bias=False)
            for k, c in pyramid_channels.items()
        })

        # Convenient handles
        self.patch_embed = getattr(self.vit, "patch_embed", None)
        self.pos_embed = getattr(self.vit, "pos_embed", None)  # may be None
        self.pos_drop = getattr(self.vit, "pos_drop", nn.Identity())
        self.blocks = self.vit.blocks
        self.norm = getattr(self.vit, "norm", nn.Identity())

        # Optional: swap MLP norms / add pre-proj LN (your "sub_norm" tweak)
        if sub_norm and hasattr(self.vit, "blocks"):
            for blk in self.vit.blocks:
                if hasattr(blk, "mlp") and hasattr(blk.mlp, "fc1"):
                    new_mlp = type(blk.mlp)(
                        in_features=blk.mlp.fc1.in_features,
                        hidden_features=blk.mlp.fc1.out_features,
                        act_layer=type(blk.mlp.act),
                        drop=getattr(blk.mlp, "drop1", nn.Dropout(0.0)).p,
                        norm_layer=nn.LayerNorm,
                    )
                    try:
                        new_mlp.load_state_dict(blk.mlp.state_dict(),
                                                strict=False)
                    except Exception:
                        pass
                    blk.mlp = new_mlp
                if hasattr(blk, "attn") and hasattr(blk.attn, "proj"):
                    blk.attn.proj = nn.Sequential(
                        nn.LayerNorm(blk.attn.proj.in_features), blk.attn.proj)

        # If model has a "neck", neutralize it (we only need features)
        if hasattr(self.vit, "neck"):
            self.vit.neck = nn.Identity()

        # Load checkpoint if provided
        if ckpt_path:
            sd = torch.load(ckpt_path, map_location="cpu")
            self.vit.load_state_dict(sd, strict=False)

        # --- One-time surgery for fixed size ---
        # 1) Patch embed: if you change patch size vs. pretrained, resample weights
        if hasattr(self.vit, "patch_embed"):
            pe = self.vit.patch_embed
            # sanity: pretrained patch/grid must be square for resampling helpers
            if (pe.grid_size[0] != pe.grid_size[1]) or (pe.patch_size[0]
                                                        != pe.patch_size[1]):
                raise ValueError(
                    "Pretrained patch/grid must be square for resample helpers."
                )

            # update conv params
            pe.patch_size = (self.ps, self.ps)
            pe.proj.kernel_size = (self.ps, self.ps)
            pe.proj.stride = (self.ps, self.ps)
            pe.proj.weight = nn.Parameter(
                resample_patch_embed(pe.proj.weight, [self.ps, self.ps]))

            # update grid metadata
            pe.grid_size = self.grid_size
            pe.num_patches = self.grid_size[0] * self.grid_size[1]
            pe.img_size = self.img_size

        # 2) Absolute pos embed: resample to new grid if present as a Parameter
        if isinstance(self.pos_embed, nn.Parameter):
            if self.pos_embed.dim() == 4:
                # NHWC style
                pe = resample_abs_pos_embed_nhwc(
                    self.pos_embed,
                    [max(self.grid_size),
                     max(self.grid_size)
                     ])[:, :self.grid_size[0], :self.grid_size[1], :]
            else:
                # [1, num_tokens, D]; handle prefix tokens
                num_prefix_tokens = getattr(self.vit, "num_prefix_tokens",
                                            1 if self.has_cls else 0)
                no_embed_class = getattr(self.vit, "no_embed_class", False)
                if no_embed_class:
                    num_prefix_tokens = 0
                pe = resample_abs_pos_embed(
                    self.pos_embed,
                    [max(self.grid_size),
                     max(self.grid_size)],
                    num_prefix_tokens=num_prefix_tokens,
                )
                prefix = pe[:, :num_prefix_tokens, :]
                grid = pe[:, num_prefix_tokens:, :]
                grid = grid.reshape(1, max(self.grid_size),
                                    max(self.grid_size), -1)
                grid = grid[:, :self.grid_size[0], :self.grid_size[1], :]
                pe = torch.cat([prefix, grid.flatten(1, 2)], dim=1)
            # set once; from now on, just use it in forward
            self.vit.pos_embed = nn.Parameter(pe)

        # If your model has windowed attention / relative pos params (e.g., Swin),
        # you'd adjust those here once for (img_size, patch_size).
        # For plain ViTs/UNI2 this is usually not needed.

    # ---- helpers ----
    def _prepare_tokens(self,
                        x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # Tokenize
        x = self.vit.patch_embed(x)  # [B, D, Ht, Wt]
        B, D, Ht, Wt = x.shape
        tok = x.flatten(2).transpose(1, 2)  # [B, Ht*Wt, D]

        # Add prefix tokens (CLS + register tokens) if present
        num_reg = self.n_reg if getattr(self.vit, "reg_token",
                                        None) is not None else 0
        if getattr(self.vit, "cls_token", None) is not None:
            cls = self.vit.cls_token.expand(B, -1, -1)  # [B,1,D]
            if num_reg > 0:
                regs = self.vit.reg_token.expand(B, num_reg, -1)  # [B,R,D]
                tok = torch.cat([cls, regs, tok], dim=1)
            else:
                tok = torch.cat([cls, tok], dim=1)

        # Add absolute PE if present (we’ve already resampled it to match grid_size)
        if isinstance(getattr(self.vit, "pos_embed", None), nn.Parameter):
            tok = tok + self.vit.pos_embed

        tok = self.pos_drop(tok)
        return tok, Ht, Wt

    def _tokens_to_map(self, tokens: torch.Tensor, Ht: int,
                       Wt: int) -> torch.Tensor:
        # drop CLS + register tokens → [B, Ht*Wt, D] → [B, D, Ht, Wt]
        start = 1 if self.has_cls else 0
        num_reg = self.n_reg if getattr(self.vit, "reg_token",
                                        None) is not None else 0
        spatial = tokens[:, start + num_reg:, :]  # [B, Ht*Wt, D]
        return spatial.transpose(1, 2).reshape(tokens.size(0), self.D, Ht, Wt)

    # ---- forward ----
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # If you truly always use the fixed size, you can assert here:
        # assert x.shape[-2:] == self.img_size, f"Expected {self.img_size}, got {x.shape[-2:]}"
        B, _, H, W = x.shape
        tokens, Ht, Wt = self._prepare_tokens(x)

        extracted: List[torch.Tensor] = []
        for depth, blk in enumerate(self.blocks, start=1):
            tokens = blk(tokens)
            if depth in self.extract_layers:
                extracted.append(tokens)
        assert len(extracted) == 4, "extract_layers must pick 4 blocks"

        # assign shallow→deep to s4..s32
        layer_to_scale = {
            "s4": extracted[0],
            "s8": extracted[1],
            "s16": extracted[2],
            "s32": extracted[3]
        }
        target = {
            "s4": (H // 4, W // 4),
            "s8": (H // 8, W // 8),
            "s16": (H // 16, W // 16),
            "s32": (H // 32, W // 32)
        }

        out: Dict[str, torch.Tensor] = {}
        for k, t in layer_to_scale.items():
            fmap = self._tokens_to_map(t, Ht, Wt)  # [B,D,Ht,Wt]
            fmap = self.proj[k](fmap)  # D -> C_k
            out[k] = F.interpolate(fmap,
                                   size=target[k],
                                   mode="bilinear",
                                   align_corners=False)
        return out


class ViTEncoderPyramidHooks(nn.Module):
    """
    ViT/UNI2 adapter using forward hooks (dynamic input size friendly).

    Contract:
        forward(x) -> {
            "s4":  [B, C4,  H/4,  W/4],
            "s8":  [B, C8,  H/8,  W/8],
            "s16": [B, C16, H/16, W/16],
            "s32": [B, C32, H/32, W/32],
        }

    Notes:
    - Assumes the ViT handles size dynamically (abs PE is handled internally or is relative/RoPE).
    - Works across timm ViTs where blocks live in `vit.blocks`.
    - No model surgery; purely reads intermediate tokens via hooks.
    """

    def __init__(
            self,
            vit: nn.Module,
            extract_layers: Iterable[int] = (6, 12, 18, 24),  # 1-based indices
            has_cls: bool = True,
            embed_dim: Optional[
                int] = None,  # auto if None: vit.embed_dim or vit.num_features
            pyramid_channels: Optional[Dict[
                str, int]] = None,  # per-scale channels (tapered)
    ):

        super().__init__()

        self.vit = vit
        self.layers = tuple(extract_layers)
        self.has_cls = has_cls

        if hasattr(vit, "patch_embed"):
            self.patch_size = vit.patch_embed.patch_size
        else:
            raise ValueError(
                "vit.patch_embed not found; cannot infer patch size.")

        if hasattr(vit, "num_prefix_tokens"):
            self.num_prefix_tokens = vit.num_prefix_tokens
        else:
            raise ValueError(
                "vit.num_prefix_tokens not found; needed to count extra tokens."
            )

        # infer D (token dim)
        if embed_dim is None:
            if hasattr(vit, "embed_dim"):
                embed_dim = vit.embed_dim
            else:
                embed_dim = getattr(vit, "num_features", None)
        if embed_dim is None:
            raise ValueError(
                "Couldn't infer embed_dim; please pass embed_dim explicitly.")
        self.D = int(embed_dim)

        # default tapered channel schedule (edit as you like)
        if pyramid_channels is None:
            pyramid_channels = {"s4": 64, "s8": 128, "s16": 256, "s32": 256}
        self.pyramid_channels = pyramid_channels

        # 1x1 projections per scale (project BEFORE resize for efficiency)
        self.proj = nn.ModuleDict({
            k: nn.Conv2d(self.D, c, 1, bias=False)
            for k, c in self.pyramid_channels.items()
        })

        # storage for hooked tensors: index -> tokens [B, N, D]
        self._stash: Dict[int, torch.Tensor] = {}

        # register hooks on vit.blocks (assumed nn.Sequential)
        blocks = getattr(self.vit, "blocks", None)
        if blocks is None:
            raise ValueError(
                "vit.blocks not found; this adapter expects a timm-style ViT with .blocks"
            )
        depth = len(blocks)
        self._handles: List[torch.utils.hooks.RemovableHandle] = [
        ]  # type: ignore
        for idx in self.layers:
            i = idx - 1
            if not (0 <= i < depth):
                raise ValueError(
                    f"extract_layers has out-of-range index {idx} for depth {depth}"
                )
            self._handles.append(blocks[i].register_forward_hook(
                self._make_hook(i)))

    # ----- hooks -----
    def _make_hook(self, idx: int):

        def fn(mod, inp, out):
            # Some blocks may return tuples; we want the token sequence
            if isinstance(out, (tuple, list)):
                out = out[0]
            self._stash[idx] = out  # [B, N, D]

        return fn

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _tokens_to_map(self, tokens: torch.Tensor, Ht: int,
                       Wt: int) -> torch.Tensor:
        """
        tokens: [B, N, D] including extra tokens (cls/reg if present)
        returns: [B, D, Ht, Wt]
        """
        B, N, D = tokens.shape
        spatial = tokens[:, self.num_prefix_tokens:, :]  # drop extra tokens
        Nsp = spatial.shape[1]
        if Nsp != Ht * Wt:
            raise RuntimeError(
                f"Token count {Nsp} != Ht*Wt ({Ht*Wt}). "
                "Pad input to multiples of patch size or handle custom PatchEmbed."
            )
        fmap = spatial.transpose(1, 2).reshape(B, D, Ht, Wt)
        return fmap

    # ----- forward -----
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run the ViT once, grab chosen layers via hooks, and build the pyramid.
        """
        self._stash.clear()

        B, _, H, W = x.shape
        Ht, Wt = H // self.patch_size[0], W // self.patch_size[1]

        # Run model (timm ViTs expose forward_features; fallback to forward)
        if hasattr(self.vit, "forward_features"):
            _ = self.vit.forward_features(x)
        else:
            _ = self.vit(x)

        # Collect in the order of extract_layers
        captured: List[torch.Tensor] = []
        for idx in self.layers:
            t = self._stash.get(idx - 1, None)
            if t is None:
                raise RuntimeError(
                    f"Hook for block {idx} didn't capture output.")
            captured.append(t)

        if len(captured) != 4:
            raise RuntimeError(
                f"Expected 4 layers, got {len(captured)}. Got indices: {self.layers}"
            )

        assign = {
            "s4": captured[0],
            "s8": captured[1],
            "s16": captured[2],
            "s32": captured[3]
        }
        target = {
            "s4": (H // 4, W // 4),
            "s8": (H // 8, W // 8),
            "s16": (H // 16, W // 16),
            "s32": (H // 32, W // 32)
        }

        out: Dict[str, torch.Tensor] = {}
        for k, tok in assign.items():
            fmap = self._tokens_to_map(tok, Ht, Wt)  # [B, D, Ht, Wt] inferred
            fmap = self.proj[k](fmap)  # D -> C_k (per scale)
            out[k] = F.interpolate(fmap,
                                   size=target[k],
                                   mode="bilinear",
                                   align_corners=False)
        return out


class ResNetPyramidAdapter(nn.Module):
    """
    Adapter for torchvision ResNet-50.

    Contract:
        forward(x) -> {
            "s4":  [B, C4,  H/4,  W/4],   # layer1
            "s8":  [B, C8,  H/8,  W/8],   # layer2
            "s16": [B, C16, H/16, W/16],  # layer3
            "s32": [B, C32, H/32, W/32],  # layer4
        }

    Notes:
    - Stem: conv1(stride=2) -> bn1 -> relu -> maxpool(stride=2) gives /4.
    - Channels before projection: {256, 512, 1024, 2048}.
    - You can freeze the backbone by setting `freeze_backbone=True`.
    """

    def __init__(
        self,
        resnet: nn.Module,
        pyramid_channels: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.backbone = resnet

        if pyramid_channels is None:
            pyramid_channels = {"s4": 64, "s8": 128, "s16": 256, "s32": 256}
        self.pyramid_channels = dict(pyramid_channels)

        in_ch = {"s4": 256, "s8": 512, "s16": 1024, "s32": 2048}
        self.proj = nn.ModuleDict({
            k:
            nn.Conv2d(in_ch[k],
                      self.pyramid_channels[k],
                      kernel_size=1,
                      bias=False)
            for k in ("s4", "s8", "s16", "s32")
        })

    def _forward_stem(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)  # /4
        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        B, _, H, W = x.shape
        # stem (/4)
        c1 = self._forward_stem(x)
        # stages
        c2 = self.backbone.layer1(c1)  # /4   (256 ch)
        c3 = self.backbone.layer2(c2)  # /8   (512 ch)
        c4 = self.backbone.layer3(c3)  # /16  (1024 ch)
        c5 = self.backbone.layer4(c4)  # /32  (2048 ch)  (or /16 if dilated)

        # project to decoder channels
        s4 = self.proj["s4"](c2)
        s8 = self.proj["s8"](c3)
        s16 = self.proj["s16"](c4)
        s32 = self.proj["s32"](c5)

        return {"s4": s4, "s8": s8, "s16": s16, "s32": s32}
