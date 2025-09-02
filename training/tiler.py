from typing import Tuple
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torchvision.transforms import CenterCrop


class Tiler(ABC):

    @abstractmethod
    def window(self, imgs: list[torch.Tensor]):
        pass

    @abstractmethod
    def stitch(self, crop_logits, origins, img_sizes):
        pass


class FixedTileTiler(Tiler):

    def __init__(self, tile: int, stride: int, weighted_blend: bool = False):
        self.tile = tile
        self.stride = stride
        self.weighted_blend = weighted_blend

    @torch.compiler.disable
    def window(self, imgs, tile_size: int, stride: int):
        """
        imgs: list[Tensor CxH×W]
        tile_size: int (model input height == width)
        stride: int (<= tile_size). stride == tile_size ⇒ no overlap
        Returns:
        crops: Tensor [N, C, tile, tile]
        origins: list of (img_idx, top, left, h, w) for each crop BEFORE padding
        img_sizes: list of (H, W) for each image
        """
        crops, origins, img_sizes = [], [], []
        for i, img in enumerate(imgs):
            C, H, W = img.shape
            img_sizes.append((H, W))

            # Compute start positions that fully cover H and W.
            def starts(L):
                if L <= tile_size:
                    return [0]
                s = list(range(0, max(L - tile_size, 0) + 1, stride))
                if s[-1] != L - tile_size:
                    s.append(L - tile_size)
                return s

            for top in starts(H):
                for left in starts(W):
                    crop = img[:, top:top + tile_size, left:left + tile_size]

                    # If at borders, pad to tile_size so model always sees fixed size
                    pad_h = tile_size - crop.shape[1]
                    pad_w = tile_size - crop.shape[2]
                    if pad_h > 0 or pad_w > 0:
                        crop = torch.nn.functional.pad(crop,
                                                       (0, pad_w, 0, pad_h),
                                                       mode="constant",
                                                       value=0)

                    crops.append(crop)
                    # Keep the *valid* size (h, w) before padding for proper unpadding later
                    valid_h = min(tile_size, H - top)
                    valid_w = min(tile_size, W - left)
                    origins.append((i, top, left, valid_h, valid_w))

        return torch.stack(crops, dim=0), origins, img_sizes

    @torch.compiler.disable
    def stitch(self, crop_logits, origins, img_sizes):
        """
        crop_logits: Tensor [N, C, tile, tile] – model outputs per crop
        origins: list of (img_idx, top, left, valid_h, valid_w)
        img_sizes: list of (H, W) for each image

        Returns: list[Tensor C×H×W] per image
        """
        device = crop_logits.device
        C = crop_logits.shape[1]

        sum_logits = []
        hit_count = []
        for (H, W) in img_sizes:
            sum_logits.append(
                torch.zeros((C, H, W), device=device, dtype=crop_logits.dtype))
            hit_count.append(
                torch.zeros((1, H, W), device=device, dtype=crop_logits.dtype))

        for n, (img_i, top, left, valid_h, valid_w) in enumerate(origins):
            sl = crop_logits[n, :, :valid_h, :valid_w]
            sum_logits[img_i][:, top:top + valid_h, left:left + valid_w] += sl
            hit_count[img_i][:, top:top + valid_h, left:left + valid_w] += 1

        merged = []
        for sums, counts in zip(sum_logits, hit_count):
            # Safe divide: where counts==0 (shouldn't happen), just leave zeros
            merged.append(sums / torch.clamp(counts, min=1.0))
        return merged  # list of C×H×W


class GridPadTiler(Tiler):
    """
    Pad-to-grid tiler:
      - If H,W don't align to (tile,stride), pad bottom/right to H_pad,W_pad so that
        (H_pad - tile) % stride == 0 and (W_pad - tile) % stride == 0.
      - Tile padded image into perfect T×T crops.
      - Stitch logits on padded canvas by (weighted) averaging overlaps.
      - Crop back to original size (H,W).
    """

    def __init__(self,
                 tile: int,
                 stride: int,
                 weighted_blend: bool = False,
                 pad_mode: str = "constant",
                 pad_value: float = 0.0):
        assert 1 <= stride <= tile, "stride must be in [1, tile]"
        self.tile = tile
        self.stride = stride
        self.weighted_blend = weighted_blend
        self.pad_mode = pad_mode
        self.pad_value = pad_value

    # ---- helpers ----
    @staticmethod
    def _ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    def _padded_len(self, L: int) -> int:
        """Smallest L_pad >= L such that (L_pad - tile) % stride == 0 and L_pad >= tile."""
        T, S = self.tile, self.stride
        if L <= T:
            return T
        k = self._ceil_div(L - T,
                           S)  # number of strides needed beyond first tile
        return k * S + T

    def _hann(self, h: int, w: int, device, dtype) -> torch.Tensor:
        wy = torch.hann_window(h, periodic=False, device=device,
                               dtype=dtype).clamp_min(1e-3)
        wx = torch.hann_window(w, periodic=False, device=device,
                               dtype=dtype).clamp_min(1e-3)
        return (wy[:, None] * wx[None, :])[None]  # [1,h,w]

    def _get_padding(self, img_size: Tuple[int,
                                           int]) -> Tuple[int, int, int, int]:
        H, W = img_size
        H_pad = self._padded_len(H)
        W_pad = self._padded_len(W)
        pad_h = H_pad - H
        pad_w = W_pad - W
        pad_top = pad_h // 2
        pad_bot = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return (H_pad, W_pad, pad_h, pad_w, pad_top, pad_bot, pad_left,
                pad_right)

    @torch.compiler.disable
    def window(self, imgs):
        crops, origins, img_sizes, pads = [], [], [], [
        ]  # <— save pads per image
        T, S = self.tile, self.stride

        for i, img in enumerate(imgs):
            C, H, W = img.shape
            img_sizes.append((H, W))

            (H_pad, W_pad, pad_h, pad_w, pad_top, pad_bot, pad_left,
             pad_right) = self._get_padding((H, W))

            if pad_h or pad_w:
                img_padded = F.pad(img,
                                   (pad_left, pad_right, pad_top, pad_bot),
                                   mode=self.pad_mode,
                                   value=self.pad_value
                                   if self.pad_mode == "constant" else 0.0)
            else:
                img_padded = img

            for top in range(0, H_pad - T + 1, S):
                for left in range(0, W_pad - T + 1, S):
                    crops.append(img_padded[:, top:top + T, left:left + T])
                    origins.append((i, top, left))  # coords on padded canvas

        return torch.stack(crops, 0), origins, img_sizes

    @torch.compiler.disable
    def stitch(self, crop_logits, origins, img_sizes):
        """
        crop_logits: [N, C, T, T]
        origins:     list of (img_i, top, left)  OR  (img_i, top, left, valid_h, valid_w)
                    (valid_h/valid_w optional if you sometimes emit partial tiles)
        img_sizes:   list of (H, W)
        """
        crop_logits = crop_logits.float()
        device, dtype = crop_logits.device, torch.float32
        C = crop_logits.shape[1]
        T = self.tile

        sums_list = []
        counts_list = []
        for img_i, img_size in enumerate(img_sizes):
            H, W = img_size
            H_pad, W_pad, _, _, _, _, _, _ = self._get_padding((H, W))
            sums_list.append(
                torch.zeros((C, H_pad, W_pad), device=device, dtype=dtype))
            counts_list.append(
                torch.zeros((C, H_pad, W_pad), device=device, dtype=dtype))

        # precompute weight map if weighted
        w_tile = None
        if self.weighted_blend:
            wy = torch.hann_window(T,
                                   periodic=False,
                                   device=device,
                                   dtype=dtype).clamp_min(1e-6)
            wx = torch.hann_window(T,
                                   periodic=False,
                                   device=device,
                                   dtype=dtype).clamp_min(1e-6)
            w_tile = (wy[:, None] * wx[None, :])[None]  # [1, T, T]

        for n, origin in enumerate(origins):
            # support either 3-tuple or 5-tuple origins
            if len(origin) == 3:
                img_i, top, left = origin
                valid_h = valid_w = T
            else:
                img_i, top, left, valid_h, valid_w = origin

            sl = crop_logits[n, :, :valid_h, :valid_w]  # [C, h, w]
            if w_tile is None:
                sums_list[img_i][:, top:top + valid_h,
                                 left:left + valid_w] += sl
                counts_list[img_i][:, top:top + valid_h,
                                   left:left + valid_w] += 1.0
            else:
                w = w_tile[:, :valid_h, :valid_w]  # [1, h, w]
                sums_list[img_i][:, top:top + valid_h,
                                 left:left + valid_w] += sl * w
                counts_list[img_i][:, top:top + valid_h,
                                   left:left + valid_w] += w

        outs = []
        for i, (sums, counts) in enumerate(zip(sums_list, counts_list)):
            H, W = img_sizes[i]
            sums = CenterCrop((H, W))(sums)
            counts = CenterCrop((H, W))(counts)
            denom = torch.where(
                counts > 0, counts,
                torch.ones_like(counts))  # avoid /0 only where uncovered
            outs.append(sums / denom)
        return outs
