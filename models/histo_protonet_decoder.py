from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize, one_hot, pad

from models.histo_encoder import Encoder


@torch.no_grad()
def masks_to_token_soft_from_semantic(
        targets_sem: torch.Tensor,  # [B,1,H,W] or [B,H,W] (long)
        num_classes: int,  # C (includes background, e.g., bg=0)
        grid_size: Tuple[int, int],  # (Ht, Wt)
        ignore_idx: int = 255,
        purity_thresh: float
    | None = 0.9,  # set None to disable purity filtering
        bg_idx: int = 0,  # background class id in [0..C-1]
        renorm_exclude_ignore:
    bool = True,  # renormalize class probs over non-ignore
        drop_background_only: bool = True,  # drop tokens with only background
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      y_soft: [B, Q, C]  pooled class fractions (over real classes 0..C-1)
      keep:   [B, Q]     True if token passes purity filter (and optional bg-only drop)

    Logic:
      - Build 1-hot over (C+1) classes: last channel is 'ignore'.
      - Pool to tokens.
      - Optionally renormalize class probs over non-ignore pixels.
      - keep = (max_class_prob >= purity_thresh) AND (optionally not background-only).
    """
    Ht, Wt = grid_size

    if targets_sem.ndim == 4:
        targets_sem = targets_sem.squeeze(1)
    assert targets_sem.ndim == 3, "targets_sem must be [B,H,W] or [B,1,H,W]"
    B, H, W = targets_sem.shape
    device = targets_sem.device

    # Map ignore to extra class C (so one_hot stays in-range)
    IGN_CH = num_classes  # index of ignore channel
    safe = targets_sem.clone()
    safe[targets_sem == ignore_idx] = IGN_CH
    # One-hot on C+1, then pool
    oh = F.one_hot(safe.long(), num_classes=num_classes + 1)  # [B,H,W,C+1]
    oh = oh.permute(0, 3, 1, 2).float()  # [B,C+1,H,W]
    pooled = F.adaptive_avg_pool2d(oh, (Ht, Wt))  # [B,C+1,Ht,Wt]

    # Split real classes vs ignore
    class_probs = pooled[:, :num_classes]  # [B,C,Ht,Wt]
    ignore_frac = pooled[:, IGN_CH:IGN_CH + 1]  # [B,1,Ht,Wt]

    # Optionally renormalize over non-ignore to get true "purity" among valid pixels
    if renorm_exclude_ignore:
        denom = class_probs.sum(dim=1, keepdim=True).clamp_min(
            1e-8)  # sum over classes
        class_purity = class_probs / denom
    else:
        # purity measured vs all pixels in the token (ignore contributes to lowering purity)
        class_purity = class_probs

    # Foreground presence (exclude background) if requested
    if drop_background_only and num_classes > 1:
        # any foreground after pooling?
        fg = torch.cat([class_probs[:, :bg_idx], class_probs[:, bg_idx + 1:]],
                       dim=1)  # [B,C-1,Ht,Wt]
        has_fg = (fg.sum(dim=1) > 0)  # [B,Ht,Wt] bool
    else:
        has_fg = torch.ones((B, Ht, Wt), dtype=torch.bool, device=device)

    # Purity thresholding
    if purity_thresh is not None:
        # max over *all* real classes (including background unless you don’t want that)
        max_purity, argmax_cls = class_purity.max(dim=1)  # [B,Ht,Wt]
        pass_thresh = (max_purity >= float(purity_thresh))
        # If you don’t want background-only to pass purity (e.g., 0.99 bg), combine with has_fg:
        keep_mask = pass_thresh & has_fg
    else:
        keep_mask = has_fg

    # Prepare outputs
    y_soft = class_probs.flatten(2).transpose(1, 2).contiguous()  # [B,Q,C]
    keep = keep_mask.flatten(1).contiguous()  # [B,Q]
    return y_soft, keep


@torch.no_grad()
def masks_to_token_hard_from_semantic(
    targets_sem: torch.Tensor,  # [B,1,H,W] or [B,H,W] (long)
    num_classes: int,  # C (incl. background; e.g., bg=0)
    grid_size: Tuple[int, int],  # (Ht, Wt)
    ignore_idx: int = 255,
    *,
    bg_idx: int = 0,
    renorm_exclude_ignore: bool = True,  # renormalize over non-ignore pixels
    drop_background_only: bool = True,  # mark tokens with only bg as invalid
    purity_thresh: Optional[
        float] = None,  # e.g., 0.9 ⇒ require ≥90% purity; None disables
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      y_hard: [B, Q] long   token class ids in [0..C-1] (argmax over pooled per-class fractions)
      valid:  [B, Q] bool   True if token passes filters (non-ignore, purity, foreground if requested)

    Notes:
      - If renorm_exclude_ignore=True, class fractions are computed over non-ignore pixels only.
      - If drop_background_only=True, tokens with no foreground (any class != bg_idx) are invalid.
      - If purity_thresh is set, max class purity must be >= threshold.
    """
    Ht, Wt = grid_size

    if targets_sem.ndim == 4:
        targets_sem = targets_sem.squeeze(1)
    assert targets_sem.ndim == 3, "targets_sem must be [B,H,W] or [B,1,H,W]"
    B, H, W = targets_sem.shape
    device = targets_sem.device

    # Map ignore to extra channel C so one_hot stays in range
    IGN_CH = num_classes
    safe = targets_sem.clone()
    safe[targets_sem == ignore_idx] = IGN_CH

    # One-hot on C+1, then pool to token grid
    oh = F.one_hot(safe.long(), num_classes=num_classes + 1)  # [B,H,W,C+1]
    oh = oh.permute(0, 3, 1, 2).float()  # [B,C+1,H,W]
    pooled = F.adaptive_avg_pool2d(oh, (Ht, Wt))  # [B,C+1,Ht,Wt]

    class_probs = pooled[:, :num_classes]  # [B,C,Ht,Wt]
    ignore_frac = pooled[:, IGN_CH:IGN_CH + 1]  # [B,1,Ht,Wt]

    # Optionally renormalize class fractions over non-ignore pixels
    if renorm_exclude_ignore:
        denom = class_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        class_frac = class_probs / denom  # [B,C,Ht,Wt]
    else:
        class_frac = class_probs  # relative to (valid + ignore)

    # Foreground presence (exclude background)
    if drop_background_only and num_classes > 1:
        fg_probs = torch.cat(
            [class_probs[:, :bg_idx], class_probs[:, bg_idx + 1:]],
            dim=1)  # [B,C-1,Ht,Wt]
        has_fg = (fg_probs.sum(dim=1) > 0)  # [B,Ht,Wt] bool
    else:
        has_fg = torch.ones((B, Ht, Wt), dtype=torch.bool, device=device)

    # Argmax class (hard label) and max purity
    max_purity, y_argmax = class_frac.max(dim=1)  # [B,Ht,Wt], [B,Ht,Wt]

    # Purity criterion
    if purity_thresh is not None:
        pass_purity = (max_purity >= float(purity_thresh))
    else:
        pass_purity = torch.ones_like(max_purity, dtype=torch.bool)

    keep = pass_purity & has_fg  # [B,Ht,Wt] bool

    # Flatten to tokens
    Q = Ht * Wt
    y_hard = y_argmax.flatten(1).contiguous().long()  # [B,Q]
    valid = keep.flatten(1).contiguous()  # [B,Q]

    return y_hard, valid


class ProtoNetLayer(nn.Module):

    def __init__(self,
                 metric="L2",
                 center_feats=True,
                 normalize_feats=True,
                 num_prototypes=7,
                 embedding_dim=1024):
        super().__init__()
        m = metric.lower()
        assert m in ("l2", "cosine")
        self.metric, self.center_feats, self.normalize_feats = m, center_feats, normalize_feats
        self.num_prototypes, self.embedding_dim = num_prototypes, embedding_dim

        self.register_buffer("prototypes",
                             torch.zeros(num_prototypes, embedding_dim),
                             persistent=True)
        self.register_buffer("proto_counts",
                             torch.zeros(num_prototypes),
                             persistent=True)
        self.register_buffer("mean",
                             torch.zeros(embedding_dim),
                             persistent=True)
        self.register_buffer("support_count", torch.zeros(1), persistent=True)

    def _preprocess(self, X: torch.Tensor) -> torch.Tensor:
        if self.center_feats:
            X = X - self.mean
        if self.normalize_feats:
            X = F.normalize(X, dim=-1)
        return X

    @torch.no_grad()
    def update_mean(self, X: torch.Tensor) -> None:
        """X: [N,D] valid tokens only."""
        Xm = F.normalize(X, dim=-1) if self.normalize_feats else X
        n = float(Xm.shape[0])
        tot = float(self.support_count.item())
        self.mean.copy_((self.mean * tot + Xm.sum(0)) / (tot + n))
        self.support_count += n

    @torch.no_grad()
    def update_prototype(self, X: torch.Tensor, y_soft: torch.Tensor) -> None:
        """X: [N,D], y_soft: [N,C] (soft rows; needn’t sum to 1)."""
        Xp = self._preprocess(X)
        class_sums = y_soft.T @ Xp  # [C,D]
        class_counts = y_soft.sum(0)  # [C]
        new_cnt = self.proto_counts + class_counts
        mask = class_counts > 0
        denom = new_cnt.clamp_min(1e-12).unsqueeze(1)

        P = self.prototypes.clone()
        P[mask] = (self.prototypes[mask] * self.proto_counts[mask].unsqueeze(1)
                   + class_sums[mask]) / denom[mask]
        self.prototypes.copy_(P)
        self.proto_counts.copy_(new_cnt)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Xp, P = self._preprocess(X), self.prototypes
        if self.metric == "l2":
            return -(Xp.unsqueeze(2) - P).norm(dim=-1, p=2)
        Xn, Pn = F.normalize(Xp, dim=-1), F.normalize(P, dim=-1)
        return Xn @ Pn.T


class ProtoNetDecoder(Encoder):

    def __init__(self,
                 encoder_name="hf-hub:MahmoodLab/UNI2-h",
                 num_classes=7,
                 img_size=(448, 448),
                 sub_norm=False,
                 patch_size=14,
                 pretrained=True,
                 ckpt_path="",
                 metric="L2",
                 center_feats=True,
                 normalize_feats=True,
                 ignore_idx=255):
        super().__init__(encoder_name=encoder_name,
                         img_size=img_size,
                         sub_norm=sub_norm,
                         patch_size=patch_size,
                         pretrained=pretrained,
                         ckpt_path=ckpt_path)
        self.ignore_idx = ignore_idx
        self.num_classes = num_classes
        self.head = ProtoNetLayer(metric=metric,
                                  center_feats=center_feats,
                                  normalize_feats=normalize_feats,
                                  num_prototypes=num_classes,
                                  embedding_dim=self.embed_dim)

    @torch.no_grad()
    def tokens_from_images(self, imgs: torch.Tensor) -> torch.Tensor:
        # Expect Encoder.forward -> [B,Q,D]
        return super().forward(imgs)

    @torch.no_grad()
    def update_mean_from_batch(self, imgs: torch.Tensor,
                               targets_sem: torch.Tensor):
        """imgs: [B,3,T,T]; targets_sem: [B,1,T,T] or [B,T,T] longs with ignore."""
        device = next(self.parameters()).device
        tokens = self.tokens_from_images(imgs.to(device))  # [B,Q,D]
        B, Q, D = tokens.shape
        y_soft, valid = masks_to_token_soft_from_semantic(
            targets_sem.to(device),
            self.num_classes,
            self.grid_size,
            self.ignore_idx,
        )
        X = tokens.reshape(B * Q, D)
        m = valid.reshape(B * Q)
        self.head.update_mean(X[m])

    @torch.no_grad()
    def update_prototypes_from_batch(self, imgs: torch.Tensor,
                                     targets_sem: torch.Tensor):
        device = next(self.parameters()).device
        tokens = self.tokens_from_images(imgs.to(device))  # [B,Q,D]
        B, Q, D = tokens.shape
        y_soft, valid = masks_to_token_soft_from_semantic(
            targets_sem.to(device),
            self.num_classes,
            self.grid_size,
            self.ignore_idx,
        )
        X = tokens.reshape(B * Q, D)
        Y = y_soft.reshape(B * Q, self.head.num_prototypes)
        m = valid.reshape(B * Q)
        self.head.update_prototype(X[m], Y[m])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        x = self.head(x)
        x = x.transpose(1, 2)

        return x.reshape(x.shape[0], -1, *self.grid_size)
