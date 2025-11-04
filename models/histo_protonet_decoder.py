from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize, one_hot, pad

from models.histo_encoder import Uni2Encoder


@torch.no_grad()
def masks_to_token_soft_from_semantic(
    targets_sem: torch.Tensor,  # [B,1,H,W] or [B,H,W], long
    num_classes: int,
    grid_size: tuple[int, int],  # (Ht,Wt)
    ignore_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      y_soft: [B, Q, C]  pooled class fractions per token
      valid:  [B, Q]     True if any non-ignore class present in that token
    """
    Ht, Wt = grid_size
    if targets_sem.ndim == 4:
        targets_sem = targets_sem.squeeze(1)
    assert targets_sem.ndim == 3
    B, H, W = targets_sem.shape

    tgt = targets_sem.clone()
    valid_pix = tgt != ignore_idx
    tgt[~valid_pix] = 0

    oh = F.one_hot(tgt.long(), num_classes=num_classes)  # [B,H,W,C]
    oh = oh.permute(0, 3, 1, 2).float()  # [B,C,H,W]
    oh = oh * valid_pix.unsqueeze(1).float()

    pooled = F.adaptive_avg_pool2d(oh, (Ht, Wt))  # [B,C,Ht,Wt]
    valid = (pooled.sum(dim=1) > 0)  # [B,Ht,Wt]

    y_soft = pooled.flatten(2).transpose(1, 2).contiguous()  # [B,Q,C]
    valid = valid.flatten(1).contiguous()  # [B,Q]
    return y_soft, valid


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


class ProtoNetDecoder(Uni2Encoder):

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
        self.head = ProtoNetLayer(metric=metric,
                                  center_feats=center_feats,
                                  normalize_feats=normalize_feats,
                                  num_prototypes=num_classes,
                                  embedding_dim=self.embed_dim)

    @torch.no_grad()
    def tokens_from_images(self, imgs: torch.Tensor) -> torch.Tensor:
        # Expect Uni2Encoder.forward -> [B,Q,D]
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
            self.head.num_prototypes,
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
            self.head.num_prototypes,
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
