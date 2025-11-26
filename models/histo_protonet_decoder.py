from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.histo_encoder import Encoder


class ProtoNetLayer(nn.Module):
    def __init__(
        self,
        prototypes: torch.Tensor,
        mean: Optional[torch.Tensor] = None,
        class_counts: Optional[torch.Tensor] = None,
        proj_matrix: Optional[torch.Tensor] = None,
        metric: str = "l2",
        center_feats: bool = True,
        normalize_feats: bool = True,
    ):
        super().__init__()
        m = metric.lower()
        assert m in ("l2", "cosine")
        self.metric = m
        self.center_feats = center_feats
        self.normalize_feats = normalize_feats

        if proj_matrix is not None:
            self.register_buffer("proj_matrix", proj_matrix, persistent=True)
        else:
            self.proj_matrix = None

        if mean is None:
            mean = torch.zeros(prototypes.shape[1], dtype=torch.float32)
        self.register_buffer("mean", mean, persistent=True)

        self.register_buffer("prototypes", prototypes, persistent=True)

        if class_counts is None:
            class_counts = torch.zeros(prototypes.shape[0], dtype=torch.long)
        self.register_buffer("class_counts", class_counts, persistent=True)

    def _project(self, X: torch.Tensor) -> torch.Tensor:
        if self.center_feats:
            X = X - self.mean
        if self.proj_matrix is not None:
            Z = X @ self.proj_matrix
        else:
            Z = X
        if self.normalize_feats:
            Z = F.normalize(Z, dim=-1)
        return Z

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 3:
            B, Q, D = X.shape
            X_flat = X.reshape(B * Q, D)
        elif X.ndim == 2:
            X_flat = X
            B = Q = None
        else:
            raise ValueError("X must be [B,Q,D] or [N,D]")

        Z = self._project(X_flat)
        P = self.prototypes  # [C, D_proj]

        if self.metric == "l2":
            diff = Z.unsqueeze(1) - P.unsqueeze(0)  # [N, C, D_proj]
            scores = -diff.norm(dim=-1, p=2)       # [N, C]
        else:
            Z_n = F.normalize(Z, dim=-1)
            P_n = F.normalize(P, dim=-1)
            scores = Z_n @ P_n.T                   # [N, C]

        if B is not None:
            scores = scores.view(B, Q, -1)         # [B, Q, C]

        return scores


class ProtoNetDecoder(Encoder):
    """
    Encoder → token features → ProtoNetLayer → per-patch logits.

    `prototypes_path` should point to a `.pt` file created by the CLI, containing:
      - mean: [D_in]
      - proj_matrix: [D_in, D_proj]
      - prototypes: [C, D_proj]
      - class_counts: [C] (optional)
      - head_config: {metric, center_feats, normalize_feats, ...}
      - meta: {encoder_id, img_size, ckpt_path, sub_norm, ...}
    """

    def __init__(
        self,
        # encoder config (used only if not overridden by payload)
        encoder_id: str = "h0-mini",
        img_size: Tuple[int, int] = (448, 448),
        ckpt_path: str = "",
        sub_norm: bool = False,
        # proto stats (either from prototypes_path or passed directly)
        prototypes_path: Optional[str] = None,
        prototypes: Optional[torch.Tensor] = None,
        mean: Optional[torch.Tensor] = None,
        proj_matrix: Optional[torch.Tensor] = None,
        class_counts: Optional[torch.Tensor] = None,
        # head behavior (may be overridden by payload)
        metric: str = "l2",
        center_feats: bool = True,
        normalize_feats: bool = True,
    ):
        # ----- load everything from payload if prototypes_path is provided -----
        if prototypes_path is not None:
            payload = torch.load(prototypes_path, map_location="cpu")

            # stats
            prototypes = payload["prototypes"]
            mean = payload["mean"]
            proj_matrix = payload.get("proj_matrix", None)
            class_counts = payload.get("class_counts", None)

            # encoder meta
            meta = payload.get("meta", {})
            encoder_id = meta.get("encoder_id", encoder_id)
            img_size = tuple(meta.get("img_size", img_size))
            ckpt_path = meta.get("ckpt_path", ckpt_path)
            sub_norm = meta.get("sub_norm", sub_norm)

            # head config
            head_cfg = payload.get("head_config", {})
            metric = head_cfg.get("metric", metric).lower()
            center_feats = bool(head_cfg.get("center_feats", center_feats))
            normalize_feats = bool(head_cfg.get("normalize_feats", normalize_feats))

        # ----- build encoder -----
        super().__init__(
            encoder_id=encoder_id,
            img_size=img_size,
            ckpt_path=ckpt_path,
            sub_norm=sub_norm,
        )

        # ----- build head -----
        if prototypes is None:
            raise ValueError(
                "ProtoNetDecoder needs prototypes: pass `prototypes_path` "
                "or provide `prototypes` directly."
            )

        self.num_classes = int(prototypes.shape[0])

        self.head = ProtoNetLayer(
            proj_matrix=proj_matrix,
            mean=mean,
            prototypes=prototypes,
            class_counts=class_counts,
            metric=metric,
            center_feats=center_feats,
            normalize_feats=normalize_feats,
        )

    @classmethod
    def from_prototypes_path(
        cls,
        prototypes_path: str,
        **overrides,
    ) -> "ProtoNetDecoder":
        """
        Convenience constructor for scripts (not used by Lightning CLI).
        All config is read from the payload; `overrides` can be used to
        forcibly override things like img_size if needed.
        """
        return cls(prototypes_path=prototypes_path, **overrides)

    @torch.no_grad()
    def tokens_from_images(self, imgs: torch.Tensor) -> torch.Tensor:
        return super().forward(imgs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokens_from_images(x)  # [B, Q, D]
        B, Q, D = tokens.shape
        Ht, Wt = self.grid_size
        assert Q == Ht * Wt, f"Q={Q} must equal Ht*Wt={Ht * Wt}"

        scores = self.head(tokens)          # [B, Q, C]
        scores = scores.transpose(1, 2)     # [B, C, Q]
        return scores.reshape(B, self.num_classes, Ht, Wt)
