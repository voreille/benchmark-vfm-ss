from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.histo_encoder import Encoder


class ProtoNetLayer(nn.Module):
    """
    Prototype head operating in a (optionally) projected feature space.

    Pipeline (for a single feature vector x):
      1) x_centered = x - mean               (if center_feats=True)
      2) z = x_centered @ proj_matrix        (if proj_matrix is set, else identity)
      3) z = normalize(z)                    (if normalize_feats=True)
      4) scores = similarity(z, prototypes)

    Buffers meant to be loaded from a .pt file (CLI):
      - mean:          [D_in]           (feature mean in encoder space)
      - proj_matrix:   [D_in, D_proj]   (e.g., PCA components^T)
      - prototypes:    [C, D_proj]      (class prototypes in projected space)
    """

    def __init__(
        self,
        num_prototypes: int,
        embedding_dim: int,
        proj_dim: Optional[int] = None,
        metric: str = "l2",
        center_feats: bool = True,
        normalize_feats: bool = True,
        learnable_proj: bool = False,
    ):
        super().__init__()
        m = metric.lower()
        assert m in ("l2", "cosine")
        self.metric = m
        self.center_feats = center_feats
        self.normalize_feats = normalize_feats

        self.embedding_dim = embedding_dim  # D_in
        self.proj_dim = proj_dim or embedding_dim  # D_proj

        # --- Projection: either fixed matrix (buffer) or learnable Linear ---
        if learnable_proj:
            # This is for the future LACE-style learning
            self.proj = nn.Linear(embedding_dim, self.proj_dim, bias=False)
            self.register_buffer("proj_matrix", None, persistent=False)
        else:
            self.proj = None
            # Default: identity projection
            I = torch.eye(embedding_dim, self.proj_dim)
            self.register_buffer("proj_matrix", I, persistent=True)

        # --- Statistics to be filled by CLI ---
        self.register_buffer(
            "mean", torch.zeros(embedding_dim), persistent=True
        )  # [D_in]

        # Prototypes live in projected space [C, D_proj]
        self.register_buffer(
            "prototypes",
            torch.zeros(num_prototypes, self.proj_dim),
            persistent=True,
        )
        # Optional: counts (useful for debugging or analysis)
        self.register_buffer(
            "class_counts",
            torch.zeros(num_prototypes),
            persistent=True,
        )

    # --------- Helpers for the CLI / loading ----------

    @torch.no_grad()
    def load_stats(
        self,
        mean: torch.Tensor,
        prototypes: torch.Tensor,
        proj_matrix: Optional[torch.Tensor] = None,
        class_counts: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Load precomputed statistics (from CLI):

          mean:         [D_in]
          prototypes:   [C, D_proj]
          proj_matrix:  [D_in, D_proj] (optional)
        """
        assert mean.shape[-1] == self.embedding_dim
        assert prototypes.shape[1] == self.proj_dim

        self.mean.copy_(mean)
        self.prototypes.copy_(prototypes)

        if proj_matrix is not None:
            assert (
                proj_matrix.shape[0] == self.embedding_dim
                and proj_matrix.shape[1] == self.proj_dim
            )
            if self.proj is not None:
                # if projection is learnable, you *could* initialize it with this matrix
                self.proj.weight.copy_(proj_matrix.T)
            else:
                self.proj_matrix.copy_(proj_matrix)

        if class_counts is not None:
            self.class_counts.copy_(class_counts)

    # --------- Core preprocessing & forward ----------

    def _project(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [N, D_in] in encoder feature space.
        Returns: Z: [N, D_proj] in projected space.
        """
        if self.center_feats:
            X = X - self.mean  # broadcast over batch

        if self.proj is not None:
            Z = self.proj(X)
        elif self.proj_matrix is not None:
            Z = X @ self.proj_matrix
        else:
            # identity
            Z = X

        if self.normalize_feats:
            Z = F.normalize(Z, dim=-1)

        return Z

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [B, Q, D_in] or [N, D_in]
        Returns:
          scores: [B, Q, C] (or [N, C]) – similarity per prototype
        """
        orig_shape = X.shape
        if X.ndim == 3:
            B, Q, D = X.shape
            X_flat = X.reshape(B * Q, D)
        elif X.ndim == 2:
            X_flat = X
            B = Q = None
        else:
            raise ValueError("X must be [B,Q,D] or [N,D]")

        Z = self._project(X_flat)  # [N, D_proj]
        P = self.prototypes  # [C, D_proj]

        if self.metric == "l2":
            # negative L2 distance as similarity
            # (N, C) = -( (Z - P)^2 summed over dim )
            # we do this efficiently: ||Z-P|| = norm(Z-P, p=2)
            diff = Z.unsqueeze(1) - P.unsqueeze(0)  # [N, C, D_proj]
            scores = -diff.norm(dim=-1, p=2)  # [N, C]
        else:
            # cosine similarity
            Z_n = F.normalize(Z, dim=-1)
            P_n = F.normalize(P, dim=-1)
            scores = Z_n @ P_n.T  # [N, C]

        if B is not None:
            scores = scores.view(B, Q, -1)  # [B, Q, C]

        return scores


class ProtoNetDecoder(Encoder):
    """
    Encoder → token features → ProtoNetLayer → per-patch logits.

    Expects Encoder.forward(imgs) -> [B, Q, D_in].
    grid_size is used only to reshape tokens to spatial map [Ht, Wt].
    """

    def __init__(
        self,
        encoder_id: str = "h0-mini",
        img_size: Tuple[int, int] = (448, 448),
        ckpt_path: str = "",
        sub_norm: bool = False,
        num_classes: int = 7,
        grid_size: Tuple[int, int] = (14, 14),  # (Ht, Wt) so that Q = Ht * Wt
        metric: str = "l2",
        center_feats: bool = True,
        normalize_feats: bool = True,
        proj_dim: Optional[int] = None,
        prototypes_path: Optional[str] = None,
    ):
        super().__init__(
            encoder_id=encoder_id,
            img_size=img_size,
            ckpt_path=ckpt_path,
            sub_norm=sub_norm,
        )
        self.num_classes = num_classes
        self.grid_size = grid_size

        self.head = ProtoNetLayer(
            num_prototypes=num_classes,
            embedding_dim=self.embed_dim,
            proj_dim=proj_dim,
            metric=metric,
            center_feats=center_feats,
            normalize_feats=normalize_feats,
            learnable_proj=False,  # for now, PCA-style
        )

        # Optionally load mean, prototypes, projection matrix from file
        if prototypes_path is not None:
            payload = torch.load(prototypes_path, map_location="cpu")
            mean = payload["mean"]  # [D_in]
            prototypes = payload["prototypes"]  # [C, D_proj]
            proj_matrix = payload.get("proj_matrix", None)  # [D_in, D_proj] or None
            counts = payload.get("class_counts", None)

            with torch.no_grad():
                self.head.load_stats(mean, prototypes, proj_matrix, counts)

    @torch.no_grad()
    def tokens_from_images(self, imgs: torch.Tensor) -> torch.Tensor:
        # Expect Encoder.forward -> [B, Q, D_in]
        return super().forward(imgs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,H,W]

        Returns:
          logits: [B, C, Ht, Wt]
        """
        tokens = self.tokens_from_images(x)  # [B,Q,D_in]
        B, Q, D = tokens.shape
        Ht, Wt = self.grid_size
        assert Q == Ht * Wt, f"Q={Q} must equal Ht*Wt={Ht * Wt}"

        scores = self.head(tokens)  # [B,Q,C]
        scores = scores.transpose(1, 2)  # [B,C,Q]
        return scores.reshape(B, self.num_classes, Ht, Wt)
