"""
Implementation adapted from: https://github.com/mbanani/lgssl/blob/df45bae647fc24dce8a6329eb697944053e9a8a0/lgssl/evaluation/fewshot.py#L9C3-L9C3
"""
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize


class ProtoNet:
    """
    Sklearn-like class for SimpleShot.

    Implementation adapted from: https://github.com/mbanani/lgssl/blob/df45bae647fc24dce8a6329eb697944053e9a8a0/lgssl/evaluation/fewshot.py#L9C3-L9C3
    """

    def __init__(self,
                 metric: str = 'L2',
                 center_feats: bool = True,
                 normalize_feats: bool = True) -> None:
        self.metric = metric
        self.center_feats = center_feats
        self.normalize_feats = normalize_feats
        self.mean_ = None
        self.prototype_embeddings_ = None
        self.support_counts_ = None
        self.prototype_labels_ = None

    def fit(self, support_dataloader) -> None:
        """
        support_dataloader must output X and y, where y is a one-hot encoding so NxC
        and X is a feature matrix of shape NxD
        """

        self.fit_mean(support_dataloader)
        self.fit_prototype(support_dataloader)

    def fit_prototype(self, support_dataloader):
        for batch in support_dataloader:
            X, y = batch
            self.aggregate_prototypes(X, y)

        self.prototype_embeddings_ = self.prototype_embeddings_ / self.support_counts_[:,
                                                                                       None]

    def aggregate_prototypes(self, X: torch.Tensor, y: torch.Tensor) -> None:

        if self.prototype_embeddings_ is None:
            self.n_classes_ = y.shape[1]
            self.feature_dim_ = X.shape[1]
            self.prototype_embeddings_ = torch.zeros(
                self.n_classes_, self.feature_dim_).to(X.device)
            self.support_counts_ = torch.zeros(self.n_classes_).to(X.device)

        ### Apply centering and normalization (if set)
        if self.center_feats:
            X = X - self.mean_

        if self.normalize_feats:
            X = normalize(X, dim=-1, p=2)

        self.prototype_embeddings_ = self.prototype_embeddings_ + y.T @ X
        self.support_counts_ = self.support_counts_ + y.sum(dim=0)

    def fit_mean(self, support_dataloader) -> None:
        for batch in support_dataloader:
            X, y = batch

            if self.prototype_labels_ is None:
                self.prototype_labels_ = set(torch.unique(y))
            self.prototype_labels_ = self.prototype_labels_.union(
                set(torch.unique(y)))

            b, feature_dim = X.shape
            if self.mean_ is None:
                self.mean_ = torch.zeros(feature_dim).to(X.device)
                self.support_size_ = 0
            self.mean_ += X.sum(dim=0)
            self.support_size_ += b

        self.mean_ = self.mean_ / self.support_size_
        self.prototype_labels_ = torch.tensor(self.prototype_labels_).to(
            X.device)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Gets the closest prototype for each query in X.

        Args:
            X (torch.Tensor): [N x D]-dim query feature matrix (N: num samples, D: feature dim)

        Returns:
            labels_pred (torch.Tensor): N-dim label vector for X (labels assigned from cloest prototype for each query in X)
        """

        ### Apply centering and normalization (if set)
        if self.center_feats:
            X = X - self.mean_

        if self.normalize_feats:
            X = normalize(X, dim=-1, p=2)

        ### Compute distances, and get the closest prototype for each query as the label
        X = X[:, None]  # [N x 1 x D]
        prototype_embeddings = self.prototype_embeddings[
            None, :]  # [1 x C x D]
        pw_dist = (X - prototype_embeddings).norm(
            dim=-1, p=2)  # [N x C x D] distance/sim matrix
        labels_pred = self.prototype_labels[pw_dist.min(
            dim=1).indices]  # [N,] label vector
        return labels_pred
