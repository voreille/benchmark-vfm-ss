# TODO: save the right param for the protonet head
# TODO: fix that norm, centering, pca etc should be optional
# TODO: add the option to add another projection matrix for LEACE
# TODO: add the option to select the n of PCA by explained variance ratio

from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

from datasets.anorak import ANORAKFewShot
from models.histo_protonet_decoder import ProtoNetDecoder
from training.tiler import GridPadTiler, Tiler


# ---------------------------
# helpers
# ---------------------------


def parse_devices(devs: List[int]) -> torch.device:
    """Same behavior as your previous CLI: first GPU id, or CPU."""
    if torch.cuda.is_available() and len(devs) > 0:
        torch.cuda.set_device(devs[0])
        return torch.device(f"cuda:{devs[0]}")
    return torch.device("cpu")


@torch.compiler.disable
def to_per_pixel_targets_semantic(
    targets: list[dict],
    ignore_idx: int,
) -> list[torch.Tensor]:
    """Convert list of instance masks into a single-channel semantic map per image."""
    out: list[torch.Tensor] = []
    for t in targets:
        h, w = t["masks"].shape[-2:]
        y = torch.full(
            (h, w),
            ignore_idx,
            dtype=t["labels"].dtype,
            device=t["labels"].device,
        )
        for i, m in enumerate(t["masks"]):
            y[m] = t["labels"][i]
        out.append(y)  # [H,W] long
    return out


@torch.no_grad()
def masks_to_token_hard_nearest(
    targets_sem: torch.Tensor,  # [B,1,H,W] or [B,H,W] long
    grid_size: Tuple[int, int],  # (Ht, Wt)
    ignore_idx: int = 255,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Very simple mask -> token mapping:

      - Downsample semantic mask with nearest neighbor to token grid.
      - Each token gets one hard label in [0..C-1] or ignore_idx.

    Returns:
      y_tokens: [B, Q] long   (token labels; may include ignore_idx)
      valid:    [B, Q] bool   (True if label != ignore_idx)
    """
    Ht, Wt = grid_size

    if targets_sem.ndim == 3:
        targets_sem = targets_sem.unsqueeze(1)  # [B,1,H,W]
    assert targets_sem.ndim == 4, "targets_sem must be [B,H,W] or [B,1,H,W]"

    B, _, H, W = targets_sem.shape

    small = F.interpolate(
        targets_sem.float(),
        size=(Ht, Wt),
        mode="nearest",
    ).long()  # [B,1,Ht,Wt]
    small = small.squeeze(1)  # [B,Ht,Wt]

    y_tokens = small.flatten(1)  # [B,Q]
    valid = y_tokens != ignore_idx  # [B,Q] bool
    return y_tokens, valid


def subsample_tokens_balanced_by_image(
    X: torch.Tensor,  # [N, D]
    y: torch.Tensor,  # [N]
    img_ids: torch.Tensor,  # [N] (global image index for each token)
    num_classes: int,
    max_tokens_per_class: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For each class c, sample up to max_tokens_per_class tokens,
    while covering as many different images as possible:

      - build mapping (class c) -> {image_id -> list(token_indices)}
      - for each class, round-robin over images: pick 1 token from each image
        until we reach max_tokens_per_class or run out.
    """
    if max_tokens_per_class <= 0:
        return X, y, img_ids

    N = X.shape[0]
    assert y.shape[0] == N and img_ids.shape[0] == N

    selected_indices: List[torch.Tensor] = []

    for c in range(num_classes):
        class_mask = y == c
        idx_c = torch.where(class_mask)[0]  # token indices for class c

        if idx_c.numel() == 0:
            continue

        if idx_c.numel() <= max_tokens_per_class:
            selected_indices.append(idx_c)
            continue

        # Map image_id -> list of token indices (for this class)
        img_ids_c = img_ids[idx_c]
        per_img: dict[int, torch.Tensor] = {}

        for local_i, token_idx in enumerate(idx_c):
            img_id_int = int(img_ids_c[local_i].item())
            if img_id_int not in per_img:
                per_img[img_id_int] = []
            per_img[img_id_int].append(token_idx)

        # convert lists to shuffled tensors
        for img_id in per_img:
            arr = torch.tensor(per_img[img_id], dtype=torch.long)
            perm = torch.randperm(arr.numel())
            per_img[img_id] = arr[perm]

        # round-robin sampling
        chosen_for_c: List[torch.Tensor] = []
        count = 0
        # keep iterating until we either fill the quota or everything is empty
        while count < max_tokens_per_class:
            all_empty = True
            for img_id, arr in per_img.items():
                if arr.numel() == 0:
                    continue
                all_empty = False
                chosen_for_c.append(arr[0:1])  # pick one token from this image
                per_img[img_id] = arr[1:]  # remove it
                count += 1
                if count >= max_tokens_per_class:
                    break
            if all_empty:
                break

        if chosen_for_c:
            selected_indices.append(torch.cat(chosen_for_c, dim=0))

    if not selected_indices:
        # fallback: no tokens selected at all
        return X.new_empty((0, X.shape[1])), y.new_empty((0,)), img_ids.new_empty((0,))

    idx = torch.cat(selected_indices, dim=0)
    return X[idx], y[idx], img_ids[idx]


@torch.no_grad()
def accumulate_features_and_labels(
    decoder: ProtoNetDecoder,
    dataloader: DataLoader,
    img_tiler: Tiler,
    target_tiler: Tiler,
    grid_size: Tuple[int, int],
    ignore_idx: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tile images & masks, extract token features, map masks->tokens, accumulate.

    Returns:
      X_all:     [N, D_in]  encoder features (CPU)
      y_all:     [N]        hard token labels
      img_ids_all: [N]      image index id for each token
    """
    X_list: list[torch.Tensor] = []
    y_list: list[torch.Tensor] = []
    img_ids_list: list[torch.Tensor] = []

    decoder.eval()
    decoder.to(device)

    global_img_offset = 0  # to give each original image a unique id across batches

    for imgs, targets in tqdm(dataloader, desc="collect tokens", leave=False):
        # imgs: batch of variable-size images (Tensor or list); targets: list[dict]
        B_batch = len(imgs)

        # assign global image ids for this batch
        batch_img_ids = torch.arange(
            global_img_offset, global_img_offset + B_batch, dtype=torch.long
        )
        global_img_offset += B_batch

        # 1) tile images
        img_crops, origins, img_sizes = img_tiler.window(
            imgs
        )  # [Nc,3,T,T], list(origins)
        img_crops = img_crops.to(device) / 255.0

        Nc = img_crops.shape[0]

        # map each crop to its original image id using origins[i][0] = index in batch
        crop_img_ids = torch.empty(Nc, dtype=torch.long)
        for i, ori in enumerate(origins):
            # Assuming origins[i] = (batch_index, y, x) or similar
            b_idx = int(ori[0])
            crop_img_ids[i] = batch_img_ids[b_idx]

        # 2) targets (instance->semantic->tile)
        sem_full = to_per_pixel_targets_semantic(targets, ignore_idx)  # list of [H,W]
        sem_full = [y.unsqueeze(0) for y in sem_full]  # list of [1,H,W]
        tgt_crops, _, _ = target_tiler.window(sem_full)  # [Nc,1,T,T]
        tgt_crops = tgt_crops.to(device)

        # 3) tokens from encoder
        tokens = decoder.tokens_from_images(img_crops)  # [Nc, Q, D_in]
        Nc, Q, D = tokens.shape

        # 4) simple mask->token mapping (downsample to ViT grid)
        y_tokens, valid = masks_to_token_hard_nearest(
            tgt_crops,  # [Nc,1,T,T]
            grid_size=grid_size,
            ignore_idx=ignore_idx,
        )  # y_tokens: [Nc,Q], valid: [Nc,Q]

        X = tokens.reshape(Nc * Q, D)  # [Nc*Q, D]
        y = y_tokens.reshape(Nc * Q)  # [Nc*Q]
        m = valid.reshape(Nc * Q)  # [Nc*Q]

        # 5) expand crop_img_ids to per-token img_ids
        token_img_ids = crop_img_ids.unsqueeze(1).expand(Nc, Q).reshape(Nc * Q)

        # keep only valid tokens
        X = X[m].cpu()
        y = y[m].cpu()
        img_ids = token_img_ids[m].cpu()

        if X.numel() == 0:
            continue

        X_list.append(X)
        y_list.append(y)
        img_ids_list.append(img_ids)

    if not X_list:
        return (
            torch.empty(0, decoder.embed_dim),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )

    X_all = torch.cat(X_list, dim=0)  # [N,D_in]
    y_all = torch.cat(y_list, dim=0)  # [N]
    img_ids_all = torch.cat(img_ids_list, 0)  # [N]

    return X_all, y_all, img_ids_all


def build_train_loader(
    root_dir: str,
    devices: List[int],
    batch_size: int,
    num_workers: int,
    img_size: Tuple[int, int],
    num_classes: int,
    num_metrics: int,
    ignore_idx: int,
    fold: int,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Reuse ANORAKFewShot exactly like your previous CLI."""
    dm = ANORAKFewShot(
        root=root_dir,
        devices=devices,  # list[int]
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        num_classes=num_classes,
        num_metrics=num_metrics,
        ignore_idx=ignore_idx,
        prefetch_factor=prefetch_factor,
        fold=fold,
    )
    dm.setup("fit")
    return dm.train_dataloader()


# ---------------------------
# main CLI
# ---------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit ProtoNet mean, PCA projection and prototypes from ANORAK segmentation data."
    )

    # --- data / DM related ---
    parser.add_argument(
        "--root-dir", type=str, required=True, help="ANORAK dataset root"
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=[448, 448],
        metavar=("H", "W"),
        help="ViT working size inside the model (grid mapping).",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--ignore-idx", type=int, default=255)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--num-metrics", type=int, default=1)

    # tiler config
    parser.add_argument(
        "--tile", type=int, default=448, help="Crop size (pixels) at source res."
    )
    parser.add_argument("--stride", type=int, default=448, help="Stride between crops.")
    parser.add_argument(
        "--weighted-blend",
        action="store_true",
        help="Use Hann weighting during stitching (for images only).",
    )

    # model/encoder
    parser.add_argument(
        "--encoder-id",
        type=str,
        default="h0-mini",
        help="Backbone encoder id.",
    )
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--ckpt-path", type=str, default="")
    parser.add_argument(
        "--metric",
        type=str,
        default="L2",
        choices=["L2", "cosine"],
    )
    parser.add_argument("--no-center-feats", action="store_true")
    parser.add_argument("--no-normalize-feats", action="store_true")

    # PCA / proto
    parser.add_argument(
        "--grid-h", type=int, required=True, help="Token grid height Ht (ViT output)."
    )
    parser.add_argument(
        "--grid-w", type=int, required=True, help="Token grid width Wt (ViT output)."
    )
    parser.add_argument(
        "--proj-dim",
        type=int,
        default=None,
        help="Projection dimension; None or <=0 = no reduction.",
    )
    parser.add_argument(
        "--max-tokens-per-class",
        type=int,
        default=0,
        help="Max tokens per class (balanced across images). <=0 means use all tokens.",
    )

    # device / output
    parser.add_argument(
        "--device",
        type=int,
        nargs="+",
        default=[0],
        help="GPU ids (first is used for encoder).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Where to save mean/proj_matrix/prototypes (.pt).",
    )

    args = parser.parse_args()

    device = parse_devices(args.device)
    img_size = (args.img_size[0], args.img_size[1])
    grid_size = (args.grid_h, args.grid_w)

    # 1) Build encoder/decoder (same style as your old script)
    decoder = (
        ProtoNetDecoder(
            encoder_id=args.encoder_id,
            num_classes=args.num_classes,
            img_size=img_size,
            sub_norm=False,
            ckpt_path=args.ckpt_path,
            metric=args.metric,
            center_feats=not args.no_center_feats,
            normalize_feats=not args.no_normalize_feats,
        )
        .to(device)
        .eval()
    )

    # 2) Build tilers (image: replicate; targets: constant ignore_idx)
    img_tiler = GridPadTiler(
        tile=args.tile,
        stride=args.stride,
        weighted_blend=args.weighted_blend,
        pad_mode="replicate",
        pad_value=0.0,
    )
    tgt_tiler = GridPadTiler(
        tile=args.tile,
        stride=args.stride,
        weighted_blend=False,
        pad_mode="constant",
        pad_value=float(args.ignore_idx),
    )

    # 3) Build train dataloader
    train_loader = build_train_loader(
        root_dir=args.root_dir,
        devices=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=img_size,
        num_classes=args.num_classes,
        num_metrics=args.num_metrics,
        ignore_idx=args.ignore_idx,
        fold=args.fold,
        prefetch_factor=args.prefetch_factor,
    )

    # 4) Accumulate ALL token features + labels + image ids
    print("Accumulating features and token labels...")
    X_all, y_all, img_ids_all = accumulate_features_and_labels(
        decoder=decoder,
        dataloader=train_loader,
        img_tiler=img_tiler,
        target_tiler=tgt_tiler,
        grid_size=grid_size,
        ignore_idx=args.ignore_idx,
        device=device,
    )

    if X_all.shape[0] == 0:
        raise RuntimeError("No valid tokens collected – check masks & ignore_idx.")

    print(f"Collected {X_all.shape[0]} valid tokens, feature dim = {X_all.shape[1]}.")

    # 5) Optional: rebalance tokens by class with maximum image coverage
    X_used, y_used, img_ids_used = subsample_tokens_balanced_by_image(
        X_all,
        y_all,
        img_ids_all,
        num_classes=args.num_classes,
        max_tokens_per_class=args.max_tokens_per_class,
    )
    print(
        f"Using {X_used.shape[0]} tokens after balancing "
        f"(max_tokens_per_class={args.max_tokens_per_class})."
    )

    N_used, D_in = X_used.shape

    # 6) Compute mean in encoder space (on used tokens)
    mean = X_used.mean(dim=0)  # [D_in]
    X_centered = X_used - mean

    # 7) PCA / projection
    proj_dim = args.proj_dim
    if proj_dim is not None and proj_dim > 0 and proj_dim < D_in:
        if PCA is None:
            raise ImportError(
                "scikit-learn is required for PCA. Install with `pip install scikit-learn`."
            )
        print(f"Fitting PCA to dimension {proj_dim}...")
        pca = PCA(n_components=proj_dim)
        Z_np = pca.fit_transform(X_centered.numpy())  # [N_used, D_proj]
        Z = torch.from_numpy(Z_np).float()  # [N_used, D_proj]
        proj_matrix = torch.from_numpy(pca.components_.T).float()  # [D_in, D_proj]
    else:
        proj_dim_eff = D_in if (proj_dim is None or proj_dim <= 0) else proj_dim
        print(f"No dimensionality reduction, proj_dim={proj_dim_eff}.")
        proj_matrix = torch.eye(D_in, proj_dim_eff, dtype=torch.float32)
        Z = X_centered @ proj_matrix  # [N_used, D_proj]

    # 8) Normalize in projected space
    Z = F.normalize(Z, dim=-1)  # [N_used, D_proj]
    D_proj = Z.shape[1]

    # 9) Compute class prototypes (on used tokens)
    print("Computing prototypes...")
    C = args.num_classes
    prototypes = torch.zeros(C, D_proj, dtype=torch.float32)
    class_counts = torch.zeros(C, dtype=torch.float32)

    for c in range(C):
        mask_c = y_used == c
        if mask_c.any():
            Zc = Z[mask_c]
            prototypes[c] = Zc.mean(dim=0)
            class_counts[c] = float(mask_c.sum().item())

    # 10) Save everything
    payload = {
        "mean": mean,  # [D_in]
        "proj_matrix": proj_matrix,  # [D_in, D_proj]
        "prototypes": prototypes,  # [C, D_proj]
        "class_counts": class_counts,  # [C]
        "num_classes": C,
        "embedding_dim": D_in,
        "proj_dim": D_proj,
        "meta": {
            "encoder_id": args.encoder_id,
            "ckpt_path": args.ckpt_path,
            "img_size": img_size,
            "grid_size": grid_size,
            "ignore_idx": args.ignore_idx,
            "tile": args.tile,
            "stride": args.stride,
            "weighted_blend": args.weighted_blend,
        },
    }
    torch.save(payload, args.output_path)
    print(
        f"[OK] saved to {args.output_path}  "
        f"prototypes={list(prototypes.shape)}, mean={list(mean.shape)}, proj_dim={D_proj}"
    )


if __name__ == "__main__":
    main()
