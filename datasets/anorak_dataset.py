from pathlib import Path

from PIL import Image
import torch
from torchvision import tv_tensors
import numpy as np
import pandas as pd


class Dataset(torch.utils.data.Dataset):
    """
    Custom dataset for loading images and masks.
    for the ANORAK data found here https://zenodo.org/records/10016027
    the dataset was preprocessed so to have mask as one int ranging from 0 to 6,
    reprensenting the per-pixel classes.
    """

    def __init__(
        self,
        image_ids,
        images_directory,
        masks_directory,
        ignore_idx=-1,
        return_background=True,
        transforms=None,
        class_mapping=None,
    ):
        self.image_ids = np.array(image_ids)
        self.images_directory = Path(images_directory)
        self.masks_directory = Path(masks_directory)
        self.transforms = transforms
        self.ignore_idx = ignore_idx
        self.data_df = self._get_data_df()
        self.return_background = return_background
        self.class_mapping = class_mapping

    def _get_data_df(self):
        df = pd.DataFrame(columns=["image_id", "image_path", "mask_path"])
        data_list = []
        for image_id in self.image_ids:
            image_path = list(self.images_directory.glob(f"{image_id}.*"))[0]
            mask_path = self.masks_directory / f"{image_id}.png"
            data_list.append(
                {"image_id": image_id, "image_path": image_path, "mask_path": mask_path}
            )
        df = pd.DataFrame(data_list)
        return df.set_index("image_id")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.data_df.index[idx]
        image = tv_tensors.Image(
            Image.open(self.data_df.loc[image_id, "image_path"]).convert("RGB")
        )
        mask = tv_tensors.Mask(
            Image.open(self.data_df.loc[image_id, "mask_path"]).convert("L")
        )

        mask = torch.squeeze(mask, 0)

        unique_labels = torch.unique(mask)
        masks, labels = [], []
        for label_id in unique_labels:
            class_id = label_id.item()

            if class_id != self.ignore_idx and (
                self.return_background or class_id != 0
            ):
                masks.append(mask == label_id)
                labels.append(torch.tensor([class_id]))

        target = {}

        if len(masks) > 0:
            target["masks"] = tv_tensors.Mask(torch.stack(masks))
            target["labels"] = torch.cat(labels)
        else:
            # image has only background or ignore regions
            target["masks"] = tv_tensors.Mask(
                torch.zeros((0, *mask.shape), dtype=torch.bool)
            )
            target["labels"] = torch.zeros((0,), dtype=torch.int64)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


class PredictDataset(torch.utils.data.Dataset):
    """
    Custom dataset for loading images and masks.
    for the ANORAK data found here https://zenodo.org/records/10016027
    the dataset was preprocessed so to have mask as one int ranging from 0 to 6,
    reprensenting the per-pixel classes.
    """

    def __init__(
        self,
        image_ids,
        images_directory,
        masks_directory,
        ignore_idx=-1,
        transforms=None,
    ):
        self.image_ids = np.array(image_ids)
        self.images_directory = Path(images_directory)
        self.masks_directory = Path(masks_directory)
        self.transforms = transforms
        self.ignore_idx = ignore_idx
        self.data_df = self._get_data_df()

    def _get_data_df(self):
        df = pd.DataFrame(columns=["image_id", "image_path", "mask_path"])
        data_list = []
        for image_id in self.image_ids:
            image_path = list(self.images_directory.glob(f"{image_id}.*"))[0]
            mask_path = self.masks_directory / f"{image_id}.png"
            data_list.append(
                {"image_id": image_id, "image_path": image_path, "mask_path": mask_path}
            )
        df = pd.DataFrame(data_list)
        return df.set_index("image_id")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.data_df.index[idx]
        image = tv_tensors.Image(
            Image.open(self.data_df.loc[image_id, "image_path"]).convert("RGB")
        )
        mask = tv_tensors.Mask(
            Image.open(self.data_df.loc[image_id, "mask_path"]).convert("L")
        )

        mask = torch.squeeze(mask, 0)

        unique_labels = torch.unique(mask)
        masks, labels = [], []
        for label_id in unique_labels:
            class_id = label_id.item()

            if class_id != self.ignore_idx:
                masks.append(mask == label_id)
                labels.append(torch.tensor([class_id]))

        target = {
            "masks": tv_tensors.Mask(torch.stack(masks)),
            "labels": torch.cat(labels),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, image_id
