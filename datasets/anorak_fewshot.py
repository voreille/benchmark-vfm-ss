from pathlib import Path
from typing import Union

import pandas as pd
from torch.utils.data import DataLoader

from datasets.anorak_dataset import Dataset
from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import CustomTransforms


class ANORAK_FS(LightningDataModule):

    def __init__(
        self,
        root,
        devices,
        num_workers: int,
        fold: int = 0,
        img_size: tuple[int, int] = (448, 448),
        batch_size: int = 1,
        num_classes: int = 7,
        num_metrics: int = 1,
        scale_range=(0.8, 1.2),
        ignore_idx: int = 255,
        overwrite_root: str = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        pin_memory: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            devices=devices,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            num_metrics=num_metrics,
            ignore_idx=ignore_idx,
            img_size=img_size,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
        )
        self.save_hyperparameters()

        # self.transforms = CustomTransforms(img_size=img_size,
        #                                    scale_range=scale_range)

        if overwrite_root:
            root = overwrite_root

        root_dir = Path(root)
        self.fold = fold

        self.images_dir = root_dir / "image"
        self.masks_dir = root_dir / "mask"

        split_df = pd.read_csv(root_dir / "split_df.csv")
        self.split_df = split_df[split_df["fold"] == fold]

        self.batch_size = batch_size
        self.num_workers = num_workers

    def _get_split_ids(self):
        return (
            self.split_df[
                self.split_df["is_train"]]["image_id"].unique().tolist(),
            self.split_df[
                self.split_df["is_val"]]["image_id"].unique().tolist(),
            self.split_df[self.split_df["is_test"]]
            ["image_id"].unique().tolist(),
        )

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        train_ids, val_ids, test_ids = self._get_split_ids()

        if stage == "fit" or stage == "validate" or stage is None:
            self.train_dataset = Dataset(
                train_ids,
                self.images_dir,
                self.masks_dir,
                # transforms=self.transforms,
            )
            self.val_dataset = Dataset(
                val_ids,
                self.images_dir,
                self.masks_dir,
            )

        if stage == "test" or stage is None:
            self.test_dataset = Dataset(
                test_ids,
                self.images_dir,
                self.masks_dir,
            )

        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=False,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
