"""Manifest-based dataset for avatar detection.

Reads from a single CSV manifest with columns:
    path,label,dataset,split,generator

Filters by split, applies transforms, and returns per-sample dicts.
Label encoding: 0=real, 1=fake (matches v11 convention).
"""

import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

LABEL_MAP = {"real": 0, "fake": 1}


class AvatarDataset(Dataset):
    """PyTorch Dataset backed by a manifest CSV.

    Args:
        manifest_path: Path to the CSV manifest file.
        split: One of 'train', 'val', 'test', 'ood_test'.
        transform: Optional transform callable. If None, default transforms
            are applied based on the split.
        img_size: Target image size (default 256).
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        split: str,
        transform: Optional[Callable] = None,
        img_size: int = 256,
    ):
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.img_size = img_size
        self.transform = transform

        self.df = pd.read_csv(self.manifest_path)
        self.df = self.df[self.df["split"] == self.split].reset_index(drop=True)

        self._validate_paths()

        if len(self.df) == 0:
            raise ValueError(
                f"No samples found for split='{split}' in {self.manifest_path}. "
                f"Available splits: {self.df['split'].unique().tolist() if len(self.df) > 0 else 'empty manifest'}"
            )

    def _validate_paths(self) -> None:
        """Remove rows whose image files don't exist, emitting warnings."""
        valid_mask = self.df["path"].apply(lambda p: Path(p).exists())
        n_invalid = (~valid_mask).sum()
        if n_invalid > 0:
            invalid_paths = self.df.loc[~valid_mask, "path"].tolist()
            for p in invalid_paths:
                warnings.warn(
                    f"Skipping missing image file: {p}", UserWarning, stacklevel=2
                )
            self.df = self.df[valid_mask].reset_index(drop=True)

    def _default_transform(self) -> A.Compose:
        if self.split == "train":
            return A.Compose(
                [
                    A.RandomResizedCrop(
                        size=(self.img_size, self.img_size), scale=(0.7, 1.0)
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(height=self.img_size, width=self.img_size),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        img_path = row["path"]
        label_str = str(row["label"]).strip().lower()
        if label_str in LABEL_MAP:
            label = LABEL_MAP[label_str]
        else:
            label = int(float(label_str))

        image = np.array(Image.open(img_path).convert("RGB"))

        transform = (
            self.transform if self.transform is not None else self._default_transform()
        )
        try:
            result = transform(image=image)
        except TypeError:
            result = transform(image)

        if isinstance(result, dict):
            image_tensor = result["image"]
        else:
            image_tensor = result

        return {
            "image": image_tensor,
            "label": label,
            "dataset": str(row["dataset"]),
            "path": str(img_path),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
        """Collate function for DataLoader — stacks tensors, preserves metadata."""
        images = torch.stack([item["image"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        datasets = [item["dataset"] for item in batch]
        paths = [item["path"] for item in batch]

        return {
            "image": images,
            "label": labels,
            "dataset": datasets,
            "path": paths,
        }
