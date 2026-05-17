import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms


class VOCSemantic(LightningDataModule):
    """
    PASCAL VOC 2012 Semantic Segmentation Dataset.

    Reads from **directory** (not zip). Returns raw ``[0, 255]`` uint8 images.
    Normalization is handled by the model internally.
    """

    def __init__(
        self,
        path,
        batch_size=2,
        num_workers=4,
        img_size=(512, 512),
        num_classes=21,
        check_empty_targets=True,
        ignore_idx=255,
        pin_memory=True,
        persistent_workers=True,
    ):
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=img_size,
            num_classes=num_classes,
            check_empty_targets=check_empty_targets,
            ignore_idx=ignore_idx,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.num_classes = num_classes
        self.check_empty_targets = check_empty_targets
        self.ignore_idx = ignore_idx
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.voc_root = Path(path) / "VOCdevkit" / "VOC2012"

    def setup(self, stage=None):
        train_ids = self._read_split("train")
        val_ids = self._read_split("val")

        self.train_dataset = _VOCDataset(
            voc_root=self.voc_root,
            image_ids=train_ids,
            transforms=Transforms(
                img_size=self.img_size,
                color_jitter_enabled=True,
                scale_range=(0.1, 2.0),
            ),
        )
        self.val_dataset = _VOCDataset(
            voc_root=self.voc_root,
            image_ids=val_ids,
        )

    def _read_split(self, split_name):
        split_path = self.voc_root / "ImageSets" / "Segmentation" / f"{split_name}.txt"
        with open(split_path) as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.train_collate,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.eval_collate,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )

    @staticmethod
    def target_parser(target, **kwargs):
        masks, labels = [], []
        for label_id in target[0].unique():
            cls_id = label_id.item()
            if cls_id == 255:
                continue
            masks.append(target[0] == label_id)
            labels.append(cls_id)
        return masks, labels, [False for _ in range(len(masks))]


class _VOCDataset(Dataset):
    """Internal VOC dataset that reads from directory (no zip)."""

    def __init__(self, voc_root, image_ids, transforms=None):
        self.voc_root = Path(voc_root)
        self.image_ids = image_ids
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        img_path = self.voc_root / "JPEGImages" / f"{image_id}.jpg"
        label_path = self.voc_root / "SegmentationClass" / f"{image_id}.png"

        img = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)

        from torchvision.transforms.v2 import functional as F
        img_tensor = F.to_image(img)
        label_tensor = F.to_image(label).to(torch.long)

        label_tensor = tv_tensors.Mask(label_tensor)

        masks, labels, is_crowd = VOCSemantic.target_parser(target=label_tensor)

        if len(masks) > 0:
            masks_tensor = tv_tensors.Mask(torch.stack(masks))
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            is_crowd_tensor = torch.tensor(is_crowd, dtype=torch.bool)
        else:
            masks_tensor = tv_tensors.Mask(torch.zeros((0, *img_tensor.shape[-2:]), dtype=torch.bool))
            labels_tensor = torch.zeros(0, dtype=torch.long)
            is_crowd_tensor = torch.zeros(0, dtype=torch.bool)

        target = {"masks": masks_tensor, "labels": labels_tensor, "is_crowd": is_crowd_tensor}

        if self.transforms is not None:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target
