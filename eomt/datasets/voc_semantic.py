# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from datasets.lightning_data_module import LightningDataModule


class VOCSemantic(LightningDataModule):
    """
    PASCAL VOC 2012 Semantic Segmentation Dataset.

    Reads from DIRECTORY (not zip). VOC data structure::

        voc_root/
            VOCdevkit/VOC2012/
                JPEGImages/          # .jpg images
                SegmentationClass/   # .png labels (pixel values = class ID 0-20, 255=ignore)
                ImageSets/Segmentation/
                    train.txt        # training image IDs
                    val.txt          # validation image IDs

    Returns raw [0, 255] uint8 images. Normalization is handled by the model.
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
        self.voc_root = Path(path) / "VOCdevkit" / "VOC2012"

    def setup(self, stage=None):
        """Create train and val datasets."""
        train_ids = self._read_split("train")
        val_ids = self._read_split("val")

        self.train_dataset = _VOCDataset(
            voc_root=self.voc_root,
            image_ids=train_ids,
            img_size=self.img_size,
        )
        self.val_dataset = _VOCDataset(
            voc_root=self.voc_root,
            image_ids=val_ids,
            img_size=self.img_size,
        )

    def _read_split(self, split_name):
        """Read image IDs from ``ImageSets/Segmentation/{split_name}.txt``."""
        split_path = self.voc_root / "ImageSets" / "Segmentation" / f"{split_name}.txt"
        with open(split_path) as f:
            return [line.strip() for line in f if line.strip()]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=self.train_collate,
            drop_last=True,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )

    @staticmethod
    def target_parser(target, **kwargs):
        """
        Convert VOC label PNG to mask/label list.

        VOC ``SegmentationClass`` PNG has pixel values = class ID (0-20, 255=ignore).

        Returns ``(masks_list, labels_list, is_crowd_list)`` format required by EoMT.
        """
        masks, labels = [], []
        for label_id in target[0].unique():
            cls_id = label_id.item()
            if cls_id == 255:  # ignore / void
                continue
            masks.append(target[0] == label_id)
            labels.append(cls_id)  # VOC class IDs are already 0-20
        return masks, labels, [False for _ in range(len(masks))]


class _VOCDataset(Dataset):
    """Internal VOC dataset that reads images and labels from directory."""

    def __init__(self, voc_root, image_ids, img_size=(512, 512)):
        self.voc_root = Path(voc_root)
        self.image_ids = image_ids
        self.img_size = img_size

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Load image (RGB, raw uint8)
        img_path = self.voc_root / "JPEGImages" / f"{image_id}.jpg"
        img = Image.open(img_path).convert("RGB")

        # Load label (single-channel PNG, pixel values = class IDs)
        label_path = self.voc_root / "SegmentationClass" / f"{image_id}.png"
        label = Image.open(label_path)

        # Resize to target size
        img = F.resize(img, self.img_size, interpolation=F.InterpolationMode.BILINEAR)
        label = F.resize(label, self.img_size, interpolation=F.InterpolationMode.NEAREST)

        # Convert to tensors
        img_tensor = F.to_image(img)  # (3, H, W), uint8, [0, 255]
        label_tensor = F.to_image(label).to(torch.long)  # (1, H, W), int64

        # Wrap as tv_tensors for transform compatibility
        img_tensor = tv_tensors.Image(img_tensor)
        label_tensor = tv_tensors.Mask(label_tensor)

        # Parse targets
        masks, labels, is_crowd = VOCSemantic.target_parser(target=label_tensor)

        # Stack masks
        if len(masks) > 0:
            masks_tensor = tv_tensors.Mask(torch.stack(masks))
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            is_crowd_tensor = torch.tensor(is_crowd, dtype=torch.bool)
        else:
            # Handle empty targets (all pixels are 255 / ignore)
            masks_tensor = tv_tensors.Mask(
                torch.zeros((0, *self.img_size), dtype=torch.bool)
            )
            labels_tensor = torch.zeros(0, dtype=torch.long)
            is_crowd_tensor = torch.zeros(0, dtype=torch.bool)

        target = {
            "masks": masks_tensor,
            "labels": labels_tensor,
            "is_crowd": is_crowd_tensor,
        }

        return img_tensor, target
