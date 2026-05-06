import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from dataset import VOCSegDataset
from eomt.model import EoMT


class EoMTLoss(nn.Module):
    """Simplified per-pixel loss (no Hungarian matching)."""

    def __init__(self, num_classes=21, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, mask_logits, class_logits, masks):
        """
        mask_logits: (B, num_q, H, W) - raw logits
        class_logits: (B, num_q, num_classes+1) - raw logits
        masks: (B, H, W) - ground truth labels (0-20, 255=ignore)
        """
        # Convert to per-pixel predictions
        mask_probs = mask_logits.sigmoid()  # (B, num_q, H, W)
        class_probs = class_logits.softmax(dim=-1)  # (B, num_q, C+1)
        # Remove "no object" class for per-pixel
        per_pixel = torch.einsum("bqhw,bqc->bchw", mask_probs, class_probs[..., :-1])

        # Upsample to match ground truth resolution (128→512 for VOC)
        per_pixel = F.interpolate(per_pixel, size=masks.shape[-2:], mode='bilinear', align_corners=False)

        # Cross-entropy loss
        ce_loss = self.ce(per_pixel, masks)

        # Dice loss (simplified, per-class)
        pred = torch.argmax(per_pixel, dim=1)  # (B, H, W)
        smooth = 1.0
        dice_loss = 0.0
        for c in range(per_pixel.shape[1]):  # num_classes
            pred_c = (pred == c).float()
            mask_c = (masks == c).float()
            intersection = (pred_c * mask_c).sum()
            dice_loss += 1 - (2 * intersection + smooth) / (pred_c.sum() + mask_c.sum() + smooth)
        dice_loss = dice_loss / per_pixel.shape[1]

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def mask_annealing(start_step, current_step, end_step, power=3.0):
    """Polynomial annealing from 1.0 to 0.0."""
    if current_step < start_step:
        return 1.0
    elif current_step >= end_step:
        return 0.0
    progress = (current_step - start_step) / (end_step - start_step)
    return (1.0 - progress) ** power


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def validate(model, val_loader, device):
    model.eval()
    hist = np.zeros((21, 21))
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            mask_logits_list, class_logits_list = model(images)
            # Convert to per-pixel
            mask_probs = mask_logits_list[-1].sigmoid()
            class_probs = class_logits_list[-1].softmax(dim=-1)
            per_pixel = torch.einsum("bqhw,bqc->bchw", mask_probs, class_probs[..., :-1])
            per_pixel = F.interpolate(per_pixel, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            preds = torch.argmax(per_pixel, dim=1)
            for b in range(preds.shape[0]):
                hist += fast_hist(
                    masks[b].cpu().numpy().flatten(),
                    preds[b].cpu().numpy().flatten(),
                    21,
                )
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)
    miou = np.nanmean(iou)
    return miou


def train(batch_size=2, num_epochs=50, learning_rate=1e-3, num_q=100,
          num_blocks=4, pretrained=True, dry_run=False):
    # ── Device ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ─────────────────────────────────────────────────────────
    print("Loading PASCAL VOC 2012 dataset...")
    train_dataset = VOCSegDataset(
        root='../voc_data', year='2012', image_set='train', download=False,
    )
    val_dataset = VOCSegDataset(
        root='../voc_data', year='2012', image_set='val', download=False,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
    )
    print(f"Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} images, {len(val_loader)} batches")

    # ── Model ───────────────────────────────────────────────────────────
    model = EoMT(
        num_classes=21, num_q=num_q, num_blocks=num_blocks,
        pretrained=pretrained, masked_attn_enabled=True,
    ).to(device)
    model.train()

    # ── Optimizer & Loss ────────────────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,
    )
    criterion = EoMTLoss(num_classes=21)

    # ── Annealing schedule ──────────────────────────────────────────────
    # VOC train ~1464 images, batch_size=2 → ~732 steps/epoch
    annealing_start = [0, 5856, 10248, 14640]    # block 0-3
    annealing_end = [8784, 14640, 20496, 26352]  # block 0-3

    # ── Training loop ───────────────────────────────────────────────────
    global_step = 0
    best_miou = 0.0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            mask_logits_list, class_logits_list = model(images)

            # Use final layer output
            loss = criterion(mask_logits_list[-1], class_logits_list[-1], masks)
            loss.backward()
            optimizer.step()

            # Update mask annealing probabilities
            for i in range(model.num_blocks):
                model.attn_mask_probs[i] = mask_annealing(
                    annealing_start[i], global_step, annealing_end[i],
                )

            epoch_loss += loss.item()
            global_step += 1

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}  "
              f"Step: {global_step}  "
              f"Mask probs: {model.attn_mask_probs.tolist()}")

        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            miou = validate(model, val_loader, device)
            print(f"  >>> Val mIoU: {miou:.4f} (best: {max(best_miou, miou):.4f})")
            if miou > best_miou:
                best_miou = miou
                torch.save(model.state_dict(), "eomt_voc_weights.pth")
                print(f"  >>> Saved best weights (mIoU={miou:.4f})")

        # Dry run: break after 1 epoch
        if dry_run:
            print("[DRY RUN] Breaking after 1 epoch")
            break

    print(f"Training complete. Best val mIoU: {best_miou:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Run 1 epoch then exit')
    args = parser.parse_args()
    train(dry_run=args.dry_run)
