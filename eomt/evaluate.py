import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VOCSegDataset
from eomt.model import EoMT


def fast_hist(a, b, n):
    """Build confusion matrix from flattened predictions and ground truth."""
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def evaluate(weight_path="eomt_voc_weights.pth", batch_size=2):
    # ── Setup ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 21

    # ── Error handling ───────────────────────────────────────────────────
    if not os.path.exists(weight_path):
        print(f"❌ 权重文件 {weight_path} 不存在！")
        print("请先运行 python train.py 训练模型。")
        return

    # ── Dataset & DataLoader ─────────────────────────────────────────────
    print("📊 正在加载验证集 (Validation Set)...")
    val_dataset = VOCSegDataset(root='../voc_data', year='2012', image_set='val', download=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ── Model ────────────────────────────────────────────────────────────
    print("🧠 正在加载 EoMT 模型并进入考试模式...")
    model = EoMT(num_classes=num_classes, pretrained=False, masked_attn_enabled=False).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # ── Evaluation loop ──────────────────────────────────────────────────
    hist = np.zeros((num_classes, num_classes))

    print("⏳ 考试开始！请耐心等待模型做完所有的题...")
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            mask_logits_list, class_logits_list = model(images)

            # Use final layer predictions
            mask_logits = mask_logits_list[-1]
            class_logits = class_logits_list[-1]

            # Convert EoMT output to per-pixel predictions
            mask_probs = mask_logits.sigmoid()          # (B, num_q, H, W)
            class_probs = class_logits.softmax(dim=-1)  # (B, num_q, 22)
            # Remove "no object" class for per-pixel
            per_pixel = torch.einsum("bqhw,bqc->bchw", mask_probs, class_probs[..., :-1])
            preds = torch.argmax(per_pixel, dim=1)      # (B, H, W)

            # Accumulate confusion matrix
            for b in range(preds.shape[0]):
                hist += fast_hist(
                    masks[b].cpu().numpy().flatten(),
                    preds[b].cpu().numpy().flatten(),
                    num_classes,
                )

    # ── Metrics ──────────────────────────────────────────────────────────
    pixel_accuracy = np.diag(hist).sum() / hist.sum()

    iou_per_class = np.diag(hist) / (
        hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10
    )

    valid = ~np.isnan(iou_per_class)
    miou = np.nanmean(iou_per_class[valid])

    # ── Output ───────────────────────────────────────────────────────────
    class_names = ['Background', 'Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                   'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable', 'Dog', 'Horse',
                   'Motorbike', 'Person', 'Pottedplant', 'Sheep', 'Sofa', 'Train', 'Tvmonitor']

    print("\n" + "=" * 40)
    print("🏆 EoMT 语义分割成绩单 🏆")
    print("=" * 40)
    print(f"🎯 全局像素准确率 (Pixel Accuracy): {pixel_accuracy * 100:.2f}%")
    print(f"🎯 平均交并比 (mIoU): {miou * 100:.2f}%")
    print("-" * 40)
    print("详细类别 IoU 得分：")
    for i in range(num_classes):
        if valid[i]:
            print(f" - {class_names[i].ljust(15)}: {iou_per_class[i] * 100:.2f}%")
    print("=" * 40)


if __name__ == '__main__':
    evaluate()
