import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# 添加项目根目录到 sys.path，以便导入 dataset.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from eomt.model import EoMT

# ==========================================
# 1. PASCAL VOC 的调色板 (给不同物体涂上专属颜色)
# ==========================================
VOC_COLORMAP = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128]
]


def decode_segmap(image, nc=21):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = VOC_COLORMAP[l][0]
        g[idx] = VOC_COLORMAP[l][1]
        b[idx] = VOC_COLORMAP[l][2]
    return np.stack([r, g, b], axis=2)


def predict(image_path, weight_path="eomt_voc_weights.pth"):
    # ── 错误处理 ──────────────────────────────────────────────────────
    if not os.path.exists(weight_path):
        print(f"❌ 权重文件 {weight_path} 不存在！")
        print("请先运行 python train.py 训练模型。")
        return
    if not os.path.exists(image_path):
        print(f"❌ 图片文件 {image_path} 不存在！")
        return

    # ── 设备 ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    # ── 加载模型 ──────────────────────────────────────────────────────
    model = EoMT(num_classes=21, pretrained=False, masked_attn_enabled=False).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # ── 图片预处理 ────────────────────────────────────────────────────
    raw_image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(raw_image).unsqueeze(0).to(device)

    # ── 推理 ──────────────────────────────────────────────────────────
    print("🧠 推理中...")
    with torch.no_grad():
        mask_logits_list, class_logits_list = model(img_tensor)

    # 转换为逐像素预测
    mask_probs = mask_logits_list[-1].sigmoid()
    class_probs = class_logits_list[-1].softmax(dim=-1)
    per_pixel = torch.einsum("bqhw,bqc->bchw", mask_probs, class_probs[..., :-1])
    pred = torch.argmax(per_pixel, dim=1).squeeze().cpu().numpy()
    colorized_pred = decode_segmap(pred)

    # ── 可视化 ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(raw_image.resize((512, 512)))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(colorized_pred)
    axes[1].set_title('EoMT Prediction')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig("result.png")
    print("🎉 预测完成！结果已保存为 result.png")


if __name__ == "__main__":
    predict("test.jpg")
