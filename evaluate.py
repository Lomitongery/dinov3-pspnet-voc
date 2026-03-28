import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VOCSegDataset
from model import DINOSegmenter


# 这是一个极其高效的数学工具，用来统计“AI 猜的”和“真实答案”之间重合了多少个像素
def fast_hist(a, b, n):
    # 过滤掉标签为 255 的边缘无效像素
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def evaluate():
    # 1. 考试准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 21
    weight_path = "pspnet_dino_weights.pth"  # 你之前炼好的仙丹

    print("📊 正在准备期末考试数据 (Validation Set)...")
    # 注意这里：image_set='val'，我们要拿模型没见过的验证集来考它！
    val_dataset = VOCSegDataset(root="./voc_data", year='2012', image_set='trainval', download=False)
    # 考试不需要打乱顺序 (shuffle=False)，每次批改 4 张
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    print("🧠 正在唤醒模型并进入考试模式...")
    model = DINOSegmenter(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()  # 极其重要：锁死脑子，开启绝对公平的考试模式！

    # 准备一个 21x21 的大表格，用来记录所有像素的对错情况
    hist = np.zeros((num_classes, num_classes))

    # 2. 开始疯狂做题并自动批改
    print("⏳ 考试开始！请耐心等待模型做完所有的题...")
    with torch.no_grad():  # 考试时不做学习笔记（不计算梯度），节省显存
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)  # 真实答案

            outputs = model(images)  # AI 做的答案
            # 提取 AI 最确定的颜色选项
            preds = torch.argmax(outputs, dim=1)

            # 将显卡里的张量拉回到内存里，转成 numpy 方便算分
            preds = preds.cpu().numpy().flatten()
            masks = masks.cpu().numpy().flatten()

            # 将这张图片的对错记录累计到总表格里
            hist += fast_hist(masks, preds, num_classes)

    # 3. 统计最终得分
    # 计算 PA (像素准确率): 对角线上的数字（全蒙对的）除以 总像素
    pixel_accuracy = np.diag(hist).sum() / hist.sum()

    # 计算每个类别的 IoU: 交集 / (预测总数 + 真实总数 - 交集)
    iou_per_class = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

    # 过滤掉无效类别（比如某个类别在验证集里没出现，避免除以0的错误，虽然VOC里基本都有）
    valid_classes = np.isnan(iou_per_class) == False
    miou = np.nanmean(iou_per_class[valid_classes])

    # 4. 打印极其专业的成绩单
    print("\n" + "=" * 40)
    print("🏆 定量分析期末成绩单 🏆")
    print("=" * 40)
    print(f"🎯 全局像素准确率 (Pixel Accuracy): {pixel_accuracy * 100:.2f}%")
    print(f"🎯 平均交并比 (mIoU): {miou * 100:.2f}%")
    print("-" * 40)
    print("详细类别 IoU 得分 (可直接填入报告表格)：")

    # 打印前几个代表性类别（0是背景，1是飞机，15是人）
    class_names = ['Background', 'Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                   'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable', 'Dog', 'Horse',
                   'Motorbike', 'Person', 'Pottedplant', 'Sheep', 'Sofa', 'Train', 'Tvmonitor']

    for i in range(num_classes):
        if valid_classes[i]:
            print(f" - {class_names[i].ljust(15)}: {iou_per_class[i] * 100:.2f}%")
    print("=" * 40)


if __name__ == '__main__':
    evaluate()