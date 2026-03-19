import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms.functional as F

# --- 自定义数据集类，继承自官方的 VOCSegmentation ---
class VOCSegDataset(datasets.VOCSegmentation):
    def __getitem__(self, index):
        # 获取原图和标签图 (PIL 图片格式)
        img, mask = super().__getitem__(index)
        
        # 1. 统一大小：缩放到 512x512
        # 原图可以平滑过渡 (双线性插值)
        img = F.resize(img, (512, 512), interpolation=F.InterpolationMode.BILINEAR)
        # 标签图绝对不能平滑！必须用“最近邻插值”，保证类别编号(0, 1, 2...)还是整数
        mask = F.resize(mask, (512, 512), interpolation=F.InterpolationMode.NEAREST)
        
        # 2. 格式转换与标准化
        # 把原图变成 PyTorch 的 Tensor，并按照 DINOv3 的习惯调整色彩分布
        img = F.to_tensor(img)
        img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # 把标签图变成整数矩阵 (Tensor)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        
        return img, mask

if __name__ == "__main__":
    print("🚀 准备开始下载/加载 VOC2012 数据集...")
    print("💡 注意：压缩包大约有 2GB，首次运行下载可能需要 10-30 分钟，请耐心等待！")
    
    # 定义数据存放的文件夹
    data_dir = "./voc_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 实例化数据集，download=True 表示如果本地没有就自动联网下载
    dataset = VOCSegDataset(root=data_dir, year='2012', image_set='trainval', download=False)
    
    print(f"✅ 太棒了！数据集加载成功，一共包含 {len(dataset)} 张图片。")
    
    # --- 抽查第一张图片，画出来看看效果 ---
    img, mask = dataset[0]
    
    # 把 Tensor 转换回普通图片格式用于可视化
    img_vis = img.permute(1, 2, 0).numpy()
    # 反标准化（把颜色调回人眼看的正常色彩）
    img_vis = img_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_vis = np.clip(img_vis, 0, 1)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Input Image (What DINOv3 sees)")
    plt.imshow(img_vis)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Segmentation Mask (Ground Truth)")
    # 用彩色显示不同的分类标签
    plt.imshow(mask.numpy(), cmap='nipy_spectral', vmin=0, vmax=20) 
    plt.axis('off')
    
    # 保存可视化结果
    plt.savefig("check_data.png")
    print("🎉 抽查图片已保存为 check_data.png，快去文件夹里看看吧！")
