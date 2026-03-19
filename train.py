import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # 用来显示漂亮的进度条

# 导入我们之前写的两个文件里的精华！
from dataset import VOCSegDataset
from model import DINOSegmenter

def train():
    # ==========================================
    # 1. 考场准备 (设置参数与显卡)
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 当前使用的计算设备: {device}")
    
    batch_size = 4  # 每次喂4张图。如果你的显卡显存大，可以改8；如果报错 OOM，改成2。
    num_epochs = 50  # 把整本教材（所有图片）看几遍。我们先看5遍试试水。
    learning_rate = 1e-3 # 纠错的步子迈多大

    # ==========================================
    # 2. 召唤装卸工 (加载数据)
    # ==========================================
    print("📦 正在加载 PASCAL VOC 数据集...")
    # download=False！因为我们已经下好解压了
    train_dataset = VOCSegDataset(root="./voc_data", year='2012', image_set='trainval', download=False)
    
    # 实例化装卸工：打乱顺序(shuffle=True)喂给模型
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"✅ 数据加载完毕！共有 {len(train_dataset)} 张图片，分为 {len(train_loader)} 批。")

    # ==========================================
    # 3. 考生入座，准备答题 (加载模型与优化器)
    # ==========================================
    model = DINOSegmenter(num_classes=21).to(device) # 把模型搬到 GPU 上
    model.train() # 告诉模型：现在是学习模式，请认真算方差！
    
    # 判卷老师：忽略像素值为 255 的边缘区域
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    # 学委：只更新 PSPNet 的参数 (因为 DINOv3 被我们冻结了，所以这里过滤出需要梯度的参数)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # ==========================================
    # 4. 正式开炉炼丹 (Training Loop)
    # ==========================================
    print("\n🚀 开始炼丹！...")
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        # 使用 tqdm 包装一下装卸工，显示漂亮的进度条
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        
        for images, masks in pbar:
            # A. 把图片和答案搬到 GPU 上
            images = images.to(device)
            masks = masks.to(device)

            # B. 考生开始作答 (前向传播)
            outputs = model(images)
            
            # C. 老师批改试卷 (计算 Loss)
            loss = criterion(outputs, masks)
            
            # D. 学委开始纠错 (反向传播与参数更新)
            optimizer.zero_grad() # 清空上一次的纠错记录
            loss.backward()       # 计算怎么调整参数最好 (反向传播)
            optimizer.step()      # 正式更新参数

            # 更新进度条上显示的 Loss 值，让你看着它慢慢变小！
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        # 算一下这一轮看书的平均错误率
        avg_loss = total_loss / len(train_loader)
        print(f"✨ 第 {epoch+1} 轮学习结束，平均 Loss: {avg_loss:.4f}")

    # ==========================================
    # 5. 炼丹结束，保存仙丹 (权重文件)
    # ==========================================
    torch.save(model.state_dict(), "pspnet_dino_weights.pth")
    print("🎉 恭喜！模型训练完成，脑子里的知识已保存为 pspnet_dino_weights.pth")

if __name__ == '__main__':
    train()
