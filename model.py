import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  # 👈 引入新买的神器！

# ==========================================
# 模块 1：PSPNet 的核心 - 空间金字塔池化 (PPM)
# ==========================================
class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, bin_sizes=(1, 2, 3, 6)):
        super().__init__()
        reduced_channels = in_channels // len(bin_sizes)
        
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_size),
                nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduced_channels),
                nn.ReLU(inplace=True)
            ) for bin_size in bin_sizes
        ])
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + reduced_channels * len(bin_sizes), out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )

    def forward(self, x):
        h, w = x.shape[2:] 
        features = [x]     
        
        for stage in self.stages:
            pooled_feat = stage(x)
            upsampled_feat = F.interpolate(pooled_feat, size=(h, w), mode='bilinear', align_corners=True)
            features.append(upsampled_feat)
            
        out = torch.cat(features, dim=1)
        return self.bottleneck(out)

# ==========================================
# 模块 2：终极缝合怪 - DINOv3 (timm版) + PSPNet
# ==========================================
class DINOSegmenter(nn.Module):
    def __init__(self, num_classes=21): 
        super().__init__()
        
        print("🚀 正在从 timm 开源库调取免审核版 DINOv3 (ViT-Small)...")
        # 【神级优化】：用 timm 加载模型，设置 features_only=True
        # 它会自动去非受限的 HuggingFace 镜像站下载权重
        # 并且会自动帮我们把 Token 剔除并重塑成 2D 图像矩阵，代码直降十行！
        self.backbone = timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=True, features_only=True)
        
        # 冻结 DINOv3 权重，只当特征提取器
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        embed_dim = 384 
        self.psp_head = PyramidPoolingModule(in_channels=embed_dim, out_channels=256)
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # timm 开启 features_only=True 后，输出的是一个列表，里面装好了 2D 矩阵
        # 我们直接取最后一层特征 [-1] 即可！
        features = self.backbone(x)[-1]
        
        # 让 PSPNet 小弟去涂色
        out = self.psp_head(features)
        out = self.classifier(out)
        
        # 放大回原图尺寸
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out

# ==========================================
# 模块 3：空跑测试 (Dummy Test)
# ==========================================
if __name__ == "__main__":
    print("模型组装完毕，正在生成两张假图片进行体检...")
    dummy_image = torch.randn(2, 3, 512, 512) 
    
    model = DINOSegmenter(num_classes=21)
    output = model(dummy_image)
    
    print("\n" + "="*40)
    print("🎯 体检成功！代码完全跑通！")
    print(f"输入的图片形状: {dummy_image.shape}")
    print(f"模型输出的形状: {output.shape}  --> (Batch大小, 类别数, 高度, 宽度)")
    print("="*40 + "\n")
