# 官方 EoMT 仓库集成说明

## 第一章：项目背景与目标

### 1.1 原有项目

本项目最初的目标是在 PASCAL VOC 2012 数据集上使用 DINOv3 骨干网络进行语义分割。项目根目录位于 `/home/xia/DINO_Project/`，其中 `pspnet/` 子目录包含了基于 PSPNet 架构的原始实现。该实现使用 DINOv3 作为编码器，PSPNet 池化模块作为解码器，在 VOC 2012 数据集上训练。

### 1.2 第一次尝试：自己写简化版 EoMT

在了解到 EoMT（Emergent Open-World Mask Transformers，CVPR 2025 Highlight）这一先进架构后，我们决定尝试将其应用于 VOC 语义分割任务。最初的方案是参考 EoMT 论文的核心思想，自己编写一个简化版实现。这个阶段创建了以下文件：

- `model.py`（251 行）：简化版 EoMT 模型，包含 query tokens、mask 预测头、class 预测头等核心组件
- `train.py`（191 行）：简化版训练脚本，包含数据加载、训练循环、日志记录
- `evaluate.py`（101 行）：简化版评估脚本，计算 mIoU 指标
- `predict.py`（95 行）：简化版推理脚本，对单张图片进行预测和可视化

这个简化版虽然能够运行，但存在几个问题：

1. **缺少官方关键特性**：没有实现 masked attention 机制、attention mask annealing、匈牙利匹配损失等论文核心创新点
2. **训练效果不理想**：由于缺少上述关键特性，模型收敛速度和最终精度都不理想
3. **维护成本高**：需要自己维护大量训练基础设施代码（优化器调度、日志、checkpoint 管理等）

### 1.3 最终决定：直接使用官方 EoMT 仓库

经过评估，我们决定放弃自己编写的简化版，直接集成官方 EoMT 仓库（github.com/tue-mps/eomt）。这个决定基于以下考虑：

1. **官方代码质量高**：EoMT 是 CVPR 2025 Highlight 论文，官方代码经过同行评审，实现完整且正确
2. **PyTorch Lightning 框架**：官方使用 PyTorch Lightning 组织训练流程，代码结构清晰，易于理解和修改
3. **配置驱动**：官方使用 LightningCLI + YAML 配置文件管理所有超参数，无需硬编码
4. **MIT License**：官方采用 MIT 开源协议，可以自由使用和修改

### 1.4 核心目标

本次集成的核心目标是：

1. 保留官方 EoMT 的核心模型架构和训练流程
2. 适配本地 PASCAL VOC 2012 数据集（已解压到目录，非 zip 压缩包）
3. 使用 TensorBoard 替代 wandb 进行日志记录
4. 支持单 GPU 训练（RTX 5070）
5. 保持代码可维护性和可扩展性

### 1.5 集成策略

在集成过程中，我们遵循以下原则：

**最小改动原则**：尽可能保留官方代码原样，只修改必要的部分。对于官方代码中与本地环境不兼容的部分（如 DINOv2 vs DINOv3 的 RoPE 差异），做最小化的适配修改。

**分层适配**：将本地特有的逻辑（如 VOC 数据集读取）封装在新增文件中，不修改官方核心代码。这样当官方仓库更新时，可以方便地合并上游变更。

**配置驱动**：所有超参数通过 YAML 配置文件管理，避免硬编码。训练、评估、恢复等操作通过命令行参数控制，无需修改代码。

---

## 第二章：官方仓库概览

### 2.1 仓库基本信息

- **仓库地址**：github.com/tue-mps/eomt
- **许可证**：MIT License
- **论文**：EoMT: Emergent Open-World Mask Transformers（CVPR 2025 Highlight）
- **编程语言**：Python（100%）
- **框架**：PyTorch + PyTorch Lightning

### 2.2 整体架构

官方仓库的代码组织非常清晰，按功能模块划分目录：

```
eomt/
├── main.py                          # LightningCLI 入口
├── models/
│   ├── eomt.py                      # EoMT 核心模型
│   ├── vit.py                       # ViT 编码器封装
│   └── scale_block.py               # 上采样模块
├── training/
│   ├── lightning_module.py          # LightningModule 基类
│   ├── mask_classification_loss.py  # Mask2Former 损失函数
│   ├── mask_classification_semantic.py  # 语义分割 LightningModule
│   └── two_stage_warmup_poly_schedule.py  # 学习率调度
├── datasets/
│   ├── dataset.py                   # 基于 zip 的数据集基类
│   ├── lightning_data_module.py     # LightningDataModule 基类
│   └── transforms.py                # 数据增强
├── configs/                         # YAML 配置文件
│   └── dinov2/                      # 按 backbone 组织
│       └── voc/                     # 按数据集组织
│           └── semantic/            # 按任务组织
│               └── eomt_small_512.yaml
├── requirements.txt
└── .gitignore
```

### 2.3 各模块功能详解

#### main.py — 程序入口

官方 `main.py` 约 183 行，使用 `LightningCLI` 解析命令行参数和 YAML 配置文件。它包含了：

- wandb 日志集成
- `torch.compile` 编译优化
- 复杂的验证调度逻辑
- `jsonargparse` 类型补丁

#### models/eomt.py — 核心模型

EoMT 的核心创新在于向 ViT 骨干网络的深层 block 中插入可学习的 query tokens，并通过 masked attention 机制让这些 queries 关注图像的前景区域。模型结构如下：

1. **ViT 编码器**：提取图像特征（patch tokens + prefix tokens）
2. **Query tokens**：在最后 N 个 block 之前插入 100 个可学习的 query embeddings
3. **Masked attention**：每个 query 只关注预测的前景 patch，减少计算量并提高定位精度
4. **Mask 预测头**：将 query 特征上采样到原始分辨率，预测每个 query 对应的 mask
5. **Class 预测头**：对每个 query 预测类别 logits

#### models/vit.py — ViT 封装

支持两种加载方式：

1. **timm 模型**：通过 `timm.create_model()` 加载，支持 DINOv2、DINOv3 等预训练权重
2. **HuggingFace Transformers 模型**：通过 `AutoModel.from_pretrained()` 加载，自动转换为 timm 兼容接口

#### models/scale_block.py — 上采样模块

由转置卷积（2x 上采样）、GELU 激活、Depthwise 卷积（3x3）、LayerNorm 组成。用于将 patch 级别的特征上采样到更高分辨率。

#### training/lightning_module.py — 训练基类

约 911 行，包含：

- `configure_optimizers()`：AdamW 优化器 + LLRD（层间学习率衰减）+ 两阶段 warmup 多项式调度
- `training_step()`：单步训练逻辑
- `validation_step()`：验证逻辑
- `mask_annealing()`：attention mask annealing 概率计算
- 评估指标：mIoU（语义）、mAP（实例）、PQ（全景）
- 可视化：`plot_semantic()` 绘制预测结果
- 大图处理：`window_imgs_semantic()` 滑动窗口推理

#### training/mask_classification_loss.py — 损失函数

继承自 HuggingFace Transformers 的 `Mask2FormerLoss`，包含：

- **匈牙利匹配**：`Mask2FormerHungarianMatcher` 在预测 queries 和 ground truth masks 之间建立一一对应
- **Mask 损失**：BCE（二值交叉熵）+ Dice 系数
- **Class 损失**：交叉熵（CE）
- **损失加权**：各分量按系数加权求和

#### training/mask_classification_semantic.py — 语义分割模块

继承 `LightningModule`，为语义分割任务定制：

- 初始化语义分割指标（`MulticlassJaccardIndex`）
- `eval_step()`：滑动窗口推理 + 逐像素预测
- `on_validation_epoch_end()`：计算并记录 mIoU

#### training/two_stage_warmup_poly_schedule.py — 学习率调度

两阶段 warmup + 多项式衰减：

1. **非 ViT 参数**：从 step 0 开始线性 warmup 到 `warmup_steps[0]`
2. **ViT 参数**：前 `warmup_steps[0]` 步学习率为 0，然后线性 warmup `warmup_steps[1]` 步
3. **多项式衰减**：warmup 结束后，学习率按 `(1 - progress)^poly_power` 衰减

#### datasets/dataset.py — 数据集基类

基于 zip 文件的通用数据集类，支持：

- 从 zip 压缩包中读取图像和标签
- 语义、实例、全景三种标注格式
- COCO JSON 标注解析
- 嵌套 zip 支持
- 多 worker 安全加载

#### datasets/lightning_data_module.py — DataModule 基类

提供 `train_collate()` 和 `eval_collate()` 两个 collate 函数，以及共享的 `dataloader_kwargs` 字典。

#### datasets/transforms.py — 数据增强

包含 ColorJitter、随机翻转、ScaleJitter、Pad、随机裁剪等增强操作。使用 `tv_tensors` 确保变换同时作用于图像和 mask。

### 2.4 依赖项

官方仓库的核心依赖：

- `lightning>=2.0.0`：PyTorch Lightning 框架，提供 Trainer、LightningModule、LightningDataModule、LightningCLI 等基础设施
- `transformers>=4.38.0`：HuggingFace Transformers 库，提供 Mask2Former 损失函数（匈牙利匹配 + BCE/Dice/CE）
- `timm>=1.0.0`：PyTorch Image Models 库，提供 ViT 骨干网络（DINOv2、DINOv3 等预训练权重）
- `wandb`：Weights & Biases 实验跟踪平台
- `jsonargparse[signatures]`：基于 Python 类型注解的 YAML 配置解析器，被 LightningCLI 使用
- `torchmetrics`：PyTorch 评估指标库，提供 mIoU、mAP、PQ 等指标
- `fvcore`：Facebook 基础工具库，提供一些实用函数

### 2.5 官方代码的数据流

理解官方代码的数据流有助于后续的修改工作。一次完整的训练迭代涉及以下步骤：

1. **数据加载**：`Dataset.__getitem__()` 从 zip 中读取图像和标签，调用 `target_parser` 转换为 mask/label 列表格式
2. **数据增强**：`Transforms.forward()` 对图像和 mask 应用 ColorJitter、翻转、缩放、裁剪等操作
3. **批处理**：`LightningDataModule.train_collate()` 将单张图像堆叠为 batch
4. **模型前向**：`LightningModule.forward()` 除以 255 -> `EoMT.forward()` 归一化 -> ViT 编码 -> 插入 queries -> L2 blocks -> 预测
5. **损失计算**：`MaskClassificationLoss.forward()` 匈牙利匹配 + BCE/Dice/CE
6. **反向传播**：PyTorch 自动求导 + AdamW 优化器更新参数
7. **日志记录**：`self.log()` 将损失和指标输出到 wandb/TensorBoard

---

## 第三章：保留不变的官方文件

以下文件从官方仓库原样复制，未做任何修改：

### 3.1 models/vit.py（68 行）

ViT 编码器封装。支持通过 timm 或 HuggingFace Transformers 加载预训练视觉骨干网络。核心功能：

- 根据 `backbone_name` 是否包含 `/` 判断加载方式
- 对 HuggingFace 模型，将 `embeddings`、`layer` 等属性映射为 timm 兼容接口
- 注册 `pixel_mean` 和 `pixel_std` 缓冲区，用于输入归一化

### 3.2 models/scale_block.py（38 行）

上采样模块。由 `ConvTranspose2d`（2x 上采样）、`GELU`、`Conv2d`（3x3 depthwise）、`LayerNorm2d` 组成。用于将 patch 级别的 mask logits 上采样到更高分辨率。

### 3.3 training/mask_classification_loss.py（120 行）

Mask2Former 损失函数。继承自 HuggingFace Transformers 的 `Mask2FormerLoss`，包含：

- 匈牙利匹配器 `Mask2FormerHungarianMatcher`
- Mask 损失（BCE + Dice）
- Class 损失（交叉熵）
- `loss_total()` 方法：对各层损失加权求和

### 3.4 training/mask_classification_semantic.py（116 行）

语义分割专用的 LightningModule。继承 `LightningModule` 基类，添加：

- 语义分割指标初始化（`MulticlassJaccardIndex`）
- `eval_step()`：滑动窗口推理 + 逐像素预测 + 指标更新
- `on_validation_epoch_end()`：计算并记录 mIoU

### 3.5 training/two_stage_warmup_poly_schedule.py（49 行）

两阶段 warmup + 多项式衰减学习率调度器。ViT 参数和非 ViT 参数使用不同的 warmup 策略。

### 3.6 datasets/dataset.py（308 行）

基于 zip 文件的通用数据集基类。虽然本地项目不使用 zip 读取（VOC 数据已解压），但保留该文件作为参考和潜在扩展用途。

### 3.7 datasets/lightning_data_module.py（52 行）

LightningDataModule 基类。提供 `train_collate()` 和 `eval_collate()` 两个静态方法，以及共享的 `dataloader_kwargs` 字典。

### 3.8 datasets/transforms.py（118 行）

数据增强模块。包含 ColorJitter、随机翻转、ScaleJitter、Pad、随机裁剪等操作。使用 `tv_tensors` 确保变换同时作用于图像和 mask。

---

## 第四章：修改的官方文件及原因

### 4.1 models/eomt.py — 核心模型（改动最大）

这是整个集成过程中改动最大的文件。官方 `eomt.py` 为 DINOv2 设计，而本地项目使用 DINOv3。DINOv3 的 `EvaAttention` 实现在 RoPE（旋转位置编码）处理上与 DINOv2 有显著差异。

#### 改动 1：`_pos_embed` 返回元组处理（第 215-220 行）

**原代码**：
```python
x = self.encoder.backbone._pos_embed(x)
```

**新代码**：
```python
pos_out = self.encoder.backbone._pos_embed(x)
if isinstance(pos_out, tuple):
    x, rope = pos_out
else:
    x = pos_out
```

**原因**：DINOv3 的 `_pos_embed` 方法返回 `(x, rope)` 元组，其中 `rope` 是旋转位置编码。而 DINOv2 的 `_pos_embed` 只返回 `x`。需要兼容两种行为。

#### 改动 2：`_attn` 方法重写——手动 RoPE 处理（第 84-175 行）

这是最核心的改动。官方代码在 `rope is not None` 时直接调用 `module(x, mask, rope)[0]`，将 RoPE 处理委托给 attention 模块内部。但 DINOv3 的 `EvaAttention` 在内部对 `num_prefix_tokens` 之后的 token 应用 RoPE，当 EoMT 在位置 0 插入 100 个 query tokens 后，token 顺序变为：

```
[queries(100), prefix(5), patches(1024)]
```

此时 `EvaAttention` 内部的 RoPE 索引偏移导致两个问题：

1. **RoPE 应用到错误的 token**：RoPE 本应只应用于 patches，但由于 queries 的插入，索引计算错误
2. **RoPE 尺寸不匹配**：`EvaAttention` 生成的 RoPE 尺寸基于原始 patch 数量，插入 queries 后序列变长

**解决方案**：手动拆分 queries 和 prefix+patches，仅对 patches 应用 RoPE：

```python
if N > num_prefix + num_patches_expected:
    # Queries 存在——拆分并手动处理 RoPE
    queries = x[:, :self.num_q, :]
    prefix_and_patches = x[:, self.num_q:, :]

    # 对 prefix+patches 应用 attention（带 RoPE）
    qkv = module.qkv(prefix_and_patches).reshape(...)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    q, k = module.q_norm(q), module.k_norm(k)

    # 仅对 patches 应用 RoPE，跳过 prefix
    from timm.layers.pos_embed_sincos import apply_rot_embed_cat
    q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope)], dim=2)
    k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope)], dim=2)

    # 对 queries 单独处理（不应用 RoPE）
    qkv_q = module.qkv(queries).reshape(...)
    q_q, k_q, v_q = qkv_q.permute(2, 0, 3, 1, 4).unbind(0)

    # 拼接：queries（无 RoPE）+ prefix+patches（有 RoPE）
    q = torch.cat([q_q, q], dim=2)
    k = torch.cat([k_q, k], dim=2)
    v = torch.cat([v_q, v], dim=2)
```

#### 改动 3：`head_dim` 动态计算（第 108、153 行）

**原代码**：`module.head_dim`

**新代码**：`module.qkv.weight.shape[0] // 3 // module.num_heads`

**原因**：DINOv3 的 `EvaAttention` 可能没有 `head_dim` 属性。从 QKV 权重形状动态计算更鲁棒，兼容不同版本的 timm 实现。

#### 改动 4：`forward` 方法中 RoPE 的获取方式（第 209-220 行）

**原代码**：
```python
rope = None
if hasattr(self.encoder.backbone, "rope_embeddings"):
    rope = self.encoder.backbone.rope_embeddings(x)
```

**新代码**：
```python
rope = None
if hasattr(self.encoder.backbone, "rope_embeddings"):
    rope = self.encoder.backbone.rope_embeddings(x)

x = self.encoder.backbone.patch_embed(x)

if hasattr(self.encoder.backbone, "_pos_embed"):
    pos_out = self.encoder.backbone._pos_embed(x)
    if isinstance(pos_out, tuple):
        x, rope = pos_out  # 从 _pos_embed 返回值中获取 rope
    else:
        x = pos_out
```

**原因**：DINOv3 的 RoPE 由 `_pos_embed` 方法生成并返回，而不是通过独立的 `rope_embeddings` 方法。需要同时支持两种获取方式。

### 4.2 main.py — 入口文件（大幅简化）

**改动**：从约 183 行简化为 34 行

**移除了**：
- wandb 代码日志集成
- `torch.compile` 编译优化
- 复杂验证调度逻辑
- `jsonargparse` 类型补丁

**保留了**：
- `LightningCLI` 设置
- `subclass_mode_model=True` 和 `subclass_mode_data=True`
- `link_arguments` 参数链接

**改为 `EoMTCLI` 子类**：将 `link_arguments` 移到 `add_arguments_to_parser` 方法中，使代码更清晰。

**原因**：官方 `main.py` 为完整论文复现设计，包含 wandb 集成、多 GPU 训练、编译优化等特性。本地项目只需要 TensorBoard 日志 + 单 GPU 训练 + 简化配置。精简后的 `main.py` 更易于理解和维护。

### 4.3 training/lightning_module.py — 训练基类（微小改动）

**改动**：wandb 导入改为条件导入

**原代码**：
```python
import wandb
```

**新代码**：
```python
try:
    import wandb
except ImportError:
    wandb = None
```

**原因**：官方代码使用 wandb 进行实验跟踪和可视化（`plot_semantic` 方法中调用 `wandb.Image`）。本地项目使用 TensorBoard 作为主要日志工具，不需要强制安装 wandb。条件导入确保未安装 wandb 时程序不会崩溃。

注意：`plot_semantic` 方法仍然保留 wandb 调用，但仅在安装了 wandb 时生效。如果未安装 wandb，可视化功能将被跳过，不影响训练流程。

### 4.4 requirements.txt — 依赖（适配本地环境）

**改动**：
- 移除了 `wandb`：本地使用 TensorBoard
- 移除了 `gitignore_parser`：本地不需要
- 添加了 `tensorboard`：替代 wandb 进行日志记录
- 取消版本锁定：`torch`、`torchvision` 使用系统安装版本
- 放宽版本要求：`lightning>=2.0.0`、`transformers>=4.38.0`、`timm>=1.0.0` 等

### 4.5 .gitignore — 简化

**改动**：从约 180 行简化为 25 行

官方 `.gitignore` 包含了大量针对特定 IDE、操作系统、编程语言的规则，总计约 180 行。对于本地项目来说，这些规则过于冗余。

**保留的核心规则**：
- Python 编译缓存（`__pycache__/`、`*.pyc`）
- IDE 配置（`.idea/`、`.vscode/`）
- Lightning 日志和 checkpoint（`lightning_logs/`、`checkpoints/`、`wandb/`）
- 模型权重文件（`*.pth`、`*.ckpt`、`*.pt`）
- 环境配置（`.env`、`venv/`）

**删除的规则**：移除了针对 Windows、macOS、Linux 系统文件的规则，以及各种编辑器（Sublime、Vim、Emacs 等）的临时文件规则。这些规则在 Linux 服务器环境下没有实际作用。

---

## 第五章：新增的自定义文件

### 5.1 datasets/voc_semantic.py（174 行）— 核心自定义文件

#### 为什么需要这个文件？

官方 EoMT 的数据集基类 `Dataset`（`datasets/dataset.py`）强制从 zip 压缩包中读取数据。它通过 `zipfile.ZipFile` 打开压缩包，从中提取图像和标签文件。

但我们的 VOC 2012 数据已经解压到 `/home/xia/DINO_Project/voc_data/` 目录中，目录结构如下：

```
voc_data/
├── VOCdevkit/VOC2012/
│   ├── JPEGImages/              # .jpg 图像文件
│   ├── SegmentationClass/       # .png 标签文件（像素值=类别ID 0-20, 255=忽略）
│   └── ImageSets/Segmentation/
│       ├── train.txt            # 1464 张训练图片 ID
│       └── val.txt              # 1449 张验证图片 ID
└── VOCtrainval_11-May-2012.tar  # 原始压缩包（已解压）
```

重新打包成 zip 既浪费磁盘空间又浪费时间。因此我们决定写一个独立的数据集类，直接从目录读取数据。

#### 方案设计

**继承关系**：`VOCSemantic` 继承自 `LightningDataModule`（官方基类），但不继承 `Dataset`（zip 基类）。

**内部数据集类**：`_VOCDataset` 继承自 `torch.utils.data.Dataset`，使用 `PIL.Image.open()` 直接从目录读取图像和标签。

**返回格式**：与官方一致，返回 `(img_tensor, target_dict)`，其中 `target_dict` 包含：
- `"masks"`：二值 mask 张量堆叠
- `"labels"`：类别 ID 列表
- `"is_crowd"`：是否为 crowd（VOC 语义分割中全部为 False）

#### 关键设计决策

**1. 不归一化**

`_VOCDataset.__getitem__()` 返回 raw [0, 255] uint8 图像。归一化分两步完成：

- `LightningModule.forward()` 中除以 255：`x = imgs / 255.0`
- `EoMT.forward()` 中减去 mean 除以 std：`x = (x - pixel_mean) / pixel_std`

这种设计的好处是数据加载路径保持简单，归一化逻辑集中在模型代码中。

**2. train/val 划分**

从 `ImageSets/Segmentation/{train,val}.txt` 读取图片 ID 列表：

```python
def _read_split(self, split_name):
    split_path = self.voc_root / "ImageSets" / "Segmentation" / f"{split_name}.txt"
    with open(split_path) as f:
        return [line.strip() for line in f if line.strip()]
```

VOC 2012 数据集包含 1464 张训练图片和 1449 张验证图片。

**3. target_parser**

将 VOC 标签 PNG（像素值 = 类别 ID 0-20，255 = 忽略）转换为 EoMT 所需的 `(masks_list, labels_list, is_crowd_list)` 格式：

```python
@staticmethod
def target_parser(target, **kwargs):
    masks, labels = [], []
    for label_id in target[0].unique():
        cls_id = label_id.item()
        if cls_id == 255:  # 忽略 / void
            continue
        masks.append(target[0] == label_id)
        labels.append(cls_id)
    return masks, labels, [False for _ in range(len(masks))]
```

**4. tv_tensors**

使用 `tv_tensors.Image` 和 `tv_tensors.Mask` 包装图像和 mask，兼容 torchvision v2 变换系统。这使得未来如果需要添加数据增强，可以直接使用 `datasets/transforms.py` 中的 `Transforms` 类。

**5. 空目标处理**

当标签 PNG 中所有像素都是 255（忽略区域）时，返回空的 mask/label 张量：

```python
if len(masks) > 0:
    masks_tensor = tv_tensors.Mask(torch.stack(masks))
    labels_tensor = torch.tensor(labels, dtype=torch.long)
else:
    masks_tensor = tv_tensors.Mask(torch.zeros((0, *self.img_size), dtype=torch.bool))
    labels_tensor = torch.zeros(0, dtype=torch.long)
```

### 5.2 configs/dinov3/voc/semantic/eomt_small_512.yaml（48 行）

这是本地项目新增的训练配置文件，使用 YAML 格式，由 LightningCLI 解析。

#### 配置详解

```yaml
trainer:
  max_epochs: 50
  accelerator: auto
  strategy: auto
  devices: 1
  precision: 16-mixed
  logger:
    class_path: lightning.pytorch.loggers.tensorboard.TensorBoardLogger
    init_args:
      save_dir: "lightning_logs"
      name: "voc_semantic_eomt_small_512"
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        dirpath: "checkpoints"
        filename: "eomt_voc_{epoch}"
        monitor: "metrics/val_iou_all"
        mode: "max"
        save_last: true
        save_top_k: 3
    - class_path: lightning.pytorch.callbacks.LearningRateMonitor
      init_args:
        logging_interval: "step"
```

**Trainer 配置**：
- `max_epochs: 50`：训练 50 个 epoch
- `precision: 16-mixed`：混合精度训练（FP16），减少显存占用
- `devices: 1`：单 GPU 训练（RTX 5070）
- `TensorBoardLogger`：使用 TensorBoard 记录日志，保存到 `lightning_logs/voc_semantic_eomt_small_512/`

**Callbacks**：
- `ModelCheckpoint`：监控 `metrics/val_iou_all`，保存最优的 3 个 checkpoint + 最后一个
- `LearningRateMonitor`：在 step 级别记录学习率变化

```yaml
model:
  class_path: training.mask_classification_semantic.MaskClassificationSemantic
  init_args:
    num_classes: 21
    img_size: [512, 512]
    attn_mask_annealing_enabled: True
    attn_mask_annealing_start_steps: [0, 5856, 10248, 14640]
    attn_mask_annealing_end_steps: [8784, 14640, 20496, 26352]
    network:
      class_path: models.eomt.EoMT
      init_args:
        num_q: 100
        num_blocks: 4
        masked_attn_enabled: True
        encoder:
          class_path: models.vit.ViT
          init_args:
            img_size: [512, 512]
            backbone_name: "vit_small_patch16_dinov3.lvd1689m"
```

**Model 配置**：
- `num_classes: 21`：PASCAL VOC 2012 包含 20 个物体类别 + 1 个背景类别
- `img_size: [512, 512]`：输入图像尺寸
- `num_q: 100`：语义分割使用 100 个 query tokens（全景分割使用 200 个）
- `num_blocks: 4`：在最后 4 个 ViT block 中插入 queries
- `backbone_name: "vit_small_patch16_dinov3.lvd1689m"`：DINOv3 ViT-Small 预训练权重

#### Attention Mask Annealing 步数计算

Attention mask annealing 是 EoMT 的关键训练策略。在训练初期，mask 概率为 1（所有 queries 关注所有 patches），随着训练进行，mask 概率逐渐降低到 0（queries 只关注预测的前景区域）。

计算过程：
- VOC train 数据集：1464 张图片
- `batch_size: 2`
- `steps_per_epoch: 1464 / 2 = 732`
- 50 epochs 总步数：732 x 50 = 36,600

各 block 的 annealing 区间：

| Block | Start Step | Start Epoch | End Step | End Epoch | 持续时间 |
|:-----:|:----------:|:-----------:|:--------:|:---------:|:--------:|
| 0 | 0 | 0 | 8784 | 12 | 12 epochs |
| 1 | 5856 | 8 | 14640 | 20 | 12 epochs |
| 2 | 10248 | 14 | 20496 | 28 | 14 epochs |
| 3 | 14640 | 20 | 26352 | 36 | 16 epochs |

```yaml
data:
  class_path: datasets.voc_semantic.VOCSemantic
  init_args:
    img_size: [512, 512]
    num_classes: 21
    ignore_idx: 255
```

**Data 配置**：
- `class_path: datasets.voc_semantic.VOCSemantic`：使用自定义的 VOC 数据集类
- `ignore_idx: 255`：VOC 标签中 255 表示忽略区域
- `data.path` 和 `data.batch_size` 可在命令行中覆盖

---

## 第六章：删除的旧文件

在集成官方仓库后，以下旧文件被删除：

| 文件 | 行数 | 说明 |
|:----|:----:|:-----|
| `model.py` | 251 | 旧简化版 EoMT 模型 |
| `train.py` | 191 | 旧简化版训练脚本 |
| `evaluate.py` | 101 | 旧简化版评估脚本 |
| `predict.py` | 95 | 旧简化版推理脚本 |
| `eomt_voc_weights.pth` | — | 旧权重文件（不兼容官方代码） |
| `test.jpg` | — | 旧测试图片 |

这些文件被删除的原因：

1. **功能被官方代码覆盖**：官方 `main.py` + `LightningCLI` 提供了更完善的训练/评估/推理流程
2. **架构不兼容**：旧实现缺少 masked attention、匈牙利匹配等核心机制
3. **权重不兼容**：旧权重文件的参数结构与官方模型不匹配
4. **维护成本**：保留两份实现会导致混淆和维护负担

---

## 第七章：保留的文件

以下文件在集成过程中被保留：

| 文件 | 说明 |
|:----|:-----|
| `__init__.py` | 空文件，使 `eomt/` 成为 Python 包 |
| `docs/01_论文精讲_EoMT.md` | EoMT 论文精讲（中文） |
| `docs/02_零基础学习知识文档.md` | 零基础学习知识文档（中文） |
| `docs/03_实验操作指南.md` | 实验操作指南（中文） |

这三份中文文档来自之前的开发阶段，分别面向不同读者：
- 01：面向有深度学习基础的读者，讲解 EoMT 论文的技术细节
- 02：面向初学者，介绍语义分割和 EoMT 的基础知识
- 03：面向实验人员，提供具体的操作步骤和命令

---

## 第八章：训练与评估指南

### 8.1 环境准备

确保 conda 环境 `dinoseg` 已激活，并安装了所有依赖：

```bash
conda activate dinoseg
cd ~/DINO_Project/eomt
pip install -r requirements.txt
```

### 8.2 训练命令

#### 快速测试（1 batch）

用于验证代码正确性，只运行 1 个 batch：

```bash
env -i HOME=$HOME PATH=$PATH /home/xia/miniconda3/envs/dinoseg/bin/python main.py fit \
  -c configs/dinov3/voc/semantic/eomt_small_512.yaml \
  --data.path /home/xia/DINO_Project/voc_data \
  --trainer.devices 1 --data.batch_size 2 --trainer.fast_dev_run True
```

#### 正式训练（50 epochs）

```bash
env -i HOME=$HOME PATH=$PATH /home/xia/miniconda3/envs/dinoseg/bin/python main.py fit \
  -c configs/dinov3/voc/semantic/eomt_small_512.yaml \
  --data.path /home/xia/DINO_Project/voc_data \
  --trainer.devices 1 --data.batch_size 2
```

#### 从 checkpoint 恢复训练

```bash
env -i HOME=$HOME PATH=$PATH /home/xia/miniconda3/envs/dinoseg/bin/python main.py fit \
  -c configs/dinov3/voc/semantic/eomt_small_512.yaml \
  --data.path /home/xia/DINO_Project/voc_data \
  --trainer.devices 1 --data.batch_size 2 \
  --ckpt_path checkpoints/last.ckpt
```

### 8.3 评估已训练的模型

```bash
env -i HOME=$HOME PATH=$PATH /home/xia/miniconda3/envs/dinoseg/bin/python main.py validate \
  -c configs/dinov3/voc/semantic/eomt_small_512.yaml \
  --data.path /home/xia/DINO_Project/voc_data \
  --data.batch_size 2 \
  --model.init_args.ckpt_path checkpoints/eomt_voc_epoch=X.ckpt \
  --model.network.masked_attn_enabled False
```

注意：评估时设置 `masked_attn_enabled=False`，因为推理时不需要 masked attention。

### 8.4 关于代理环境变量

系统设置了 socks5 代理，而 `httpx` 库（HuggingFace Transformers 的依赖）不支持 socks 协议。训练前需要清除代理环境变量：

```bash
unset ALL_PROXY HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
```

或者使用 `env -i HOME=$HOME PATH=$PATH` 启动干净环境（如上文的训练命令所示）。

### 8.5 训练流程说明

完整的训练流程如下：

1. **配置解析**：`main.py` 解析 YAML 配置文件和命令行参数，创建 `EoMTCLI` 实例
2. **数据准备**：`VOCSemantic.setup()` 读取 VOC train/val 划分，创建 `_VOCDataset` 实例
3. **模型初始化**：`MaskClassificationSemantic` 实例化 `EoMT(ViT(vit_small_patch16_dinov3))`，加载 DINOv3 预训练权重
4. **优化器配置**：`configure_optimizers()` 设置 AdamW 优化器 + LLRD（层间学习率衰减）+ 两阶段 warmup 多项式调度
5. **每个 training step**：
   - 前向传播：图像 -> ViT 编码 -> 插入 queries -> L2 blocks -> 预测 mask/class logits
   - 损失计算：匈牙利匹配 + BCE/Dice/CE
   - 反向传播更新参数
   - 更新 mask annealing 概率
6. **验证**：每 N 个 epoch 在验证集上计算 mIoU，保存最佳 checkpoint
7. **日志**：所有指标输出到 TensorBoard，保存在 `lightning_logs/voc_semantic_eomt_small_512/`

### 8.6 查看训练日志

```bash
tensorboard --logdir lightning_logs/voc_semantic_eomt_small_512
```

然后在浏览器中打开 `http://localhost:6006` 查看训练曲线。

TensorBoard 中可以看到以下指标：

- **损失曲线**：`losses/train_loss_total` 总损失，以及各分量损失（mask、dice、cross_entropy）
- **mIoU 曲线**：`metrics/val_iou_all` 验证集 mIoU，以及各 block 的 mIoU
- **学习率**：`lr-*` 各参数组的学习率变化
- **Attention mask 概率**：`attn_mask_prob_*` 各 block 的 mask 概率变化
- **可视化**：如果安装了 wandb，还会显示预测结果的可视化图像

### 8.7 常见问题

**Q：训练时遇到代理错误怎么办？**

A：使用 `env -i HOME=$HOME PATH=$PATH` 启动干净环境，或者在训练前执行 `unset ALL_PROXY HTTP_PROXY HTTPS_PROXY http_proxy https_proxy`。

**Q：显存不足怎么办？**

A：可以尝试以下方法：
- 降低 `batch_size`（从 2 改为 1）
- 降低 `img_size`（从 512 改为 384 或 256）
- 关闭 `precision: 16-mixed` 使用 FP32（不推荐，显存占用更大）

**Q：如何加载预训练权重？**

A：在配置文件中设置 `model.init_args.network.init_args.encoder.init_args.ckpt_path`，或者在命令行中通过 `--model.init_args.ckpt_path` 指定。

**Q：如何调整学习率？**

A：在配置文件中修改 `model.init_args.lr`（基础学习率）和 `model.init_args.llrd`（层间学习率衰减系数）。默认值分别为 `1e-4` 和 `0.8`。

---

## 附录：文件清单总表

| 类别 | 文件 | 来源 | 行数 | 说明 |
|:----|:----|:----:|:----:|:-----|
| 入口 | `main.py` | 官方 -> 修改 | 34 | 简化 CLI，TensorBoard |
| 配置 | `requirements.txt` | 官方 -> 修改 | 12 | 适配本地环境 |
| 配置 | `.gitignore` | 官方 -> 修改 | 25 | 简化 |
| 模型 | `models/eomt.py` | 官方 -> 修改 | 264 | RoPE 兼容修复 |
| 模型 | `models/vit.py` | 官方 -> 原样 | 68 | ViT 封装 |
| 模型 | `models/scale_block.py` | 官方 -> 原样 | 38 | 上采样 |
| 训练 | `training/lightning_module.py` | 官方 -> 修改 | 911 | wandb 条件导入 |
| 训练 | `training/mask_classification_loss.py` | 官方 -> 原样 | 120 | 损失函数 |
| 训练 | `training/mask_classification_semantic.py` | 官方 -> 原样 | 116 | 语义分割模块 |
| 训练 | `training/two_stage_warmup_poly_schedule.py` | 官方 -> 原样 | 49 | 学习率调度 |
| 数据 | `datasets/dataset.py` | 官方 -> 原样 | 308 | zip 基类 |
| 数据 | `datasets/lightning_data_module.py` | 官方 -> 原样 | 52 | DataModule 基类 |
| 数据 | `datasets/transforms.py` | 官方 -> 原样 | 118 | 数据增强 |
| **自定义** | **`datasets/voc_semantic.py`** | **新增** | **174** | **VOC 数据集（目录读取）** |
| **自定义** | **`configs/dinov3/voc/semantic/eomt_small_512.yaml`** | **新增** | **48** | **VOC 训练配置** |
| 文档 | `docs/01_论文精讲_EoMT.md` | 保留 | — | 论文精讲 |
| 文档 | `docs/02_零基础学习知识文档.md` | 保留 | — | 基础知识 |
| 文档 | `docs/03_实验操作指南.md` | 保留 | — | 操作指南 |
| 包 | `__init__.py` | 保留 | 0 | 空包文件 |

### 统计摘要

- **官方文件（原样保留）**：8 个，共 849 行
- **官方文件（修改）**：4 个（含 `requirements.txt` 和 `.gitignore`），共 335 行
- **新增自定义文件**：2 个，共 222 行
- **保留的旧文件**：5 个（含 3 份文档 + `__init__.py`）
- **删除的旧文件**：6 个（旧实现代码 + 旧权重 + 测试图片）
- **总计**：约 1400 行代码
