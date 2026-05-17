# 官方 EoMT 仓库集成说明

## 一、项目架构

```
DINO_Project/
├── dataset.py                     # 旧 PSPNet 用的 dataset（已废弃）
├── voc_data/                      # VOC 2012 数据集（共享）
│   └── VOCdevkit/VOC2012/
│       ├── JPEGImages/            # 1464+1449 张 .jpg
│       ├── SegmentationClass/     # .png 标签
│       └── ImageSets/Segmentation/{train,val}.txt
├── pspnet/                        # PSPNet 实验（不动）
└── eomt/                          # EoMT 实验（本目录）
    ├── main.py                    # 入口（LightningCLI）
    ├── requirements.txt
    ├── .gitignore
    │
    ├── models/                    # 模型层
    │   ├── eomt.py                # 核心：EoMT 类（queries + mask/class head + masked attention）
    │   ├── vit.py                 # ViT 编码器封装（timm / HuggingFace）
    │   └── scale_block.py         # 上采样 Block（ConvTranspose2d ×2）
    │
    ├── training/                  # 训练层
    │   ├── lightning_module.py    # 基类（AdamW+LLRD、mask annealing、checkpoint、指标）
    │   ├── mask_classification_semantic.py  # 语义分割 LightningModule
    │   ├── mask_classification_loss.py      # Mask2Former 损失（匈牙利匹配+BCE+Dice+CE）
    │   └── two_stage_warmup_poly_schedule.py # 两阶段 warmup + 多项式 LR 衰减
    │
    ├── datasets/                  # 数据层
    │   ├── dataset.py             # 官方 zip-based Dataset 基类（未使用）
    │   ├── lightning_data_module.py  # LightningDataModule 基类（提供 collate）
    │   ├── transforms.py          # 数据增强（ColorJitter, Flip, ScaleJitter, Crop）
    │   └── voc_semantic.py        # ⭐ VOC 数据集（从目录读取，不做归一化）
    │
    ├── configs/dinov3/voc/semantic/
    │   └── eomt_small_512.yaml    # ⭐ 训练配置
    │
    └── docs/                      # 文档（不变）
```

### 数据流

```
VOCSemantic.setup()
  → _VOCDataset.__getitem__()
    → PIL.Image.open() 读取原图 (uint8 [0,255]) + 标签 PNG
    → tv_tensors 包装，无归一化
    → target_parser() 转换标签 → {masks, labels, is_crowd}
    → 返回 (img, target_dict)

LightningModule.forward(imgs)
  → imgs / 255.0                           # 第一步归一化
  → EoMT.forward(x)
    → (x - pixel_mean) / pixel_std         # 第二步归一化
    → patch_embed → _pos_embed → L1 blocks
    → 拼接 100 个 queries → L2 blocks
    → _predict() → mask_logits + class_logits

MaskClassificationLoss(mask_logits, class_logits, targets)
  → HungarianMatcher 匹配 queries ↔ GT masks
  → BCE mask loss × 5.0
  → Dice loss × 5.0
  → CE class loss × 2.0
```

---

## 二、官方仓库工作流

官方 EoMT（[github.com/tue-mps/eomt](https://github.com/tue-mps/eomt)）是 PyTorch Lightning 项目，核心流程：

```
python main.py fit -c configs/xxx.yaml --data.path /path/to/data

1. LightningCLI 解析 YAML 配置 + 命令行参数
2. 实例化 LightningDataModule（数据集类）
3. 实例化 LightningModule（MaskClassificationSemantic）
4. Trainer.fit() 启动训练
   - training_step: 前向 → 匈牙利匹配 → 损失计算 → 反向传播
   - on_train_batch_end: 更新 mask annealing 概率
   - validation_step: windowed inference → mIoU 计算
5. ModelCheckpoint 自动保存（monitor=val_iou_all）
```

### 数据集约定

官方所有数据集从 **zip 文件** 读取，结构例如：

```
--data.path=/data/
  ├── ADEChallengeData2016.zip
  │   └── ADEChallengeData2016/images/training/  # 原图
  │   └── ADEChallengeData2016/annotations/training/  # 标签
  └── VOCtrainval_11-May-2012.zip    # ← 官方无 VOC 支持
```

每个数据集类继承 `LightningDataModule`，在其 `setup()` 中创建官方 `Dataset` 实例（zip-based），并实现 `target_parser()` 静态方法。

### 官方支持的数据集

| 数据集 | 任务 | 配置路径 |
|--------|------|----------|
| COCO | panoptic / instance | `configs/dinov3/coco/` |
| ADE20K | panoptic / semantic | `configs/dinov3/ade20k/` |
| Cityscapes | semantic | `configs/dinov2/cityscapes/` |

**VOC 不在官方支持列表中。**

---

## 三、我们的改动

### 3.1 文件来源

| 来源 | 文件 | 说明 |
|------|------|------|
| **官方原样** | `models/vit.py`, `models/scale_block.py` | 无修改 |
| **官方原样** | `training/mask_classification_loss.py`, `training/mask_classification_semantic.py`, `training/two_stage_warmup_poly_schedule.py` | 无修改 |
| **官方原样** | `datasets/dataset.py`, `datasets/lightning_data_module.py`, `datasets/transforms.py` | 无修改 |
| **官方→修改** | `models/eomt.py` | RoPE 兼容（DINOv3 EvaAttention） |
| **官方→修改** | `training/lightning_module.py` | wandb→条件导入；plot_semantic 支持 TensorBoard；on_save_checkpoint 清空超参 |
| **官方→修改** | `main.py` | 简化 CLI，link_arguments 进子类 |
| **官方→修改** | `requirements.txt` | 去掉 wandb/gitignore_parser，加 tensorboard |
| **新增** | `datasets/voc_semantic.py` | VOC 数据集（目录读取） |
| **新增** | `configs/dinov3/voc/semantic/eomt_small_512.yaml` | VOC 训练配置 |

### 3.2 关键修改详解

#### (1) `models/eomt.py` — DINOv3 兼容

| 问题 | 修改 |
|------|------|
| `_pos_embed()` 返回 `(x, rope)` 元组 | 解包处理 |
| `EvaAttention` 不支持 `head_dim` 属性 | 从 `qkv.weight.shape[0]` 动态计算 |
| RoPE 与插入的 queries 冲突 | 手动拆分 queries 和 patches，仅对 patches 应用 RoPE |

#### (2) `training/lightning_module.py` — 日志和超参

| 问题 | 修改 |
|------|------|
| 硬依赖 wandb | 改为 `try: import wandb except: wandb=None` |
| `plot_semantic()` 只用 wandb API | 加 `SummaryWriter.add_image()` 分支 |
| checkpoint 超参导致恢复失败 | `on_save_checkpoint` 清空超参字典 |

#### (3) `main.py` — CLI 简化

- 移除官方 183 行中的 wandb 日志、torch.compile、复杂验证调度
- 改为 34 行的 `EoMTCLI(LightningCLI)` 子类，`link_arguments` 在 `add_arguments_to_parser` 中

#### (4) `datasets/voc_semantic.py` — 核心新增

官方 `Dataset` 强制从 zip 读取，但我们的 VOC 数据已解压。方案：写独立 `LightningDataModule`，内部 `_VOCDataset` 直接 `PIL.Image.open()` 读目录。

关键设计：
- 返回 **raw uint8 [0,255]**，模型内两步归一化（/255 → mean/std）
- 从 `ImageSets/Segmentation/{train,val}.txt` 读取划分
- `target_parser` 将标签 PNG（像素值=类别ID，255=忽略）转为 `{masks, labels, is_crowd}`

#### (5) `configs/dinov3/voc/semantic/eomt_small_512.yaml`

| 参数 | 值 | 说明 |
|------|-----|------|
| `backbone_name` | `vit_small_patch16_dinov3.lvd1689m` | DINOv3 ViT-S/16 |
| `num_classes` | 21 | VOC 20+背景 |
| `num_q` | 100 | 可学习 queries |
| `num_blocks` | 4 | L2 blocks |
| `max_epochs` | 50 | |
| `lr` | 1e-4 | 默认值 |
| `llrd` | 0.8 | backbone 逐层衰减 |
| `annealing_steps` | 0→8784, …, 14640→26352 | 4 blocks 错开 |
| `logger` | TensorBoard | 替代 wandb |

---

## 四、改动后的工作流

### 训练

```bash
cd ~/DINO_Project/eomt
python main.py fit -c configs/dinov3/voc/semantic/eomt_small_512.yaml \
  --data.path /home/xia/DINO_Project/voc_data \
  --trainer.devices 1 --data.batch_size 2
```

内部流程：

```
main.py → EoMTCLI 解析 YAML
  → VOCSemantic.setup() 读取 train.txt/val.txt
  → MaskClassificationSemantic(EoMT(ViT(dinov3_vits16)))
  → Trainer.fit()
    ├─ training_step: 前向 → 匈牙利匹配 → 损失 → 反向
    ├─ on_train_batch_end: 更新 attn_mask_probs（annealing）
    ├─ validation_step: windowed inference → mIoU
    └─ ModelCheckpoint: 保存最佳 3 个 + last.ckpt
```

### 恢复训练

```bash
python main.py fit -c configs/dinov3/voc/semantic/eomt_small_512.yaml \
  --data.path /home/xia/DINO_Project/voc_data \
  --trainer.devices 1 --data.batch_size 2 \
  --ckpt_path checkpoints/last.ckpt
```

**注意**：`on_save_checkpoint` 已清空超参，恢复不会因 LightningCLI 参数冲突失败。

### 评估

```bash
python main.py validate -c configs/dinov3/voc/semantic/eomt_small_512.yaml \
  --data.path /home/xia/DINO_Project/voc_data \
  --data.batch_size 2 \
  --ckpt_path checkpoints/eomt_voc_epoch=X.ckpt \
  --model.network.masked_attn_enabled False
```

### 与官方工作流的主要差异

| 方面 | 官方 | 我们 |
|------|------|------|
| 数据集 | zip 文件读取 | **目录读取**（voc_semantic.py） |
| 日志 | wandb | **TensorBoard** |
| CLI | 183 行，复杂配置 | **34 行**，仅核心链接 |
| 数据集 | COCO/ADE20K/Cityscapes | **PASCAL VOC 2012** |
| backbone | DINOv2/DINOv3 L/g | **DINOv3 ViT-S/16** |
| compile | torch.compile 启用 | 禁用 |
| 超参恢复 | 原生支持 | **on_save_checkpoint 清理** |
