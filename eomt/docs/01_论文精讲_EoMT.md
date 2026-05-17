# 论文精讲：Your ViT is Secretly an Image Segmentation Model (EoMT)

> **CVPR 2025 Highlight** | 作者：Tommie Kerssies, Niccolò Cavagnero, Alexander Hermans, Narges Norouzi, Giuseppe Averta, Bastian Leibe, Gijs Dubbelman, Daan de Geus  
> 机构：埃因霍温理工大学、都灵理工大学、亚琛工业大学  
> 代码：https://www.tue-mps.org/eomt/（MIT 协议）

---

## 第一章：论文核心贡献

### 1.1 一句话概括

这篇论文的核心结论可以用标题直接概括：**你的 ViT 其实就是一个图像分割模型**。换句话说，你不需要在 ViT 外面挂一大堆复杂的专用模块（Adapter、Pixel Decoder、Transformer Decoder），只要把 ViT 本身稍加改造，它就能自己完成分割任务。

### 1.2 三大贡献

**贡献一：系统性地验证了"专用组件并非必要"这一假设。**

论文通过逐步消融实验，从当前最复杂的 SOTA 方案（ViT-Adapter + Mask2Former）开始，一步一步拆掉所有任务专用组件，最终得到一个几乎就是纯 ViT 的模型。结果令人惊讶：拆掉所有组件后，全景分割质量（PQ）只下降了 1.1 个点（从 57.1 降到 56.0），但推理速度提升了 **4.4 倍**。

**贡献二：提出了极简架构 EoMT（Encoder-only Mask Transformer）。**

EoMT 的核心设计极其简单：在 ViT 的 L1 个 Block 之后，把一组可学习的 Query 向量拼接到 Patch Token 序列中，让剩下的 L2 个 Block 同时处理图像特征和 Query。最后用一个极小的预测头（一个 Linear 层做分类，一个 3 层 MLP 做 Mask）输出结果。整个模型没有 Adapter、没有 Pixel Decoder、没有 Transformer Decoder，就是一个"加了几个可学习向量的 ViT"。

**贡献三：提出了 Mask Annealing 训练策略。**

Mask2Former 等模型在训练时使用 Masked Attention（让每个 Query 只关注自己预测的 Mask 区域），这能帮助收敛，但推理时计算 Mask 并施加 Attention Mask 非常慢。EoMT 的 Mask Annealing 策略在训练初期完全启用 Masked Attention，然后随着训练进度逐步退火，到训练结束时完全关闭。这样模型既享受了 Masked Attention 的收敛好处，又在推理时彻底摆脱了它的速度负担。

### 1.3 核心结论

论文的核心结论可以总结为三句话：

1. **大规模预训练是关键。** 使用 DINOv2 或 EVA-02 这类视觉基础模型（VFM）做预训练时，EoMT 和复杂方案的性能差距极小（仅 0.7-1.1 PQ）。但如果用 ImageNet-1K 这种小规模预训练，差距会扩大到 6.1 PQ。
2. **大模型是放大器。** ViT-g 级别下 EoMT 和基线差距仅 0.7 PQ，但 ViT-S 下差距达 5.8 PQ。模型越大，专用组件越多余。
3. **计算资源应该投在扩大 ViT 上，而不是增加架构复杂度。** EoMT 用 ViT-L 跑 128 FPS 还能达到 56.0 PQ，而基线用 ViT-B 跑 32 FPS 只有 54.4 PQ。更大的模型 + 更简单的架构 = 又快又好。

---

## 第二章：研究动机与问题

### 2.1 当前 SOTA 的复杂度问题

在 EoMT 之前，用 ViT 做图像分割的 SOTA 方案长什么样？以 ViT-Adapter + Mask2Former 为例，整个流程是这样的：

1. **ViT Backbone**：标准的 ViT 提取图像特征，输出单尺度的 Patch Token。
2. **ViT-Adapter**：在 ViT 旁边并联一个 CNN 分支，用可变形注意力（Deformable Attention）在 ViT 和 CNN 特征之间反复交互，生成多尺度特征 {F4, F8, F16, F32}。
3. **Pixel Decoder**：对多尺度特征进一步融合增强，输出处理后的多尺度特征。
4. **Transformer Decoder**：引入一组可学习的 Object Queries，通过 Cross-Attention 与多尺度特征交互，经过 J 层 Block 后输出每个 Query 的分类和 Mask 预测。

这个流程的问题很明显：**太复杂了**。每个组件都有自己的参数、自己的计算逻辑、自己的工程优化需求。这不仅让模型变得很重（ViT-L 级别 349M 参数、830 GFLOPs），还让推理速度上不去（29 FPS）。

更重要的是，这些组件之间互相耦合，没法单独利用 ViT 生态的优化工具（FlashAttention、torch.compile、Token Merging 等）。

### 2.2 两个核心假设

论文提出了两个核心假设，这也是整个工作的出发点：

**假设一：大规模预训练让 ViT 自己学会了密集语义特征。**

像 DINOv2 这样的视觉基础模型，使用 Masked Image Modeling（MIM）目标进行预训练。MIM 要求模型根据可见的 Patch 去预测被 Mask 掉的 Patch 的内容，这本质上就是在学习像素级别的密集语义关系。论文引用了一篇 2024 年的工作（也是同一批作者），证明 DINOv2 预训练后的 ViT 已经具备了很强的密集预测能力。因此，Adapter 这类帮助 ViT 提取多尺度特征的组件可能就不再需要了。

**假设二：大模型有足够的容量自己学会分割。**

更大的 ViT 有更多的参数、更强的表达能力。论文认为，当模型足够大时，它完全可以在自己的参数空间里"模拟"出 Adapter、Pixel Decoder、Transformer Decoder 的功能，而不需要显式地添加这些模块。

### 2.3 逐步消融的思路

为了验证这两个假设，论文设计了一个非常优雅的实验思路：**从最复杂的方案开始，一步一步拆掉组件，观察性能变化。**

具体来说，他们定义了 5 个步骤：

- **Step 0**：完整版 ViT-Adapter + Mask2Former（基线）
- **Step 1**：去掉 ViT-Adapter
- **Step 2**：去掉 Pixel Decoder
- **Step 3**：去掉多尺度特征处理
- **Step 4**：去掉 Transformer Decoder（把 Query 放进 Encoder）
- **Step 5**：去掉 Masked Attention（用 Mask Annealing 替代）

每一步都记录 PQ、GFLOPs、FPS 的变化，最终得到了 EoMT。

---

## 第三章：背景知识（技术铺垫）

### 3.1 图像分割的三种任务

在深入 EoMT 之前，有必要先搞清楚图像分割的三种主要任务：

**语义分割（Semantic Segmentation）**：给图像中的每个像素分配一个类别标签。比如"这个像素属于道路，那个像素属于天空"。同一类别的不同实例不做区分。评价指标是 mIoU（mean Intersection over Union）。

**实例分割（Instance Segmentation）**：只分割可数的物体（things），比如人、车、猫。每个实例单独一个 Mask，不同实例即使类别相同也要分开。评价指标是 AP（Average Precision）。

**全景分割（Panoptic Segmentation）**：语义分割 + 实例分割的统合。对于可数物体（things），每个实例单独分割；对于不可数物体（stuff，如天空、道路），每个类别一个 Mask。评价指标是 PQ（Panoptic Quality）。

EoMT 是一个统一架构，三种任务都能做。

### 3.2 ViT-Adapter

https://zhuanlan.zhihu.com/p/608272954

ViT-Adapter 是 Chen 等人在 2023 年提出的模块，目的是给 ViT 引入 CNN 的归纳偏置和多尺度特征提取能力。

它的工作方式是这样的：

1. 用一个小 CNN 从输入图像中提取多尺度特征 {F4, F8, F16, F32}。
2. 在 ViT 的每个 Block 中，通过**可变形注意力（Deformable Attention）** 在 ViT 的 Patch Token 和 CNN 的多尺度特征之间进行信息交互。
3. 从 ViT 中提取增强后的特征，注入回 CNN 的多尺度特征中。

最终输出一组融合了 ViT 全局语义和 CNN 局部细节的多尺度特征。

ViT-Adapter 的问题在于：它引入了大量额外的参数和计算，而且它的可变形注意力操作没法利用 FlashAttention 等高度优化的 Transformer 算子。

### 3.3 Mask2Former 框架

Mask2Former 是 Cheng 等人在 2022 年提出的统一分割框架，包含三个核心部分：

**Pixel Decoder**：接收多尺度特征，用多尺度可变形注意力层进一步融合增强，输出处理后的多尺度特征 {F4_hat, F8_hat, F16_hat, F32_hat}。

**Transformer Decoder**：包含 J 层 Block，每层 Block 的核心操作是 Masked Cross-Attention。具体来说：

1. 每个 Query 先通过一个 MLP 预测一个中间 Mask。
2. 这个 Mask 被用来约束 Cross-Attention：每个 Query 只能关注自己预测 Mask 区域内的图像特征。
3. 同时 Query 之间也做 Self-Attention，让它们互相协调各自负责的区域。

**预测头**：经过 J 层 Block 后，每个 Query 通过一个 Linear 层预测类别，通过 MLP + Dot Product 预测 Mask。

Mask2Former 的关键创新是 Masked Attention，它让每个 Query 聚焦于自己的区域，避免了 Query 之间的冲突。但这也带来了推理时的额外开销。

### 3.4 DINOv2 预训练

DINOv2 是 Meta 在 2024 年发布的视觉基础模型，使用自监督学习在大量未标注数据上训练 ViT。

它的预训练目标包括：

- **Masked Image Modeling（MIM）**：随机 Mask 掉一部分 Patch，让模型根据可见 Patch 预测被 Mask 掉 Patch 的内容。这迫使模型学习像素级别的密集语义关系。
- **自蒸馏（Self-Distillation）**：让 Student 模型的输出接近 Teacher 模型的输出，Teacher 由 Student 的指数移动平均得到。

DINOv2 预训练对分割任务特别重要，因为 MIM 目标本质上就是在训练模型做密集预测。论文的实验也证实了这一点：使用 DINOv2 预训练时，EoMT 和复杂方案的差距最小。

### 3.5 ViT 基础结构

ViT（Vision Transformer）的结构其实很简单，由 L 个相同的 Block 堆叠而成。每个 Block 包含：

1. **Layer Normalization（LayerNorm）**：对每个 Token 做归一化。
2. **Multi-Head Self-Attention（MHSA）**：Token 之间两两计算注意力，捕捉全局依赖关系。
3. **残差连接**：Add & Norm 的结构。
4. **MLP**：两层全连接网络，中间有 GELU 激活函数。

数学上可以写成：

```
Z^i = X^i + MHSA(Norm(X^i))
X^{i+1} = Z^i + MLP(Norm(Z^i))
```

ViT 的输入处理流程：

1. **Patch Embedding**：将图像切成 N 个 p×p 的 Patch，每个 Patch 线性投影到 D 维空间，得到 Patch Token。
2. **加上位置编码**：给每个 Patch Token 加上位置信息。
3. **加上 [CLS] Token**：可选，用于分类任务。
4. **经过 L 个 Block**：逐层处理。

对于 ViT-S/384，具体参数是：Patch Size = 16×16，Embed Dim = 384，Block 数 = 12。

---

## 第四章：从复杂到简单：逐步消融实验

### 4.1 消融实验概览

这是论文最精彩的部分之一。作者从 ViT-Adapter + Mask2Former 出发，分 5 步逐步拆掉所有任务专用组件，每一步都记录性能变化。

所有实验使用 DINOv2-L 预训练，输入 640×640，ViT-L 架构。

### 4.2 五步消融详解

**Step 0：基线（ViT-Adapter + Mask2Former）**

- 参数：349M
- 计算量：830 GFLOPs
- 速度：29 FPS
- 精度：57.1 PQ

这是当前 SOTA 的配置，完整包含 ViT-Adapter、Pixel Decoder、多尺度特征、Transformer Decoder、Masked Attention。

**Step 1：去掉 ViT-Adapter**

- 参数：342M（↓7M）
- 计算量：700 GFLOPs（↓130）
- 速度：36 FPS（↑7）
- 精度：56.7 PQ（↓0.4）

去掉 Adapter 后，多尺度特征怎么来？论文参考了 ViTDet 的做法：直接用转置卷积把 ViT 输出的 F16 上采样到 F4 和 F8，用普通卷积下采样到 F32。这个简单的特征金字塔替代了复杂的 ViT-Adapter。

结果：PQ 只降了 0.4，速度提升了 7 FPS。说明 Adapter 的作用确实有限。

**Step 2：去掉 Pixel Decoder**

- 参数：337M（↓5M）
- 计算量：685 GFLOPs（↓15）
- 速度：61 FPS（↑25）
- 精度：56.9 PQ（↑0.2）

去掉 Pixel Decoder 后，多尺度特征不再需要融合增强，直接送入 Transformer Decoder。有趣的是 PQ 反而涨了 0.2（可能是减少了过拟合），速度大幅提升 25 FPS。

**Step 3：去掉多尺度特征处理**

- 参数：328M（↓9M）
- 计算量：673 GFLOPs（↓12）
- 速度：64 FPS（↑3）
- 精度：56.7 PQ（↓0.2）

这一步只保留 F16 作为 Cross-Attention 的特征，F4 仅用于计算 Mask Logits 的 Dot Product。PQ 降了 0.2，速度提升 3 FPS。

**Step 4：去掉 Transformer Decoder（把 Query 放进 Encoder）**

- 参数：316M（↓12M）
- 计算量：828 GFLOPs（↑155）
- 速度：61 FPS（↓3）
- 精度：56.2 PQ（↓0.5）

这是最关键的一步。不再使用独立的 Transformer Decoder，而是把 Query 拼接到 Patch Token 后面，让 ViT 的 Encoder Block 同时处理图像特征和 Query。

注意这里计算量反而上升了（828 vs 673 GFLOPs），速度也降了（61 vs 64 FPS）。这是因为在训练时还需要在每个 L2 Block 前预测中间 Mask 来做 Masked Attention，而 Upscale 操作（把 F16 上采样到 F4 算 Mask）是计算密集的。

**Step 5：去掉 Masked Attention = EoMT**

- 参数：316M
- 计算量：669 GFLOPs（↓159）
- 速度：128 FPS（↑67）
- 精度：56.0 PQ（↓0.2）

通过 Mask Annealing 策略，推理时不再需要 Masked Attention。计算量大幅下降（不再需要反复 Upscale 算 Mask），速度直接翻倍（61 → 128 FPS），PQ 只降了 0.2。

### 4.3 关键发现

从 Step 0 到 Step 5，总的来看：

- **PQ 下降**：57.1 → 56.0，仅降 1.1
- **速度提升**：29 → 128 FPS，提升 4.4 倍
- **参数减少**：349M → 316M，减少 9.5%

有意思的是，FPS 的提升（4.4×）远大于 FLOPs 的下降（830 → 669，约 1.24×）。这是因为 EoMT 完全基于纯 ViT 架构，可以充分利用：

- **FlashAttention**：高度优化的 Attention 计算
- **torch.compile**：PyTorch 2.0 的 JIT 编译优化
- 没有自定义 CUDA Kernel 的瓶颈

而那些被去掉的组件（可变形注意力、Pixel Decoder 等）都有自己的自定义算子，没法享受这些优化。

---

## 第五章：EoMT 架构详解（核心章节）

### 5.1 整体架构

EoMT 的整体架构可以用一句话说清楚：**一个 ViT + 一组可学习的 Query + 一个极小的预测头**。

具体流程：

```
输入图像 → Patch Embedding → L1 个 Encoder Block → 拼接 Queries → L2 个 Encoder Block → Class Head + Mask Head → 输出
```

### 5.2 核心创新：把 Query 放进 Encoder

这是 EoMT 最巧妙的设计。

在传统的 Mask Transformer 中，Query 的处理是独立于图像特征提取的：

1. ViT Encoder 只负责提取图像特征。
2. Transformer Decoder 负责让 Query 通过 Cross-Attention 与图像特征交互。

这两个阶段是**串行**的，而且 Cross-Attention 的计算效率低于 Self-Attention。

EoMT 的做法完全不同：**把 Query 直接拼接到 Patch Token 序列中，让 Self-Attention 同时完成三件事**：

1. **Query-to-Query**：Query 之间互相通信，协调各自负责的区域（相当于 Decoder 中的 Self-Attention）。
2. **Query-to-Patch**：Query 从 Patch Token 中读取信息（相当于 Decoder 中的 Cross-Attention）。
3. **Patch-to-Patch**：Patch Token 之间继续做正常的 Self-Attention，保持图像特征提取能力。

这三种交互在同一个 MHSA 操作中**并行完成**，而不是像传统方案那样分步串行。这是 EoMT 高效的核心原因。

### 5.3 Query 的细节

Query 的实现非常简单：

```python
self.queries = nn.Embedding(num_queries, embed_dim)
```

就是一个可学习的 Embedding 矩阵，形状为 (num_q, embed_dim)。在 L1 个 Block 处理完之后，把 Query 拼接到 Token 序列的末尾。

Token 序列的顺序是：

```
[queries, cls_token, register_tokens, patch_tokens]
```

其中 prefix_tokens = 1（cls_token）+ num_registers（register_tokens，DINOv2 中使用的特殊 Token）。

对于 ViT-S/384，embed_dim = 384，num_queries 是一个超参数（论文中默认 100 或 200，取决于数据集）。

### 5.4 Class Head

Class Head 极其简单：

```python
self.class_head = nn.Linear(embed_dim, num_classes + 1)
```

就是一个单层线性层，输入是 Query 的最终表示，输出是每个类别的 logits。+1 表示"无对象"类（no object），用于处理那些没有匹配到任何真实物体的 Query。

### 5.5 Mask Head

Mask Head 稍微复杂一点，是一个 3 层 MLP：

```python
self.mask_head = nn.Sequential(
    nn.Linear(embed_dim, embed_dim),
    nn.GELU(),
    nn.Linear(embed_dim, embed_dim),
    nn.GELU(),
    nn.Linear(embed_dim, embed_dim),
)
```

输入是 Query 的最终表示，输出是 Mask Embedding，形状为 (batch, num_q, embed_dim)。

### 5.6 Mask 预测

Mask 的预测通过一个 Einstein Summation（einsum）操作完成：

```python
mask_logits = torch.einsum("bqc, bchw -> bqhw", mask_head(queries), upscaled_features)
```

这里 upscaled_features 是 ViT 输出的 F16 特征经过 Upscale 模块上采样到 F4（原图的 1/4 分辨率）后的结果。

### 5.7 Upscale 模块

Upscale 模块负责把 ViT 输出的 F16（16× 下采样）上采样到 F4（4× 下采样），用于计算高分辨率的 Mask Logits。

它的结构是：

```
ConvTranspose2d (2×2, stride 2) → GELU → ConvTranspose2d (2×2, stride 2) → GELU → Depthwise Conv3×3 → LayerNorm2d
```

两次转置卷积把分辨率从 1/16 提升到 1/4，Depthwise Conv3×3 做平滑，LayerNorm2d 做归一化。

### 5.8 具体配置示例：ViT-S/384

以我们项目使用的 ViT-S/384 为例：

- **Patch Size**：16×16
- **Embed Dim**：384
- **总 Block 数**：12
- **L1**：8（前 8 个 Block 只处理图像）
- **L2**：4（后 4 个 Block 同时处理图像和 Query）
- **num_queries**：100（默认）
- **输入分辨率**：512×512
- **Patch Token 数**：(512/16)² = 1024

在 L1 和 L2 的分界处，Token 序列从 1024（+ cls + registers）变成 1024 + 100 + cls + registers。

### 5.9 训练时的额外操作

训练时，在每个 L2 Block 之前，会额外执行一次 Mask Head 的前向传播，预测中间 Mask，然后用这个 Mask 来约束 Self-Attention 中的 Query-to-Patch 部分。这就是 Masked Attention。

但在推理时，这个操作被完全去掉了（通过 Mask Annealing 实现），所以推理时的 EoMT 就是一个纯 ViT 前向传播 + 最后两个小预测头。

---

## 第六章：Mask Annealing 策略

### 6.1 为什么需要 Masked Attention

在训练初期，Query 是随机初始化的，它们不知道应该关注图像的哪个区域。如果没有约束，多个 Query 可能会竞争同一个区域，导致训练不稳定、收敛缓慢。

Masked Attention 解决了这个问题：每个 Query 先预测一个粗糙的中间 Mask，然后在 Self-Attention 中，这个 Query 只能关注自己 Mask 区域内的 Patch Token。这相当于给每个 Query 划分了"势力范围"，让它们各司其职。

### 6.2 Masked Attention 如何工作

具体来说，在每个 L2 Block 中：

1. 用当前的 Query 表示，通过 Mask Head 预测一个中间 Mask（形状为 num_q × H/4 × W/4）。
2. 把这个 Mask 插值（Interpolate）到 Attention 的尺寸。
3. 对于每个 Query q，生成一个 Attention Mask：Mask 中值大于 0 的位置允许 Attention，小于等于 0 的位置被 Mask 掉。
4. 在 Self-Attention 的 Query-to-Patch 部分应用这个 Mask。

这样每个 Query 就只能看到自己感兴趣的区域了。

### 6.3 为什么推理时要去掉

推理时保留 Masked Attention 有两个问题：

1. **慢**：每个 Block 都要额外算一次 Mask Head 和 Upscale，这很耗时。
2. **无法利用优化**：Masked Attention 需要修改 Attention 的计算逻辑，没法直接用 FlashAttention 等高效实现。

### 6.4 Mask Annealing 的核心思想

Mask Annealing 的核心思想是：**训练初期完全使用 Masked Attention 帮助收敛，然后逐步退火，到训练结束时完全不用**。

退火的方式是多项式衰减：

```
P_mask = (1 - progress)^poly_power
```

其中 progress 是当前训练进度（0 到 1），poly_power 是多项式指数（论文默认 0.9）。

P_mask 表示"在当前 Block 中施加 Mask 的概率"。训练开始时 P_mask = 1.0（总是 Mask），训练结束时 P_mask = 0.0（从不 Mask）。

### 6.5 错开退火

论文还做了一个很巧妙的设计：**不同 Block 的退火进度是错开的**。

对于 L2 = 4 的情况（ViT-L），4 个 Block 各有不同的 start/end 步数：

- Block 21（最浅）：最早开始退火，最早结束
- Block 22：稍晚开始，稍晚结束
- Block 23：更晚
- Block 24（最深）：最晚开始，最晚结束

这样做的直觉是：浅层 Block 的 Query 已经学到了比较稳定的注意力模式，可以更早摆脱 Mask 的约束；深层 Block 的 Query 还在精细调整，需要更长时间的 Mask 保护。

### 6.6 _disable_attn_mask 机制

在具体实现中，每个训练步骤会为每个 Block 独立决定是否施加 Mask：

```python
if random.random() < P_mask:
    # 施加 Masked Attention
    attn_mask = compute_attention_mask(intermediate_mask)
else:
    # 不使用 Mask
    attn_mask = None
```

这个随机机制让模型在训练过程中逐渐适应"没有 Mask 也能正常工作"的状态。

### 6.7 对比实验

论文在 Table 7 中对比了四种策略：

| 训练策略 | 推理策略 | GFLOPs | FPS | PQ |
|---------|---------|--------|-----|-----|
| 全程 Mask | 全程 Mask | 828 | 61 | 56.2 |
| 全程 Mask | 去掉 Mask | 669 | 128 | 27.4（↓28.8）|
| 全程无 Mask | 无 Mask | 669 | 128 | 53.2（↓3.0）|
| Mask Annealing | 无 Mask | 669 | 128 | 56.0（↓0.2）|

关键发现：

- **训练 Mask + 推理去掉**：灾难性失败，PQ 从 56.2 暴跌到 27.4。说明模型完全依赖 Mask，去掉后直接崩溃。
- **全程无 Mask**：能工作，但 PQ 只有 53.2，比全程 Mask 低 3.0。说明 Masked Attention 确实有帮助。
- **Mask Annealing**：PQ 56.0，只比全程 Mask 低 0.2，但推理速度和全程无 Mask 一样快。这就是 Mask Annealing 的价值。

---

## 第七章：实验结果与分析

### 7.1 预训练的影响（最关键因素）

论文 Table 2 展示了不同预训练策略下 EoMT 和基线的对比（ViT-L，COCO val2017）：

| 预训练 | 基线 PQ | EoMT PQ | 差距 |
|--------|---------|---------|------|
| DINOv2 | 57.1 | 56.0 | 1.1 |
| EVA-02 | 56.7 | 55.5 | 1.2 |
| ImageNet-21K | 53.9 | 50.0 | 3.9 |
| ImageNet-1K | 50.4 | 44.3 | 6.1 |

这个表格清晰地验证了论文的核心假设：

- **大规模自监督/弱监督预训练（DINOv2、EVA-02）**：差距极小（1.1-1.2 PQ）。这些预训练教会了 ViT 密集语义特征，所以专用组件变得多余。
- **中等规模监督预训练（IN21K）**：差距扩大到 3.9 PQ。预训练质量不够，专用组件还能帮上忙。
- **小规模监督预训练（IN1K）**：差距扩大到 6.1 PQ。预训练不足时，专用组件的作用就很明显了。

这也解释了为什么之前的方案需要那么多专用组件：因为它们大多使用 IN21K 或 IN1K 预训练，预训练质量不够，必须靠额外组件来补。

### 7.2 模型大小的影响

论文 Table 3 展示了不同模型大小下的对比（DINOv2 预训练，COCO val2017）：

| 模型 | 基线 PQ | EoMT PQ | 差距 | EoMT FPS |
|------|---------|---------|------|----------|
| ViT-g | 57.7 | 57.0 | 0.7 | 55 |
| ViT-L | 57.1 | 56.0 | 1.1 | 128 |
| ViT-B | 54.4 | 50.6 | 3.8 | 261 |
| ViT-S | 50.5 | 44.7 | 5.8 | 330 |

趋势非常明显：**模型越大，差距越小**。

ViT-g 的差距只有 0.7 PQ，而 ViT-S 的差距高达 5.8 PQ。这说明大模型有足够的容量"内化"那些专用组件的功能。

同时注意 FPS 的变化：ViT-g 的 EoMT 跑 55 FPS，比 ViT-B 的基线（32 FPS）还快，但 PQ 更高（57.0 vs 54.4）。这就是"用更大的模型但更简单的架构"的优势。

### 7.3 语义分割结果

**Cityscapes 验证集**：

| 方法 | mIoU | FPS |
|------|------|-----|
| ViT-Adapter + M2F（基线） | 84.5 | 7 |
| EoMT（Ours） | 84.2 | 25 |

EoMT 的 mIoU 只比基线低 0.3，但速度快了 3.6 倍（25 vs 7 FPS）。

**ADE20K 验证集**：

| 方法 | mIoU | FPS |
|------|------|-----|
| ViT-Adapter + M2F（基线） | 58.9 | 21 |
| EoMT（Ours） | 58.4 | 92 |

EoMT 的 mIoU 只低 0.5，但速度快了 4.4 倍（92 vs 21 FPS）。

### 7.4 全景分割结果

在 COCO 全景分割上（Table 4），EoMT 和当前 SOTA 方法对比：

- EoMT ViT-L（640²）：56.0 PQ @ 128 FPS
- EoMT ViT-L（1280²）：58.3 PQ @ 30 FPS
- EoMT ViT-g（640²）：57.0 PQ @ 55 FPS
- EoMT ViT-g（1280²）：59.2 PQ @ 12 FPS

对比基线 ViT-Adapter + M2F ViT-L（640²）：57.1 PQ @ 29 FPS。

EoMT ViT-L（1280²）在 30 FPS 下达到 58.3 PQ，已经超过了基线 640² 的 57.1 PQ，而且速度相当。

### 7.5 Token Merging 兼容性

因为 EoMT 完全基于纯 ViT，它可以无缝利用 ViT 生态的优化工具。论文测试了 ALGM（一种 Token Merging 方法）：

| 方法 | 无 Token Merging | 有 Token Merging | 吞吐量提升 |
|------|-----------------|-----------------|-----------|
| ViT-Adapter + M2F | 9 img/s | 9 img/s | 0% |
| EoMT | 29 img/s | 38 img/s | 31% |

ViT-Adapter + M2F 用了 Token Merging 后 FLOPs 虽然降了，但吞吐量完全没变，因为它的瓶颈在 Adapter 和 Decoder 的自定义操作上。而 EoMT 的吞吐量直接提升了 31%。

### 7.6 OOD 泛化

论文在 BRAVO 基准上测试了分布外（Out-of-Distribution）泛化能力：

| 方法 | 骨干网络 | 预训练 | mIoU_ID | mIoU_OOD |
|------|---------|--------|---------|----------|
| M2F | Swin-L | IN21K | 83.3 | 69.4 |
| M2F | ViT-Adapter-L | DINOv2 | 84.5 | 78.0 |
| EoMT | ViT-L | DINOv2 | 84.2 | 77.2 |

DINOv2 基模型在 OOD 上的表现远超 Swin 基模型（77.2-78.0 vs 69.4），差距超过 7.8 mIoU。这是因为 DINOv2 的 MIM 预训练学到了更通用的视觉特征。

EoMT 的 OOD 表现（77.2）和 ViT-Adapter + M2F（78.0）几乎一样，说明去掉专用组件并没有损害泛化能力。

---

## 第八章：关键结论与启示

### 8.1 简单就是快

EoMT 最直接的启示是：**在深度学习架构设计中，简单不仅意味着优雅，还意味着快**。

EoMT 比 ViT-Adapter + Mask2Former 快 4.4 倍，不是因为用了什么神奇的加速技巧，而是因为去掉了所有不必要的组件。每个被去掉的组件都贡献了一部分速度提升：

- 去掉 Adapter：+7 FPS
- 去掉 Pixel Decoder：+25 FPS
- 去掉多尺度：+3 FPS
- 去掉 Masked Attention：+67 FPS

这些组件单独看都不算太慢，但堆在一起就成了巨大的速度瓶颈。

### 8.2 资源投在扩大 ViT 上

论文的核心建议是：**把计算资源花在扩大 ViT 和预训练上，而不是增加架构复杂度**。

这个建议有充分的数据支持：

- EoMT ViT-L（56.0 PQ @ 128 FPS）vs 基线 ViT-B（54.4 PQ @ 32 FPS）：EoMT 用更大的模型但更简单的架构，在 PQ 和 FPS 上都赢了。
- EoMT ViT-g（57.0 PQ @ 55 FPS）vs 基线 ViT-L（57.1 PQ @ 29 FPS）：EoMT 用 ViT-g 在接近的 PQ 下速度快了将近一倍。

### 8.3 充分利用 ViT 生态

EoMT 的纯 ViT 设计让它能无缝利用 ViT 生态的所有优化：

- **FlashAttention**：Attention 计算加速 2-4 倍。
- **torch.compile**：PyTorch 2.0 的 JIT 编译，自动融合算子。
- **Token Merging**：ALGM 让 EoMT 吞吐量提升 31%。
- **未来优化**：任何 Transformer 的改进都能直接用在 EoMT 上。

相比之下，ViT-Adapter + Mask2Former 的自定义算子（可变形注意力、Pixel Decoder 等）无法享受这些优化，成了性能瓶颈。

### 8.4 对分割领域的启示

EoMT 可能预示着分割领域的一个趋势：**未来的分割模型可能就是一个纯 ViT**。

就像在 NLP 中，BERT、GPT 等模型证明了"一个足够大的 Transformer 可以解决所有任务"，EoMT 在视觉分割领域做了类似的论证。它告诉我们：不要急着给 ViT 加各种专用模块，先试试把 ViT 本身做大、把预训练做好。

当然，这个结论目前主要适用于大规模预训练 + 大模型的场景。对于小模型（ViT-S）或弱预训练（IN1K），专用组件仍然有价值。但随着预训练技术的进步和模型规模的扩大，这个适用范围会越来越广。

---

## 第九章：与我们的项目关系

### 9.1 我们的项目概况

我们的项目是 **DINOv3（ViT-S/384）+ EoMT 在 PASCAL VOC 2012 上的语义分割**。

具体来说：

- **骨干网络**：DINOv3 ViT-S/384（timm 加载，pretrained=True）
- **分割架构**：EoMT（Encoder-only Mask Transformer）
- **数据集**：PASCAL VOC 2012（20 类 + 1 背景 = 21 类）
- **任务**：语义分割（不是全景分割）
- **共享代码**：dataset.py 和 voc_data/ 目录

### 9.2 与论文的差异

**任务差异**：论文主要做 COCO/ADE20K 的全景分割（PQ 指标），我们做 PASCAL VOC 的语义分割（mIoU 指标）。语义分割比全景分割简单一些（不需要区分实例），所以我们的实现可以更轻量。

**模型差异**：论文主要用 ViT-L/g（304M/1.16B 参数），我们用 ViT-S（22M 参数）。根据论文的结论，小模型下 EoMT 和基线的差距会更大（ViT-S 差 5.8 PQ）。但我们的优势在于 DINOv3 预训练仍然是大规模自监督的，这应该能缩小一些差距。

**预训练差异**：论文用 DINOv2，我们用 DINOv3。DINOv3 是 DINOv2 的升级版，在相同架构下有更好的特征质量。这对我们的 EoMT 实现是有利的。

### 9.3 我们的优势

尽管 ViT-S 较小，但我们有几个有利因素：

1. **DINOv3 预训练**：大规模自监督预训练，根据论文结论，这是缩小 EoMT 和基线差距的关键因素。
2. **PASCAL VOC 相对简单**：只有 20 个类别，图像场景相对单一，不像 COCO 有 133 类、ADE20K 有 150 类。
3. **语义分割而非全景分割**：不需要区分实例，Query 只需要学会覆盖不同类别区域即可。

### 9.4 实现注意事项

根据论文的发现，我们在实现时需要注意：

1. **num_queries 的选择**：论文默认 100-200，但 VOC 只有 21 类，可以适当减少。不过 Query 数量也影响模型容量，需要实验调优。
2. **Mask Annealing 的参数**：论文用 poly_power=0.9，但我们的训练 epoch 数可能不同，需要调整退火进度。
3. **L1/L2 的分割**：ViT-S 有 12 个 Block，论文建议 L2=4（即 L1=8, L2=4），这个配置可以直接沿用。
4. **输入分辨率**：论文用 640×640，我们用 512×512。更低的分辨率意味着更少的 Patch Token（1024 vs 1600），计算量更小，但可能影响小物体的分割质量。

### 9.5 共享代码

我们的 dataset.py 和 voc_data/ 目录与 PSPNet 项目共享。这意味着：

- 数据加载逻辑一致（512×512 缩放、ImageNet 标准化、ignore_index=255）
- 可以方便地对比 EoMT 和 PSPNet 在相同数据条件下的表现
- 训练脚本可以复用大部分数据加载代码

---

> **参考文献**：Kerssies, T., Cavagnero, N., Hermans, A., et al. Your ViT is Secretly an Image Segmentation Model. CVPR 2025.
