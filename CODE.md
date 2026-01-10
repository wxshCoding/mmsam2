# 项目代码详细说明文档

本文档对 **MMSAM2** 项目中的核心代码文件、类与函数进行了详细的解析，旨在帮助开发者深入理解模型架构、训练流程及评估机制。

---

## 1. 核心训练与模型文件

### `train.py`
**功能**：主训练脚本，负责整个实验的生命周期管理。

**关键模块详情**：
- **参数解析 (`argparse`)**：
    - `--exp_name`: 实验名称，决定日志和模型保存路径（如 `Polyp`, `Camouflaged`）。
    - `--data_path`: 数据集根目录路径。
    - `--valid_list`: 验证集子数据集列表（如 `CVC-300`, `CAMO` 等），用于训练过程中的在线验证。
    - `--hiera_path`: SAM2 预训练权重路径（默认 `sam2.pt`）。
    - `--batch_size`, `--lr`, `--weight_decay`: 训练超参数。
- **损失函数 (`structure_loss`)**：
    - 结合了 **加权二元交叉熵损失 (Weighted BCE Loss)** 和 **加权 IoU 损失 (Weighted IoU Loss)**。
    - 关注于结构信息的保留，特别是针对边界模糊的目标。
    - 计算公式涉及对局部像素的加权，以增强对困难样本的关注。
- **优化器与调度器**：
    - 使用 `AdamW` 优化器。对不同部分的参数应用了不同的策略（如 Backbone 可能冻结或使用更小学习率，特定层使用不同的权重衰减）。
    - 使用 `CosineAnnealingLR` 进行余弦退火学习率调整。
- **训练循环**：
    - 加载 `FullDataset_new` 数据集。
    - 每个 Epoch 进行前向传播 `pred0, pred1, pred2 = model(x, point)`，计算多级监督损失 `loss = loss0 + loss1 + loss2`。
    - 定期（`valid_interval`）在验证集上评估模型性能（mDice, mIoU）。

### `mmsam2.py`
**功能**：定义整个 MMSAM2 网络架构，是对 SAM2 的定制化改进。

**关键类详情**：
- **`DoubleConv`**：
    - 基础卷积块，包含两次 `Conv2d -> BatchNorm2d -> ReLU` 操作。用于特征提取和通道数调整。
- **`Up`**：
    - 上采样模块。使用双线性插值 (`bilinear`) 进行上采样，然后拼接跳跃连接的特征（skip connection），最后通过 `DoubleConv` 融合特征。类似于 U-Net 的解码器结构。
- **`Adapter`**：
    - 简单的适配器模块，包含两个线性层和 GELU 激活函数，用于特征空间的转换或微调。
- **`DynamicMemoryBank` (动态记忆库)**：
    - **创新点**：基于 Mamba 架构思路改进的记忆库。
    - **`update` 方法**：
        - 计算新特征与现有记忆的余弦相似度。
        - **动态更新策略**：如果新特征与现有记忆高度相似（超过阈值），则通过加权移动平均的方式融合更新该记忆，而不是简单的追加。
        - 融合权重取决于 IoU 分数和记忆的使用频率，保证记忆库的高质量和多样性。
- **`MMSAM2` 类**：
    - 整合了 SAM2 的 Image Encoder、Prompt Encoder 和 Mask Decoder。
    - 替换了原有的 FPN Neck 为 **MFB (Multi-Field Bottleneck Fusion)** 模块（在 `sam2.modeling.backbones.MFB` 中定义）。
    - 实现了双路解码器逻辑（Dual-Path Decoder），融合语义先验与高频结构信息。

### `dataset.py`
**功能**：处理医学/伪装目标数据的加载与增强。

**关键类详情**：
- **数据增强类**：
    - `Resize`: 将图像和掩码调整到统一尺寸（如 352x352）。
    - `RandomHorizontalFlip` / `RandomVerticalFlip`: 随机水平/垂直翻转，增加数据多样性。
    - `ToTensor`: 将 PIL 图像转换为 PyTorch Tensor。
    - `Normalize`: 使用 ImageNet 的均值和方差进行标准化。
- **`FullDataset_new`**：
    - 支持 `mode='train'` 和 `mode='valid'`。
    - 假设数据结构为标准的 `images/` 和 `masks/` 文件夹对应的组织形式。
    - 在训练模式下应用完整的数据增强流水线。

### `_function.py`
**功能**：辅助工具库，封装了评估指标和通用功能。

**关键函数详情**：
- **`init_metrics`**：
    - 初始化 `py_sod_metrics` 库中的多种指标处理器，包括 F-measure (FM), Weighted F-measure (WFM), S-measure (SM), E-measure (EM), MAE 等。
    - 配置了用于灰度图（`sample_gray`）和二值图（`sample_bin`）的不同评估模式。
- **`evaluate`**：
    - 计算基础的 TP, FP, TN, FN。
    - 计算 Dice 系数。
    - 提供了快速的评估逻辑，用于训练过程中的监控。

---

## 2. 脚本与自动化工具 (sh/)

### 训练脚本 (`sh/train/*.sh`)
这些脚本是启动 `train.py` 的快捷方式，预设了针对不同任务的最佳超参数：
- **`train_camouflaged.sh`**: 针对伪装目标检测（CAMO, COD10K等）。
- **`train_marine.sh`**: 针对海洋生物分割（RMAS, MAS3K）。
- **`train_polyp.sh`**: 针对息肉分割，包含多个数据集的联合验证列表。
- **`train_salient.sh`**: 针对显著性目标检测。

### 评估与测试脚本 (`sh/eval/`)
- **`test.py`**:
    - **功能**：推理生成。
    - **流程**：加载 `.pth` 权重 -> 读取测试图片 -> 模型推理 -> 保存预测的掩码图片（`.png`）到指定文件夹。
- **`eval.py`**:
    - **功能**：指标计算。
    - **流程**：读取预测文件夹和 GT 文件夹 -> 使用 `py_sod_metrics` 计算各项 SOD 指标 -> 输出最终报表。
- **自动化脚本 (`*_auto.sh`)**:
    - 串联了 `test.py` 和 `eval.py`。
    - 例如 `polyp_auto.sh` 会自动遍历 CVC-300, CVC-ClinicDB 等多个子数据集，依次完成“预测-评估”的全流程，极大简化了批量测试的工作量。

---

## 3. SAM2 底层集成

本项目基于 Meta 的 **Segment Anything 2 (SAM2)** 构建。
- **`sam2/build_sam.py`**: 使用 `hydra` 和 `omegaconf` 从 YAML 配置文件加载模型配置，是模型初始化的入口。
- **`sam2/sam2_image_predictor.py`**: 封装了图像层面的推理逻辑，处理图像编码缓存，支持基于点/框提示的交互式分割。
- **`sam2/sam2_video_predictor.py`**: 虽然本项目主要关注单帧图像分割，但该模块保留了处理时序信息和帧间记忆传播的能力，为未来扩展视频分割任务提供了基础。
- **`sam2/automatic_mask_generator.py`**: 提供全图自动分割功能，通过网格点提示（Grid Pormpts）生成所有可能的掩码，并进行后处理过滤。