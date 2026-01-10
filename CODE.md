# 项目代码结构与文件功能说明

本文档详细解释了项目中各个文件和脚本的功能及作用。

## 根目录文件 (Root Directory)

### `train.py`
**功能**：MMSAM2模型的训练脚本。
**主要作用**：
1. 解析命令行参数，设置实验名称、数据路径、预训练模型路径等。
2. 加载数据集（FullDataset_new）和模型（MMSAM2）。
3. 定义包括多尺度记忆库改进、ImageEncoder MFBFpnNeck替换、基于Mamba的解码器改进等模型训练流程。
4. 执行训练循环，计算损失，更新模型权重，并保存模型检查点。

### `dataset.py`
**功能**：定义数据加载和预处理类。
**主要作用**：
1. 提供Dataset类（如FullDataset_new, TestDataset等），用于读取图像和标签数据。
2. 实现多种数据增强变换类（ToTensor, Resize, RandomHorizontalFlip, RandomVerticalFlip等），用于在训练过程中增强数据的多样性。

### `mmsam2.py`
**功能**：定义MMSAM2模型架构。
**主要作用**：
1. 构建基于SAM2的基础模型。
2. 定义卷积模块（DoubleConv）和上采样模块（Up），用于构建改进的解码器或特征融合网络。
3. 整合编码器和解码器部分，形成完整的图像分割/预测网络。

### `_function.py`
**功能**：提供训练和评估过程中的辅助函数。
**主要作用**：
1. 初始化评估指标（init_metrics），包括F-measure, S-measure, E-measure等。
2. 提供保存检查点、调整学习率、模型验证等辅助功能（如果包含）。
3. 封装与py_sod_metrics库的交互，用于计算显著性目标检测（SOD）相关的性能指标。

---

## 脚本目录 (sh/)

### 训练脚本 (`sh/train/`)

#### `train_camouflaged.sh`
**功能**：针对Camouflaged伪装目标检测数据集启动训练。
**作用**：设置环境变量（如CUDA设备），调用train.py脚本，并传入数据集路径、验证集列表、预训练模型路径、超参数等。

#### `train_marine.sh`
**功能**：针对Marine海洋生物检测数据集启动训练。
**作用**：设置环境变量，调用train.py脚本，配置数据集路径和验证列表（MAS3K, RMAS）。

#### `train_polyp.sh`
**功能**：针对Polyp息肉检测数据集启动训练。
**作用**：设置环境变量，调用train.py脚本，配置数据集路径和多个验证集列表。

#### `train_salient.sh`
**功能**：针对Salient显著性检测数据集启动训练。
**作用**：设置环境变量，调用train.py脚本，配置数据集路径和多个验证集列表。

### 评估脚本 (`sh/eval/`)

#### `test.py`
**功能**：模型推理测试脚本。
**主要作用**：
1. 加载训练好的模型检查点。
2. 读取测试集图像。
3. 如果有Ground Truth，读取对应的掩码。
4. 运行模型推理生成预测掩码，并将结果保存到指定目录。

#### `eval.py`
**功能**：模型性能评估脚本。
**主要作用**：
1. 读取模型预测的掩码结果和真实的Ground Truth掩码。
2. 使用多种评估指标（如MAE, F-measure, S-measure, E-measure等）计算模型性能。
3. 输出评估结果，用于量化分析模型效果。

#### `camouflaged_auto.sh`
**功能**：针对Camouflaged数据集进行自动化的测试与评估流程。
**作用**：
1. 依次对多个测试子集（如CAMO, CHAMELEON等）调用test.py生成预测掩码。
2. 对生成的掩码调用eval.py进行性能评估。

#### `marine_auto.sh`
**功能**：针对Marine数据集的自动化评估脚本。
**作用**：调用test.py进行推理生成（针对RMAS等子集），并调用eval.py计算评估指标。

#### `polyp_auto.sh`
**功能**：针对Polyp数据集的自动化评估脚本。
**作用**：遍历多个子集（CVC-300, etc.），调用test.py生成预测，再调用eval.py进行指标评估。

---

## SAM2 核心模块 (sam2/)

### `sam2/build_sam.py`
**功能**：构建SAM2模型的工厂函数。
**主要作用**：
1. `build_sam2`: 读取配置文件，应用覆盖项，实例化SAM2模型，并加载检查点。
2. 负责模型的初始化和从YAML配置到PyTorch模型的转换。

### `sam2/sam2_image_predictor.py`
**功能**：SAM2图像预测器类。
**主要作用**：
1. 封装了SAM2模型，专门用于图像分割任务。
2. 处理图像的编码（embedding calculation）。
3. 提供基于提示（prompts）的掩码预测接口。

### `sam2/sam2_video_predictor.py`
**功能**：SAM2视频预测器类。
**主要作用**：
1. 处理视频序列的分割任务，继承自SAM2Base。
2. 管理视频帧的推理状态和记忆库。
3. 处理用户交互（如点击、框选）在视频帧上的传播。

### `sam2/automatic_mask_generator.py`
**功能**：自动掩码生成器。
**主要作用**：
1. 在图像上自动生成网格状的点提示。
2. 使用SAM2ImagePredictor批量预测掩码。
3. 应用非极大值抑制（NMS）等后处理步骤，过滤重复或低质量的掩码，生成整张图的分割结果。
