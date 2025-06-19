
# YOLOv8 抽烟与表情识别模型训练指南

本文档详细介绍了如何训练本项目中使用的**抽烟检测**和**表情识别**模型。

## 1. 环境与文件结构准备

在开始训练之前，请确保您的项目目录拥有以下结构。如果目录或文件不存在，请手动创建。

```
/workspace/proj_shi/
|
├── datasets/             # 用于存放所有原始数据集的目录
│   ├── smoking_dataset/    # (训练后) 解压后的吸烟数据集应在此
│   └── emotion_dataset/    # (训练后) 解压后的表情数据集应在此
|
├── runs/                 # YOLOv8 训练结果的默认输出目录
│   └── detect/
|
└── yolov8_weights/       # 存放下载的YOLOv8预训练权重
    └── yolov8s.pt        # 示例：s模型的权重文件
```

## 2. 下载 YOLOv8 预训练权重

YOLOv8 提供了多种尺寸的预训练模型（如 n, s, m, l, x）。我们的实验主要使用了 `yolov8s.pt`。您可以通过以下两种方式获取它：

### 方式一：使用命令行 (推荐)

最简单的方式是直接在您的终端运行 `yolo` 命令。如果您本地没有对应的权重文件，YOLOv8 会**自动下载**。

例如，运行一个简单的预测命令即可触发下载：
```bash
# 确保已进入 yolov8_weights 目录，这样下载的文件会保存在这里
cd /workspace/proj_shi/yolov8_weights

# 运行一个任意的预测任务，YOLO会自动检查并下载 yolov8s.pt
yolo predict model=yolov8s.pt source='https://ultralytics.com/images/bus.jpg'
```
命令执行后，`yolov8s.pt` 文件就会出现在 `yolov8_weights` 目录下。

### 方式二：手动下载

您也可以直接从 Ultralytics 的 GitHub Release 页面下载。

1.  访问 [YOLOv8 GitHub Releases](https://github.com/ultralytics/ultralytics/releases)。
2.  找到最新版本的 Release。
3.  在 "Assets" 部分，找到 `yolov8s.pt` 并点击下载。
4.  将下载好的 `yolov8s.pt` 文件移动到项目的 `yolov8_weights/` 目录下。

## 3. 数据集准备

本项目使用了两个自定义数据集。假设您的原始数据集文件（`.zip` 格式）已经准备好，请按照以下步骤操作：

1.  将数据集压缩包上传到服务器。
2.  使用 `unzip` 命令将其解压到 `datasets/` 目录对应的子文件夹中。

   ```bash
   # 示例：解压吸烟数据集
   unzip /path/to/your/smoking_dataset.zip -d /workspace/proj_shi/datasets/smoking_dataset/

   # 示例：解压表情识别数据集
   unzip /path/to/your/emotion_dataset.zip -d /workspace/proj_shi/datasets/emotion_dataset/
   ```

3.  解压后，每个数据集中都应包含一个 `data.yaml` 文件，该文件描述了数据集的路径和类别信息。请**检查并确保 `data.yaml` 文件中的路径是正确的绝对路径**。

   **示例 `data.yaml` 内容:**
   ```yaml
   train: /workspace/proj_shi/datasets/emotion_dataset/train/images
   val: /workspace/proj_shi/datasets/emotion_dataset/valid/images

   names:
     0: angry
     1: happy
     2: sad
     3: surprised
   ```

## 4. 模型训练执行命令

当权重文件和数据集都准备就绪后，您可以执行以下命令开始训练。

### 训练吸烟检测模型

该模型我们使用 `yolov8n.pt`（一个更小的模型）进行训练，并设定了100个轮次。

```bash
yolo task=detect mode=train \
  model=/workspace/proj_shi/yolov8_weights/yolov8n.pt \
  data=/workspace/proj_shi/datasets/smoking_dataset/data.yaml \
  epochs=100 \
  imgsz=640 \
  name=smoking_detection_run
```
*   训练结果将保存在 `/workspace/proj_shi/runs/detect/smoking_detection_run/`。
*   我们的项目中实际使用的是 `train3` 文件夹，您可以将 `name` 参数修改为 `train3` 来复现。

### 训练表情识别模型

该模型我们使用了 `yolov8s.pt`，并采用了**带有早停机制**的优化策略，以防止过拟合并找到最佳模型。

```bash
yolo task=detect mode=train \
  model=/workspace/proj_shi/yolov8_weights/yolov8s.pt \
  data=/workspace/proj_shi/datasets/emotion_dataset/data.yaml \
  epochs=200 \
  imgsz=640 \
  patience=50 \
  name=emotion_recognition_optimized
```
*   `epochs=200`: 设置一个较高的训练上限。
*   `patience=50`: **核心优化**。如果在连续50轮内，验证集性能没有提升，训练将自动停止。
*   训练结果将保存在 `/workspace/proj_shi/runs/detect/emotion_recognition_optimized/`。
*   我们的项目中实际使用的是 `train2` 文件夹，您可以将 `name` 参数修改为 `train2` 来复现。



## 5. 课内功能实现和图像风格迁移项目

### 4.1. 下载项目代码

本项目直接使用开源实现。首先，需要将该项目克隆到您的 `final_project` 目录下。
> **来源**: [gordicaleksa/pytorch-neural-style-transfer on GitHub](https://github.com/gordicaleksa/pytorch-neural-style-transfer)

```bash
# 进入总项目目录
cd /workspace/proj_shi/final_project/
# 克隆仓库
git clone https://github.com/gordicaleksa/pytorch-neural-style-transfer.git
```

