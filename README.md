# AI Hologram Rearrangement

利用深度学习实现大规模中性原子阵列的常数时间全息图生成。

## 项目简介

本项目复现了论文 *"利用人工智能实现大规模中性原子阵列的常数时间重排"* 中的核心方法。

在中性原子量子计算中，需要使用空间光调制器 (SLM) 生成全息图来产生光镊阵列。传统方法使用 **加权 Gerchberg-Saxton (WGS)** 迭代算法计算 SLM 相位掩模，迭代过程耗时较长。本项目训练深度学习模型直接从目标振幅分布预测最优相位，实现常数时间的全息图生成。

## 项目结构

```
├── GS算法/              # 传统 WGS 算法实现（含 GPU 加速版本）
│   ├── GS_01.py         # 基础 WGS 实现
│   ├── GS_03_success.py # 验证成功的 WGS 版本
│   ├── GS_05_GPU_01.py  # CuPy GPU 加速版本
│   ├── GS_09~13_*.py    # 训练数据生成脚本
│   └── GS_14~22_*.py    # 进阶版本
├── AI/                  # 深度学习模型
│   ├── AI_01~02.py      # 数据加载与基础训练
│   ├── AI_03.py         # ResNet 残差网络模型
│   ├── AI_04_triconv.py # Tri-Conv 架构
│   ├── AI_05_multichan.py # 多通道模型
│   ├── AI_06~10.py      # U-Net 等进阶架构
│   └── results_unet*/   # 训练好的模型权重
├── Hungary/             # 匈牙利算法（原子重排匹配）
├── tests/               # 数据验证与测试脚本
├── training_result/     # 训练结果（loss 曲线、对比图）
└── gs_training_data*/   # 训练数据集（由 GS 算法生成）
```

## 原理概述

1. **问题定义**：已知 SLM 平面和目标焦平面的振幅分布，求 SLM 平面的相位分布
2. **传统方法 (WGS)**：通过傅里叶变换在两平面之间迭代投影，逐步优化相位
3. **AI 方法**：将目标振幅作为输入，训练神经网络直接预测最优相位掩模

## 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.x + CuPy（GPU 加速的 GS 算法）
- PyTorch（深度学习训练）

### 安装

```bash
pip install -r requirements.txt
```

> **注意**: `cupy-cuda11x` 需根据你的 CUDA 版本调整。如 CUDA 12.x 请改为 `cupy-cuda12x`。

### 运行步骤

1. **生成训练数据**（使用 WGS 算法）：

```bash
cd GS算法
python GS_09_traindata_03.py   # 基础训练数据
python GS_12.py                 # 多通道训练数据
```

2. **训练 AI 模型**：

```bash
cd AI
python AI_03.py                 # ResNet 模型
python AI_06.py                 # U-Net 模型
```

3. **查看结果**：训练完成后在 `training_result/` 目录查看 loss 曲线和预测对比图。

> 所有脚本需从**项目根目录**运行，或确保工作目录为项目根。

## 参考文献

利用人工智能实现大规模中性原子阵列的常数时间重排 (2024).

## License

This project is for research and educational purposes.
