[English](./README.md) | [简体中文](./README.zh-CN.md)  
[![Project Page](https://img.shields.io/badge/Project%20Page-1f6feb?style=flat-square)](https://kudouzala.github.io/ion_detect_page/)
[![Dataset](https://img.shields.io/badge/HuggingFace%20Dataset-FFD21E?style=flat-square)](https://huggingface.co/datasets/KudouZala/PEM_electrolyzer-ion_detect)

# ion_detect

基于电压 + EIS 时间序列的 PEM 电解槽离子污染检测，支持可解释分析与物理约束训练。

![可解释结果示例](./github_png/20241006_2ppm铬离子污染测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_pred4_attribution_plot.png)

## 快速开始（推荐 `SWC-PSWM`）

以下是当前仓库中最推荐、效果较理想的一条复现路径。

### 1) 环境安装

```bash
conda create -n ion_detect python=3.10 -y
conda activate ion_detect
pip install -r requirements.txt
```

如需 GPU，请安装与你 CUDA 匹配的 PyTorch（示例为 CUDA 12.1）：

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 2) 数据准备

- 从 Hugging Face 下载数据：<https://huggingface.co/datasets/KudouZala/PEM_electrolyzer-ion_detect>
- 按项目约定目录放置到 `data/` 与 `datasets/` 下。

### 3) 训练 / 搜索 / 测试

```bash
python scripts/machine_learning_code/main.py --config scripts/machine_learning_code/configs/paper/SWC-PSWM.yaml --train
python scripts/machine_learning_code/main.py --config scripts/machine_learning_code/configs/paper/SWC-PSWM.yaml --search
python scripts/machine_learning_code/main.py --config scripts/machine_learning_code/configs/paper/SWC-PSWM.yaml --test
```

输出位置：

- 权重与训练过程：`output/trained_model_save/SWC-PSWM/`
- 推理与可解释结果：`output/inference_results/SWC-PSWM/`

## 项目结构

- `scripts/machine_learning_code/`：训练、测试、配置文件
- `datasets/`：机器学习数据集与统计信息
- `data/`：原始实验数据处理脚本
- `output/`：本地生成的训练与推理输出
- `github_png/`：README 使用的图片资源

## 关键配置

- 论文主模型：`scripts/machine_learning_code/configs/paper/SWC-PSWM.yaml`
- 论文消融：`scripts/machine_learning_code/configs/paper/exp_a.yaml` 到 `exp_d.yaml`
- 内部探索配置（非论文主线）已归档到 `scripts/machine_learning_code/configs/internal/`。

## 可选：EIS 拟合流程（AutoEIS）

如果你只做机器学习复现，可以跳过本节。

- AutoEIS 项目：<https://github.com/AUTODIAL/AutoEIS>
- 仓库中已包含 `autoeis/`，需要时安装：

```bash
cd autoeis
pip install -e .
```

## 致谢

本项目包含改编自 [AutoEIS](https://github.com/AUTODIAL/AutoEIS) 的代码，遵循 MIT 许可证。