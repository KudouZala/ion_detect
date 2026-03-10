#!/usr/bin/env python3
"""
统计 SWC-PSWM 模型的可训练参数数量，并计算样本-参数比与正则化策略。
用于回复论文审稿意见 Comment 12。
"""
import sys
from pathlib import Path

# 确保可以 import 同目录模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml
import torch
from model_new_models import IonDetectModel


def main():
    config_path = Path(__file__).resolve().parent / "configs" / "paper" / "SWC-PSWM.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = IonDetectModel(cfg)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    n_samples = 1337
    ratio = n_samples / trainable_params if trainable_params else float("inf")

    # 从配置读取正则化相关
    tcfg = cfg.get("train", {})
    tc = tcfg.get("trainer", {})
    nm = cfg.get("new_model", {})

    print("=" * 60)
    print("SWC-PSWM 模型参数量与正则化（用于论文 Comment 12）")
    print("=" * 60)
    print(f"总参数量:           {total_params:,}")
    print(f"可训练参数量:       {trainable_params:,}")
    print(f"训练样本数:         {n_samples:,}")
    print(f"样本-参数比:        {ratio:.4f} (samples / trainable parameters)")
    print()
    print("正则化策略 (configs/paper/SWC-PSWM.yaml):")
    print(f"  - Dropout:         {nm.get('dropout', 'N/A')}")
    print(f"  - Weight decay:    {tcfg.get('weight_decay', 'N/A')}")
    print(f"  - Label smoothing: {tc.get('label_smoothing', 0)}")
    print(f"  - Gradient clip:   {tc.get('grad_clip', 'N/A')} (max norm)")
    print(f"  - Cosine LR:       {nm.get('use_cosine_lr', False)} (warmup_epochs={nm.get('warmup_epochs', 0)})")
    print("=" * 60)


if __name__ == "__main__":
    main()
