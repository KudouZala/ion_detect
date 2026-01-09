# -*- coding: utf-8 -*-
"""
从一段日志文本中解析 predict / truth 并生成“行=真实，列=预测”的概率混淆矩阵。
固定类别顺序（横/纵坐标）为：钙离子, 钠离子, 镍离子, 铬离子, 铁离子, 铜离子
用法：把你的日志粘贴到 log_text 变量中，运行脚本。
依赖：python3, pandas, numpy, matplotlib, seaborn
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from io import StringIO

# ====== 在这里把你的整段日志粘贴为字符串（保持三引号） ======
log_text = r"""
[189/200] 🔍 评测 checkpoint: trained_model_epoch_1890.pth  (epoch=1890)
→ volt requires_grad: True
→ impe requires_grad: True
✅ Structured parameter table saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240827_10ppm铬离子污染和恢复测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_phys_params_structured.csv
✅ Attention heatmap (CLS) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240827_10ppm铬离子污染和恢复测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_attn_pred3.csv
✅ Attention heatmap (PARAM) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240827_10ppm铬离子污染和恢复测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_param_attn_pred3.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240827_10ppm铬离子污染和恢复测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_saliency_pred3.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240827_10ppm铬离子污染和恢复测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_ig_pred3.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240827_10ppm铬离子污染和恢复测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_saliency_time_aggregates.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240827_10ppm铬离子污染和恢复测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_ig_time_aggregates.csv
correct: True
predict: 铬离子
truth: 铬离子
→ volt requires_grad: True
→ impe requires_grad: True
✅ Structured parameter table saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_phys_params_structured.csv
✅ Attention heatmap (CLS) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_attn_pred0.csv
✅ Attention heatmap (PARAM) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_param_attn_pred0.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_saliency_pred0.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_ig_pred0.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_saliency_time_aggregates.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_ig_time_aggregates.csv
correct: False
predict: 钙离子
truth: 铜离子
→ volt requires_grad: True
→ impe requires_grad: True
✅ Structured parameter table saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240823_10ppm钙离子污染和恢复测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_phys_params_structured.csv
✅ Attention heatmap (CLS) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240823_10ppm钙离子污染和恢复测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_attn_pred0.csv
✅ Attention heatmap (PARAM) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240823_10ppm钙离子污染和恢复测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_param_attn_pred0.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240823_10ppm钙离子污染和恢复测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_saliency_pred0.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240823_10ppm钙离子污染和恢复测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_ig_pred0.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240823_10ppm钙离子污染和恢复测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_saliency_time_aggregates.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240823_10ppm钙离子污染和恢复测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_ig_time_aggregates.csv
correct: True
predict: 钙离子
truth: 钙离子
→ volt requires_grad: True
→ impe requires_grad: True
✅ Structured parameter table saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240907_10ppm铁离子污染测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_phys_params_structured.csv
✅ Attention heatmap (CLS) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240907_10ppm铁离子污染测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_attn_pred5.csv
✅ Attention heatmap (PARAM) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240907_10ppm铁离子污染测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_param_attn_pred5.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240907_10ppm铁离子污染测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_saliency_pred5.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240907_10ppm铁离子污染测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_ig_pred5.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240907_10ppm铁离子污染测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_saliency_time_aggregates.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240907_10ppm铁离子污染测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_ig_time_aggregates.csv
correct: True
predict: 铁离子
truth: 铁离子
→ volt requires_grad: True
→ impe requires_grad: True
✅ Structured parameter table saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240831_10ppm镍离子污染测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_phys_params_structured.csv
✅ Attention heatmap (CLS) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240831_10ppm镍离子污染测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_attn_pred2.csv
✅ Attention heatmap (PARAM) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240831_10ppm镍离子污染测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_param_attn_pred2.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240831_10ppm镍离子污染测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_saliency_pred2.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240831_10ppm镍离子污染测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_ig_pred2.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240831_10ppm镍离子污染测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_saliency_time_aggregates.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240831_10ppm镍离子污染测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_ig_time_aggregates.csv
correct: True
predict: 镍离子
truth: 镍离子
→ volt requires_grad: True
→ impe requires_grad: True
✅ Structured parameter table saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240823_10ppm钙离子污染和恢复测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_phys_params_structured.csv
✅ Attention heatmap (CLS) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240823_10ppm钙离子污染和恢复测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_attn_pred0.csv
✅ Attention heatmap (PARAM) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240823_10ppm钙离子污染和恢复测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_param_attn_pred0.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240823_10ppm钙离子污染和恢复测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_saliency_pred0.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240823_10ppm钙离子污染和恢复测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_ig_pred0.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240823_10ppm钙离子污染和恢复测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_saliency_time_aggregates.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240823_10ppm钙离子污染和恢复测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_ig_time_aggregates.csv
correct: True
predict: 钙离子
truth: 钙离子
→ volt requires_grad: True
→ impe requires_grad: True
✅ Structured parameter table saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_phys_params_structured.csv
✅ Attention heatmap (CLS) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_attn_pred4.csv
✅ Attention heatmap (PARAM) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_param_attn_pred4.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_saliency_pred4.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_ig_pred4.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_saliency_time_aggregates.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_ig_time_aggregates.csv
correct: True
predict: 铜离子
truth: 铜离子
→ volt requires_grad: True
→ impe requires_grad: True
✅ Structured parameter table saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240827_10ppm铬离子污染和恢复测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_phys_params_structured.csv
✅ Attention heatmap (CLS) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240827_10ppm铬离子污染和恢复测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_attn_pred3.csv
✅ Attention heatmap (PARAM) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240827_10ppm铬离子污染和恢复测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_param_attn_pred3.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240827_10ppm铬离子污染和恢复测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_saliency_pred3.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240827_10ppm铬离子污染和恢复测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_ig_pred3.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240827_10ppm铬离子污染和恢复测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_saliency_time_aggregates.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240827_10ppm铬离子污染和恢复测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_ig_time_aggregates.csv
correct: True
predict: 铬离子
truth: 铬离子
→ volt requires_grad: True
→ impe requires_grad: True
✅ Structured parameter table saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240907_10ppm铁离子污染测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_phys_params_structured.csv
✅ Attention heatmap (CLS) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240907_10ppm铁离子污染测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_attn_pred5.csv
✅ Attention heatmap (PARAM) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240907_10ppm铁离子污染测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_param_attn_pred5.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240907_10ppm铁离子污染测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_saliency_pred5.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240907_10ppm铁离子污染测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_ig_pred5.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240907_10ppm铁离子污染测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_saliency_time_aggregates.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240907_10ppm铁离子污染测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_ig_time_aggregates.csv
correct: True
predict: 铁离子
truth: 铁离子
→ volt requires_grad: True
→ impe requires_grad: True
✅ Structured parameter table saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_column_gamry_[12, 14, 16, 18]_phys_params_structured.csv
✅ Attention heatmap (CLS) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_column_gamry_[12, 14, 16, 18]_attn_pred6.csv
✅ Attention heatmap (PARAM) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_column_gamry_[12, 14, 16, 18]_param_attn_pred6.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_column_gamry_[12, 14, 16, 18]_saliency_pred6.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_column_gamry_[12, 14, 16, 18]_ig_pred6.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_column_gamry_[12, 14, 16, 18]_saliency_time_aggregates.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_column_gamry_[12, 14, 16, 18]_ig_time_aggregates.csv
correct: True
predict: 无污染
truth: 无污染
→ volt requires_grad: True
→ impe requires_grad: True
✅ Structured parameter table saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240831_10ppm镍离子污染测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_phys_params_structured.csv
✅ Attention heatmap (CLS) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240831_10ppm镍离子污染测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_attn_pred2.csv
✅ Attention heatmap (PARAM) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240831_10ppm镍离子污染测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_param_attn_pred2.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240831_10ppm镍离子污染测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_saliency_pred2.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240831_10ppm镍离子污染测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_ig_pred2.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240831_10ppm镍离子污染测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_saliency_time_aggregates.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240831_10ppm镍离子污染测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_ig_time_aggregates.csv
correct: True
predict: 镍离子
truth: 镍离子
→ volt requires_grad: True
→ impe requires_grad: True
✅ Structured parameter table saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20241010_2ppm钠离子污染测试_新版电解槽_ion_firecloud_[0, 2, 4, 6]_phys_params_structured.csv
✅ Attention heatmap (CLS) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20241010_2ppm钠离子污染测试_新版电解槽_ion_firecloud_[0, 2, 4, 6]_attn_pred1.csv
✅ Attention heatmap (PARAM) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20241010_2ppm钠离子污染测试_新版电解槽_ion_firecloud_[0, 2, 4, 6]_param_attn_pred1.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20241010_2ppm钠离子污染测试_新版电解槽_ion_firecloud_[0, 2, 4, 6]_saliency_pred1.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20241010_2ppm钠离子污染测试_新版电解槽_ion_firecloud_[0, 2, 4, 6]_ig_pred1.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20241010_2ppm钠离子污染测试_新版电解槽_ion_firecloud_[0, 2, 4, 6]_saliency_time_aggregates.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20241010_2ppm钠离子污染测试_新版电解槽_ion_firecloud_[0, 2, 4, 6]_ig_time_aggregates.csv
correct: True
predict: 钠离子
truth: 钠离子
→ volt requires_grad: True
→ impe requires_grad: True
✅ Structured parameter table saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240827_10ppm铬离子污染和恢复测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_phys_params_structured.csv
✅ Attention heatmap (CLS) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240827_10ppm铬离子污染和恢复测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_attn_pred3.csv
✅ Attention heatmap (PARAM) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240827_10ppm铬离子污染和恢复测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_param_attn_pred3.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240827_10ppm铬离子污染和恢复测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_saliency_pred3.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240827_10ppm铬离子污染和恢复测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_ig_pred3.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240827_10ppm铬离子污染和恢复测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_saliency_time_aggregates.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240827_10ppm铬离子污染和恢复测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_ig_time_aggregates.csv
correct: True
predict: 铬离子
truth: 铬离子
→ volt requires_grad: True
→ impe requires_grad: True
✅ Structured parameter table saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240823_10ppm钙离子污染和恢复测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_phys_params_structured.csv
✅ Attention heatmap (CLS) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240823_10ppm钙离子污染和恢复测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_attn_pred2.csv
✅ Attention heatmap (PARAM) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240823_10ppm钙离子污染和恢复测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_param_attn_pred2.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240823_10ppm钙离子污染和恢复测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_saliency_pred2.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240823_10ppm钙离子污染和恢复测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_ig_pred2.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240823_10ppm钙离子污染和恢复测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_saliency_time_aggregates.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240823_10ppm钙离子污染和恢复测试_新版电解槽_ion_gamry_[10, 12, 14, 16]_ig_time_aggregates.csv
correct: False
predict: 镍离子
truth: 钙离子
→ volt requires_grad: True
→ impe requires_grad: True
✅ Structured parameter table saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240907_10ppm铁离子污染测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_phys_params_structured.csv
✅ Attention heatmap (CLS) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240907_10ppm铁离子污染测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_attn_pred5.csv
✅ Attention heatmap (PARAM) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240907_10ppm铁离子污染测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_param_attn_pred5.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240907_10ppm铁离子污染测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_saliency_pred5.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240907_10ppm铁离子污染测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_ig_pred5.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240907_10ppm铁离子污染测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_saliency_time_aggregates.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240907_10ppm铁离子污染测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_ig_time_aggregates.csv
correct: True
predict: 铁离子
truth: 铁离子
→ volt requires_grad: True
→ impe requires_grad: True
✅ Structured parameter table saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20241010_2ppm钠离子污染测试_新版电解槽_ion_firecloud_[10, 12, 14, 16]_phys_params_structured.csv
✅ Attention heatmap (CLS) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20241010_2ppm钠离子污染测试_新版电解槽_ion_firecloud_[10, 12, 14, 16]_attn_pred1.csv
✅ Attention heatmap (PARAM) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20241010_2ppm钠离子污染测试_新版电解槽_ion_firecloud_[10, 12, 14, 16]_param_attn_pred1.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20241010_2ppm钠离子污染测试_新版电解槽_ion_firecloud_[10, 12, 14, 16]_saliency_pred1.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20241010_2ppm钠离子污染测试_新版电解槽_ion_firecloud_[10, 12, 14, 16]_ig_pred1.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20241010_2ppm钠离子污染测试_新版电解槽_ion_firecloud_[10, 12, 14, 16]_saliency_time_aggregates.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20241010_2ppm钠离子污染测试_新版电解槽_ion_firecloud_[10, 12, 14, 16]_ig_time_aggregates.csv
correct: True
predict: 钠离子
truth: 钠离子
→ volt requires_grad: True
→ impe requires_grad: True
✅ Structured parameter table saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_column_gamry_[10, 12, 14, 16]_phys_params_structured.csv
✅ Attention heatmap (CLS) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_column_gamry_[10, 12, 14, 16]_attn_pred6.csv
✅ Attention heatmap (PARAM) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_column_gamry_[10, 12, 14, 16]_param_attn_pred6.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_column_gamry_[10, 12, 14, 16]_saliency_pred6.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_column_gamry_[10, 12, 14, 16]_ig_pred6.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_column_gamry_[10, 12, 14, 16]_saliency_time_aggregates.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_column_gamry_[10, 12, 14, 16]_ig_time_aggregates.csv
correct: True
predict: 无污染
truth: 无污染
→ volt requires_grad: True
→ impe requires_grad: True
✅ Structured parameter table saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_phys_params_structured.csv
✅ Attention heatmap (CLS) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_attn_pred4.csv
✅ Attention heatmap (PARAM) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_param_attn_pred4.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_saliency_pred4.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_ig_pred4.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_saliency_time_aggregates.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_ig_time_aggregates.csv
correct: True
predict: 铜离子
truth: 铜离子
→ volt requires_grad: True
→ impe requires_grad: True
✅ Structured parameter table saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20241010_2ppm钠离子污染测试_新版电解槽_ion_firecloud_[12, 14, 16, 18]_phys_params_structured.csv
✅ Attention heatmap (CLS) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20241010_2ppm钠离子污染测试_新版电解槽_ion_firecloud_[12, 14, 16, 18]_attn_pred1.csv
✅ Attention heatmap (PARAM) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20241010_2ppm钠离子污染测试_新版电解槽_ion_firecloud_[12, 14, 16, 18]_param_attn_pred1.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20241010_2ppm钠离子污染测试_新版电解槽_ion_firecloud_[12, 14, 16, 18]_saliency_pred1.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20241010_2ppm钠离子污染测试_新版电解槽_ion_firecloud_[12, 14, 16, 18]_ig_pred1.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20241010_2ppm钠离子污染测试_新版电解槽_ion_firecloud_[12, 14, 16, 18]_saliency_time_aggregates.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20241010_2ppm钠离子污染测试_新版电解槽_ion_firecloud_[12, 14, 16, 18]_ig_time_aggregates.csv
correct: True
predict: 钠离子
truth: 钠离子
→ volt requires_grad: True
→ impe requires_grad: True
✅ Structured parameter table saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_column_gamry_[0, 2, 4, 6]_phys_params_structured.csv
✅ Attention heatmap (CLS) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_column_gamry_[0, 2, 4, 6]_attn_pred6.csv
✅ Attention heatmap (PARAM) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_column_gamry_[0, 2, 4, 6]_param_attn_pred6.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_column_gamry_[0, 2, 4, 6]_saliency_pred6.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_column_gamry_[0, 2, 4, 6]_ig_pred6.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_column_gamry_[0, 2, 4, 6]_saliency_time_aggregates.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240822_10ppm铜离子污染测试_新版电解槽_ion_column_gamry_[0, 2, 4, 6]_ig_time_aggregates.csv
correct: True
predict: 无污染
truth: 无污染
→ volt requires_grad: True
→ impe requires_grad: True
✅ Structured parameter table saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240831_10ppm镍离子污染测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_phys_params_structured.csv
✅ Attention heatmap (CLS) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240831_10ppm镍离子污染测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_attn_pred0.csv
✅ Attention heatmap (PARAM) saved: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240831_10ppm镍离子污染测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_param_attn_pred0.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240831_10ppm镍离子污染测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_saliency_pred0.csv
✅ 保存到 /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240831_10ppm镍离子污染测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_ig_pred0.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240831_10ppm镍离子污染测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_saliency_time_aggregates.csv
✅ 时间聚合归因保存: /home/cagalii/Application/ion_detect/output/inference_results/20251213b/20240831_10ppm镍离子污染测试_新版电解槽_ion_gamry_[12, 14, 16, 18]_ig_time_aggregates.csv
correct: False
predict: 钙离子
truth: 镍离子
🎯 epoch=1890 | 正确 18/21 | 准确率=85.71%
"""
# ==================================================================
# --------------------------------------------
# 0. Chinese → English mapping table
# --------------------------------------------
ch_to_en = {
    "钙离子": "Ca2+",
    "钠离子": "Na+",
    "镍离子": "Ni2+",
    "铬离子": "Cr3+",
    "铜离子": "Cu2+",
    "铁离子": "Fe3+",
    "无污染": "no_ion",
    # 兼容日志中可能出现的英文（若日志有英文则直接映射）
    "Ca2+": "Ca2+",
    "Na+": "Na+",
    "Ni2+": "Ni2+",
    "Cr3+": "Cr3+",
    "Cu2+": "Cu2+",
    "Fe3+": "Fe3+",
    "no_ion": "no_ion",
}

# The final 7-class label order used in confusion matrix
labels = ["Ca2+", "Na+", "Ni2+", "Cr3+", "Cu2+", "Fe3+", "no_ion"]


# --------------------------------------------
# 1. Extract predictions and ground truths
# --------------------------------------------
predicts_raw = re.findall(r'predict:\s*([^\s\r\n]+)', log_text)
truths_raw   = re.findall(r'truth:\s*([^\s\r\n]+)', log_text)

if len(truths_raw) != len(predicts_raw):
    n = min(len(truths_raw), len(predicts_raw))
    truths_raw = truths_raw[:n]
    predicts_raw = predicts_raw[:n]
    print(f"Warning: numbers of predict/truth not equal. Truncated to {n} pairs.", file=sys.stderr)


# --------------------------------------------
# 2. Convert Chinese labels → English labels
# --------------------------------------------
truths = []
predicts = []

for t in truths_raw:
    if t in ch_to_en:
        truths.append(ch_to_en[t])
    else:
        print(f"Warning: unknown truth label: {t}, skipped", file=sys.stderr)

for p in predicts_raw:
    if p in ch_to_en:
        predicts.append(ch_to_en[p])
    else:
        print(f"Warning: unknown predict label: {p}, skipped", file=sys.stderr)


# --------------------------------------------
# 3. Build index map using English labels
# --------------------------------------------
label_to_idx = {lab: i for i, lab in enumerate(labels)}
n_classes = len(labels)


# --------------------------------------------
# 4. Count confusion matrix
# --------------------------------------------
counts = np.zeros((n_classes, n_classes), dtype=int)
true_counts = np.zeros(n_classes, dtype=int)

for t, p in zip(truths, predicts):

    if t not in label_to_idx:
        continue

    i = label_to_idx[t]
    true_counts[i] += 1

    if p in label_to_idx:
        j = label_to_idx[p]
        counts[i, j] += 1
    # if predicted outside classes, do nothing to counts


# --------------------------------------------
# 5. Row-normalized probability matrix
# --------------------------------------------
probs = np.zeros_like(counts, dtype=float)
for i in range(n_classes):
    if true_counts[i] > 0:
        probs[i, :] = counts[i, :] / float(true_counts[i])
    else:
        probs[i, :] = 0.0


# --------------------------------------------
# 6. Create DataFrames
# --------------------------------------------
counts_df = pd.DataFrame(counts, index=labels, columns=labels)
probs_df  = pd.DataFrame(np.round(probs, 4), index=labels, columns=labels)

print("Class order (rows=true, cols=pred):", labels)
print("\nCount Matrix:")
print(counts_df)
print("\nRow-normalized Probabilities:")
print(probs_df)
print("\nSamples per true class:")
print(dict(zip(labels, true_counts.tolist())))


# --------------------------------------------
# 7. Save CSV + Images in script directory
# --------------------------------------------
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

counts_path = os.path.join(current_dir, "confusion_counts.csv")
probs_path  = os.path.join(current_dir, "confusion_probs.csv")
fig_path    = os.path.join(current_dir, "confusion_matrix_probabilities.png")

counts_df.to_csv(counts_path, encoding="utf-8-sig")
probs_df.to_csv(probs_path,  encoding="utf-8-sig")


# --------------------------------------------
# 8. Draw heatmap (English labels)
# --------------------------------------------
plt.figure(figsize=(8, 6))
ax = sns.heatmap(
    probs_df,
    annot=True,
    fmt='.4f',
    linewidths=0.5,
    linecolor='gray',
    cmap='Blues',
    cbar_kws={'label': 'Probability (row-normalized)'}
)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
plt.title("Confusion Matrix (Row-normalized Probabilities)\nRows=True, Cols=Predicted")
plt.tight_layout()
plt.savefig(fig_path, dpi=200, bbox_inches='tight')

print("\nSaved files:")
print(" ", counts_path)
print(" ", probs_path)
print(" ", fig_path)
