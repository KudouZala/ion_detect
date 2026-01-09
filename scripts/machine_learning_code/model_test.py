import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from captum.attr import Saliency, IntegratedGradients
import re
import json
# 标签映射：数字索引 -> 英文离子名称
LABEL_MAPPING = {0: 'Ca2+_ion', 1: 'Na+_ion', 2: 'Ni2+_ion', 3: 'Cr3+_ion',4: 'Cu2+_ion',5:'Fe3+_ion',  6:'no_ion'}

# 英文离子名称 -> 中文离子名称映射
ion_chinese_map = {
    'Ca2+_ion': '钙离子',
    'Na+_ion': '钠离子',
    'Ni2+_ion': '镍离子',
    'Cr3+_ion': '铬离子',
    'Cu2+_ion': '铜离子',
    'Fe3+_ion': '铁离子',
    'no_ion':'无污染'
}

# ===============================
# 功能：对预测结果进行评估
# 输入：
#   xlsx_path: 样本文件路径
#   predicted_label: 模型预测的标签(英文离子名字符串)
# 输出：
#   correct: 布尔值，预测是否正确
#   predicted_ion_chinese: 预测的中文离子名
#   true_ion_chinese: 真实的中文离子名
# ===============================
def evaluate_prediction(xlsx_path, predicted_label):
    # 获取预测结果对应的中文离子名
    if predicted_label is not None:
        predicted_ion_chinese = ion_chinese_map.get(predicted_label, "未知")
    else:
        predicted_ion_chinese = "无预测"

    # 从文件路径中提取真实标签对应的中文离子名
    true_ion_chinese = "未知"
    if 'ion_column' in xlsx_path:
         true_ion_chinese = '无污染'
    else:
        for name in ion_chinese_map.values():
            if name in str(xlsx_path):  # 匹配路径字符串
                true_ion_chinese = name
                break

    # 判断预测是否正确
    correct = (predicted_ion_chinese == true_ion_chinese)
    return correct, predicted_ion_chinese, true_ion_chinese


# ===============================
# 功能：对单个xlsx文件进行推理并生成可解释性结果
# 输入：
#   xlsx_path: 单个样本文件路径
#   model: 已加载好的深度学习模型
#   device: torch.device
#   num_time_points: 时间点数量
#   num_freq_points: 阻抗频率点数量
#   folder_name: 输出结果保存文件夹名
# 输出：
#   correct: 是否预测正确True orFalse
#   predict: 预测的中文离子名
#   truth: 真实的中文离子名
# ===============================


def test_single_xlsx_and_generate_explanations_three_system_1117(
    xlsx_path, model, device, num_time_points, num_freq_points, folder_name
):
    """
    针对 Model_three_system_1117 的单文件测试函数：
    - 读取 xlsx
    - 做归一化
    - 前向推理
    - 输出预测类别、物性参数、影响因子
    - 保存 attention / Saliency / IG 到 CSV
    """
    # 获取项目根目录
    base_dir = Path(__file__).resolve().parent.parent.parent
    model.eval()  # 设为评估模式

    # === 0. 设置输出路径 ===
    output_dir = base_dir / "output" / "inference_results" / folder_name
    os.makedirs(output_dir, exist_ok=True)

    # === 1. 加载单个 xlsx 文件 ===
    df = pd.read_excel(xlsx_path)
    label = df['Label'].values[0]

    # === 1.1 提取浓度值 (ppm) ===
    file_name = os.path.basename(xlsx_path)
    if 'ppm' in df.columns:
        try:
            concentration = float(df['ppm'].iloc[0])
            concentration = torch.tensor([[concentration]], dtype=torch.float32, device=device).expand(1, 1)
        except Exception as e:
            print(f"Warning: Failed to parse ppm from df in {file_name}, set to -1. Error: {e}")
            concentration = torch.tensor([[ -1.0 ]], dtype=torch.float32, device=device)
    else:
        print(f"Warning: No ppm column in {file_name}, set to -1")
        concentration = torch.tensor([[ -1.0 ]], dtype=torch.float32, device=device)

    # === 1.2 构建 4 个时间点的数据 ===
    if num_time_points==4:
        time_points = [0, 2, 4, 6]
    elif num_time_points==3:
        time_points = [0, 2, 4]
    elif num_time_points==2:
        time_points = [0, 2]
    elif num_time_points==1:
        time_points = [0]
    else:
        print("no num_time_points")

    volt_list, impe_list = [], []
    for t in time_points:
        t_df = df[df['Time(h)'] == t].copy()
        if t_df.empty:
            print(f"❌ 时间点 {t} 缺失")
            return

        # 频率：从高到低排序（降序）
        if 'Freq' not in t_df.columns:
            raise ValueError("Missing column 'Freq' for frequency sorting.")
        t_df = t_df.sort_values('Freq', ascending=False).reset_index(drop=True)

        # （可选但建议）保证频点数足够
        if len(t_df) < num_freq_points:
            raise ValueError(f"Impedance points {len(t_df)} < {num_freq_points} for time={t}.")

        # 电压
        volt = torch.tensor([t_df['mean_voltage'].iloc[0]], dtype=torch.float32)

        # 阻抗 (Zreal, Zimag)：此时顺序已是高频->低频
        impedance_np = t_df.loc[:num_freq_points-1, ['Zreal', 'Zimag']].to_numpy()

        volt_list.append(volt)
        impe_list.append(torch.tensor(impedance_np, dtype=torch.float32))

    # 组合成张量
    volt_tensor = torch.stack(volt_list).unsqueeze(0).to(device)   # (1, T, 1)
    true_volt = volt_tensor[0, -1, 0]
    impe_tensor = torch.stack(impe_list).unsqueeze(0).to(device)   # (1, T, F, 2)


    # 环境参数
    env_param = torch.tensor(
        [[df['temperature'].mean(), df['flow'].mean(), df['current'].mean()]],
        dtype=torch.float32
    ).to(device)

    # === 2. 加载归一化参数 stats_dataset2.json ===
    stats_path = base_dir / "datasets" / "stats_dataset2.json"
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"统计参数文件 {stats_path} 不存在，请先运行 Dataset_2_Stable 保存该文件。"
        )

    with open(stats_path, 'r') as f:
        stats = json.load(f)

    # 电压：min-max（和训练保持一致）
    volt_min = stats["volt_min"]
    volt_max = stats["volt_max"]

    # 阻抗：mean / std（和 Dataset_2_Stable 里的统计方式一致）
    impe_mag_mean = torch.tensor(
        stats["impe_mag_mean"], dtype=torch.float32, device=device
    )   # (F,)
    impe_mag_std = torch.tensor(
        stats["impe_mag_std"], dtype=torch.float32, device=device
    )    # (F,)
    impe_phase_mean = torch.tensor(
        stats["impe_phase_mean"], dtype=torch.float32, device=device
    ) # (F,)
    impe_phase_std = torch.tensor(
        stats["impe_phase_std"], dtype=torch.float32, device=device
    )  # (F,)

    # --- 电压归一化 ---
    denom_v = max(volt_max - volt_min, 1e-8)
    volt_tensor = (volt_tensor - volt_min) / denom_v

    # --- 阻抗 -> 幅值 + 相位，再用 mean/std 归一化 ---
    real = impe_tensor[..., 0]   # (B, 4, F)
    imag = impe_tensor[..., 1]   # (B, 4, F)

    # 1) 复数形式
    z_complex = real + 1j * imag            # (B, 4, F)

    # 2) 幅值/相位预处理（和训练完全一致）
    z_mag = torch.log1p(torch.abs(z_complex))   # log(1 + |Z|), (B, 4, F)
    z_phase = torch.angle(z_complex)            # [-π, π]
    z_phase = (z_phase + np.pi) / (2 * np.pi)   # → [0, 1]

    # 3) z-score 标准化（freq 维广播）
    z_mag = (z_mag - impe_mag_mean) / (impe_mag_std + 1e-8)
    z_phase = (z_phase - impe_phase_mean) / (impe_phase_std + 1e-8)

    # 4) 拼回最终张量
    impe_tensor = torch.stack((z_mag, z_phase), dim=-1)   # (B, 4, F, 2)

    # === 2.1 电解槽参数，根据文件名选择 ===
    if "新版电解槽" in file_name:
        electrolyzer_parameters = torch.tensor([
            0.012, 0.012, 0.002, 135e-6,
            2.38e6, 2.38e6, 5.96e7, 4
        ], dtype=torch.float32)
    elif "旧版电解槽" in file_name:
        electrolyzer_parameters = torch.tensor([
            0.012, 0.012, 0.002, 135e-6,
            2.38e6, 2.38e6, 5.96e7, 4
        ], dtype=torch.float32)
    else:
        electrolyzer_parameters = torch.tensor([
            0.012, 0.012, 0.002, 135e-6,
            2.38e6, 2.38e6, 5.96e7, 4
        ], dtype=torch.float32)
    electrolyzer_parameters = electrolyzer_parameters.unsqueeze(0).to(device)   # (1, P)

    # === 3. 启用梯度，准备解释性分析 ===
    volt_tensor.requires_grad_()
    impe_tensor.requires_grad_()
    print("→ volt requires_grad:", volt_tensor.requires_grad)
    print("→ impe requires_grad:", impe_tensor.requires_grad)

    # === 3.1 前向推理（兼容 1117 多输出） ===
    outputs = model(volt_tensor, impe_tensor, env_param, electrolyzer_parameters, concentration)
    # 只取前 10 个，后面即使有 rule_pred, group_logits 等也会被忽略：
    (prob_output,
     predicted_voltage,
     predicted_conc,
     wuxing,
     cls_attn_mean,
     raw_prob_output,
     yingxiang,
     param_attn_mean,
     freq_attn,
     freq_attn_param,
     *extra_outputs) = outputs

    predicted_class = torch.argmax(prob_output.detach(), dim=1).item()

    # === 3.2 提取五个物性参数 & 五个影响因子 ===
    wuxing_names = ['sigma_mem', 'alpha_ca', 'alpha_an', 'i_0ca', 'i_0an']
    yingxiang_names = ['psi', 'theta_ca', 'theta_an', 'phi_ca', 'phi_an']

    if isinstance(wuxing, list):
        wuxing_values = []
        for t in wuxing:
            if torch.is_tensor(t):
                wuxing_values.append(t.flatten()[0].item())
            else:
                wuxing_values.append(float(t))
    else:
        # 如果是 tensor 形式 (5,1) 或 (1,5) 等，也做一下兼容
        wuxing_tensor = torch.cat(wuxing, dim=1) if isinstance(wuxing, tuple) else wuxing
        wuxing_values = [float(x) for x in wuxing_tensor.flatten()[:5]]

    if isinstance(yingxiang, list):
        yingxiang_values = []
        for t in yingxiang:
            if torch.is_tensor(t):
                yingxiang_values.append(t.flatten()[0].item())
            else:
                try:
                    yingxiang_values.append(float(t))
                except:
                    continue
    else:
        yingxiang_tensor = torch.cat(yingxiang, dim=1) if isinstance(yingxiang, tuple) else yingxiang
        yingxiang_values = [float(x) for x in yingxiang_tensor.flatten()[:5]]

    # === 3.3 构造结果行 ===
    base_filename = os.path.splitext(os.path.basename(xlsx_path))[0]
    rows = []
    for i in range(5):
        rows.append([wuxing_names[i], wuxing_values[i], yingxiang_names[i], yingxiang_values[i]])

    label_mapping = {
        'Ca2+_ion': 0, 'Na+_ion': 1, 'Ni2+_ion': 2,'Cr3+_ion': 3,'Cu2+_ion': 4,'Fe3+_ion': 5, "no_ion": 6
    }
    inverse_label_mapping = {v: k for k, v in label_mapping.items()}

    predicted_class_int = int(predicted_class)
    predicted_label = inverse_label_mapping.get(predicted_class_int, f"Unknown({predicted_class_int})")
    rows.append(['predicted_class', predicted_label, '', ''])

    # 各类别概率
    prob_values = prob_output.detach().cpu().numpy().flatten()
    for ion, idx in label_mapping.items():
        prob = prob_values[idx]
        rows.append([f"prob_{ion}", prob, '', ''])

    # 电压 & 浓度
    rows.append(["predicted_volt", predicted_voltage.squeeze().item(), "", ""])
    rows.append(["true_volt", true_volt.item(), "", ""])
    rows.append(["predicted_conc", predicted_conc.squeeze().item(), "", ""])
    rows.append(["true_conc", concentration.item(), "", ""])

    # === 3.4 保存结构化参数表格 ===
    df_out = pd.DataFrame(rows, columns=['wuxing_name', 'wuxing_value', 'yingxiang_name', 'yingxiang_value'])
    combined_path = os.path.join(output_dir, f"{base_filename}_phys_params_structured.csv")
    df_out.to_csv(combined_path, index=False)
    print(f"✅ Structured parameter table saved: {combined_path}")

    # === 4. 保存 Attention heatmap ===
    attn_path = os.path.join(output_dir, f"{base_filename}_attn_pred{predicted_class}.csv")
    pd.DataFrame(cls_attn_mean[0].detach().cpu().numpy().reshape(1, -1)).to_csv(attn_path, index=False, header=False)
    print(f"✅ Attention heatmap (CLS) saved: {attn_path}")

    param_attn_path = os.path.join(output_dir, f"{base_filename}_param_attn_pred{predicted_class}.csv")
    pd.DataFrame(param_attn_mean[0].detach().cpu().numpy().reshape(1, -1)).to_csv(param_attn_path, index=False, header=False)
    print(f"✅ Attention heatmap (PARAM) saved: {param_attn_path}")

    # === 5. 定义 forward_func，用于梯度归因（使用 raw_prob_output：outputs[5]） ===
    def forward_func(v, i, e, p, c):
        out = model(v, i, e, p, c)
        return out[5]  # raw_prob_output: [B, C]

    # 准备可导输入
    volt_in = volt_tensor.clone().detach().requires_grad_(True)
    impe_in = impe_tensor.clone().detach().requires_grad_(True)
    env_in  = env_param.clone().detach().requires_grad_(True)
    para_in = electrolyzer_parameters.clone().detach().requires_grad_(True)
    conc_arg = concentration

    # === 6. Saliency ===
    saliency = Saliency(forward_func)
    sal_attr = saliency.attribute(
        inputs=(volt_in, impe_in, env_in, para_in),
        additional_forward_args=(conc_arg,),
        target=predicted_class
    )
    sal_v    = sal_attr[0].detach().cpu().numpy()[0]  # (T,1)
    sal_i    = sal_attr[1].detach().cpu().numpy()[0]  # (T,F,2)
    sal_env  = sal_attr[2].detach().cpu().numpy()[0]  # (3,)
    sal_para = sal_attr[3].detach().cpu().numpy()[0]  # (P,)

    # === 7. Integrated Gradients ===
    ig = IntegratedGradients(forward_func)
    ig_attr = ig.attribute(
        inputs=(volt_in, impe_in, env_in, para_in),
        baselines=(
            torch.zeros_like(volt_in),
            torch.zeros_like(impe_in),
            torch.zeros_like(env_in),
            torch.zeros_like(para_in),
        ),
        additional_forward_args=(conc_arg,),
        target=predicted_class,
        internal_batch_size=6
    )
    ig_v    = ig_attr[0].detach().cpu().numpy()[0]
    ig_i    = ig_attr[1].detach().cpu().numpy()[0]
    ig_env  = ig_attr[2].detach().cpu().numpy()[0]
    ig_para = ig_attr[3].detach().cpu().numpy()[0]

    # === 8. 保存梯度可解释性结果 ===
    def _ensure_2d(x):
        if isinstance(x, np.ndarray) and x.ndim == 1:
            return x[:, None]
        return x

    def save_combined_csv(voltage_grad, impedance_grad, env_grad, para_grad, out_path):
        voltage_grad = _ensure_2d(voltage_grad)
        env_grad     = _ensure_2d(env_grad)
        para_grad    = _ensure_2d(para_grad)

        rows = []

        # 电压
        T = voltage_grad.shape[0]
        for t in range(T):
            rows.append([t, "volt", "", "", float(voltage_grad[t, 0])])

        # 阻抗
        T, F, D = impedance_grad.shape
        for t in range(T):
            for f in range(F):
                for d in range(D):
                    rows.append([t, "impe", f, d, float(impedance_grad[t, f, d])])

        # 环境参数
        for idx in range(env_grad.shape[0]):
            rows.append([0, "env", idx, "", float(env_grad[idx, 0])])

        # 电解槽参数
        for idx in range(para_grad.shape[0]):
            rows.append([0, "electrolyzer_param", idx, "", float(para_grad[idx, 0])])

        pd.DataFrame(rows, columns=["time_idx", "type", "freq_idx", "dim", "value"]).to_csv(out_path, index=False)
        print(f"✅ 保存到 {out_path}")

    def save_time_aggregates_csv(impedance_grad, voltage_grad, out_path):
        volt_time_abs = np.abs(voltage_grad.squeeze(-1))           # (T,)
        impe_time_sum = np.abs(impedance_grad).sum(axis=(1, 2))    # (T,)
        df = pd.DataFrame({
            "t": np.arange(volt_time_abs.shape[0]),
            "volt_abs": volt_time_abs,
            "impe_abs_sum": impe_time_sum
        })
        df.to_csv(out_path, index=False)
        print(f"✅ 时间聚合归因保存: {out_path}")

    sal_path = os.path.join(output_dir, f"{base_filename}_saliency_pred{predicted_class}.csv")
    ig_path  = os.path.join(output_dir, f"{base_filename}_ig_pred{predicted_class}.csv")
    save_combined_csv(sal_v, sal_i, sal_env, sal_para, sal_path)
    save_combined_csv(ig_v, ig_i, ig_env, ig_para, ig_path)

    sal_time_agg_path = os.path.join(output_dir, f"{base_filename}_saliency_time_aggregates.csv")
    ig_time_agg_path  = os.path.join(output_dir, f"{base_filename}_ig_time_aggregates.csv")
    save_time_aggregates_csv(sal_i, sal_v, sal_time_agg_path)
    save_time_aggregates_csv(ig_i, ig_v, ig_time_agg_path)

    # === 9. 评估预测结果 ===
    correct, predict, truth = evaluate_prediction(xlsx_path=xlsx_path, predicted_label=predicted_label)
    print("correct:", correct)
    print("predict:", predict)
    print("truth:", truth)

    return correct, predict, truth


