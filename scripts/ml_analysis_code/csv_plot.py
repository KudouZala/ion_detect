import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import subprocess

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib
# 设置字体为 Noto Sans CJK，这种字体支持中文
# rcParams['font.family'] = ['Noto Sans CJK SC']  # 简体中文
import os
from matplotlib import rcParams
import matplotlib.font_manager as fm

# 设置支持中文的字体
# 根据操作系统选择合适的中文字体
if os.name == 'posix':
    # Linux 或 macOS，尝试使用常见的开源中文字体
    # .ttc 是一种字体集合格式（包含多个字体）
    # matplotlib 默认无法直接解析 .ttc 中的多个子字体名，因此需要通过 FontProperties 精确指定一个字体。
    # 手动加载ttc字体
    font_prop = fm.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font_name = font_prop.get_name()
    rcParams['font.sans-serif'] = font_name
elif os.name == 'nt':
    # Windows 系统，默认使用 SimHei
    rcParams['font.sans-serif'] = ['SimHei']

def plot_attributions_from_folder(folder_path, save_fig=False):
    print(f"📂 正在检查文件夹: {folder_path}")
    files = os.listdir(folder_path)
    print(f"📁 找到 {len(files)} 个文件")

    # 自动找 pred ID 和 prefix
    pred_pattern = re.compile(r"(.*)_(saliency|ig)_pred(\d+)\.csv")
    pred_dict = {}
    for f in files:
        match = pred_pattern.match(f)
        if match:
            prefix, typ, pred_num = match.groups()
            key = (prefix, pred_num)
            pred_dict.setdefault(key, {})[typ] = f

    print(f"🔍 找到 {len(pred_dict)} 个预测编号组")

    # ==== 自动检测时间点数 & 频率点数 ====
    # 假设 Attention 文件的 token 数满足：
    # token_count = 1(CLASS) + T(时间token) + T*freq_points(频率token)
    #            = T*(freq_points + 1) + 1
    attn_files = [f for f in files if "_attn_" in f and "param_attn_" not in f]
    if len(attn_files) > 0:
        sample_attn_file = os.path.join(folder_path, attn_files[0])
        sample_data = pd.read_csv(sample_attn_file, header=None)
        token_count = sample_data.shape[1] if sample_data.shape[0] == 1 else sample_data.shape[0]
        print(f"🔎 检测到示例 Attention 文件: {attn_files[0]}, token 数={token_count}")

        freq_points = None
        num_time_points = None

        # 枚举候选组合：freq_points ∈ {64,63}, 时间点 ∈ {4,3,2}
        for f in [64, 63]:
            for t in [4, 3, 2]:
                if token_count == t * (f + 1) + 1:
                    freq_points = f
                    num_time_points = t
                    break
            if freq_points is not None:
                break

        if freq_points is None:
            # 兜底：保持原来的判断逻辑，默认为 4 个时间点
            if token_count == 4 * 64 + 4 + 1:
                freq_points = 64
            elif token_count == 4 * 63 + 4 + 1:
                freq_points = 63
            else:
                print("⚠️ 未识别列数，默认使用 4 时间点 + 64 频率点")
                freq_points = 64
            num_time_points = 4

        print(f"✅ 检测结果：{num_time_points} 个时间点, {freq_points} 个频率点")

    else:
        print("⚠️ 未找到任何 Attention 文件，默认使用 4 个时间点 + 64 频率点")
        freq_points = 64
        num_time_points = 4

    # 总 token 数：时间 token + (时间点 * 频率点 * 2[模 + 相])
    numbers = num_time_points + num_time_points * freq_points * 2
    short_numbers = 1 + 1 * freq_points * 2


    for (prefix, pred_num), file_map in pred_dict.items():
        pred_id = f"pred{pred_num}"
        attn_path = os.path.join(folder_path, f"{prefix}_attn_{pred_id}.csv")
        param_attn_path = os.path.join(folder_path, f"{prefix}_param_attn_{pred_id}.csv")
        sal_path = os.path.join(folder_path, f"{prefix}_saliency_{pred_id}.csv")
        ig_path = os.path.join(folder_path, f"{prefix}_ig_{pred_id}.csv")

        missing_files = []
        if not os.path.exists(sal_path):
            missing_files.append(sal_path)
        if not os.path.exists(ig_path):
            missing_files.append(ig_path)

        if missing_files:
            print(f"❌ 缺少以下文件，跳过：{', '.join(missing_files)}")
            continue

        # === Attention 读取 ===
        has_attn = os.path.exists(attn_path)
        if has_attn:
            attn = pd.read_csv(attn_path, header=None).values.flatten()
            # 前 num_time_points 个为时间 token
            time_tokens = attn[0:num_time_points]
            freq_tokens_raw = attn[num_time_points:]
            freq_tokens_expanded = np.empty(freq_tokens_raw.size * 2)
            freq_tokens_expanded[::2] = freq_tokens_raw
            freq_tokens_expanded[1::2] = freq_tokens_raw
            attn_expanded = np.concatenate([time_tokens, freq_tokens_expanded])[:numbers]

            attn_expanded = np.concatenate([time_tokens, freq_tokens_expanded])[:numbers]
        else:
            print(f"⚠️ 未找到 Attention 文件，将不绘制 Attention 曲线：{prefix}_{pred_id}")
            attn_expanded = np.zeros(numbers)

        # === Param_Attn 读取 ===
        has_param_attn = os.path.exists(param_attn_path)
        if has_param_attn:
            param_attn = pd.read_csv(param_attn_path, header=None).values.flatten()
            # 跳过第一列无效数据
            if len(param_attn) > 1:
                # ==== 1️⃣ 解析电压token ====
                volt_token = param_attn[1]  # 电压时间token(单值)

                # ==== 2️⃣ 解析阻抗tokens ====
                freq_tokens_param_raw = param_attn[2:]  # 频率tokens


                # 扩展频率tokens（复制一次用于强度和相位）
                freq_tokens_param_expanded = np.empty(freq_tokens_param_raw.size * 2)
                freq_tokens_param_expanded[::2] = freq_tokens_param_raw
                freq_tokens_param_expanded[1::2] = freq_tokens_param_raw

                # ==== 3️⃣ 构造拼接结果 ====
                # 电压部分: [有效值, 0, 0, 0]
                                # ==== 3️⃣ 构造拼接结果（适配任意时间点数） ====
                # 电压部分: 长度 = num_time_points，只有第 1 个时间点非零
                volt_block = np.array([volt_token] + [0] * (num_time_points - 1))

                # 阻抗部分: 第 1 个时间点为有效频率 token，其余时间点为 0
                zero_block = np.zeros_like(freq_tokens_param_expanded)
                blocks = [freq_tokens_param_expanded] + [zero_block] * (num_time_points - 1)
                impe_block = np.concatenate(blocks)

                # 最终拼接
                param_all_time = np.concatenate([volt_block, impe_block])


                # 截断或填充到 numbers 长度
                if len(param_all_time) != numbers:
                    raise ValueError(
                        f"❌ Param_Attn 长度不匹配: 期望 {numbers}，实际 {len(param_all_time)} "
                        f"(prefix={prefix}, pred_id={pred_id})"
                    )
                else:
                    param_attn_expanded = param_all_time

            else:
                print(f"⚠️ Param_Attn 文件数据不足，将使用零填充：{prefix}_{pred_id}")
                param_attn_expanded = np.zeros(numbers)
        else:
            print(f"⚠️ 未找到 Param_Attn 文件，将不绘制 Param_Attn 曲线：{prefix}_{pred_id}")
            param_attn_expanded = np.zeros(numbers)


        sal = pd.read_csv(sal_path)["value"].values
        ig = pd.read_csv(ig_path)["value"].values

        def clean_numeric_array(arr, target_len):
            arr = pd.to_numeric(pd.Series(arr), errors='coerce').dropna().values
            if len(arr) > target_len:
                arr = arr[:target_len]
            elif len(arr) < target_len:
                arr = np.pad(arr, (0, target_len - len(arr)), 'constant')
            return arr

        sal_clean = clean_numeric_array(sal, numbers)
        ig_clean = clean_numeric_array(ig, numbers)

        
                # =============== 额外绘制：按频率求和图 ===============
                # =============== 额外绘制：四类频率求和对比图 ===============
        try:
            x_freq = np.arange(freq_points * 2)
            expected_len = numbers  # = num_time_points + num_time_points * freq_points * 2

            # 初始化存储
            volt_sum_attn = volt_sum_param = volt_sum_sal = volt_sum_ig = None
            freq_sum_attn = freq_sum_param = freq_sum_sal = freq_sum_ig = None

            # 1️⃣ Attention
            if has_attn:
                if len(attn_expanded) >= num_time_points:
                    volt_sum_attn = np.sum(attn_expanded[:num_time_points])
                if len(attn_expanded) == expected_len:
                    freq_sum_attn = attn_expanded[num_time_points:].reshape(
                        num_time_points, freq_points * 2
                    ).sum(axis=0)

            # 2️⃣ Param_Attention
            if has_param_attn:
                if len(param_attn_expanded) >= num_time_points:
                    volt_sum_param = np.sum(param_attn_expanded[:num_time_points])
                if len(param_attn_expanded) == expected_len:
                    freq_sum_param = param_attn_expanded[num_time_points:].reshape(
                        num_time_points, freq_points * 2
                    ).sum(axis=0)

            # 3️⃣ Saliency
            if len(sal_clean) >= num_time_points:
                volt_sum_sal = np.sum(sal_clean[:num_time_points])
            if len(sal_clean) == expected_len:
                freq_sum_sal = sal_clean[num_time_points:].reshape(
                    num_time_points, freq_points * 2
                ).sum(axis=0)

            # 4️⃣ Integrated Gradients
            if len(ig_clean) >= num_time_points:
                volt_sum_ig = np.sum(ig_clean[:num_time_points])
            if len(ig_clean) == expected_len:
                freq_sum_ig = ig_clean[num_time_points:].reshape(
                    num_time_points, freq_points * 2
                ).sum(axis=0)

            else:
                print(f"⚠️ IG 数据长度不匹配: 期望 {expected_len}，实际 {len(ig_clean)} ({prefix}_pred{pred_num})")

            # 检查是否有数据
            if all(v is None for v in [freq_sum_attn, freq_sum_param, freq_sum_sal, freq_sum_ig]):
                print(f"⚠️ 所有频率求和数据为空，跳过绘图 ({prefix}_pred{pred_num})")
            else:
                # ========== 构建横坐标：0 表示电压求和，后面是频率索引 ==========
                x_all = np.concatenate([[0], x_freq + 1])  # 0=Voltage, 其余是频率索引
                fig_freq, ax1 = plt.subplots(figsize=(18, 5))
                ax2 = ax1.twinx()

                lines_freq = []
                labels_freq = []

                # 左轴曲线
                if volt_sum_attn is not None and freq_sum_attn is not None:
                    y_attn = np.concatenate([[volt_sum_attn], freq_sum_attn])
                    l1, = ax1.plot(x_all, y_attn, label="Attention", color='blue')
                    lines_freq.append(l1); labels_freq.append("Attention")

                if volt_sum_param is not None and freq_sum_param is not None:
                    y_param = np.concatenate([[volt_sum_param], freq_sum_param])
                    l2, = ax1.plot(x_all, y_param, label="Param_Attention", color='orange', linestyle='-.')
                    lines_freq.append(l2); labels_freq.append("Param_Attention")

                # 右轴曲线
                if volt_sum_sal is not None and freq_sum_sal is not None:
                    y_sal = np.concatenate([[volt_sum_sal], freq_sum_sal])
                    l3, = ax2.plot(x_all, y_sal, label="Saliency", color='green')
                    lines_freq.append(l3); labels_freq.append("Saliency")

                if volt_sum_ig is not None and freq_sum_ig is not None:
                    y_ig = np.concatenate([[volt_sum_ig], freq_sum_ig])
                    l4, = ax2.plot(x_all, y_ig, label="Integrated Gradients", color='red')
                    lines_freq.append(l4); labels_freq.append("Integrated Gradients")

                # 标签和网格
                ax1.set_ylabel("Voltage+Freq Attribution (Attn/Param)", color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                ax2.set_ylabel("Voltage+Freq Attribution (Saliency/IG)", color='green')
                ax2.tick_params(axis='y', labelcolor='green')

                ax1.set_title(f"Voltage + Frequency Attribution Summed Over Time ({prefix}_pred{pred_num})")
                ax1.set_xlabel("0=VoltageSum, 1~N=Frequency Token Index")
                ax1.grid(True, linestyle='--', alpha=0.5)
                fig_freq.legend(lines_freq, labels_freq, loc='upper right')
                fig_freq.tight_layout()

                if save_fig:
                    fig_freq.savefig(os.path.join(folder_path, f"{prefix}_pred{pred_num}_volt_freq_sum_plot.png"), dpi=300)
                plt.close(fig_freq)

                # ========== CSV 数据 ==========
                # 构建频率标签（含电压）
                                # ========== CSV 数据 ==========
                # 构建频率标签（含电压），这里用索引而不是具体频率值，避免依赖 freq_values_hz
                freq_labels = ["Voltage_Sum"] + [
                    f"FreqToken_{i+1}" for i in range(freq_points * 2)
                ]


                if len(freq_labels) != len(x_all):
                    print(f"⚠️ 频率标签长度不匹配: 期望 {len(x_all)}, 实际 {len(freq_labels)}")
                    freq_labels = [""] * len(x_all)

                csv_data = {
                    "Index": x_all,
                    "Label": freq_labels
                }
                if volt_sum_attn is not None and freq_sum_attn is not None:
                    csv_data["Attention"] = np.concatenate([[volt_sum_attn], freq_sum_attn])
                if volt_sum_param is not None and freq_sum_param is not None:
                    csv_data["Param_Attention"] = np.concatenate([[volt_sum_param], freq_sum_param])
                if volt_sum_sal is not None and freq_sum_sal is not None:
                    csv_data["Saliency"] = np.concatenate([[volt_sum_sal], freq_sum_sal])
                if volt_sum_ig is not None and freq_sum_ig is not None:
                    csv_data["Integrated_Gradients"] = np.concatenate([[volt_sum_ig], freq_sum_ig])

                df_csv = pd.DataFrame(csv_data)
                csv_path = os.path.join(folder_path, f"{prefix}_pred{pred_num}_volt_freq_sum_data.csv")
                df_csv.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"✅ 电压+频率求和数据已保存: {csv_path}")

        except Exception as e:
            print(f"⚠️ 无法绘制电压+频率求和对比图 ({prefix}_pred{pred_num}): {e}")

        
        
        
        
        
        
        # === 绘图 ===
        fig, ax1 = plt.subplots(figsize=(22, 6))
        x = np.arange(numbers)
        lines = []
        labels = []

        print(f"绘图 {prefix}_{pred_id}: x.shape={x.shape}, attn_expanded.shape={attn_expanded.shape}")

        # 初始化数据字典
        csv_data = {
            "Token_Index": x
        }

        # 1️⃣ Attention
        if has_attn:
            l1, = ax1.plot(x, attn_expanded, label="Attention", color='blue')
            csv_data["Attention"] = attn_expanded
            lines.append(l1)
            labels.append("Attention")

        # 2️⃣ Param_Attention
        if has_param_attn and param_attn_expanded is not None:
            l4, = ax1.plot(np.arange(len(param_attn_expanded)), param_attn_expanded,
                           label="Param_Attention", color='orange', linestyle='-.')
            csv_data["Param_Attention"] = param_attn_expanded
            lines.append(l4)
            labels.append("Param_Attention")

        ax1.set_ylabel("Attention", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # 3️⃣ Saliency
        ax2 = ax1.twinx()
        l2, = ax2.plot(x, sal_clean, label="Saliency", color='green')
        csv_data["Saliency"] = sal_clean
        ax2.set_ylabel("Saliency", color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        lines.append(l2)
        labels.append("Saliency")

        # 4️⃣ Integrated Gradients
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.05))
        l3, = ax3.plot(x, ig_clean, label="Integrated Gradients", color='red')
        csv_data["Integrated_Gradients"] = ig_clean
        ax3.set_ylabel("IG", color='red')
        ax3.tick_params(axis='y', labelcolor='red')
        lines.append(l3)
        labels.append("Integrated Gradients")

        # 区域分割标注
                # 区域分割标注（适配任意时间点数）
        segment_labels = ["Time Tokens"] + [f"Freq@T{i+1}" for i in range(num_time_points)]
        block = freq_points * 2  # 每个时间点对应的频率 token 数（模+相）
        segment_positions = [0, num_time_points]
        for k in range(1, num_time_points + 1):
            segment_positions.append(num_time_points + k * block)

        for i in range(1, len(segment_positions) - 1):
            ax1.axvline(segment_positions[i], color='gray', linestyle='--', alpha=0.4)
            ax1.text(segment_positions[i] + 5, ax1.get_ylim()[1] * 0.9,
                     segment_labels[i], fontsize=12)


        ax1.set_xlim(0, numbers)
        ax1.set_xlabel("Input Token Index (Time + Frequency Domain)")
        ax1.set_title(f"Attribution Visualization ({prefix}_pred{pred_num})")
        ax1.legend(lines, labels, loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.5)
        fig.tight_layout()

        # === 保存图像 ===
        if save_fig:
            fig.savefig(os.path.join(folder_path, f"{prefix}_pred{pred_num}_attribution_plot.png"), dpi=300)
        plt.close(fig)
                # === 保存 CSV 数据 ===
        csv_data = {
            "Token_Index": x,
            "Token_Type": ["Time"] * num_time_points + ["Freq"] * (numbers - num_time_points)
        }


        # 添加频率标签
        if freq_points == 64:
            freq_values_hz = [
                19950, 15850, 12590, 10000, 7943, 6310, 5010, 3980, 3160, 2510, 1990, 1590, 1260, 1000,
                794.3, 631.0, 501.2, 398.1, 316.2, 251.2, 199.5, 158.5, 125.9, 100.0,
                79.43, 63.10, 50.12, 39.81, 31.62, 25.12, 19.95, 15.85, 12.59, 10.0,
                7.94, 6.31, 5.01, 3.98, 3.16, 2.51, 1.99, 1.59, 1.26, 1.0,
                0.7943, 0.6310, 0.5012, 0.3981, 0.3162, 0.2512, 0.1995, 0.1585, 0.1259, 0.1,
                0.07943, 0.06310, 0.05012, 0.03981, 0.03162, 0.02512, 0.01995, 0.01585, 0.01259, 0.01
            ]
        else:
            freq_values_hz = [
                19950, 15850, 12590, 10000, 7943, 6310, 5010, 3980, 3160, 2510, 1990, 1590, 1260, 1000,
                794.3, 631.0, 501.2, 398.1, 316.2, 251.2, 199.5, 158.5, 125.9, 100.0,
                79.43, 63.10, 50.12, 39.81, 31.62, 25.12, 19.95, 15.85, 12.59, 10.0,
                7.94, 6.31, 5.01, 3.98, 3.16, 2.51, 1.99, 1.59, 1.26, 1.0,
                0.7943, 0.6310, 0.5012, 0.3981, 0.3162, 0.2512, 0.1995, 0.1585, 0.1259, 0.1,
                0.07943, 0.06310, 0.05012, 0.03981, 0.03162, 0.02512, 0.01995, 0.01585, 0.01259
            ]

                # 构建 Token_Label（按检测到的时间点数/频率点数动态生成）
        token_labels = [f"T{i+1}" for i in range(num_time_points)]
        for t in range(1, num_time_points + 1):
            for f in freq_values_hz[:freq_points]:
                token_labels.append(f"{t}_time_{f}_Hz_mag")
                token_labels.append(f"{t}_time_{f}_Hz_phase")

        if len(token_labels) != numbers:
            print(f"⚠️ Token_Label 长度不匹配: 期望 {numbers}, 实际 {len(token_labels)}")
            # 用空字符串填充避免报错
            token_labels = [""] * numbers

        csv_data["Token_Label"] = token_labels


        # 加入各曲线数据
        if has_attn:
            csv_data["Attention"] = attn_expanded
        if has_param_attn:
            csv_data["Param_Attention"] = param_attn_expanded
        csv_data["Saliency"] = sal_clean
        csv_data["Integrated_Gradients"] = ig_clean

        df_csv = pd.DataFrame(csv_data)
        csv_path = os.path.join(folder_path, f"{prefix}_pred{pred_num}_attribution_plot_data.csv")
        df_csv.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ Attribution 图数据(含频率标签)已保存: {csv_path}")

        
def main():
    parser = argparse.ArgumentParser(description="可视化测试结果")
    parser.add_argument("--load_run", type=str, required=True, help="指定要加载的运行结果文件夹名称，例如20250729a")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[2] / "output" / "inference_results"
    folder_path = base_dir / args.load_run

    if not folder_path.exists():
        raise FileNotFoundError(f"❌ 指定的运行结果目录不存在: {folder_path}")

    print(f"📂 正在可视化: {folder_path}")
    plot_attributions_from_folder(str(folder_path), save_fig=True)

if __name__ == "__main__":
    main()
