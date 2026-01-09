import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ========== 通用配置 ==========
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# ===== 用户配置部分 =====
use_directory = True
input_dir = '/home/cagalii/Application/train_machine_learning/最新_绘图和输出文件代码_更新位置/inference_results_freq_and_time_embedding_all_freq_20250607c_final'  # TODO: 替换为你的实际路径
output_dir = 'output_charts_explanations_20250607c_final'
os.makedirs(output_dir, exist_ok=True)

# 文件模式配置
target_suffix = {
    'saliency': '_saliency_map.csv',
    'ig': '_integrated_gradients.csv',
    'attention': '_attention_values.xlsx'
}

# F1-F64 对应的频率值（Hz）
freq_values = [
    19952, 15850, 12590, 10000,
    7943, 6310, 5012, 3981, 3162, 2512, 1995, 1585, 1259, 1000,
    794.3, 631.0, 501.2, 398.1, 316.2, 251.2, 199.5, 158.5, 125.9, 100.0,
    79.43, 63.10, 50.12, 39.81, 31.62, 25.12, 19.95, 15.85, 12.59, 10.0,
    7.94, 6.31, 5.01, 3.98, 3.16, 2.51, 1.99, 1.59, 1.26, 1.0,
    0.7943, 0.6310, 0.5012, 0.3981, 0.3162, 0.2512, 0.1995, 0.1585, 0.1259, 0.1,
    0.07943, 0.06310, 0.05012, 0.03981, 0.03162, 0.02512, 0.01995, 0.01585, 0.01259, 0.01
]

# ========== 文件缓存 ==========
file_data = {
    'saliency': {},
    'ig': {},
    'attention': {}
}

# ========== 通用分析函数 ==========
def process_file(file_path, tag):
    try:
        filename_key = os.path.basename(file_path).replace(target_suffix[tag], '')

        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            volt_df = df[df['type'] == 'voltage'].groupby('time')['value'].sum()
            voltage_sum = volt_df.sum()
            impe_df = df[df['type'] == 'impe']
            impe_sum_by_freq = impe_df.groupby('freq')['value'].sum()
            if tag == 'ig':
                frequency_sums = [impe_sum_by_freq.get(freq + 1, 0.0) for freq in range(64)]
            else:
                frequency_sums = [np.abs(impe_sum_by_freq.get(freq + 1, 0.0)) for freq in range(64)]

        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, header=0, engine='openpyxl')
            last_row = df.iloc[-1]
            voltage_sum = last_row.iloc[0:4].sum()
            num_freq = 64
            frequency_sums = []
            for i in range(num_freq):
                cols = [4 + i + num_freq * t for t in range(4)]
                values = last_row.iloc[cols]
                frequency_sums.append(np.abs(values.sum()))

        values = [np.abs(voltage_sum)] + frequency_sums
        labels = ['Voltage'] + [f'F{i+1}' for i in range(64)]
        file_data[tag][filename_key] = frequency_sums

        df_csv = pd.DataFrame([values], columns=labels)
        df_csv.to_csv(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_data.csv"), index=False)

        # 画图：柱状图
        plt.figure(figsize=(18, 6))
        sns.barplot(x=labels, y=values)
        plt.xticks(rotation=90)
        plt.ylabel('Summed Value')
        plt.title(f"Voltage & Frequency Magnitude - {os.path.basename(file_path)}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_chart.png"), dpi=300)
        plt.close()

        # 折线图（log 频率）
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=freq_values, y=frequency_sums, marker='o', linewidth=2.5)
        plt.axhline(0, linestyle='--', color='green' if tag == 'ig' else ('blue' if tag == 'attention' else 'red'), linewidth=1.2, alpha=0.4)
        plt.xscale('log')
        plt.xlabel('Frequency (Hz, log scale)')
        plt.ylabel('Importance')
        plt.title(f"Frequency Attribution - {os.path.basename(file_path)}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_freq_log_lineplot.png"), dpi=300)
        plt.close()

        return {
            'filename': os.path.basename(file_path),
            'voltage_sum': float(np.abs(voltage_sum)),
            'impedance_sum': float(np.sum(frequency_sums))
        }

    except Exception as e:
        print(f"[Error] Processing {file_path}: {e}")
        return None

# ========== 主执行流程 ==========
for tag, suffix in target_suffix.items():
    print(f"\n--- Processing {tag.upper()} files ---")
    summary_data = []

    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
             if f.endswith(suffix) and not f.startswith('~$')]

    for f in files:
        result = process_file(f, tag)
        if result:
            summary_data.append(result)

    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(os.path.join(output_dir, f'{tag}_summary_comparison.csv'), index=False)

# ========== 合并图绘制（三轴） ==========
all_keys = set.intersection(*[set(d.keys()) for d in file_data.values()])

for key in all_keys:
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xscale('log')
    ax1.set_xlabel('Frequency (Hz, log scale)')
    ax1.set_ylabel('IG Attribution', color='green')
    l1 = ax1.plot(freq_values, file_data['ig'][key], 's-', label='IG', color='green')
    ax1.axhline(0, linestyle='--', color='green', linewidth=1.0, alpha=0.4)
    ax1.tick_params(axis='y', labelcolor='green')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Saliency Attribution', color='red')
    l2 = ax2.plot(freq_values, file_data['saliency'][key], 'o-', label='Saliency', color='red')
    ax2.axhline(0, linestyle='--', color='red', linewidth=1.0, alpha=0.4)
    ax2.tick_params(axis='y', labelcolor='red')

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))  # move third y-axis out
    ax3.set_frame_on(True)
    ax3.patch.set_visible(False)
    for sp in ax3.spines.values():
        sp.set_visible(False)
    ax3.spines["right"].set_visible(True)
    ax3.set_ylabel('Attention Attribution', color='blue')
    l3 = ax3.plot(freq_values, file_data['attention'][key], '^-', label='Attention', color='blue')
    ax3.axhline(0, linestyle='--', color='blue', linewidth=1.0, alpha=0.4)
    ax3.tick_params(axis='y', labelcolor='blue')

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=True)
    fig.suptitle(f"Combined Attribution Plot with Three Y Axes\n{key}")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    plt.savefig(os.path.join(output_dir, f"{key}_combined_lineplot_three_axis.png"), dpi=300)
    plt.close()
