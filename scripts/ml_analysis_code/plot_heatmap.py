import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os



# 设置 seaborn 风格
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# ====== 配置部分 ======
use_directory = True  # 是否使用目录扫描模式（True = 使用文件夹，False = 用下面的手动列表）
input_dir = '/home/cagalii/Application/train_machine_learning/inference_results_freq_and_time_embedding_all_freq'  # 如果 use_directory 为 True，这里指定你的文件夹路径
manual_file_list = [ # 要处理的 Excel 文件列表（修改为你的实际路径
    '/home/cagalii/Application/train_machine_learning/inference_results/20240822_10ppm铜离子污染测试_ion_gamry_[0, 2, 4, 6]_attention_values.xlsx',
    '/home/cagalii/Application/train_machine_learning/inference_results/20240822_10ppm铜离子污染测试_ion_gamry_[2, 4, 6, 8]_attention_values.xlsx',
    '/home/cagalii/Application/train_machine_learning/inference_results/20240822_10ppm铜离子污染测试_ion_gamry_[4, 6, 8, 10]_attention_values.xlsx',

    '/home/cagalii/Application/train_machine_learning/inference_results/20240823_10ppm钙离子污染和恢复测试_ion_gamry_[0, 2, 4, 6]_attention_values.xlsx',
    '/home/cagalii/Application/train_machine_learning/inference_results/20240823_10ppm钙离子污染和恢复测试_ion_gamry_[2, 4, 6, 8]_attention_values.xlsx',
    '/home/cagalii/Application/train_machine_learning/inference_results/20240823_10ppm钙离子污染和恢复测试_ion_gamry_[4, 6, 8, 10]_attention_values.xlsx',

    '/home/cagalii/Application/train_machine_learning/inference_results/20240831_10ppm镍离子污染测试_ion_gamry_[0, 2, 4, 6]_attention_values.xlsx',
    '/home/cagalii/Application/train_machine_learning/inference_results/20240831_10ppm镍离子污染测试_ion_gamry_[2, 4, 6, 8]_attention_values.xlsx',
    '/home/cagalii/Application/train_machine_learning/inference_results/20240831_10ppm镍离子污染测试_ion_gamry_[4, 6, 8, 10]_attention_values.xlsx',
    
    '/home/cagalii/Application/train_machine_learning/inference_results/20240907_10ppm铁离子污染测试_ion_gamry_[0, 2, 4, 6]_attention_values.xlsx',
    '/home/cagalii/Application/train_machine_learning/inference_results/20240907_10ppm铁离子污染测试_ion_gamry_[2, 4, 6, 8]_attention_values.xlsx',
    '/home/cagalii/Application/train_machine_learning/inference_results/20240907_10ppm铁离子污染测试_ion_gamry_[4, 6, 8, 10]_attention_values.xlsx',
    
    '/home/cagalii/Application/train_machine_learning/inference_results/20240910_10ppm钙离子污染测试_ion_gamry_[0, 2, 4, 6]_attention_values.xlsx',
    '/home/cagalii/Application/train_machine_learning/inference_results/20240910_10ppm钙离子污染测试_ion_gamry_[2, 4, 6, 8]_attention_values.xlsx',
    '/home/cagalii/Application/train_machine_learning/inference_results/20240910_10ppm钙离子污染测试_ion_gamry_[4, 6, 8, 10]_attention_values.xlsx',
    
    '/home/cagalii/Application/train_machine_learning/inference_results/20240915_2ppm铜离子污染测试_ion_gamry_[0, 2, 4, 6]_attention_values.xlsx',
    '/home/cagalii/Application/train_machine_learning/inference_results/20240915_2ppm铜离子污染测试_ion_gamry_[2, 4, 6, 8]_attention_values.xlsx',
    '/home/cagalii/Application/train_machine_learning/inference_results/20240915_2ppm铜离子污染测试_ion_gamry_[4, 6, 8, 10]_attention_values.xlsx',

    '/home/cagalii/Application/train_machine_learning/inference_results/20240918_2ppm钙离子污染测试_ion_gamry_[0, 2, 4, 6]_attention_values.xlsx',
    '/home/cagalii/Application/train_machine_learning/inference_results/20240918_2ppm钙离子污染测试_ion_gamry_[2, 4, 6, 8]_attention_values.xlsx',
    '/home/cagalii/Application/train_machine_learning/inference_results/20240918_2ppm钙离子污染测试_ion_gamry_[4, 6, 8, 10]_attention_values.xlsx',

    '/home/cagalii/Application/train_machine_learning/inference_results/20241001_2ppm铁离子污染测试_ion_gamry_[0, 2, 4, 6]_attention_values.xlsx',
    '/home/cagalii/Application/train_machine_learning/inference_results/20241001_2ppm铁离子污染测试_ion_gamry_[2, 4, 6, 8]_attention_values.xlsx',
    '/home/cagalii/Application/train_machine_learning/inference_results/20241001_2ppm铁离子污染测试_ion_gamry_[4, 6, 8, 10]_attention_values.xlsx',

    '/home/cagalii/Application/train_machine_learning/inference_results/20241003_2ppm镍离子污染测试_ion_gamry_[0, 2, 4, 6]_attention_values.xlsx',
    '/home/cagalii/Application/train_machine_learning/inference_results/20241003_2ppm镍离子污染测试_ion_gamry_[2, 4, 6, 8]_attention_values.xlsx',
    '/home/cagalii/Application/train_machine_learning/inference_results/20241003_2ppm镍离子污染测试_ion_gamry_[4, 6, 8, 10]_attention_values.xlsx',

    '/home/cagalii/Application/train_machine_learning/inference_results/20241006_2ppm铬离子污染测试_ion_firecloud_[0, 2, 4, 6]_attention_values.xlsx',
    '/home/cagalii/Application/train_machine_learning/inference_results/20241006_2ppm铬离子污染测试_ion_firecloud_[2, 4, 6, 8]_attention_values.xlsx',
    '/home/cagalii/Application/train_machine_learning/inference_results/20241006_2ppm铬离子污染测试_ion_firecloud_[4, 6, 8, 10]_attention_values.xlsx',

    # '/home/cagalii/Application/train_machine_learning/inference_results/20241006_2ppm铬离子污染测试_ion_gamry_[0, 2, 4, 6]_attention_values.xlsx',
    # '/home/cagalii/Application/train_machine_learning/inference_results/20241006_2ppm铬离子污染测试_ion_gamry_[2, 4, 6, 8]_attention_values.xlsx',
    # '/home/cagalii/Application/train_machine_learning/inference_results/20241006_2ppm铬离子污染测试_ion_gamry_[4, 6, 8, 10]_attention_values.xlsx',

    #     '/home/cagalii/Application/train_machine_learning/inference_results/20241003_2ppm镍离子污染测试_ion_gamry_[0, 2, 4, 6]_attention_values.xlsx',
    # '/home/cagalii/Application/train_machine_learning/inference_results/20241003_2ppm镍离子污染测试_ion_gamry_[2, 4, 6, 8]_attention_values.xlsx',
    # '/home/cagalii/Application/train_machine_learning/inference_results/20241003_2ppm镍离子污染测试_ion_gamry_[4, 6, 8, 10]_attention_values.xlsx',

    #     '/home/cagalii/Application/train_machine_learning/inference_results/20241003_2ppm镍离子污染测试_ion_gamry_[0, 2, 4, 6]_attention_values.xlsx',
    # '/home/cagalii/Application/train_machine_learning/inference_results/20241003_2ppm镍离子污染测试_ion_gamry_[2, 4, 6, 8]_attention_values.xlsx',
    # '/home/cagalii/Application/train_machine_learning/inference_results/20241003_2ppm镍离子污染测试_ion_gamry_[4, 6, 8, 10]_attention_values.xlsx',

    #     '/home/cagalii/Application/train_machine_learning/inference_results/20241003_2ppm镍离子污染测试_ion_gamry_[0, 2, 4, 6]_attention_values.xlsx',
    # '/home/cagalii/Application/train_machine_learning/inference_results/20241003_2ppm镍离子污染测试_ion_gamry_[2, 4, 6, 8]_attention_values.xlsx',
    # '/home/cagalii/Application/train_machine_learning/inference_results/20241003_2ppm镍离子污染测试_ion_gamry_[4, 6, 8, 10]_attention_values.xlsx',

    #     '/home/cagalii/Application/train_machine_learning/inference_results/20241003_2ppm镍离子污染测试_ion_gamry_[0, 2, 4, 6]_attention_values.xlsx',
    # '/home/cagalii/Application/train_machine_learning/inference_results/20241003_2ppm镍离子污染测试_ion_gamry_[2, 4, 6, 8]_attention_values.xlsx',
    # '/home/cagalii/Application/train_machine_learning/inference_results/20241003_2ppm镍离子污染测试_ion_gamry_[4, 6, 8, 10]_attention_values.xlsx',

    #     '/home/cagalii/Application/train_machine_learning/inference_results/20241003_2ppm镍离子污染测试_ion_gamry_[0, 2, 4, 6]_attention_values.xlsx',
    # '/home/cagalii/Application/train_machine_learning/inference_results/20241003_2ppm镍离子污染测试_ion_gamry_[2, 4, 6, 8]_attention_values.xlsx',
    # '/home/cagalii/Application/train_machine_learning/inference_results/20241003_2ppm镍离子污染测试_ion_gamry_[4, 6, 8, 10]_attention_values.xlsx',
]


# 创建输出文件夹
output_dir = 'output_charts_freq_and_time_embedding_all_freq'
os.makedirs(output_dir, exist_ok=True)

# 获取文件列表
if use_directory:
    excel_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                   if f.endswith('_attention_values.xlsx') and not f.startswith('~$')]
else:
    excel_files = manual_file_list

# ====== 主循环处理每个文件 ======
for file_path in excel_files:
    try:
        # 读取数据
        df = pd.read_excel(file_path, header=0, engine='openpyxl')
        last_row = df.iloc[-1]

        # 1. 电压值：前4列求和
        voltage_sum = last_row.iloc[0:4].sum()

        # 2. num_freq个频率，每个频率跨4个时间点列号
        num_freq=64
        frequency_sums = []
        for i in range(num_freq):
            cols = [4 + i + num_freq * t for t in range(4)]
            values = last_row.iloc[cols]
            frequency_sums.append(values.sum())

        # 3. 汇总数据
        values = [voltage_sum] + frequency_sums
        labels = ['Voltage'] + [f'F{i+1}' for i in range(num_freq)]

        # 4. 画图
        plt.figure(figsize=(18, 6))
        sns.barplot(x=labels, y=values)
        plt.xticks(rotation=90)
        plt.ylabel('Summed Value')
        plt.title(f"Voltage & Frequency Magnitude - {os.path.basename(file_path)}")
        plt.tight_layout()

        # 保存图像
        chart_name = os.path.splitext(os.path.basename(file_path))[0] + '_chart.png'
        chart_path = os.path.join(output_dir, chart_name)
        plt.savefig(chart_path, dpi=300)
        plt.close()
        print(f"[Saved Chart] {chart_path}")

        # 5. 保存为 CSV 文件
        csv_name = os.path.splitext(os.path.basename(file_path))[0] + '_data.csv'
        csv_path = os.path.join(output_dir, csv_name)
        df_csv = pd.DataFrame([values], columns=labels)
        df_csv.to_csv(csv_path, index=False)
        print(f"[Saved CSV]   {csv_path}")

    except Exception as e:
        print(f"[Error] Processing {file_path}: {e}")

