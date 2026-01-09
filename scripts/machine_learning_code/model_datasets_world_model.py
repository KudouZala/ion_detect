import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt
import math
from pathlib import Path
from captum.attr import Saliency, IntegratedGradients
import re
import json
# 获取当前文件的上一层目录
base_dir = Path(__file__).resolve().parent.parent

# 拼接 JSON 文件路径
json_path = base_dir / "machine_learning_code" / "label_mapping.json"

# 读取 JSON 文件
with open(json_path, "r", encoding="utf-8") as f:
    label_mapping = json.load(f)

class Dataset_2_Stable_plus(Dataset):
    def __init__(self, data_folder, stats_file=None, save_stats=True, num_time_points=4, exclude_fnames=None):
        self.data = []
        self.labels = []
        self.env_params = []
        self.true_voltages = []     #存储电压
        self.concentrations = []   # 存储浓度
        self.file_names = []       # 存储文件名
        self.exclude_fnames = set(exclude_fnames or [])


        all_voltages = []
        all_true_voltages = []

        #  用于按频率统计阻抗的 mean/std：
        #   每次 append 的元素形状为 (63,)；最后 stack 成 (N_frames, 63)
        all_impedances_mag_frames = []
        all_impedances_phase_frames = []
        
        # =============== 1️ 读取数据文件 ===============
        for file in os.listdir(data_folder):
            if not file.endswith('.xlsx'):
                continue
                        # ✅ 逻辑剔除：训练集跳过测试集文件
            if file in self.exclude_fnames:
                continue

            file_path = os.path.join(data_folder, file)
            df = pd.read_excel(file_path)

            if 'Label' not in df.columns or df['Label'].empty:
                print(f"Skipping file {file}: missing Label.")
                continue
            label = df['Label'].values[0]
            if label == 'Al3+_ion':
                print("Al3+_ion,skip")
                continue

            # ✅ 提取浓度值：优先从 df['ppm'] 读取
            if 'ppm' in df.columns:
                try:
                    concentration = float(df['ppm'].iloc[0])
                except Exception as e:
                    print(f"Warning: Failed to parse ppm from df in {file}, set to -1. Error: {e}")
                    concentration = -1.0
            else:
                print(f"Warning: No ppm column in {file}, set to -1")
                concentration = -1.0
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

            volt_data_list, impe_data_list = [], []

            # 逐时间点提取数据
            for t in time_points:
                time_data = df[df['Time(h)'] == t]
                if time_data.empty:
                    print(f"Error: Missing {t}h in {file}. Skipping file.")
                    volt_data_list, impe_data_list = [], []
                    break

                # ✅ 保证按频率排序，和模型中的 freq_values 对齐（改为：高→低）
                if 'Freq' in time_data.columns:
                    time_data = time_data.sort_values(by='Freq', ascending=False).reset_index(drop=True)
                else:
                    raise ValueError(f"Missing column 'Freq' in {file} at t={t}h.")


                voltage = time_data['mean_voltage'].values[0]
                impedance_np = time_data[['Zreal', 'Zimag']].values

                # ✅ 检查是否有 63 个点，不足则跳过该文件
                if impedance_np.shape[0] < 63:
                    print(f"Skipping file {file}: impedance points {impedance_np.shape[0]} < 63")
                    volt_data_list, impe_data_list = [], []
                    break

                truncated_real = impedance_np[:63, 0]
                truncated_imag = impedance_np[:63, 1]

                # 复数形式阻抗
                z_complex = truncated_real + 1j * truncated_imag

                # ================= 阻抗预处理 =================
                # |Z| 做 log1p，phase 映射到 [0,1]
                z_mag = np.log1p(np.abs(z_complex))  # (63,)
                z_phase = np.angle(z_complex)
                z_phase = (z_phase + np.pi) / (2 * np.pi)  # [-pi,pi]→[0,1]
                # =================================================

                # 👉 在这里先累积“原始（已做 log/phase 映射，但未归一化）”阻抗，用于后面按频率统计
                all_impedances_mag_frames.append(z_mag)      # (63,)
                all_impedances_phase_frames.append(z_phase)  # (63,)

                all_voltages.append(voltage)

                impedance_processed = np.stack((z_mag, z_phase), axis=1)  # (63, 2)
                volt_data_list.append(torch.tensor([voltage], dtype=torch.float32))
                impe_data_list.append(torch.tensor(impedance_processed, dtype=torch.float32))

            if not volt_data_list or not impe_data_list:
                continue

            try:
                volt_tensor = torch.stack(volt_data_list)   # (T, 1)
                impe_tensor = torch.stack(impe_data_list)   # (T, 63, 2)
            except RuntimeError as e:
                print(f"Error stacking {file}: {e}")
                continue

            if not all(col in df.columns and not df[col].empty for col in ['current', 'temperature', 'flow']):
                print(f"Skipping file {file}: missing env params.")
                continue

            env_param = torch.tensor(
                [df['temperature'].mean(),
                 df['flow'].mean(),
                 df['current'].mean()],
                dtype=torch.float32
            )

            if label not in label_mapping:
                print(f"Warning: Label '{label}' not in label_mapping. Skipped.")
                continue
            label_idx = torch.tensor(label_mapping[label], dtype=torch.long)

            true_voltage_val = volt_tensor[-1].unsqueeze(0)
            all_true_voltages.append(true_voltage_val.item())

            self.data.append((volt_tensor, impe_tensor))
            self.labels.append(label_idx)
            self.env_params.append(env_param)
            self.true_voltages.append(true_voltage_val)
            self.concentrations.append(torch.tensor([concentration], dtype=torch.float32))
            self.file_names.append(file)

        # 如果没有任何数据
        if len(self.data) == 0:
            print("数据为空，请注意")
            return

        # =============== 2️⃣ 加载或计算统计参数 ===============
        if stats_file and os.path.exists(stats_file) and not save_stats:
            # ✅ 从已有 stats_file 读取训练集统计量（推荐在 val/test 阶段使用）
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            self.volt_min = stats["volt_min"]
            self.volt_max = stats["volt_max"]

            # 阻抗按频率的 mean/std（列表 -> tensor）
            self.impe_mag_mean = torch.tensor(stats["impe_mag_mean"], dtype=torch.float32)
            self.impe_mag_std = torch.tensor(stats["impe_mag_std"], dtype=torch.float32)
            self.impe_phase_mean = torch.tensor(stats["impe_phase_mean"], dtype=torch.float32)
            self.impe_phase_std = torch.tensor(stats["impe_phase_std"], dtype=torch.float32)
        else:
            self.volt_min = min(all_voltages) if all_voltages else 0.0
            self.volt_max = max(all_voltages) if all_voltages else 1.0

            # ✅ 把 all_impedances_* 累积的每一帧 (63,) 堆成 (N_frames, 63)
            mag_array = np.stack(all_impedances_mag_frames, axis=0)   # (N_frames, 63)
            phase_array = np.stack(all_impedances_phase_frames, axis=0)

            mag_mean = mag_array.mean(axis=0)        # (63,)
            mag_std = mag_array.std(axis=0)          # (63,)
            phase_mean = phase_array.mean(axis=0)    # (63,)
            phase_std = phase_array.std(axis=0)      # (63,)

            # 防止除 0
            mag_std[mag_std < 1e-8] = 1e-8
            phase_std[phase_std < 1e-8] = 1e-8

            self.impe_mag_mean = torch.tensor(mag_mean, dtype=torch.float32)
            self.impe_mag_std = torch.tensor(mag_std, dtype=torch.float32)
            self.impe_phase_mean = torch.tensor(phase_mean, dtype=torch.float32)
            self.impe_phase_std = torch.tensor(phase_std, dtype=torch.float32)

            # ✅ 保存到 stats_file，便于 val/test 复用
            if save_stats and stats_file:
                stats = {
                    "volt_min": float(self.volt_min),
                    "volt_max": float(self.volt_max),
                    "impe_mag_mean": mag_mean.tolist(),
                    "impe_mag_std": mag_std.tolist(),
                    "impe_phase_mean": phase_mean.tolist(),
                    "impe_phase_std": phase_std.tolist(),
                }
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=2)

        self.true_volt_min = min(all_true_voltages) if all_true_voltages else 0.0
        self.true_volt_max = max(all_true_voltages) if all_true_voltages else 1.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        volt_data, impe_data = self.data[idx]
        env_param = self.env_params[idx]
        label = self.labels[idx]
        true_voltage = self.true_voltages[idx]
        concentration = self.concentrations[idx]
        file_name = self.file_names[idx]
        file_name_lower = file_name.lower() # 转换为小写，便于统一检查

        # --------------------------------------------------------------------------
        # 修改点 2: 核心逻辑 - 计算 use_conc_flag
        # --------------------------------------------------------------------------
        
        # 1. 检查文件名是否包含 "ion_column"
        contains_ion_column = "ion_column" in file_name_lower
        
        # 2. 检查文件名是否包含特定的时间点字符串
        # 之前的 valid_time_points_1/2 列表不再使用
        
        # 时序点满足要求 (T_valid)
        is_valid_points = (
            "_[0, 2, 4, 6].xlsx" in file_name_lower or 
            "_[2, 4, 6, 8].xlsx" in file_name_lower
        )

        # 3. 整合新逻辑: (包含 "ion_column") OR (包含有效时间点字符串)
        # 只要满足任一条件，ppm就生效
        use_conc_flag = torch.tensor(
            contains_ion_column or is_valid_points,
            dtype=torch.bool
        )

        rapid_patterns = [
            "_[0, 2, 4, 6].xlsx",
            "_[2, 4, 6, 8].xlsx",
        ]
        is_rapid_stage = any(((p in file_name_lower) and ("ion_column" not in file_name_lower)) for p in rapid_patterns)
        stage_id = torch.tensor(1 if is_rapid_stage else 0, dtype=torch.long)
        # --------------------------------------------------------------------------

        # ✅ 根据文件名判断新版/旧版电解槽参数（保持你原来的逻辑）
        if "新版电解槽" in file_name:
            electrolyzer_parameters = torch.tensor([
                0.012, 0.012, 0.002, 135e-6,
                2.38e6, 2.38e6, 5.96e7, 4
            ], dtype=torch.float32)
        elif "旧版电解槽" in file_name:  # 目前两者一样
            electrolyzer_parameters = torch.tensor([
                0.012, 0.012, 0.002, 135e-6,
                2.38e6, 2.38e6, 5.96e7, 4
            ], dtype=torch.float32)
        else:
            # 默认值（可按需调整）
            electrolyzer_parameters = torch.tensor([
                0.012, 0.012, 0.002, 135e-6,
                2.38e6, 2.38e6, 5.96e7, 4
            ], dtype=torch.float32)

        # 1️⃣ 归一化电压（仍然是全局 min-max）
        volt_data = (volt_data - self.volt_min) / max(self.volt_max - self.volt_min, 1e-8)

        # 2️⃣ 阻抗按频率做 z-score 标准化
        # impe_data: (T=4, F=63, 2)
        mag = impe_data[..., 0]    # (T, F)
        phase = impe_data[..., 1]  # (T, F)

        # self.impe_*_mean/std: (F,)，通过广播作用在最后一维
        mag = (mag - self.impe_mag_mean) / self.impe_mag_std
        phase = (phase - self.impe_phase_mean) / self.impe_phase_std

        impe_data = torch.stack((mag, phase), dim=-1)  # (T, F, 2)

        # 5️⃣ ---- 返回处理后的数据 ----
        return (
            volt_data,
            impe_data,
            env_param,
            label,
            true_voltage,
            electrolyzer_parameters,
            concentration,
            use_conc_flag, # <--- 增加 use_conc_flag 作为第 8 个元素
            stage_id,
        )
