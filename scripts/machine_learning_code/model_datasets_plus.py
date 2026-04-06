import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from model_datasets import label_mapping


def _parse_tp_list(tp_cfg) -> List[int]:
    if isinstance(tp_cfg, int):
        out = [int(tp_cfg)]
    elif isinstance(tp_cfg, (list, tuple)):
        out = [int(x) for x in tp_cfg]
    else:
        raise ValueError(
            f"data.num_time_points 必须是 int 或 list[int]，当前为: {type(tp_cfg)}"
        )
    out = sorted(set(out))
    for x in out:
        if x not in (1, 2, 3, 4, 5, 6):
            raise ValueError(f"仅支持 num_time_points in [1,2,3,4,5,6]，当前包含: {x}")
    return out


_WINDOW_REGEX = re.compile(
    r"^(?P<prefix>.*?_\[)(?P<times>\d+(?:\s*,\s*\d+)*)\]\s*(?:\.\w+)?$"
)


def _parse_window_times_from_name(fname: str) -> Optional[Tuple[str, List[int]]]:
    m = _WINDOW_REGEX.match(fname.strip())
    if not m:
        return None
    prefix = m.group("prefix")
    times = [int(x.strip()) for x in m.group("times").split(",")]
    return prefix, times


def _extract_frames_by_filename_times(df: pd.DataFrame, filename_times: List[int]):
    """
    文件名时间为真值：
    - df 里的 Time(h) 当作槽位（常见为 0/2/4/6）
    - 按槽位排序后映射到 filename_times
    """
    if "Time(h)" not in df.columns:
        return None
    slot_values = sorted(df["Time(h)"].dropna().unique().tolist())
    if len(slot_values) != len(filename_times):
        return None

    out = {}
    for i, slot_t in enumerate(slot_values):
        abs_t = int(filename_times[i])
        t_df = df[df["Time(h)"] == slot_t].copy()
        if "Freq" in t_df.columns:
            t_df = t_df.sort_values(by="Freq", ascending=False).reset_index(drop=True)
        t_df["Time(h)"] = abs_t
        out[abs_t] = t_df
    return out


class FilenameWindowDatasetPlus(Dataset):
    """
    单一 num_time_points 的文件名驱动数据集：
    - 仅使用文件名 [] 中恰好 num_time_points 个数字的样本
    - 按文件名时间映射槽位，不再依赖 df 内 Time(h) 的绝对值
    """

    def __init__(
        self,
        data_folder,
        stats_file=None,
        save_stats=True,
        num_time_points=4,
        num_freq_points=63,
        exclude_fnames=None,
    ):
        self.data = []
        self.labels = []
        self.env_params = []
        self.true_voltages = []
        self.concentrations = []
        self.file_names = []
        self.window_times = []
        self.exclude_fnames = set(exclude_fnames or [])
        self.num_time_points = int(num_time_points)
        self.num_freq_points = int(num_freq_points)

        all_voltages = []
        all_true_voltages = []
        all_impedances_mag_frames = []
        all_impedances_phase_frames = []

        for file in os.listdir(data_folder):
            if not file.endswith(".xlsx"):
                continue
            if file in self.exclude_fnames:
                continue

            parsed = _parse_window_times_from_name(file)
            if parsed is None:
                continue
            _, fname_times = parsed
            if len(fname_times) != self.num_time_points:
                continue

            file_path = os.path.join(data_folder, file)
            try:
                df = pd.read_excel(file_path)
            except Exception:
                continue

            if "Label" not in df.columns or df["Label"].empty:
                continue
            label = df["Label"].values[0]
            if label == "Al3+_ion":
                continue

            if "ppm" in df.columns:
                try:
                    concentration = float(df["ppm"].iloc[0])
                except Exception:
                    concentration = -1.0
            else:
                concentration = -1.0

            frame_map = _extract_frames_by_filename_times(df, fname_times)
            if frame_map is None:
                continue

            volt_data_list, impe_data_list = [], []
            ok = True
            for t in fname_times:
                t_df = frame_map.get(int(t), None)
                if t_df is None or t_df.empty:
                    ok = False
                    break

                voltage = t_df["mean_voltage"].values[0]
                impedance_np = t_df[["Zreal", "Zimag"]].values
                if impedance_np.shape[0] < self.num_freq_points:
                    ok = False
                    break

                truncated_real = impedance_np[: self.num_freq_points, 0]
                truncated_imag = impedance_np[: self.num_freq_points, 1]
                z_complex = truncated_real + 1j * truncated_imag
                z_mag = np.log1p(np.abs(z_complex))
                z_phase = np.angle(z_complex)
                z_phase = (z_phase + np.pi) / (2 * np.pi)

                all_impedances_mag_frames.append(z_mag)
                all_impedances_phase_frames.append(z_phase)
                all_voltages.append(voltage)

                impedance_processed = np.stack((z_mag, z_phase), axis=1)
                volt_data_list.append(torch.tensor([voltage], dtype=torch.float32))
                impe_data_list.append(torch.tensor(impedance_processed, dtype=torch.float32))

            if not ok:
                continue

            try:
                volt_tensor = torch.stack(volt_data_list)  # (T,1)
                impe_tensor = torch.stack(impe_data_list)  # (T,F,2)
            except RuntimeError:
                continue

            if not all(col in df.columns and not df[col].empty for col in ["current", "temperature", "flow"]):
                continue

            env_param = torch.tensor(
                [df["temperature"].mean(), df["flow"].mean(), df["current"].mean()],
                dtype=torch.float32,
            )

            if label not in label_mapping:
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
            self.window_times.append(list(fname_times))

        if len(self.data) == 0:
            self.volt_min = 0.0
            self.volt_max = 1.0
            self.impe_mag_mean = torch.zeros(self.num_freq_points, dtype=torch.float32)
            self.impe_mag_std = torch.ones(self.num_freq_points, dtype=torch.float32)
            self.impe_phase_mean = torch.zeros(self.num_freq_points, dtype=torch.float32)
            self.impe_phase_std = torch.ones(self.num_freq_points, dtype=torch.float32)
            self.true_volt_min = 0.0
            self.true_volt_max = 1.0
            return

        if stats_file and os.path.exists(stats_file) and not save_stats:
            with open(stats_file, "r", encoding="utf-8") as f:
                stats = json.load(f)
            self.volt_min = stats["volt_min"]
            self.volt_max = stats["volt_max"]
            self.impe_mag_mean = torch.tensor(stats["impe_mag_mean"], dtype=torch.float32)
            self.impe_mag_std = torch.tensor(stats["impe_mag_std"], dtype=torch.float32)
            self.impe_phase_mean = torch.tensor(stats["impe_phase_mean"], dtype=torch.float32)
            self.impe_phase_std = torch.tensor(stats["impe_phase_std"], dtype=torch.float32)
        else:
            self.volt_min = min(all_voltages) if all_voltages else 0.0
            self.volt_max = max(all_voltages) if all_voltages else 1.0

            mag_array = np.stack(all_impedances_mag_frames, axis=0)
            phase_array = np.stack(all_impedances_phase_frames, axis=0)
            mag_mean = mag_array.mean(axis=0)
            mag_std = mag_array.std(axis=0)
            phase_mean = phase_array.mean(axis=0)
            phase_std = phase_array.std(axis=0)
            mag_std[mag_std < 1e-8] = 1e-8
            phase_std[phase_std < 1e-8] = 1e-8

            self.impe_mag_mean = torch.tensor(mag_mean, dtype=torch.float32)
            self.impe_mag_std = torch.tensor(mag_std, dtype=torch.float32)
            self.impe_phase_mean = torch.tensor(phase_mean, dtype=torch.float32)
            self.impe_phase_std = torch.tensor(phase_std, dtype=torch.float32)

            if save_stats and stats_file:
                stats = {
                    "volt_min": float(self.volt_min),
                    "volt_max": float(self.volt_max),
                    "impe_mag_mean": mag_mean.tolist(),
                    "impe_mag_std": mag_std.tolist(),
                    "impe_phase_mean": phase_mean.tolist(),
                    "impe_phase_std": phase_std.tolist(),
                }
                with open(stats_file, "w", encoding="utf-8") as f:
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
        times = self.window_times[idx]
        file_name_lower = file_name.lower()

        contains_ion_column = "ion_column" in file_name_lower
        # early stage 仅用于非 ion_column 样本：窗口最后一个时间点 <= 6h
        max_t = max(times) if len(times) > 0 else 10**9
        if contains_ion_column:
            # ion_column 不适用 early-stage 概念
            use_conc_flag = torch.tensor(True, dtype=torch.bool)
            stage_id = torch.tensor(0, dtype=torch.long)
        else:
            is_early_stage = (max_t <= 6)
            use_conc_flag = torch.tensor(is_early_stage, dtype=torch.bool)
            stage_id = torch.tensor(1 if is_early_stage else 0, dtype=torch.long)

        if "新版电解槽" in file_name:
            electrolyzer_parameters = torch.tensor(
                [0.012, 0.012, 0.002, 135e-6, 2.38e6, 2.38e6, 5.96e7, 4],
                dtype=torch.float32,
            )
        elif "旧版电解槽" in file_name:
            electrolyzer_parameters = torch.tensor(
                [0.012, 0.012, 0.002, 135e-6, 2.38e6, 2.38e6, 5.96e7, 4],
                dtype=torch.float32,
            )
        else:
            electrolyzer_parameters = torch.tensor(
                [0.012, 0.012, 0.002, 135e-6, 2.38e6, 2.38e6, 5.96e7, 4],
                dtype=torch.float32,
            )

        volt_data = (volt_data - self.volt_min) / max(self.volt_max - self.volt_min, 1e-8)
        mag = impe_data[..., 0]
        phase = impe_data[..., 1]
        mag = (mag - self.impe_mag_mean) / self.impe_mag_std
        phase = (phase - self.impe_phase_mean) / self.impe_phase_std
        impe_data = torch.stack((mag, phase), dim=-1)

        return (
            volt_data,
            impe_data,
            env_param,
            label,
            true_voltage,
            electrolyzer_parameters,
            concentration,
            use_conc_flag,
            stage_id,
        )


class MultiTimePointsDatasetPlus(Dataset):
    """
    将同一目录样本按多个 num_time_points 复用，支持同一次训练混合 1~6 时间点。

    设计要点：
    - 采用文件名驱动读取逻辑（文件名时间为真值）。
    - 每个 tp 单独构建子数据集，再把样本索引合并成一个统一视图。
    - 为 pair 逻辑暴露 get_pair_meta()，使配对时可按 tp 分组。
    """

    def __init__(
        self,
        data_folder,
        stats_file: Optional[str] = None,
        save_stats: bool = True,
        num_time_points=4,
        num_freq_points: int = 63,
        exclude_fnames=None,
    ):
        self.tp_options: List[int] = _parse_tp_list(num_time_points)
        self.exclude_fnames = set(exclude_fnames or [])

        # 先用最大的 tp 生成（或刷新）统计量，再让其他 tp 复用该统计量
        self._datasets_by_tp: Dict[int, Dataset_2_Stable_plus] = {}
        max_tp = max(self.tp_options)
        _ = FilenameWindowDatasetPlus(
            data_folder=data_folder,
            stats_file=stats_file,
            save_stats=save_stats,
            num_time_points=max_tp,
            num_freq_points=num_freq_points,
            exclude_fnames=self.exclude_fnames,
        )

        for tp in self.tp_options:
            ds = FilenameWindowDatasetPlus(
                data_folder=data_folder,
                stats_file=stats_file,
                save_stats=False,
                num_time_points=tp,
                num_freq_points=num_freq_points,
                exclude_fnames=self.exclude_fnames,
            )
            self._datasets_by_tp[tp] = ds

        # 展平全局索引
        self._global_index = []
        self.file_names: List[str] = []
        for tp in self.tp_options:
            ds = self._datasets_by_tp[tp]
            for local_idx, fname in enumerate(ds.file_names):
                self._global_index.append(
                    {"tp": tp, "local_idx": local_idx, "fname": fname}
                )
                self.file_names.append(f"{fname}__tp{tp}")

    def __len__(self):
        return len(self._global_index)

    def __getitem__(self, idx):
        item = self._global_index[idx]
        tp = item["tp"]
        local_idx = item["local_idx"]
        sample = self._datasets_by_tp[tp][local_idx]
        # 在末尾附加一个 tp 字段，不影响旧 trainer 读取前 9 列
        return (*sample, tp)

    def get_pair_meta(self, idx: int) -> Dict:
        """
        返回配对所需元信息：
        - prefix: 文件名前缀（到 '_['）
        - times: 方括号内时间窗口
        - tp: 该样本训练时使用的 num_time_points
        """
        item = self._global_index[idx]
        fname = item["fname"]
        tp = int(item["tp"])

        m = _WINDOW_REGEX.match(fname.strip())
        if not m:
            return {
                "fname": fname,
                "prefix": None,
                "times": None,
                "tp": tp,
            }
        times = [int(x.strip()) for x in m.group("times").split(",")]
        return {
            "fname": fname,
            "prefix": m.group("prefix"),
            "times": times,
            "tp": tp,
        }
