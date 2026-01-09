import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import numpy as np
import pandas as pd
from itertools import combinations

# === 专用于 2-1 的“与 fit_res_analysis.py 一致”的实现 ===

def find_matching_rows_21_simple(df):
    """
    与 fit_res_analysis.py 相同的 row1/row2 匹配规则（仅用于 2-1）。
    返回 (row1_idx, row2_idx) 或 (None, None)。
    """
    col0 = df.iloc[:, 0].astype(str)

    row1_mask = col0.str.contains(
        r"(_ion循环1／1_工步组1\(工步组\)\(1／1\)_工步3\(阻抗\)_greater_than_0)|(_ion_0_大于0)",
        na=False
    )
    row2_mask = col0.str.contains(
        r"(_ion_3_大于0)|(_ion循环1／1_工步组2\(工步组\)\(3／80\)_工步3\(阻抗\)_greater_than_0)",
        na=False
    )

    row1_idx = row1_mask.idxmax() if row1_mask.any() else None
    row2_idx = row2_mask.idxmax() if row2_mask.any() else None
    return (row1_idx, row2_idx) if (row1_idx is not None and row2_idx is not None) else (None, None)


def compute_diff_21_simple(df, row1_idx, row2_idx, columns):
    """
    与 fit_res_analysis.py 相同的 2-1 差分逻辑：
      v1: row1 以及其上一行（若存在）的均值
      v2: 从 row2_idx 往后找与 v1 绝对差 < 1 的值，凑够最多 3 个后取均值
      返回 v2 - v1（按列）
    任一步失败返回 None。
    """
    import numpy as np
    import pandas as pd

    try:
        v1_vals, v2_vals = [], []

        # row1 及其上一行的均值
        pre_indices = [i for i in range(row1_idx - 1, row1_idx + 1) if i >= 0]

        for col in columns:
            # --- v1 ---
            v1_list = []
            for i in pre_indices:
                val = df.iloc[i][col]
                if pd.notna(val):
                    v1_list.append(float(val))
            if len(v1_list) == 0:
                # 与 1.py 一致：若在 row1 及其上一行都取不到，放弃该文件
                return None

            v1 = np.mean(v1_list)
            v1_vals.append(v1)

            # --- v2: 从 row2_idx 往后搜与 v1 |Δ| < 1 的最多 3 个值取均值 ---
            v2_candidates = []
            for i in range(row2_idx, len(df)):
                val = df.iloc[i][col]
                if pd.isna(val):
                    continue
                v2 = float(val)
                if abs(v2 - v1) < 1:
                    v2_candidates.append(v2)
                    if len(v2_candidates) == 3:
                        break

            if len(v2_candidates) == 0:
                return None

            v2_vals.append(np.mean(v2_candidates))

        v1_vals = np.array(v1_vals, dtype=float)
        v2_vals = np.array(v2_vals, dtype=float)
        return v2_vals - v1_vals

    except Exception:
        return None



def fit_res_analysis(root_path, ion_density):
    # Output directory two levels above current script
    current_dir = Path(__file__).resolve()
    output_dir = current_dir.parents[1] / "eis_fit_analysis_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ion type mapping: Chinese name → Symbol
    ion_types = {
        "钠离子": "Na⁺",
        "钙离子": "Ca²⁺",
        "铬离子": "Cr³⁺",
        "镍离子": "Ni²⁺",
        "铜离子": "Cu²⁺",
        "铁离子": "Fe³⁺",
        "铝离子": "Al³⁺"
    }

    # Containers for two diffs: (row2-row1) and (row3-row1)
    ion_data_21 = {symbol: [] for symbol in ion_types.values()}
    ion_data_31 = {symbol: [] for symbol in ion_types.values()}

    # ---------- helpers ----------
    def find_matching_rows(df):
        """Return indices (row1_idx, row2_idx, row3_idx) based on patterns in first column."""
        col0 = df.iloc[:, 0].astype(str)

        # row1
        row1_mask = col0.str.contains(
            r"(_ion_column循环1[/／]1_工步组1\(工步组\)\(3[/／]\d+\)_工步3\(阻抗\)_greater_than_0)"
            r"|(_ion_column循环1[/／]1_工步组2\(工步组\)\(3[/／]\d+\)_工步3\(阻抗\)_greater_than_0)"
            r"|(_ion_column循环1[/／]1_工步组3\(工步组\)\(3[/／]\d+\)_工步3\(阻抗\)_greater_than_0)"
            r"|(_ion_column_循环1[/／]1_工步组1\(工步组\)\(3[/／]\d+\)_工步3\(阻抗\)_greater_than_0)"
            r"|(_ion_column_循环1[/／]1_工步组2\(工步组\)\(3[/／]\d+\)_工步3\(阻抗\)_greater_than_0)"
            r"|(_ion_column_循环1[/／]1_工步组3\(工步组\)\(3[/／]\d+\)_工步3\(阻抗\)_greater_than_0)"
            r"|(_ion_column_3)",
            na=False
        )


        # row2
        row2_mask = col0.str.contains(
            r"(_ion_3_)"
            r"|(_ion循环1[/／]1_工步组2\(工步组\)\(3[/／]\d+\)_工步3\(阻抗\)_greater_than_0)"
            r"|(_ion循环1[/／]1_工步组1\(工步组\)\(3[/／]\d+\)_工步3\(阻抗\)_greater_than_0)",
            r"|(_ion循环1[/／]1_工步组3\(工步组\)\(3[/／]\d+\)_工步3\(阻抗\)_greater_than_0)",
            na=False
        )

        # row3（你的“恢复/renew”两种写法做“或”）
        # row3_mask = col0.str_contains if hasattr(col0, 'str_contains') else col0.str.contains
        row3_mask = col0.str.contains(
            r"(_ion_column_renew_H2SO4_3)"
            r"|(_ion_column_renew_3)"
            r"|(_ion_column_renew循环1[/／]1_工步组2\(工步组\)\(3[/／]\d+\)_工步3\(阻抗\)_greater_than_0)"
            r"|(_ion_column_renew循环1[/／]1_工步组1\(工步组\)\(3[/／]\d+\)_工步3\(阻抗\)_greater_than_0)"
            r"|(_ion_column_renew循环1[/／]1_工步组3\(工步组\)\(3[/／]\d+\)_工步3\(阻抗\)_greater_than_0)",
            na=False
        )

        row1_idx = row1_mask.idxmax() if row1_mask.any() else None
        row2_idx = row2_mask.idxmax() if row2_mask.any() else None
        row3_idx = row3_mask.idxmax() if row3_mask.any() else None
        return row1_idx, row2_idx, row3_idx





    def _pair_mean_from_expanding_window(df, start_idx, col, window_init=5, rel_thresh=0.05):
        """
        扩大窗口版：
        - 起点 fixed = start_idx，初始窗口大小 window_init（若不足则用能取到的行数），
            然后每次把窗口末端扩大 1 行，直到满足条件或到表尾。
        - 每个窗口内先丢弃 >5 与 NaN；若有效值 <2 则继续扩大。
        - 在窗口有效值中找到“最近的一对”（绝对差最小）；若该对相对差 <= rel_thresh(默认5%)，
            立即返回这对的均值。
        - 若到表尾仍未命中 5%，则在“起点到表尾”的整个范围内，取全局最近的一对（仍只用 ≤5）作退化返回其均值。
        - 若全范围内有效值(≤5 且非 NaN)不足两条，则抛出详细 ValueError。
        """
        n_rows = len(df)
        eps = 1e-9

        if start_idx is None:
            raise ValueError(f"[取值失败] 起点行 start_idx 为 None（列：{col}）。")
        if not (0 <= start_idx < n_rows):
            raise ValueError(f"[取值失败] 起点行 start_idx={start_idx} 超出行数范围 [0, {n_rows-1}]（列：{col}）。")

        # 起点行辅助信息（用于任何失败时输出）
        try:
            start_val = df.iloc[start_idx][col]
            start_val_str = "NA" if pd.isna(start_val) else f"{float(start_val):.6g}"
            try:
                first_col_text = str(df.iloc[start_idx, 0])
            except Exception:
                first_col_text = None
        except Exception as e:
            raise ValueError(
                f"[取值失败] 无法读取起点行数据（列：{col}，start_idx={start_idx}）。原因：{e}"
            )

        # 初始窗口终点（允许表尾处初始就 < window_init）
        end_idx = min(n_rows - 1, start_idx + max(1, window_init) - 1)

        # 全局退化候选（在整个扩张过程中维护）
        global_best = None  # (abs_diff, mean_ab, (row_i,row_j), (val_i,val_j))

        # 从初始窗口开始一路扩大到表尾
        while end_idx < n_rows:
            # 收集当前窗口内值，丢弃 >5 和 NaN
            valid_vals, valid_rows = [], []
            raw = []  # (row, val_str, tag) tag∈{"OK",">5","NA"} 仅用于报错展示
            for r in range(start_idx, end_idx + 1):
                v = df.iloc[r][col]
                if pd.isna(v):
                    raw.append((r, "NA", "NA"))
                else:
                    fv = float(v)
                    if fv > 5:
                        raw.append((r, f"{fv:.6g}", ">5"))
                    else:
                        raw.append((r, f"{fv:.6g}", "OK"))
                        valid_vals.append(fv)
                        valid_rows.append(r)

            # 当前窗口内不足两条有效值 → 扩大窗口
            if len(valid_vals) >= 2:
                # 窗口内“最近的一对”
                best_absdiff = None
                best_pair_vals = None
                best_pair_rows = None
                for i in range(len(valid_vals)):
                    a = valid_vals[i]
                    for j in range(i + 1, len(valid_vals)):
                        b = valid_vals[j]
                        ad = abs(a - b)
                        if (best_absdiff is None) or (ad < best_absdiff):
                            best_absdiff = ad
                            best_pair_vals = (a, b)
                            best_pair_rows = (valid_rows[i], valid_rows[j])

                # 更新全局退化候选
                mean_ab = 0.5 * (best_pair_vals[0] + best_pair_vals[1])
                rel = best_absdiff / max(abs(mean_ab), eps)
                if (global_best is None) or (best_absdiff < global_best[0]):
                    global_best = (best_absdiff, mean_ab, best_pair_rows, best_pair_vals)

                # 命中 5% 立即返回
                if rel <= rel_thresh:
                    return mean_ab

            # 扩大 1 行；若已到表尾则跳出循环
            if end_idx == n_rows - 1:
                break
            end_idx += 1

        # 扩到表尾仍未命中 5%：全局退化
        if global_best is not None:
            return global_best[1]

        # 全范围内没有两条有效值（≤5 且非 NaN）
        # 为了便于定位，输出起点→表尾的简要摘要（最多列出前/后的若干行）
        summary = []
        MAX_LIST = 12  # 控制报错信息长度
        all_rows = list(range(start_idx, n_rows))
        head = all_rows[:MAX_LIST//2]
        tail = all_rows[-MAX_LIST//2:] if len(all_rows) > MAX_LIST//2 else []
        show_rows = head + (["..."] if len(all_rows) > MAX_LIST else []) + tail

        for r in show_rows:
            if r == "...":
                summary.append("...")
                continue
            v = df.iloc[r][col]
            if pd.isna(v):
                summary.append((r, "NA", "NA"))
            else:
                fv = float(v)
                if fv > 5:
                    summary.append((r, f"{fv:.6g}", ">5"))
                else:
                    summary.append((r, f"{fv:.6g}", "OK"))

        raise ValueError(
            "[取值失败] 从起点行 {s} 扩大到表尾的任何窗口内，都没有两条有效值(≤5 且非 NaN)（列：{col}）。\n"
            "  - 起点行首列文本: {head}\n"
            "  - 起点行该列值: {sv}\n"
            "  - 起点→表尾(部分)值摘要(行号,值,标记): {sum}".format(
                s=start_idx, col=col, head=first_col_text, sv=start_val_str, sum=summary
            )
        )


    def compute_diff(df, row1_idx, rowx_idx, columns, window_init=5, rel_thresh=0.05):
        """
        对每个列 col：
        v1 = _pair_mean_from_expanding_window(df, row1_idx, col, ...)
        v2 = _pair_mean_from_expanding_window(df, rowx_idx, col, ...)
        返回：
        np.array(v2_vals - v1_vals, dtype=float)
        任一步失败会抛出包含“v1/v2 阶段 + 列名 + 起点信息”的详细 ValueError。
        """
        v1_vals, v2_vals = [], []
        for col in columns:
            try:
                v1 = _pair_mean_from_expanding_window(
                    df, row1_idx, col, window_init=window_init, rel_thresh=rel_thresh
                )
            except Exception as e:
                raise ValueError(f"[v1取值失败] 列：{col}，start_idx={row1_idx}。详情：{e}")

            try:
                v2 = _pair_mean_from_expanding_window(
                    df, rowx_idx, col, window_init=window_init, rel_thresh=rel_thresh
                )
            except Exception as e:
                raise ValueError(f"[v2取值失败] 列：{col}，start_idx={rowx_idx}。详情：{e}")

            v1_vals.append(v1)
            v2_vals.append(v2)

        v1_vals = np.array(v1_vals, dtype=float)
        v2_vals = np.array(v2_vals, dtype=float)
        return v2_vals - v1_vals




    def group_and_mean(diffs_list):
        """
        diffs_list: list of np.array with variable lengths.
        Return mean_vec, used_len, used_count.
        Strategy: group by length; pick the length group with most samples.
        """
        if not diffs_list:
            return None, 0, 0
        grouped = {}
        for d in diffs_list:
            if isinstance(d, (list, np.ndarray)):
                grouped.setdefault(len(d), []).append(np.array(d))
        # choose the length with the most samples
        best_len = max(grouped.keys(), key=lambda k: len(grouped[k]))
        valid = np.vstack(grouped[best_len])
        return np.mean(valid, axis=0), best_len, valid.shape[0]

    # ---------- walk & collect ----------
    for folder_name in os.listdir(root_path):
        if ion_density not in folder_name:
            continue

        for cn_name, symbol in ion_types.items():
            if cn_name not in folder_name:
                continue

            folder_path = os.path.join(root_path, folder_name)
            if not os.path.isdir(folder_path):
                continue

            for subfolder in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue

                # 仅处理父文件夹（即 file 的直接父级 subfolder）名中包含“恢复”
                # 仅处理绝对路径中包含“恢复”的文件夹
                # if "恢复" not in subfolder_path:
                #     print(f"未找到恢复字样：{subfolder_path}")
                #     continue

                for file in os.listdir(subfolder_path):
                    if not file.endswith("_sorted.xlsx"):
                        continue

                    file_path = os.path.join(subfolder_path, file)
                    try:
                        print(f"\n📄 Processing file: {file_path}")
                        df = pd.read_excel(file_path)

                        row1_idx, row2_idx, row3_idx = find_matching_rows(df)
                        if row1_idx is None:
                            print("  ⚠️ row1 未找到，跳过")
                            continue
                       
                        

                        # 哪些列可用
                        full_cols = ["R0", "P1w", "P1n", "R1", "P2w", "P2n", "R2", "P3w", "P3n", "R3"]
                        short_cols = ["R0", "P1w", "P1n", "R1", "P2w", "P2n", "R2"]

                        if all(c in df.columns for c in full_cols):
                            r_cols = ["R0", "R1", "R2", "R3"]
                        elif all(c in df.columns for c in short_cols):
                            r_cols = ["R0", "R1", "R2"]
                        else:
                            print("  ⚠️ 缺少 R 列，跳过")
                            continue

                        # 先用“简单规则”（与 fit_res_analysis.py 相同）重新定位 row1/row2 —— 仅用于 2-1
                        row1_idx_simple, row2_idx_simple = find_matching_rows_21_simple(df)

                        if row1_idx_simple is not None and row2_idx_simple is not None:
                            # 2-1 采用与 1.py 相同的差分逻辑
                            diff_21 = compute_diff_21_simple(df, row1_idx_simple, row2_idx_simple, r_cols)
                        else:
                            # 兜底：若简单规则没匹配上，退回你现在的复杂匹配结果
                            if row2_idx is None:
                                print("  ⚠️ row2 未找到，跳过 2-1")
                                diff_21 = None
                            else:
                                diff_21 = compute_diff_21_simple(df, row1_idx, row2_idx, r_cols)

                        if diff_21 is not None:
                            ion_data_21[symbol].append(diff_21)

                        # —— 下面保持原样：3-1 仍用你的扩张窗口稳健法 ——
                        if row3_idx is not None:
                            try:
                                diff_31 = compute_diff(df, row1_idx, row3_idx, r_cols)  # 原来的扩张窗口法
                                if diff_31 is not None:
                                    ion_data_31[symbol].append(diff_31)
                            except Exception as e:
                                print(f"  ❌ 3-1 失败：{e}")
                        else:
                            print("  ⚠️ row3 未找到，跳过 3-1")

                    except Exception as e:
                        print(f"❌ Failed to read: {file_path}, Error: {e}")

    # ---------- aggregate ----------
    mean_diffs_21, mean_diffs_31 = {}, {}
    counts_21, counts_31 = {}, {}
    max_r_count = 0

    print("\n====== 汇总统计 ======")
    for ion in ion_types.values():
        # 2-1
        mean21, len21, n21 = group_and_mean(ion_data_21[ion])
        # 3-1
        mean31, len31, n31 = group_and_mean(ion_data_31[ion])

        if mean21 is not None:
            mean_diffs_21[ion] = mean21
            counts_21[ion] = n21
            max_r_count = max(max_r_count, len(mean21))
            print(f"📊 {ion} 2-1 平均(样本 {n21}, 长度 {len21}): " + ", ".join([f"R{i}:{v:.6f}" for i, v in enumerate(mean21)]))
        if mean31 is not None:
            mean_diffs_31[ion] = mean31
            counts_31[ion] = n31
            max_r_count = max(max_r_count, len(mean31))
            print(f"📊 {ion} 3-1 平均(样本 {n31}, 长度 {len31}): " + ", ".join([f"R{i}:{v:.6f}" for i, v in enumerate(mean31)]))

    # 若一个离子只在其中一组有数据，也允许画/写
    # ---------- plot ----------
    plt.figure(figsize=(10, 6))

    # 用 tab10 做基色，同一离子深浅两种
    base_colors = plt.get_cmap("tab10")
    import matplotlib.colors as mcolors

    DEBUG = True  # 调试开关

    # 规范化函数：去空格、上标转普通字符、小写
    _SUPERSCRIPT_MAP = str.maketrans({"²": "2", "³": "3", "⁺": "+"})
    def norm_ion(s: str) -> str:
        return s.strip().translate(_SUPERSCRIPT_MAP).replace(" ", "").lower()

    # 你想要的颜色（用 ASCII 版本建表）
    ion_colors_raw = {
        "Ca2+": "blue",
        "Na+":  "gold",   # 比 yellow 柔和些
        "Ni2+": "green",
        "Cr3+": "red",
        "Cu2+": "purple",
        "Fe3+": "brown",
    }

    # 建立“规范化后的键”→颜色 的查表
    ion_colors = {norm_ion(k): v for k, v in ion_colors_raw.items()}

    if DEBUG:
        print("[DEBUG] ion_colors keys (normalized):", list(ion_colors.keys()))

    ion_list = [ion for ion in ion_types.values() if (ion in mean_diffs_21 or ion in mean_diffs_31)]

    if DEBUG:
        print("[DEBUG] ion_list (raw):", [repr(i) for i in ion_list])

    for ion in ion_list:
        key = norm_ion(ion)
        color = ion_colors.get(key)
        if color is None:
            if DEBUG:
                print(f"[WARN] Unmapped ion name: raw={repr(ion)} -> key={repr(key)}; fallback to black")
            color = "black"

        if DEBUG:
            print(f"[DEBUG] use color for {ion}: key={key}, color={color}, "
                f"has_21={ion in mean_diffs_21}, has_31={ion in mean_diffs_31}")

        # 2-1 (深色虚线，按你当前写法)
        if ion in mean_diffs_21:
            vals = mean_diffs_21[ion]
            if len(vals)==4:
                labels = ["R_O", "R_LF", "R_MF", "R_HF"][:len(vals)]
            if len(vals)==3:
                labels = ["R_O", "R_LF",  "R_HF"][:len(vals)]
            plt.plot(labels, vals, marker='x', label=f"{ion} contaminated",
                    color=color, linewidth=2, alpha=0.3, linestyle="--")

        # 3-1 (浅色实线，按你当前写法)
        if ion in mean_diffs_31:
            vals = mean_diffs_31[ion]
            if len(vals)==4:
                labels = ["R_O", "R_LF", "R_MF", "R_HF"][:len(vals)]
            if len(vals)==3:
                labels = ["R_O", "R_LF",  "R_HF"][:len(vals)]
            plt.plot(labels, vals, marker='o', label=f"{ion} recovered",
                    color=color, linewidth=2, alpha=0.8)

    plt.xlabel("Resistance Component")
    plt.ylabel("Change (ΔOhm)")
    plt.title(f"Impedance Change (contaminated vs recovered) ")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()

    date_suffix = Path(root_path).name
    plot_path = output_dir / f"resistance_change_plot_2v3_{date_suffix}_{ion_density}.png"
    plt.savefig(plot_path, dpi=200)
    plt.show()

    # ---------- save CSV ----------
    # 统一列头
    if max_r_count==4:
        excel_columns = ["R0", "R_LF", "R_MF", "R_HF"][:max_r_count]
    if max_r_count==3:
        excel_columns = ["R0", "R_LF",  "R_HF"][:max_r_count]

    # 组装两个 DataFrame，并在列名加后缀
    def to_df(dct, suffix):
        if not dct:
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(dct, orient="index")
        df.columns = [f"{c}_{suffix}" for c in excel_columns[:df.shape[1]]]
        return df

    df21 = to_df(mean_diffs_21, "2minus1")
    df31 = to_df(mean_diffs_31, "3minus1")

    # 样本数也保存
    s21 = pd.Series(counts_21, name="n_samples_2minus1")
    s31 = pd.Series(counts_31, name="n_samples_3minus1")

    # 合并
    df_out = pd.concat([df21, df31, s21, s31], axis=1)

    excel_filename = f"resistance_change_summary_2v3_{date_suffix}_{ion_density}.xlsx"
    excel_path = output_dir / excel_filename
    df_out.index.name = "Ion"
    df_out.to_excel(excel_path)

    print(f"\n✅ Done! Plot saved to: {plot_path}")
    print(f"           Excel saved to: {excel_path}")

# === 调用示例（与原脚本一致） ===
if __name__ == "__main__":
    from pathlib import Path
    date_folder = "20250723"   # 你的日期字符串
    current_dir = Path(__file__).resolve()
    root_path = current_dir.parents[1] / "eis_fit_results" / date_folder
    ion_density = '2ppm'
    fit_res_analysis(root_path=root_path, ion_density=ion_density)
