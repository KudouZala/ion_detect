import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
# Set root directory

# --- 强制注册系统里的 Noto CJK 字体，并设置为中文字体 ---
import os, glob
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager

# 1) 动态注册所有 NotoSansCJK*.ttc（你 fc-list 打印的这些路径就包含它们）
candidates = []
candidates += glob.glob("/usr/share/fonts/opentype/noto/NotoSansCJK-*.ttc")
candidates += glob.glob("/usr/share/fonts/truetype/noto/NotoSansCJK-*.ttc")
for p in candidates:
    try:
        font_manager.fontManager.addfont(p)
    except Exception as e:
        print("注册失败:", p, e)

# 2) 看看 Matplotlib 现在认识的字体里，有没有 Noto Sans CJK SC
families = sorted(set(f.name for f in font_manager.fontManager.ttflist))
print("可用字体族数量：", len(families))
hit = [f for f in families if "Noto Sans CJK SC" in f]
print("匹配到的 'Noto Sans CJK SC'：", hit[:5])

# 3) 指定字体（先用 SC；若没找到，就退而求其次用 JP/KR/TC，至少能显示 CJK）
target = None
for prefer in ["Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK KR", "Noto Sans CJK TC"]:
    if any(prefer == f for f in families):
        target = prefer
        break

if target is None:
    # 兜底：让 DejaVu 之外的 sans-serif 也能被搜索到
    target = "Noto Sans CJK SC"  # 仍然设置名字，下面配合 FontProperties 也能生效

rcParams["font.family"] = target
rcParams["axes.unicode_minus"] = False

print("最终使用字体：", target)


def fit_res_analysis(root_path,ion_density):


    # Output directory two levels above current script
    current_dir = Path(__file__).resolve()
    output_dir = current_dir.parents[1] / "eis_fit_analysis_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Ion type mapping: Chinese name → Symbol
    ion_types = {
        "铝离子": "Al3+",
        "钙离子": "Ca2+",
        "钠离子": "Na+",
        "镍离子": "Ni2+",
        "铬离子": "Cr3+",
        "铜离子": "Cu2+",
        "铁离子": "Fe3+",
        
    }

    # Initialize data container
    ion_data = {symbol: [] for symbol in ion_types.values()}

    def find_matching_rows(df):
        row1_mask = df.iloc[:, 0].astype(str).str.contains(
            r"(_ion循环1／1_工步组1\(工步组\)\(1／1\)_工步3\(阻抗\)_greater_than_0)|(_ion_0_大于0)", na=False)
        row2_mask = df.iloc[:, 0].astype(str).str.contains(
            r"(_ion_3_大于0)|(_ion循环1／1_工步组2\(工步组\)\(3／80\)_工步3\(阻抗\)_greater_than_0)", na=False)

        row1_idx = row1_mask.idxmax() if row1_mask.any() else None
        row2_idx = row2_mask.idxmax() if row2_mask.any() else None

        if row1_idx is not None and row2_idx is not None:
            return row1_idx, row2_idx
        else:
            return None, None


    def compute_diff(df, row1_idx, row2_idx, columns):
        try:
            v1_vals = []
            v2_vals = []

            # ---- 初始值：row1及其上1行的平均 ----
            pre_indices = [i for i in range(row1_idx - 1, row1_idx + 1) if i >= 0]

            for col in columns:
                # ---- 初始值 v1 ----
                v1_list = []
                for i in pre_indices:
                    val = df.iloc[i][col]
                    if pd.notna(val):
                        v1_list.append(float(val))
                if len(v1_list) == 0:
                    print(f"🚫 Column {col}: no valid values in row1 or its previous 2 rows.")
                    return None, row1_idx
                v1 = np.mean(v1_list)
                v1_vals.append(v1)

                # ---- 最终值 v2（从 row2_idx 往后找 3 个与 v1 差值 < 1 的值） ----
                v2_candidates = []
                for i in range(row2_idx, len(df)):
                    val = df.iloc[i][col]
                    if pd.isna(val):
                        continue
                    v2 = float(val)
                    if  abs(v2 - v1) < 1:
                        v2_candidates.append(v2)
                        print(f"✅ Column {col}: matched df[{i}][{col}] = {v2:.3f} (|Δ|={abs(v2 - v1):.3f})")
                    if len(v2_candidates) == 3:
                        break

                if len(v2_candidates) < 3:
                    print(f"⚠️  Column {col}: only found {len(v2_candidates)} values within ±1 after row2_idx={row2_idx}")
                if len(v2_candidates) == 0:
                    print(f"🚫 Column {col}: no values within ±1 found, skipping this row")
                    return None, row1_idx

                v2_vals.append(np.mean(v2_candidates))

            v1_vals = np.array(v1_vals)
            v2_vals = np.array(v2_vals)

            # print(f"    ➤ initial values (avg of row1 & above): {v1_vals}")
            # print(f"    ➤ final values (avg of matched 3 rows): {v2_vals}")
            # print(f"    ➤ diff: {v2_vals - v1_vals}")
            return v2_vals - v1_vals

        except Exception as e:
            print("Error in compute_diff:", e)
            return None



    # Walk through folders and collect results
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

                for file in os.listdir(subfolder_path):
                    if file.endswith("_sorted.xlsx"):
                        file_path = os.path.join(subfolder_path, file)
                        try:
                            print(f"\n📄 Processing file: {file_path}")
                            df = pd.read_excel(file_path)
                            row1_idx, row2_idx = find_matching_rows(df)
                            if row1_idx is not None and row2_idx is not None:
                                row1 = df.loc[row1_idx]
                                # print("  ✅ Matched row1:")
                                # print(row1)
                                # print("  ✅ Matched row2:")
                                # print(df.loc[row2_idx])

                                

    

                                full_cols = ["R0", "P1w", "P1n", "R1", "P2w", "P2n", "R2", "P3w", "P3n", "R3"]
                                short_cols = ["R0", "P1w", "P1n", "R1", "P2w", "P2n", "R2"]

                                if all(c in df.columns for c in full_cols):
                                    diff = compute_diff(df, row1_idx, row2_idx, ["R0", "R1", "R2", "R3"])
                                elif all(c in df.columns for c in short_cols):
                                    diff = compute_diff(df, row1_idx, row2_idx, ["R0", "R1", "R2"])
                                else:
                                    print("  ⚠️ Required columns not found")
                                    continue

                                if diff is not None:
                                    ion_data[symbol].append(diff)
                        except Exception as e:
                            print(f"❌ Failed to read: {file_path}, Error: {e}")

    # Compute and print averages
    # Compute and print averages
    # Compute and print averages
    mean_diffs = {}
    max_r_count = 0  # keep track of max R columns for Excel header

    for ion, diffs in ion_data.items():
        if not diffs:
            continue

        # 🔍 调试：输出每个 diff 的长度
        for i, diff in enumerate(diffs):
            if not isinstance(diff, (list, np.ndarray)):
                print(f"🚫 Error: unexpected diff type: {type(diff)}")
                continue
            if len(diff) != 3:
                print(f"⚠️ [DEBUG] {ion} sample {i} has {len(diff)} values: {diff}")

        # 按长度分组 diffs
        grouped_diffs = {}
        for diff in diffs:
            key = len(diff)
            grouped_diffs.setdefault(key, []).append(diff)

        # 选择包含样本最多的组作为主组
        best_len = max(grouped_diffs.keys(), key=lambda k: len(grouped_diffs[k]))
        valid_diffs = np.vstack(grouped_diffs[best_len])  # 仅拼接该组
        mean = np.mean(valid_diffs, axis=0)

        mean_diffs[ion] = mean
        max_r_count = max(max_r_count, len(mean))

        print(f"\n📊 Average change for {ion} (based on {len(grouped_diffs[best_len])} samples with {best_len} values):")
        for i, val in enumerate(mean):
            print(f"    R{i}: {val:.6f}")

    # Plot
    plt.figure(figsize=(10, 6))
    for ion, values in mean_diffs.items():
        if len(values)==4:
            labels = ["R_O", "R_LF", "R_MF", "R_HF"][:len(values)]
        if len(values)==3:
            labels = ["R_O", "R_LF",  "R_HF"][:len(values)]
        if ion == 'Al3+':
            plt.plot(labels, values, marker='x', markersize=6, label=ion)
        elif ion == 'Ca2+':
            plt.plot(labels, values, marker='+', markersize=6, label=ion)
        elif ion == 'Na+':
            plt.plot(labels, values, marker='+', markersize=6, label=ion)
        elif ion == 'Ni2+':
            plt.plot(labels, values, marker='+', markersize=6, label=ion)
        elif ion == 'Cr3+':
            plt.plot(labels, values, marker='o', markersize=6, label=ion)
        elif ion == 'Fe3+':
            plt.plot(labels, values, marker='o', markersize=6, label=ion)
        elif ion == 'Cu2+':
            plt.plot(labels, values, marker='o', markersize=6, label=ion)
        else:
            print("没有这个离子")


    # plt.xlabel("Resistance Component", fontsize=14)
    # plt.ylabel("Change (Δ)", fontsize=14)
    # plt.title(f"Impedance Change for {ion_density} Ion Contamination", fontsize=16)
    plt.xlabel("阻抗类型", fontsize=14)
    plt.ylabel("变化量 (ΔOhm)", fontsize=14)
    plt.title(f"离子污染引起的阻抗变化", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # Save plot
    date_suffix = Path(root_path).name
    plot_path = output_dir / f"resistance_change_plot_{date_suffix}_{ion_density}_CN.png"
    plt.savefig(plot_path)
    plt.show()


    # Save to Excel
    if max_r_count==4:
        excel_columns = ["R0", "R_LF", "R_MF", "R_HF"][:max_r_count]
    if max_r_count==3:
        excel_columns = ["R0", "R_LF",  "R_HF"][:max_r_count]
    df_out = pd.DataFrame.from_dict(mean_diffs, orient="index")
    df_out.columns = excel_columns
    # Save Excel
    # 提取 root_path 的最后一部分作为日期后缀

    excel_filename = f"resistance_change_summary_{date_suffix}_{ion_density}.xlsx"
    excel_path = output_dir / excel_filename
    df_out.to_excel(excel_path)

    print(f"\n✅ Done! Plot saved to: {plot_path}")
    print(f"           Excel saved to: {excel_path}")

    print("\n✅ Done! Plot saved as resistance_change_plot.png, table saved as resistance_change_summary.xlsx")



from pathlib import Path

# 日期字符串（你指定的）
date_folder = "20250723"

# 当前脚本文件路径
current_dir = Path(__file__).resolve()

# 构造 root_path：当前 py 的上两层路径 + eis_fit_results + 日期文件夹
root_path = current_dir.parents[1] / "eis_fit_results" / date_folder
ion_density = '2ppm'
fit_res_analysis(root_path=root_path,ion_density=ion_density)