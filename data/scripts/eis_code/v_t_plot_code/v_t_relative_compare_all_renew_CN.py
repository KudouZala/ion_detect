#一个py程序，绘制2ppm每种离子：污染前的电压，污染6h的电压，以及硫酸恢复后的电压。
#我会指定给你一个文件夹：/home/cagalii/Application/ion_detect/data/校内测试，在该文件夹下有很多的子文件夹，比如：“20241029_2ppm铁离子污染和恢复测试”，你需要找到所有带有“2ppm“和“污染和恢复测试”字样的子文件夹，
#里面会有一个文件夹，名字中含有“_gamry”或者“_firecloud”，
# 如果是“_gamry”，那么你需要找到该文件夹下面的含有“PWRGALVANOSTATIC_...”字样的文件夹，然后计算"..._ion_0.DTA"中的电压平均值作为污染前的电压，"..._ion_3.DTA"中的电压平均值作为污染6h的电压，“ion_column_renew_H2SO4_3.DTA”中的电压平均值作为恢复后的电压
# 如果是“_firecloud”，那么你需要找到该文件夹下面的含有“_ion_column”“_ion”“_ion_column_renew”字样的文件夹，然后计算“_ion_column”字样的文件夹下的"...（5／80)_工步1(CC).csv"中的电压平均值作为污染前的电压，“_ion”文件下的"(3／80)_工步1(CC).csv"中的电压平均值作为污染6h的电压，“_ion_column_renew”下的“...（5／80)_工步1(CC).csv”中的电压平均值作为恢复后的电压
#至于如何从dta或者csv文件中获取电压数值，你可以参考我上传的这个代码中的部分。
#如果每种离子有多个，那么求平均后再给我
#代码中给我加入一些文件名的调试，比方说找不到文件，找到了哪个文件等

# -*- coding: utf-8 -*-
# compute_2ppm_ion_voltage_diffs_mixed_plot_single.py
#
# 功能：跨平台混合抽取三段电压；仅计算两种差值：Δh6=(污染6h-前)、Δrenew=(恢复后-前)；
# 若同一离子有多个实验，对差值再取平均。绘图：每个离子只画**一条**折线：
#    x=["初始","污染6h","恢复后"], y=[0, Δh6_mean, Δrenew_mean]
# 颜色按映射；在点位上叠加：污染6h 用 'x' 标记、恢复后用 'o' 标记。
#

from pathlib import Path
import os

import os
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

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

# ========================= 用户可改区域 =========================
BASE_DIR = Path("/home/cagalii/Application/ion_detect/data/校内测试")
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (SCRIPT_DIR.parent.parent.parent / "volt_t_plot").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEBUG = True
# ===============================================================

def _read_lines_with_fallback(path: Path):
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.readlines()
        except Exception:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()

def extract_voltage_data(lines, voltage_key="V", current_key="A", column_indices=None):
    if column_indices is None:
        column_indices = {"D": 2, "E": 3}
    start_row = None
    for i, line in enumerate(lines):
        cols = line.strip().split()
        if len(cols) > max(column_indices.values()) and cols[column_indices["D"]] == voltage_key and cols[column_indices["E"]] == current_key:
            start_row = i + 1
            break
    if start_row is None:
        raise ValueError("未找到 D/E 表头 (D=V, E=A)")
    data = []
    for line in lines[start_row:]:
        cols = line.strip().split()
        try:
            data.append(float(cols[column_indices["D"]]))
        except (ValueError, IndexError):
            continue
    return data

def mean_voltage_from_dta(fp: Path):
    try:
        lines = _read_lines_with_fallback(fp)
        v = extract_voltage_data(lines)
        if not v:
            print(f"[警告] DTA 无有效电压：{fp}")
            return None
        return float(pd.Series(v).mean())
    except Exception as e:
        print(f"[错误] 读取 DTA 失败：{fp} -> {e}")
        return None

def mean_voltage_from_csv(fp: Path):
    try:
        df = pd.read_csv(fp)
        if "Voltage/V" not in df.columns:
            print(f"[警告] CSV 缺少 'Voltage/V' 列：{fp}")
            return None
        return float(pd.to_numeric(df["Voltage/V"], errors="coerce").dropna().mean())
    except Exception as e:
        print(f"[错误] 读取 CSV 失败：{fp} -> {e}")
        return None

def chinese_ion_to_symbol(name: str) -> str:
    mapping = {"铁":"Fe3+","镍":"Ni2+","铜":"Cu2+","铬":"Cr3+","钙":"Ca2+","钠":"Na+","铝":"Al3+","无离子":"No-ion","无":"No-ion"}
    for k, v in mapping.items():
        if k in name:
            return v
    m = re.search(r"[A-Z][a-z]?\d?[+-]?\+?", name)
    return m.group(0) if m else name

# ---- Firecloud 寻找器 ----
def scan_firecloud(fire_root: Path):
    res = {}
    ion_column_dirs = [p for p in fire_root.rglob("*") if p.is_dir() and "_ion_column" in p.name and "_ion_column_renew" not in p.name]
    ion_dirs = [p for p in fire_root.rglob("*") if p.is_dir() and p.name.endswith("_ion") and "_ion_column" not in p.name]
    renew_dirs = [p for p in fire_root.rglob("*") if p.is_dir() and "_ion_column_renew" in p.name]

    if not ion_column_dirs: print(f"[调试] _firecloud 未找到 _ion_column：{fire_root}")
    if not ion_dirs: print(f"[调试] _firecloud 未找到 _ion：{fire_root}")
    if not renew_dirs: print(f"[调试] _firecloud 未找到 _ion_column_renew：{fire_root}")

    # 兼容 ( ) 与 （ ）；兼容 ／ 与 /；并要求右括号
    pat_5_80 = re.compile(r"[\(（]5[／/ ]80[)）]_工步1\(CC\)\.csv$")
    pat_3_80 = re.compile(r"[\(（]3[／/ ]80[)）]_工步1\(CC\)\.csv$")

    def pick(dir_list, pat, tag):
        for d in dir_list:
            cands = [p for p in d.rglob("*.csv") if pat.search(p.name)]
            if cands:
                print(f"[调试] Firecloud 在 {d} 找到 {tag}：{cands[0]}")
                return cands[0]
            else:
                print(f"[调试] Firecloud 在 {d} 未匹配到 {tag}")
        return None

    pre_fp = pick(ion_column_dirs, pat_5_80, "pre (5/80)")
    h6_fp = pick(ion_dirs, pat_3_80, "h6 (3/80)")
    renew_fp = pick(renew_dirs, pat_5_80, "renew (5/80)")

    if pre_fp: res["pre"] = (mean_voltage_from_csv(pre_fp), str(pre_fp))
    if h6_fp: res["h6"] = (mean_voltage_from_csv(h6_fp), str(h6_fp))
    if renew_fp: res["renew"] = (mean_voltage_from_csv(renew_fp), str(renew_fp))
    return res

# ---- Gamry 寻找器 ----
def scan_gamry(gamry_root: Path):
    res = {}
    galvanos = [p for p in gamry_root.rglob("*") if p.is_dir() and "PWRGALVANOSTATIC_" in p.name]
    if not galvanos:
        print(f"[调试] _gamry 未找到 PWRGALVANOSTATIC_*：{gamry_root}")
        return res
    for gal_dir in galvanos:
        print(f"[调试] 尝试 Gamry 子目录：{gal_dir}")
        ion0 = list(gal_dir.rglob("*_ion_0.DTA"))
        ion3 = list(gal_dir.rglob("*_ion_3.DTA"))
        renews = list(gal_dir.rglob("*_ion_column_renew_H2SO4_3.DTA"))
        # 兜底 baseline
        ioncol_any = list(gal_dir.rglob("*ion_column*.DTA"))

        if "pre" not in res:
            if ion0:
                print(f"[调试] Gamry 找到 *_ion_0.DTA：{ion0[0]}")
                res["pre"] = (mean_voltage_from_dta(ion0[0]), str(ion0[0]))
            elif ioncol_any:
                print(f"[调试] Gamry 找到 fallback baseline *ion_column*.DTA：{ioncol_any[0]}")
                res["pre"] = (mean_voltage_from_dta(ioncol_any[0]), str(ioncol_any[0]))
            else:
                print(f"[调试] 未找到 Gamry baseline (*_ion_0.DTA 或 *ion_column*.DTA) 于：{gal_dir}")

        if "h6" not in res and ion3:
            print(f"[调试] Gamry 找到 *_ion_3.DTA：{ion3[0]}")
            res["h6"] = (mean_voltage_from_dta(ion3[0]), str(ion3[0]))
        elif "h6" not in res:
            print(f"[调试] 未找到 *_ion_3.DTA 于：{gal_dir}")

        if "renew" not in res and renews:
            print(f"[调试] Gamry 找到 ion_column_renew_H2SO4_3.DTA：{renews[0]}")
            res["renew"] = (mean_voltage_from_dta(renews[0]), str(renews[0]))
        elif "renew" not in res:
            print(f"[调试] 未找到 ion_column_renew_H2SO4_3.DTA 于：{gal_dir}")
    return res

# ---- 扫描并“组合”同一实验根目录的三段 ----
def scan_and_combine(base_dir: Path):
    results = defaultdict(list)
    for root, dirs, files in os.walk(base_dir):
        folder_name = os.path.basename(root)
        if "2ppm" in folder_name and "污染和恢复测试" in folder_name:
            ion = chinese_ion_to_symbol(folder_name)
            print(f"\n[发现实验] {root}  ->  推断离子：{ion}")
            platform_dirs = [Path(root) / d for d in dirs if ("_gamry" in d or "_firecloud" in d)]
            if not platform_dirs:
                print(f"[调试] 未在 {root} 下找到含 '_gamry' 或 '_firecloud' 的子目录。")
                continue
            from_fire = {}; from_gamry = {}
            for pdir in platform_dirs:
                if "_firecloud" in pdir.name:
                    from_fire.update(scan_firecloud(pdir))
                elif "_gamry" in pdir.name:
                    from_gamry.update(scan_gamry(pdir))
            chosen = {}
            # pre
            if "pre" in from_fire and from_fire["pre"][0] is not None:
                chosen["pre"] = (from_fire["pre"][0], from_fire["pre"][1], "firecloud")
            elif "pre" in from_gamry and from_gamry["pre"][0] is not None:
                chosen["pre"] = (from_gamry["pre"][0], from_gamry["pre"][1], "gamry")
            # h6
            if "h6" in from_gamry and from_gamry["h6"][0] is not None:
                chosen["h6"] = (from_gamry["h6"][0], from_gamry["h6"][1], "gamry")
            elif "h6" in from_fire and from_fire["h6"][0] is not None:
                chosen["h6"] = (from_fire["h6"][0], from_fire["h6"][1], "firecloud")
            # renew
            if "renew" in from_gamry and from_gamry["renew"][0] is not None:
                chosen["renew"] = (from_gamry["renew"][0], from_gamry["renew"][1], "gamry")
            elif "renew" in from_fire and from_fire["renew"][0] is not None:
                chosen["renew"] = (from_fire["renew"][0], from_fire["renew"][1], "firecloud")
            if all(k in chosen for k in ("pre","h6","renew")):
                print("[组合] 成功组装三段："
                      f"pre<-{chosen['pre'][2]} ({chosen['pre'][1]}), "
                      f"h6<-{chosen['h6'][2]} ({chosen['h6'][1]}), "
                      f"renew<-{chosen['renew'][2]} ({chosen['renew'][1]})")
                results[ion].append({"pre":chosen["pre"],"h6":chosen["h6"],"renew":chosen["renew"],"root":root})
            else:
                print("[组合] 未能在该实验中组装出完整三段，跳过。"
                      f" 可用键：pre={('pre' in chosen)}, h6={('h6' in chosen)}, renew={('renew' in chosen)}")
    return results

def norm_ion(name: str) -> str:
    return name.replace(" ", "").replace("＋", "+").replace("－", "-")
def plot_single_polyline(df_agg: pd.DataFrame, save_path: Path):
    # Color map (kept)
    ion_colors_raw = {
        "Ca2+": "blue",
        "Na+":  "gold",
        "Ni2+": "green",
        "Cr3+": "red",
        "Cu2+": "purple",
        "Fe3+": "brown",
    }
    ion_colors = {norm_ion(k): v for k, v in ion_colors_raw.items()}
    if DEBUG:
        print("[DEBUG] ion_colors keys (normalized):", list(ion_colors.keys()))

    # English ticks for research-style plots
    x = [0, 1, 2]
    # xticklabels = ["Baseline", "6h Contam.", "Post-Renewal"]
    xticklabels = ["初始状态", "污染6h", "H2SO4恢复后"]

    plt.figure(figsize=(11, 6))

    for _, row in df_agg.iterrows():
        ion = str(row["ion"])
        key = norm_ion(ion)
        color = ion_colors.get(key, "black")
        if color == "black" and DEBUG:
            print(f"[WARN] Unmapped ion name: raw={repr(ion)} -> key={repr(key)}; fallback to black")

        y = [0.0, float(row["delta_h6_mean"]), float(row["delta_renew_mean"])]
        # Main polyline (solid), one legend entry per ion
        plt.plot(x, y, label=ion, color=color, linewidth=2)

        # Overlaid markers: 6h as 'x', post-renewal as 'o', baseline as faint '.'
        plt.scatter([1], [y[1]], marker='x', color=color, s=70, alpha=0.9)
        plt.scatter([2], [y[2]], marker='o', color=color, s=60, alpha=0.9)
        plt.scatter([0], [0.0], marker='.', color=color, s=40, alpha=0.7)

    plt.xticks(x, xticklabels, fontsize=11)
    # plt.ylabel("Relative Voltage ΔV (V)", fontsize=12)
    # plt.title("2ppm Ions: Baseline → 6h Contamination → Post-Renewal (ΔV)", fontsize=14)
    plt.ylabel("相对电压变化(V)", fontsize=12)
    plt.title("离子污染: 初始状态 → 污染6h  → H2SO4恢复后 (ΔV)", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(ncol=3, fontsize=10, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Figure saved: {save_path}")


def main():
    print(f"[开始] 扫描根目录：{BASE_DIR}")
    combos = scan_and_combine(BASE_DIR)
    if not combos:
        print("[提示] 没有找到可组装的三段数据。")
        return

    # 逐实验计算差值
    rows_detail = []
    for ion, exps in combos.items():
        for exp in exps:
            pre_val, pre_fp, pre_src = exp["pre"]
            h6_val, h6_fp, h6_src = exp["h6"]
            renew_val, renew_fp, renew_src = exp["renew"]
            if None in (pre_val, h6_val, renew_val):
                continue
            rows_detail.append({
                "ion": ion,
                "delta_h6": h6_val - pre_val,
                "delta_renew": renew_val - pre_val,
                "pre_src": pre_src, "pre_file": pre_fp,
                "h6_src": h6_src, "h6_file": h6_fp,
                "renew_src": renew_src, "renew_file": renew_fp,
                "experiment_root": exp["root"],
            })

    df_detail = pd.DataFrame(rows_detail)
    detail_csv = OUTPUT_DIR / "per_experiment_diffs.csv"
    df_detail.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    print(f"\n[输出] 逐实验差值明细：{detail_csv}")

    # 按离子聚合（对差值取平均）
    agg_rows = []
    if not df_detail.empty:
        for ion, g in df_detail.groupby("ion"):
            agg_rows.append({
                "ion": ion,
                "delta_h6_mean": pd.to_numeric(g["delta_h6"], errors="coerce").dropna().mean(),
                "delta_renew_mean": pd.to_numeric(g["delta_renew"], errors="coerce").dropna().mean(),
                "n_experiments": len(g),
            })
    df_agg = pd.DataFrame(agg_rows).sort_values("ion").reset_index(drop=True)
    agg_csv = OUTPUT_DIR / "aggregated_by_ion_diffs.csv"
    df_agg.to_csv(agg_csv, index=False, encoding="utf-8-sig")
    print(f"[输出] 按离子聚合后的差值均值：{agg_csv}")

    # 绘图（单折线版）
    if not df_agg.empty:
        fig_path = OUTPUT_DIR / "relative_deltas_single_line_CN.png"
        plot_single_polyline(df_agg, fig_path)
    else:
        print("[提示] df_agg 为空，跳过绘图。")

    # 调试溯源报告
    report_path = OUTPUT_DIR / "debug_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 2ppm 各离子差值（污染6h-前、恢复后-前）— 溯源报告（单折线绘图版）\n\n")
        for ion, exps in combos.items():
            f.write(f"## {ion}\n\n")
            for i, exp in enumerate(exps, 1):
                pre_val, pre_fp, pre_src = exp["pre"]
                h6_val, h6_fp, h6_src = exp["h6"]
                renew_val, renew_fp, renew_src = exp["renew"]
                f.write(f"- 实验 {i}  根目录: `{exp['root']}`\n")
                f.write(f"  - pre   ({pre_src}): {pre_val:.6f} V  文件: `{pre_fp}`\n")
                f.write(f"  - h6    ({h6_src}): {h6_val:.6f} V  文件: `{h6_fp}`\n")
                f.write(f"  - renew ({renew_src}): {renew_val:.6f} V  文件: `{renew_fp}`\n")
                f.write(f"  - Δh6 = {h6_val - pre_val:.6f} V,  Δrenew = {renew_val - pre_val:.6f} V\n\n")
    print(f"[输出] 调试报告：{report_path}")

if __name__ == "__main__":
    main()