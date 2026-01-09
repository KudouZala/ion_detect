# -*- coding: utf-8 -*-
"""
compute_2ppm_ion_voltage_diffs.py

功能：
- 在 /home/cagalii/Application/ion_detect/data/校内测试 下递归查找所有同时包含“2ppm”和“污染和恢复测试”的子文件夹。
- 兼容两种平台：
    * _gamry：在 PWRGALVANOSTATIC_* 下读取 *_ion_0.DTA / *_ion_3.DTA / ion_column_renew_H2SO4_3.DTA
    * _firecloud：在 _ion_column / _ion / _ion_column_renew 下读取 (5／80)_工步1(CC).csv / (3／80)_工步1(CC).csv / (5／80)_工步1(CC).csv
- 对每个实验计算两项差值：
    Δh6 = (污染6h - 污染前)
    Δrenew = (硫酸恢复后 - 污染前)
- 若同一离子有多个实验，则对差值再取平均；导出两份 CSV：
    1) per_experiment_diffs.csv：逐实验的差值结果与使用到的文件路径
    2) aggregated_by_ion_diffs.csv：按离子聚合后的差值平均值与实验数量
- 输出 debug_report.md，记录找文件过程（找到/找不到、实际使用的文件）。
- 不使用 argparse，直接修改脚本顶部的常量。

作者：ChatGPT 生成
"""

import os
import re
from pathlib import Path, PureWindowsPath, PurePosixPath
from collections import defaultdict, namedtuple

import pandas as pd


# ========================= 用户可改区域 =========================
# 数据根目录
BASE_DIR = Path("/home/cagalii/Application/ion_detect/data/校内测试")

# 输出目录
OUTPUT_DIR = Path("./ion_2ppm_diff_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===============================================================


def _normalize_rel(p: str) -> Path:
    """把相对路径字符串按其分隔符解析为部件，再用本机 Path 重组。"""
    if p is None:
        return Path()
    s = str(p).strip().strip('"').strip("'").replace("\u00a0", " ")
    pure = PureWindowsPath(s) if "\\" in s else PurePosixPath(s)
    return Path(*pure.parts)


def _read_lines_with_fallback(path: Path):
    """文本读取编码兜底：utf-8 -> gbk -> latin-1 -> 最后 ignore"""
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.readlines()
        except Exception:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()


def extract_voltage_data(lines, voltage_key="V", current_key="A", column_indices=None):
    """
    参考用户上传脚本的解析方法：定位标题行（D列=V, E列=A），然后取 D 列的电压。
    """
    if column_indices is None:
        column_indices = {"D": 2, "E": 3}

    start_row = None
    for i, line in enumerate(lines):
        columns = line.strip().split()
        if len(columns) > max(column_indices.values()) and \
           columns[column_indices["D"]] == voltage_key and \
           columns[column_indices["E"]] == current_key:
            start_row = i + 1
            break

    if start_row is None:
        raise ValueError(f"未找到满足条件的表头行 (D列={voltage_key}, E列={current_key})")

    data = []
    for line in lines[start_row:]:
        columns = line.strip().split()
        try:
            data.append(float(columns[column_indices["D"]]))
        except (ValueError, IndexError):
            continue

    return data


def mean_voltage_from_dta(file_path: Path):
    try:
        lines = _read_lines_with_fallback(file_path)
        volts = extract_voltage_data(lines)
        if len(volts) == 0:
            print(f"[警告] DTA 无有效电压数据：{file_path}")
            return None
        return float(pd.Series(volts).mean())
    except Exception as e:
        print(f"[错误] 读取 DTA 失败：{file_path} -> {e}")
        return None


def mean_voltage_from_csv(file_path: Path):
    try:
        df = pd.read_csv(file_path)
        if "Voltage/V" not in df.columns:
            print(f"[警告] CSV 缺少 'Voltage/V' 列：{file_path}")
            return None
        return float(pd.to_numeric(df["Voltage/V"], errors="coerce").dropna().mean())
    except Exception as e:
        print(f"[错误] 读取 CSV 失败：{file_path} -> {e}")
        return None


def chinese_ion_to_symbol(name: str) -> str:
    """
    从中文关键词推断离子符号。若无法确定，返回原字符串。
    """
    mapping = {
        "铁": "Fe3+",
        "镍": "Ni2+",
        "铜": "Cu2+",
        "铬": "Cr3+",
        "钙": "Ca2+",
        "钠": "Na+",
        "铝": "Al3+",
        "无离子": "No-ion",
        "无": "No-ion",
    }
    for k, v in mapping.items():
        if k in name:
            return v
    m = re.search(r"[A-Z][a-z]?\d?[+-]?\+?", name)
    if m:
        return m.group(0)
    return name


ExpTriple = namedtuple("ExpTriple", ["pre", "h6", "renew", "platform", "root_folder", "used_files"])


def find_gamry_triple(gamry_root: Path):
    galvanos = [p for p in gamry_root.rglob("*") if p.is_dir() and "PWRGALVANOSTATIC_" in p.name]
    if not galvanos:
        print(f"[调试] _gamry 未找到 PWRGALVANOSTATIC_* 文件夹：{gamry_root}")
        return None

    for gal_dir in galvanos:
        print(f"[调试] 尝试 Gamry 子目录：{gal_dir}")
        pre = h6 = renew = None
        used = {}

        ion0 = list(gal_dir.rglob("*_ion_0.DTA"))
        ion3 = list(gal_dir.rglob("*_ion_3.DTA"))
        renews = list(gal_dir.rglob("*_ion_column_renew_H2SO4_3.DTA"))

        if ion0:
            print(f"[调试] 找到 *_ion_0.DTA：{ion0[0]}")
            pre = mean_voltage_from_dta(ion0[0]); used["pre"] = str(ion0[0])
        else:
            print(f"[调试] 未找到 *_ion_0.DTA 于：{gal_dir}")

        if ion3:
            print(f"[调试] 找到 *_ion_3.DTA：{ion3[0]}")
            h6 = mean_voltage_from_dta(ion3[0]); used["h6"] = str(ion3[0])
        else:
            print(f"[调试] 未找到 *_ion_3.DTA 于：{gal_dir}")

        if renews:
            print(f"[调试] 找到 ion_column_renew_H2SO4_3.DTA：{renews[0]}")
            renew = mean_voltage_from_dta(renews[0]); used["renew"] = str(renews[0])
        else:
            print(f"[调试] 未找到 ion_column_renew_H2SO4_3.DTA 于：{gal_dir}")

        if pre is not None and h6 is not None and renew is not None:
            return ExpTriple(pre, h6, renew, "gamry", str(gamry_root), used)

    return None


def find_firecloud_triple(fire_root: Path):
    ion_column_dirs = [p for p in fire_root.rglob("*") if p.is_dir() and "_ion_column" in p.name and "_ion_column_renew" not in p.name]
    ion_dirs = [p for p in fire_root.rglob("*") if p.is_dir() and p.name.endswith("_ion") and "_ion_column" not in p.name]
    renew_dirs = [p for p in fire_root.rglob("*") if p.is_dir() and "_ion_column_renew" in p.name]

    if not ion_column_dirs:
        print(f"[调试] _firecloud 未找到包含 _ion_column 的目录：{fire_root}")
    if not ion_dirs:
        print(f"[调试] _firecloud 未找到以 _ion 结尾的目录：{fire_root}")
    if not renew_dirs:
        print(f"[调试] _firecloud 未找到包含 _ion_column_renew 的目录：{fire_root}")

    pat_5_80 = re.compile(r"\(5[／/ ]80\)_工步1\(CC\)\.csv$")
    pat_3_80 = re.compile(r"\(3[／/ ]80\)_工步1\(CC\)\.csv$")

    def pick_csv_mean(dir_list, pat, tag):
        for d in dir_list:
            cands = [p for p in d.rglob("*.csv") if pat.search(p.name)]
            if cands:
                print(f"[调试] 在 {d} 找到 {tag} CSV：{cands[0]}")
                return mean_voltage_from_csv(cands[0]), str(cands[0])
            else:
                print(f"[调试] 在 {d} 未匹配到 {tag} CSV。")
        return None, None

    pre, pre_fp = pick_csv_mean(ion_column_dirs, pat_5_80, "污染前 (5/80)")
    h6, h6_fp = pick_csv_mean(ion_dirs, pat_3_80, "污染6h (3/80)")
    renew, renew_fp = pick_csv_mean(renew_dirs, pat_5_80, "恢复后 (5/80)")

    used = {}
    if pre_fp: used["pre"] = pre_fp
    if h6_fp: used["h6"] = h6_fp
    if renew_fp: used["renew"] = renew_fp

    if pre is not None and h6 is not None and renew is not None:
        return ExpTriple(pre, h6, renew, "firecloud", str(fire_root), used)

    return None


def scan_all_2ppm_experiments(base_dir: Path):
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

            for pdir in platform_dirs:
                if "_gamry" in pdir.name:
                    triple = find_gamry_triple(pdir)
                    if triple:
                        results[ion].append(triple)
                    else:
                        print(f"[调试] {pdir} 未能凑齐 Gamry 的三段文件，跳过该实验。")

                elif "_firecloud" in pdir.name:
                    triple = find_firecloud_triple(pdir)
                    if triple:
                        results[ion].append(triple)
                    else:
                        print(f"[调试] {pdir} 未能凑齐 Firecloud 的三段文件，跳过该实验。")

    return results


def main():
    print(f"[开始] 扫描根目录：{BASE_DIR}")
    exp_map = scan_all_2ppm_experiments(BASE_DIR)

    if not exp_map:
        print("[提示] 没有找到任何完整的 2ppm '污染和恢复测试' 实验三段数据。")
        return

    # 逐实验计算差值，并保存明细
    rows_detail = []
    for ion, triples in exp_map.items():
        for t in triples:
            if t.pre is None or t.h6 is None or t.renew is None:
                continue
            delta_h6 = t.h6 - t.pre
            delta_renew = t.renew - t.pre
            rows_detail.append({
                "ion": ion,
                "delta_h6": delta_h6,
                "delta_renew": delta_renew,
                "platform": t.platform,
                "root_folder": t.root_folder,
                "pre_file": t.used_files.get("pre"),
                "h6_file": t.used_files.get("h6"),
                "renew_file": t.used_files.get("renew"),
            })

    df_detail = pd.DataFrame(rows_detail)
    detail_csv = OUTPUT_DIR / "per_experiment_diffs.csv"
    df_detail.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    print(f"\n[输出] 逐实验差值明细：{detail_csv}")

    # 按离子聚合（对差值取平均）
    agg_rows = []
    if not df_detail.empty:
        for ion, group in df_detail.groupby("ion"):
            agg_rows.append({
                "ion": ion,
                "delta_h6_mean": pd.to_numeric(group["delta_h6"], errors="coerce").dropna().mean(),
                "delta_renew_mean": pd.to_numeric(group["delta_renew"], errors="coerce").dropna().mean(),
                "n_experiments": len(group),
            })
    df_agg = pd.DataFrame(agg_rows).sort_values("ion").reset_index(drop=True)
    agg_csv = OUTPUT_DIR / "aggregated_by_ion_diffs.csv"
    df_agg.to_csv(agg_csv, index=False, encoding="utf-8-sig")
    print(f"[输出] 按离子聚合后的差值均值：{agg_csv}")

    # 调试溯源报告
    report_path = OUTPUT_DIR / "debug_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 2ppm 各离子差值（污染6h-前、恢复后-前）— 溯源报告\n\n")
        for ion, triples in exp_map.items():
            f.write(f"## {ion}\n\n")
            for idx, t in enumerate(triples, 1):
                if t.pre is None or t.h6 is None or t.renew is None:
                    continue
                delta_h6 = t.h6 - t.pre
                delta_renew = t.renew - t.pre
                f.write(f"- 实验 {idx}  平台: **{t.platform}**  根目录: `{t.root_folder}`\n")
                f.write(f"  - Δh6 = {delta_h6:.6f} V  (h6 - pre)  文件: pre= `{t.used_files.get('pre')}`; h6= `{t.used_files.get('h6')}`\n")
                f.write(f"  - Δrenew = {delta_renew:.6f} V  (renew - pre)  文件: pre= `{t.used_files.get('pre')}`; renew= `{t.used_files.get('renew')}`\n\n")
    print(f"[输出] 调试报告：{report_path}")

    # 控制台简要汇总
    if not df_agg.empty:
        print("\n=== 按离子聚合的差值均值 ===")
        for _, row in df_agg.iterrows():
            print(f"{row['ion']}:  Δh6_mean={row['delta_h6_mean']:.6f} V,  Δrenew_mean={row['delta_renew_mean']:.6f} V,  n={int(row['n_experiments'])}")
    else:
        print("\n[提示] 没有可聚合的实验差值。")

if __name__ == "__main__":
    main()