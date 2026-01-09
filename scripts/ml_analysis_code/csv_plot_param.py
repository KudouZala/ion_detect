#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
csv_plot_param.py

用法:
    python csv_plot_param.py --load_run=20250812a

功能:
- 在脚本所在路径的上两级目录中定位 output/inference_results/<load_run>
- 读取该目录下所有 "*_phys_params_structured.csv"
- 根据文件名中的离子关键词（钙离子/钠离子/铁离子/镍离子/铬离子/铜离子）分类
- 从CSV中提取 psi, theta_ca, theta_an, phi_ca, phi_an 对应的“同一行下一列”的数值
- 按离子进行聚合并计算每个参数的平均值
- 输出一个统计CSV与一张折线图（不同离子不同折线）
"""

import argparse
import sys
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
PARAM_KEYS_ROMAN =["psi", "theta_ca", "theta_an", "phi_ca", "phi_an"]  # Roman numerals for parameters
# Ion colors as per the user's request
ION_COLORS = {
    "Ca": "purple",
    "Na": "yellow",
    "Ni": "cyan",
    "Cr": "green",
    "Fe": "orange",
    "Cu": "blue",
}
ION_KEYWORDS = {
    "钙离子": "Ca",
    "钠离子": "Na",
    "镍离子": "Ni",
    "铬离子": "Cr",
    "铁离子": "Fe",
    "铜离子": "Cu",
}

PARAM_KEYS = ["psi", "theta_ca", "theta_an", "phi_ca", "phi_an"]

def parse_args():
    parser = argparse.ArgumentParser(description="统计离子物理参数并绘图")
    parser.add_argument("--load_run", required=True, help="如: 20250812a")
    return parser.parse_args()

def find_target_dir(load_run: str) -> Path:
    # 脚本所在路径的上两级目录
    script_path = Path(__file__).resolve()
    base_dir = script_path.parent.parent.parent  # 上两级
    target = base_dir / "output" / "inference_results" / load_run
    if not target.exists():
        print(f"[ERROR] 目录不存在: {target}")
        sys.exit(1)
    return target

def robust_read_csv(path: Path) -> pd.DataFrame:
    """尽量鲁棒地读取 CSV（考虑编码/分隔符等常见问题）"""
    # 优先尝试utf-8/utf-8-sig
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312"]
    for enc in encodings:
        try:
            # 结构未知，先不设 header；保持所有内容为字符串便于检索
            df = pd.read_csv(path, header=None, dtype=str, encoding=enc)
            return df
        except Exception:
            continue
    # 最后一搏：python引擎
    df = pd.read_csv(path, header=None, dtype=str, engine="python")
    return df

_float_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def first_float_in_cell(cell) -> float | None:
    """从单元格文本中提取首个浮点数，若无返回None"""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None
    s = str(cell).strip()
    # 去掉常见千分位逗号（若有）
    s = s.replace(",", "")
    m = _float_pattern.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def extract_params_from_df(df: pd.DataFrame) -> dict[str, list[float]]:
    """
    在整个 DataFrame 中查找 PARAM_KEYS（大小写不敏感）。
    规则：当某个单元格文本等于目标key（忽略大小写与两端空格）时，
         取该单元格“同一行的下一列”作为其值；如果下一列不存在或非数值则忽略。
    返回: {param_key: [values...]}
    """
    results = {k: [] for k in PARAM_KEYS}
    # 全部转为小写、strip 便于比较（保留原 df 取值）
    lower_df = df.applymap(lambda x: str(x).strip().lower() if pd.notna(x) else x)

    # 为加速，先建立 param -> set(lowered) 的映射
    wanted = {k: k.lower() for k in PARAM_KEYS}

    rows, cols = df.shape
    for r in range(rows):
        for c in range(cols):
            cell = lower_df.iat[r, c]
            if cell is None or (isinstance(cell, float) and np.isnan(cell)):
                continue
            for p, p_low in wanted.items():
                if cell == p_low:
                    # 取“同一行下一列”
                    if c + 1 < cols:
                        val = first_float_in_cell(df.iat[r, c + 1])
                        if val is not None:
                            results[p].append(val)
                    # 不 break，避免一行多个参数名的极端情况
    return results

def ion_from_filename(name: str) -> str | None:
    """从文件名中识别离子，返回缩写（Ca/Na/Fe/Ni/Cr/Cu）或 None"""
    if "ion_column" in name:
        return None
    for zh, short in ION_KEYWORDS.items():
        if zh in name:
            return short
    return None
def main():
    args = parse_args()
    target_dir = find_target_dir(args.load_run)

    csv_files = sorted(target_dir.glob("*_phys_params_structured.csv"))
    if not csv_files:
        print(f"[WARN] 未找到CSV：{target_dir} 下没有 *_phys_params_structured.csv")
        sys.exit(0)

    # 聚合容器: ion -> param -> list[float]
    agg: dict[str, dict[str, list[float]]] = {
        ion: {k: [] for k in PARAM_KEYS} for ion in ION_KEYWORDS.values()
    }

    total_count = 0
    matched_count = 0

    for f in csv_files:
        total_count += 1
        ion = ion_from_filename(f.name)
        if ion is None:
            # 跳过没有离子关键词的文件
            continue
        matched_count += 1

        df = robust_read_csv(f)
        param_vals = extract_params_from_df(df)
        for k in PARAM_KEYS:
            agg[ion][k].extend(param_vals.get(k, []))

    if matched_count == 0:
        print(f"[WARN] 未在文件名中识别到任何目标离子。检查文件名是否包含：{list(ION_KEYWORDS.keys())}")
        sys.exit(0)

    # 计算均值（忽略空列表/NaN）
    summary_rows = []
    for ion in ION_KEYWORDS.values():
        means = []
        for k in PARAM_KEYS:
            arr = np.array(agg[ion][k], dtype=float) if agg[ion][k] else np.array([], dtype=float)
            if arr.size == 0:
                means.append(np.nan)
            else:
                means.append(float(np.nanmean(arr)))
        summary_rows.append([ion] + means)

    summary_df = pd.DataFrame(summary_rows, columns=["ion"] + PARAM_KEYS_ROMAN)

    # 保存CSV
    out_csv = target_dir / f"{args.load_run}_ions_param_summary.csv"
    summary_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 汇总CSV已保存: {out_csv}")

    # 画折线图
    plt.figure(figsize=(10, 6))
    x = np.arange(len(PARAM_KEYS_ROMAN))
    for _, row in summary_df.iterrows():
        ion = row["ion"]
        y = [row[k] for k in PARAM_KEYS_ROMAN]
        plt.plot(x, y, marker="o", label=ion, color=ION_COLORS[ion])

    # 字体：尽量兼容中文（若系统无对应字体，matplotlib 会回退）
    try:
        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Arial"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    PARAM_KEYS_LATEX = [r"$\psi$", r"$\theta_{ca}$", r"$\theta_{an}$", r"$\phi_{ca}$", r"$\phi_{an}$"]  # LaTeX形式的罗马字母

    # 然后在绘图部分替换参数名，改成 LaTeX 格式的显示
    plt.xticks(x, PARAM_KEYS_LATEX)
    plt.xlabel(r"$\text{Parameters}$")  # 使用 LaTeX 语法的 xlabel
    plt.ylabel(r"$\text{Average Value}$")  # 使用 LaTeX 语法的 ylabel
    plt.title(f"{args.load_run} Ion Parameter Comparison")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="Ion")
    plt.tight_layout()

    out_png = target_dir / f"{args.load_run}_ions_param_plot.png"
    plt.savefig(out_png, dpi=200)
    print(f"[OK] 折线图已保存: {out_png}")

    # 控制台摘要
    print("\n=== 汇总预览 ===")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(summary_df)

    # 额外提示
    print(f"\n从目录: {target_dir}")
    print(f"总CSV文件: {total_count}，其中识别到目标离子的文件: {matched_count}")

if __name__ == "__main__":
    main()
