#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
csv_plot_param2.py  (English plots + robust font; correct delta averaging)

Usage:
    python csv_plot_param2.py --load_run=20250729a

Outputs:
1) <load_run>_ions_initparam_timeseries.xlsx   — mean time series table per ion at t={0,2,4,6}h
2) <load_run>_timeseries_<Ion>.png            — one figure per ion, 5 curves (x={0,2,4,6})
3) <load_run>_initparam_delta_6h_minus_0h.csv — per-ion deltas Δ(6h-0h) for 5 parameters (computed as mean of per-sample differences)
4) <load_run>_delta_0h_6h.png                 — 5-subplot figure; each subplot is one parameter, colored by ion

File selection (strict):
- Scan only files whose names contain "_phys_params_structured.csv".
- Time buckets are inferred from filename by these fragments:
  [0, 2, 4, 6]  -> t=0h
  [2, 4, 6, 8]  -> t=2h
  [4, 6, 8, 10] -> t=4h
  [6, 8, 10, 12]-> t=6h
- Ion is inferred by Chinese ion keywords in the filename (e.g. "铜离子" for Copper).
  Files containing "ion_column" are ignored.

CSV value extraction:
Locate the cell whose content equals one of the parameter keys (case/space-insensitive),
then take the value in the **next column on the same row** (first float in that cell):
    sigma_mem, alpha_ca, alpha_an, i_0ca, i_0an
"""

import argparse
import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Fonts: force a guaranteed-available family to avoid warnings ----------
plt.rcParams["font.family"] = "DejaVu Sans"   # shipped with matplotlib
plt.rcParams["axes.unicode_minus"] = True     # minus sign renders correctly
# Ion colors
ION_COLORS = {
    "Ca": "purple",
    "Na": "yellow",
    "Fe": "orange",
    "Ni": "cyan",
    "Cr": "green",
    "Cu": "blue",
}

INIT_PARAM_KEYS_LATEX = [r"$\sigma_{\text{mem}}$", r"$\alpha_{x}$", r"$\alpha_{y}$", r"$i_{0,x}$", r"$i_{0,y}$"]

# --------------------------- Config ---------------------------
ION_KEYWORDS = {
    "钙离子": "Ca",
    "钠离子": "Na",
    "铁离子": "Fe",
    "镍离子": "Ni",
    "铬离子": "Cr",
    "铜离子": "Cu",
}
ION_ORDER = ["Ca", "Na", "Ni","Cr","Fe","Cu"]

INIT_PARAM_KEYS = ["sigma_mem", "alpha_ca", "alpha_an", "i_0ca", "i_0an"]

WINDOW_PATTERNS = {
    0: re.compile(r"\[\s*0\s*,\s*2\s*,\s*4\s*,\s*6\s*\]"),
    2: re.compile(r"\[\s*2\s*,\s*4\s*,\s*6\s*,\s*8\s*\]"),
    4: re.compile(r"\[\s*4\s*,\s*6\s*,\s*8\s*,\s*10\s*\]"),
    6: re.compile(r"\[\s*6\s*,\s*8\s*,\s*10\s*,\s*12\s*\]"),
}

def parse_args():
    ap = argparse.ArgumentParser(description="Aggregate initial-parameter time series (0–6h) per ion and plot, with correct delta averaging.")
    ap.add_argument("--load_run", required=True, help="e.g., 20250729a")
    return ap.parse_args()

def find_target_dir(load_run: str) -> Path:
    script_path = Path(__file__).resolve()
    base_dir = script_path.parent.parent.parent
    target = base_dir / "output" / "inference_results" / load_run
    if not target.exists():
        print(f"[ERROR] directory not found: {target}")
        sys.exit(1)
    return target

def robust_read_csv(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "gbk", "gb2312"]:
        try:
            return pd.read_csv(path, header=None, dtype=str, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path, header=None, dtype=str, engine="python")

_float_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
def first_float_in_cell(cell) -> float | None:
    if cell is None:
        return None
    if isinstance(cell, float) and np.isnan(cell):
        return None
    s = str(cell).strip().replace(",", "")
    m = _float_pattern.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def extract_params_from_df(df: pd.DataFrame, keys: list[str]) -> dict[str, list[float]]:
    res = {k: [] for k in keys}
    lower_df = df.applymap(lambda x: str(x).strip().lower() if pd.notna(x) else x)
    wanted = {k: k.lower() for k in keys}
    R, C = df.shape
    for r in range(R):
        for c in range(C):
            cell = lower_df.iat[r, c]
            if cell is None or (isinstance(cell, float) and np.isnan(cell)):
                continue
            for p, p_low in wanted.items():
                if cell == p_low:
                    if c + 1 < C:
                        v = first_float_in_cell(df.iat[r, c + 1])
                        if v is not None:
                            res[p].append(v)
    return res

def match_time_bucket(name: str) -> int | None:
    for t, pat in WINDOW_PATTERNS.items():
        if pat.search(name):
            return t
    return None

def detect_ion(name: str) -> str | None:
    if "ion_column" in name:
        return None
    for zh, abbr in ION_KEYWORDS.items():
        if zh in name:
            return abbr
    return None

# For pairing 0h and 6h belonging to the same sample
_strip_window_regex = re.compile(
    r"\[\s*0\s*,\s*2\s*,\s*4\s*,\s*6\s*\]|\[\s*2\s*,\s*4\s*,\s*6\s*,\s*8\s*\]|\[\s*4\s*,\s*6\s*,\s*8\s*,\s*10\s*\]|\[\s*6\s*,\s*8\s*,\s*10\s*,\s*12\s*\]"
)
def base_key_from_filename(name: str) -> str:
    s = _strip_window_regex.sub("", name)
    s = s.replace("__", "_")
    return s

def main():
    # 解析命令行参数
    args = parse_args()
    target_dir = find_target_dir(args.load_run)

    # 收集所有 *_phys_params_structured.csv 文件
    csv_files = sorted(target_dir.glob("*_phys_params_structured.csv"))
    if not csv_files:
        print("[WARN] no *_phys_params_structured.csv found.")
        sys.exit(0)

    # 初始化时间序列聚合器（用于 t=0, 2, 4, 6）
    agg_ts = {abbr: {t: {k: [] for k in INIT_PARAM_KEYS} for t in [0, 2, 4, 6]}
              for abbr in ION_ORDER}

    # 用于计算 Δ(6h-0h) 时，按样本计算差值的配对
    pairs = {abbr: {} for abbr in ION_ORDER}

    # 遍历所有CSV文件，进行数据提取和聚合
    for f in csv_files:
        name = f.name
        t = match_time_bucket(name)
        if t not in {0, 2, 4, 6}:
            continue
        ion = detect_ion(name)
        if ion is None:
            continue

        # 读取 CSV 文件并提取参数值
        df = robust_read_csv(f)
        vals = extract_params_from_df(df, INIT_PARAM_KEYS)

        # 聚合时间序列数据
        for k in INIT_PARAM_KEYS:
            agg_ts[ion][t][k].extend(vals.get(k, []))

        # 对 t=0 和 t=6 计算差值（用于 Δ(6h-0h)）
        if t in {0, 6}:
            bk = base_key_from_filename(name)
            if bk not in pairs[ion]:
                pairs[ion][bk] = {}
            pairs[ion][bk][t] = {k: list(vals.get(k, [])) for k in INIT_PARAM_KEYS}

    # 创建时间序列 DataFrame (timeseries_df)
    cols = [f"{k}@t{t}h" for k in INIT_PARAM_KEYS for t in [0, 2, 4, 6]]
    rows = []
    for ion in ION_ORDER:
        row_vals = []
        for k in INIT_PARAM_KEYS:
            for t in [0, 2, 4, 6]:
                arr = np.array(agg_ts[ion][t][k], dtype=float) if agg_ts[ion][t][k] else np.array([], dtype=float)
                row_vals.append(float(np.nanmean(arr)) if arr.size else np.nan)
        rows.append([ion] + row_vals)
    
    # 创建 timeseries DataFrame
    timeseries_df = pd.DataFrame(rows, columns=["ion"] + cols)

    # 保存为 Excel 文件
    out_xlsx = target_dir / f"{args.load_run}_ions_initparam_timeseries.xlsx"
    try:
        with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
            timeseries_df.to_excel(writer, index=False, sheet_name="timeseries")
    except Exception:
        with pd.ExcelWriter(out_xlsx) as writer:
            timeseries_df.to_excel(writer, index=False, sheet_name="timeseries")
    print(f"[OK] saved Excel: {out_xlsx}")

    # 绘制每个离子的时间序列图
    x = np.array([0, 2, 4, 6], dtype=float)
    for _, row in timeseries_df.iterrows():
        ion = str(row["ion"])
        plt.figure(figsize=(10, 6))
        for k, param_key in zip(INIT_PARAM_KEYS, INIT_PARAM_KEYS_LATEX):
            y = [row[f"{k}@t0h"], row[f"{k}@t2h"], row[f"{k}@t4h"], row[f"{k}@t6h"]]
            plt.plot(x, y, marker="o", label=param_key, color=ION_COLORS[ion])  # 使用不同颜色
        plt.xticks(x, [0, 2, 4, 6])
        plt.xlabel("Time (hour)")
        plt.ylabel("Value")
        plt.title(f"{args.load_run} | {ion} — Initial-state params vs time (0–6h)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(title="Parameter")
        plt.tight_layout()
        out_png = target_dir / f"{args.load_run}_timeseries_{ion}.png"
        plt.savefig(out_png, dpi=200)
        print(f"[OK] saved: {out_png}")
        plt.close()

    # 计算 Δ(6h-0h) 的差值并聚合
    delta_pool = {ion: {k: [] for k in INIT_PARAM_KEYS} for ion in ION_ORDER}
    for ion in ION_ORDER:
        for bk, d in pairs[ion].items():
            if 0 in d and 6 in d:
                for k in INIT_PARAM_KEYS:
                    lst0 = list(d[0].get(k, []))
                    lst6 = list(d[6].get(k, []))
                    if not lst0 or not lst6:
                        continue
                    n = min(len(lst0), len(lst6))
                    diffs = [float(lst6[i]) - float(lst0[i]) for i in range(n)]
                    delta_pool[ion][k].extend(diffs)

    # 创建 delta DataFrame
    delta_cols = [f"{k}@delta_6h-0h" for k in INIT_PARAM_KEYS]
    delta_rows = []
    for ion in ION_ORDER:
        vals = []
        for k in INIT_PARAM_KEYS:
            arr = np.array(delta_pool[ion][k], dtype=float) if delta_pool[ion][k] else np.array([], dtype=float)
            vals.append(float(np.nanmean(arr)) if arr.size else np.nan)
        delta_rows.append([ion] + vals)
    delta_df = pd.DataFrame(delta_rows, columns=["ion"] + delta_cols)

    # 保存为 CSV 文件
    out_delta_csv = target_dir / f"{args.load_run}_initparam_delta_6h_minus_0h.csv"
    delta_df.to_csv(out_delta_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] saved Delta CSV: {out_delta_csv}")

    # 绘制 Δ(6h-0h) 的柱状图，并修改标题为罗马字母
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
    axes = axes.flatten()
    for i, k in enumerate(INIT_PARAM_KEYS):
        ax = axes[i]
        ions = ION_ORDER
        y = []
        for ion in ions:
            v = delta_df.loc[delta_df["ion"] == ion, f"{k}@delta_6h-0h"]
            y.append(np.nan if v.empty else float(v.values[0]))
        idx = np.arange(len(ions))
        ax.bar(idx, y, color=[ION_COLORS[ion] for ion in ions])  # 为每个离子设置不同的颜色
        ax.set_xticks(idx)
        ax.set_xticklabels(ions)
        
        # 使用 LaTeX 表示罗马字母作为标题
        ax.set_title(f"$\Delta(6h–0h)$ — {INIT_PARAM_KEYS_LATEX[i]}")
        ax.set_ylabel("Delta")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # 关闭多余的子图
    if len(INIT_PARAM_KEYS) < len(axes):
        for j in range(len(INIT_PARAM_KEYS), len(axes)):
            axes[j].axis("off")

    fig.suptitle(f"{args.load_run} | 0h→6h delta: per-parameter $\Delta(6h–0h)$ (per-sample diff then mean)")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_delta_png = target_dir / f"{args.load_run}_delta_0h_6h.png"
    fig.savefig(out_delta_png, dpi=220)
    plt.close(fig)
    print(f"[OK] saved Delta figure: {out_delta_png}")

    print("Done.")


if __name__ == "__main__":
    main()
