"""
论文图表生成脚本 —— 从训练日志中提取指标，生成论文级别的图和表。

用法:
    cd scripts/machine_learning_code/logs
    python plot_results.py

输出:
    figures/   —— 高清 PDF 图片
    tables/    —— LaTeX .tex 表格 + CSV
"""

from __future__ import annotations
import re, os, sys, warnings
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────
# 0.  全局配置
# ──────────────────────────────────────────────────────────────
LOG_DIR   = Path(__file__).resolve().parent
FIG_DIR   = LOG_DIR / "figures"
TABLE_DIR = LOG_DIR / "tables"
FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

METRICS = ["Accuracy", "Precision", "Recall", "F1"]

PEAK_HALF_WINDOW = 10       # 峰值前后各 N 次评估 (±100 epoch)
BOOTSTRAP_N   = 10000       # bootstrap 重采样次数
CI_LEVEL      = 0.95        # 置信区间
CURVE_MAX_EPOCH = 600       # 训练曲线图 x 轴截断

# 颜色方案 (色盲友好)
COLORS = {
    "SVM":         "#E69F00",
    "RF":          "#56B4E9",
    "LSTM":        "#009E73",
    "Transformer": "#F0E442",
    "Exp-A":       "#0072B2",
    "Exp-B":       "#D55E00",
    "Exp-C":       "#CC79A7",
    "Exp-D":       "#999999",
    "Exp-E":       "#66A61E",
    "Exp-F":       "#E6AB02",
    "Exp-G":       "#A6761D",
    "Exp-H":       "#1B9E77",
    "Exp-I":       "#7570B3",
    "Exp-J":       "#D95F02",
}

# matplotlib 论文风格
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.labelsize":   11,
    "axes.titlesize":   12,
    "legend.fontsize":  8,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "lines.linewidth":  1.5,
    "lines.markersize": 4,
})


# ──────────────────────────────────────────────────────────────
# 1.  日志解析
# ──────────────────────────────────────────────────────────────
_METRIC_RE = re.compile(
    r"准确率 \(Accuracy\):\s+([\d.]+).*?"
    r"精确率 \(Precision\):\s+([\d.]+).*?"
    r"召回率 \(Recall\):\s+([\d.]+).*?"
    r"F1 分数 \(F1\):\s+([\d.]+)",
    re.DOTALL
)

_EXP_HEADER_RE = re.compile(r"---\s*验证集\s*\[epoch\s+(\d+)\]\s*---")
_LSTM_HEADER_RE = re.compile(r"---\s*LSTM.*?\[epoch\s+(\d+)\]\s*---")
_TF_HEADER_RE = re.compile(r"---\s*Transformer 基线.*?---")
_RF_HEADER_RE = re.compile(r"---\s*Random Forest 基线.*?---")
_SVM_HEADER_RE = re.compile(r"---\s*SVM 基线.*?---")


def _extract_metrics_after(lines: List[str], start: int) -> Optional[Tuple[float, ...]]:
    """从 start 行开始向下找 4 个指标"""
    block = "\n".join(lines[start:start + 8])
    m = _METRIC_RE.search(block)
    if m:
        return tuple(float(x) for x in m.groups())
    return None


def parse_exp_new_log(filepath: Path) -> List[Tuple[int, Tuple[float, ...]]]:
    """解析 exp_new_*.log → [(epoch, (acc, prec, rec, f1)), ...]"""
    lines = filepath.read_text(errors="replace").splitlines()
    results = []
    for i, line in enumerate(lines):
        m = _EXP_HEADER_RE.search(line)
        if m:
            epoch = int(m.group(1))
            metrics = _extract_metrics_after(lines, i + 1)
            if metrics:
                results.append((epoch, metrics))
    return results


def parse_lstm_log(filepath: Path) -> List[Tuple[int, Tuple[float, ...]]]:
    lines = filepath.read_text(errors="replace").splitlines()
    results = []
    for i, line in enumerate(lines):
        m = _LSTM_HEADER_RE.search(line)
        if m:
            epoch = int(m.group(1))
            metrics = _extract_metrics_after(lines, i + 1)
            if metrics:
                results.append((epoch, metrics))
    return results


def parse_transformer_log(filepath: Path) -> List[Tuple[int, Tuple[float, ...]]]:
    """Transformer 日志无 epoch 编号，按出现顺序推断 (每 10 epoch)"""
    lines = filepath.read_text(errors="replace").splitlines()
    results = []
    count = 0
    for i, line in enumerate(lines):
        if _TF_HEADER_RE.search(line):
            count += 1
            epoch = count * 10
            metrics = _extract_metrics_after(lines, i + 1)
            if metrics:
                results.append((epoch, metrics))
    if len(results) > 1 and results[-1][1] == results[-2][1]:
        results = results[:-1]
    return results


def parse_static_log(filepath: Path, header_re) -> Optional[Tuple[float, ...]]:
    """解析只有一次评估结果的日志 (RF/SVM)"""
    lines = filepath.read_text(errors="replace").splitlines()
    for i, line in enumerate(lines):
        if header_re.search(line):
            metrics = _extract_metrics_after(lines, i + 1)
            if metrics:
                return metrics
    return None


# ──────────────────────────────────────────────────────────────
# 2.  数据收集
# ──────────────────────────────────────────────────────────────
def _tp_from_filename(name: str) -> int:
    """从文件名推断 num_time_points"""
    if "num_tp2" in name or "_tp2" in name or "_t2" in name:
        return 2
    if "num_tp4" in name or "_tp4" in name or "_t4" in name:
        return 4
    return 3


def _exp_label(name: str) -> str:
    """exp_new_a_num_tp2.log → 'Exp-A'; 支持 a-j"""
    m = re.search(r"exp_new_([a-j])", name)
    if m:
        return f"Exp-{m.group(1).upper()}"
    return name


EXP_DESCRIPTIONS = {
    "Exp-A": "Base (CE only)",
    "Exp-B": "w/o Env Params",
    "Exp-C": "CLIP Classify",
    "Exp-D": "+ Physics",
    "Exp-E": "+ Physics + CrossAttn + CLIP",
    "Exp-F": "Full (All Aux Losses)",
    "Exp-G": "Exp-G",
    "Exp-H": "Exp-H",
    "Exp-I": "Exp-I",
    "Exp-J": "Exp-J",
}


def collect_all_data() -> dict:
    """
    返回结构:
    {
      (method, tp): {
        "curve": [(epoch, (acc, prec, rec, f1)), ...],   # 可为空
        "final": (acc, prec, rec, f1),                    # 最佳 epoch
      }
    }
    """
    data = {}

    for f in sorted(LOG_DIR.glob("exp_new_*.log")):
        tp = _tp_from_filename(f.name)
        label = _exp_label(f.name)
        curve = parse_exp_new_log(f)
        if curve:
            best = max(curve, key=lambda x: x[1][3])  # best F1
            data[(label, tp)] = {"curve": curve, "final": best[1], "best_epoch": best[0]}

    for f in sorted(LOG_DIR.glob("log_main_lstm_baseline_*.txt")):
        tp = _tp_from_filename(f.name)
        curve = parse_lstm_log(f)
        if curve:
            best = max(curve, key=lambda x: x[1][3])
            data[("LSTM", tp)] = {"curve": curve, "final": best[1], "best_epoch": best[0]}

    for f in sorted(LOG_DIR.glob("log_main_transformer_baseline_*.txt")):
        tp = _tp_from_filename(f.name)
        curve = parse_transformer_log(f)
        if curve:
            best = max(curve, key=lambda x: x[1][3])
            data[("Transformer", tp)] = {"curve": curve, "final": best[1], "best_epoch": best[0]}

    for f in sorted(LOG_DIR.glob("log_main_rf_baseline_*.txt")):
        tp = _tp_from_filename(f.name)
        metrics = parse_static_log(f, _RF_HEADER_RE)
        if metrics:
            data[("RF", tp)] = {"curve": [], "final": metrics, "best_epoch": None}

    for f in sorted(LOG_DIR.glob("log_main_svm_baseline_*.txt")):
        tp = _tp_from_filename(f.name)
        metrics = parse_static_log(f, _SVM_HEADER_RE)
        if metrics:
            data[("SVM", tp)] = {"curve": [], "final": metrics, "best_epoch": None}

    return data


def _bootstrap_ci(values: np.ndarray, n_boot: int = BOOTSTRAP_N, ci: float = CI_LEVEL) -> Tuple[float, float]:
    """Bootstrap 置信区间"""
    rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return lo, hi


def compute_stable_stats(curve: List[Tuple[int, Tuple[float, ...]]],
                         half_window: int = PEAK_HALF_WINDOW) -> dict:
    """
    以 F1 峰值为中心，取前后各 half_window 个评估点组成窗口，
    计算 Mean±Std 和 Bootstrap 95% CI。
    """
    if not curve:
        return {}
    best_idx = max(range(len(curve)), key=lambda i: curve[i][1][3])
    lo_idx = max(0, best_idx - half_window)
    hi_idx = min(len(curve), best_idx + half_window + 1)
    window = curve[lo_idx:hi_idx]
    arr = np.array([x[1] for x in window])  # (W, 4)
    stats = {}
    for j, name in enumerate(METRICS):
        col = arr[:, j]
        ci_lo, ci_hi = _bootstrap_ci(col)
        stats[name] = {
            "mean": col.mean(),
            "std":  col.std(ddof=1) if len(col) > 1 else 0.0,
            "max":  col.max(),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        }
    stats["_window_epochs"] = (window[0][0], window[-1][0])
    stats["_window_size"] = len(window)
    return stats


# ──────────────────────────────────────────────────────────────
# 3.  表格生成
# ──────────────────────────────────────────────────────────────

def _fmt(val: float, bold: bool = False) -> str:
    s = f"{val:.4f}"
    return f"\\textbf{{{s}}}" if bold else s


def generate_comparison_table(data: dict) -> None:
    """Table 1: 所有方法在不同 tp 下的最终对比表 (LaTeX)"""
    methods_order = ["SVM", "RF", "LSTM", "Transformer",
                     "Exp-A", "Exp-B", "Exp-C", "Exp-D", "Exp-E", "Exp-F",
                     "Exp-G", "Exp-H", "Exp-I", "Exp-J"]
    tp_list = [2, 3, 4]

    rows_csv = []
    rows_latex = []

    for tp in tp_list:
        tp_data = {m: data.get((m, tp)) for m in methods_order if (m, tp) in data}
        if not tp_data:
            continue

        best_per_metric = {}
        for mi in range(4):
            vals = [d["final"][mi] for d in tp_data.values() if d]
            best_per_metric[mi] = max(vals) if vals else -1

        for method in methods_order:
            key = (method, tp)
            if key not in data:
                continue
            d = data[key]
            final = d["final"]
            curve = d.get("curve", [])
            stats = compute_stable_stats(curve) if curve else {}

            csv_row = {"Method": method, "TP": tp, "Best Epoch": d.get("best_epoch", "-")}
            latex_cells = []

            for mi, mname in enumerate(METRICS):
                is_best = abs(final[mi] - best_per_metric[mi]) < 1e-6
                latex_cells.append(_fmt(final[mi], bold=is_best))
                csv_row[f"Best_{mname}"] = final[mi]

                if mname in stats:
                    s = stats[mname]
                    csv_row[f"Mean_{mname}"] = s["mean"]
                    csv_row[f"Std_{mname}"] = s["std"]
                    csv_row[f"CI_lo_{mname}"] = s["ci_lo"]
                    csv_row[f"CI_hi_{mname}"] = s["ci_hi"]

            if stats:
                f1_s = stats.get("F1", {})
                stability_str = f"${f1_s.get('mean',0):.4f} \\pm {f1_s.get('std',0):.4f}$"
            else:
                stability_str = "-"

            row_label = f"{method} (tp={tp})"
            latex_row = f"  {row_label} & " + " & ".join(latex_cells) + f" & {stability_str} \\\\"
            rows_latex.append(latex_row)
            rows_csv.append(csv_row)

        rows_latex.append("  \\midrule")

    if rows_latex and rows_latex[-1].strip() == "\\midrule":
        rows_latex.pop()

    latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Performance comparison across methods and temporal resolutions.}\n"
        "\\label{tab:comparison}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{l c c c c c}\n"
        "\\toprule\n"
        "Method & Accuracy & Precision & Recall & F1 & F1 (Mean$\\pm$Std) \\\\\n"
        "\\midrule\n"
        + "\n".join(rows_latex) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\end{table}\n"
    )

    (TABLE_DIR / "table_comparison.tex").write_text(latex)
    pd.DataFrame(rows_csv).to_csv(TABLE_DIR / "table_comparison.csv", index=False)
    print(f"  -> {TABLE_DIR / 'table_comparison.tex'}")
    print(f"  -> {TABLE_DIR / 'table_comparison.csv'}")


def _generate_ablation_bars_table(data: dict, tp: int, methods_list: List[str],
                                   display_names: Optional[Dict[str, str]], table_basename: str,
                                   caption: str, label: str) -> None:
    """生成与消融柱状图对应的表格（LaTeX + CSV），表中方法名用 display_names。"""
    methods_present = [m for m in methods_list if (m, tp) in data]
    if not methods_present:
        return

    best_per_metric = {}
    for mi in range(4):
        vals = [data[(m, tp)]["final"][mi] for m in methods_present]
        best_per_metric[mi] = max(vals) if vals else -1

    rows_latex = []
    rows_csv = []
    for method in methods_present:
        d = data[(method, tp)]
        final = d["final"]
        curve = d.get("curve", [])
        stats = compute_stable_stats(curve) if curve else {}

        display_label = (display_names or {}).get(method, method)

        latex_cells = []
        csv_row = {"Method": display_label, "TP": tp, "Best Epoch": d.get("best_epoch", "-")}
        for mi, mname in enumerate(METRICS):
            is_best = abs(final[mi] - best_per_metric[mi]) < 1e-6
            latex_cells.append(_fmt(final[mi], bold=is_best))
            csv_row[f"Best_{mname}"] = final[mi]
            if mname in stats:
                s = stats[mname]
                csv_row[f"Mean_{mname}"] = s["mean"]
                csv_row[f"Std_{mname}"] = s["std"]

        if stats and "F1" in stats:
            f1_s = stats["F1"]
            stability_str = f"${f1_s['mean']:.4f} \\pm {f1_s['std']:.4f}$"
        else:
            stability_str = "-"

        rows_latex.append(f"  {display_label} & " + " & ".join(latex_cells) + f" & {stability_str} \\\\")
        rows_csv.append(csv_row)

    latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{l c c c c c}\n"
        "\\toprule\n"
        "Method & Accuracy & Precision & Recall & F1 & F1 (Mean$\\pm$Std) \\\\\n"
        "\\midrule\n"
        + "\n".join(rows_latex) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\end{table}\n"
    )
    (TABLE_DIR / f"table_{table_basename}.tex").write_text(latex)
    pd.DataFrame(rows_csv).to_csv(TABLE_DIR / f"table_{table_basename}.csv", index=False)
    print(f"  -> {TABLE_DIR / f'table_{table_basename}.tex'}")
    print(f"  -> {TABLE_DIR / f'table_{table_basename}.csv'}")


def generate_ablation_table(data: dict) -> None:
    """Table 2: 消融实验表 (固定 tp=3)"""
    ablation_methods = ["Exp-A", "Exp-B", "Exp-C", "Exp-D", "Exp-E", "Exp-F", "Exp-G", "Exp-H", "Exp-I", "Exp-J"]
    tp = 3

    components = {
        "Exp-A": {"Env": "\\checkmark", "Physics": "", "CrossAttn": "", "CLIP": "", "AuxLoss": ""},
        "Exp-B": {"Env": "",            "Physics": "", "CrossAttn": "", "CLIP": "", "AuxLoss": ""},
        "Exp-C": {"Env": "\\checkmark", "Physics": "", "CrossAttn": "", "CLIP": "\\checkmark", "AuxLoss": ""},
        "Exp-D": {"Env": "\\checkmark", "Physics": "\\checkmark", "CrossAttn": "", "CLIP": "", "AuxLoss": ""},
        "Exp-E": {"Env": "\\checkmark", "Physics": "\\checkmark", "CrossAttn": "\\checkmark", "CLIP": "\\checkmark", "AuxLoss": ""},
        "Exp-F": {"Env": "\\checkmark", "Physics": "\\checkmark", "CrossAttn": "\\checkmark", "CLIP": "\\checkmark", "AuxLoss": "\\checkmark"},
        "Exp-G": {"Env": "", "Physics": "", "CrossAttn": "", "CLIP": "", "AuxLoss": ""},
        "Exp-H": {"Env": "", "Physics": "", "CrossAttn": "", "CLIP": "", "AuxLoss": ""},
        "Exp-I": {"Env": "", "Physics": "", "CrossAttn": "", "CLIP": "", "AuxLoss": ""},
        "Exp-J": {"Env": "", "Physics": "", "CrossAttn": "", "CLIP": "", "AuxLoss": ""},
    }

    best_per_metric = {}
    for mi in range(4):
        vals = [data[(m, tp)]["final"][mi] for m in ablation_methods if (m, tp) in data]
        best_per_metric[mi] = max(vals) if vals else -1

    rows = []
    csv_rows = []
    for method in ablation_methods:
        key = (method, tp)
        if key not in data:
            continue
        d = data[key]
        final = d["final"]
        curve = d.get("curve", [])
        stats = compute_stable_stats(curve) if curve else {}

        comp = components.get(method, {})
        comp_cells = " & ".join(comp.get(c, "") for c in ["Env", "Physics", "CrossAttn", "CLIP", "AuxLoss"])

        metric_cells = []
        csv_row = {"Method": method, "Description": EXP_DESCRIPTIONS.get(method, "")}

        for mi, mname in enumerate(METRICS):
            is_best = abs(final[mi] - best_per_metric[mi]) < 1e-6
            metric_cells.append(_fmt(final[mi], bold=is_best))
            csv_row[f"Best_{mname}"] = final[mi]

            if mname in stats:
                s = stats[mname]
                csv_row[f"Mean_{mname}"] = s["mean"]
                csv_row[f"Std_{mname}"] = s["std"]

        if stats and "F1" in stats:
            f1_s = stats["F1"]
            stab = f"${f1_s['mean']:.4f} \\pm {f1_s['std']:.4f}$"
        else:
            stab = "-"

        row = f"  {method} & {comp_cells} & " + " & ".join(metric_cells) + f" & {stab} \\\\"
        rows.append(row)
        csv_rows.append(csv_row)

    latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Ablation study (tp=3). Best values are in \\textbf{bold}.}\n"
        "\\label{tab:ablation}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{l ccccc cccc c}\n"
        "\\toprule\n"
        "  & Env & Physics & CrossAttn & CLIP & AuxLoss & Acc & Prec & Rec & F1 & F1 (Mean$\\pm$Std) \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\end{table}\n"
    )

    (TABLE_DIR / "table_ablation.tex").write_text(latex)
    pd.DataFrame(csv_rows).to_csv(TABLE_DIR / "table_ablation.csv", index=False)
    print(f"  -> {TABLE_DIR / 'table_ablation.tex'}")
    print(f"  -> {TABLE_DIR / 'table_ablation.csv'}")


# ──────────────────────────────────────────────────────────────
# 4.  图表生成
# ──────────────────────────────────────────────────────────────

def plot_training_curves(data: dict) -> None:
    """Figure 1: 按 tp 分面的训练曲线 (Accuracy + F1), 截断到 CURVE_MAX_EPOCH, SVM/RF 画水平虚线"""
    tp_list = [2, 3, 4]
    curve_methods = ["LSTM", "Transformer", "Exp-A", "Exp-B", "Exp-C", "Exp-D", "Exp-E", "Exp-F", "Exp-G", "Exp-H", "Exp-I", "Exp-J"]
    static_methods = ["SVM", "RF"]
    linestyles = {
        "LSTM": "--", "Transformer": "-.",
        "Exp-A": "-", "Exp-B": "-", "Exp-C": "-", "Exp-D": "-", "Exp-E": "-", "Exp-F": "-",
        "Exp-G": "-", "Exp-H": "-", "Exp-I": "-", "Exp-J": "-",
    }
    markers = {
        "LSTM": "s", "Transformer": "^",
        "Exp-A": "o", "Exp-B": "v", "Exp-C": "D", "Exp-D": "p", "Exp-E": "*", "Exp-F": "H",
        "Exp-G": "X", "Exp-H": "P", "Exp-I": "d", "Exp-J": "h",
    }

    for metric_idx, metric_name in [(0, "Accuracy"), (3, "F1")]:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

        for ax_i, tp in enumerate(tp_list):
            ax = axes[ax_i]

            for method in static_methods:
                key = (method, tp)
                if key not in data:
                    continue
                val = data[key]["final"][metric_idx]
                ax.axhline(y=val, color=COLORS.get(method, "#888"),
                           linestyle=":", linewidth=1.2, alpha=0.7, label=method)

            for method in curve_methods:
                key = (method, tp)
                if key not in data or not data[key]["curve"]:
                    continue
                curve = [(e, m) for e, m in data[key]["curve"] if e <= CURVE_MAX_EPOCH]
                if not curve:
                    continue
                epochs = [c[0] for c in curve]
                vals = [c[1][metric_idx] for c in curve]
                ax.plot(epochs, vals,
                        label=method,
                        color=COLORS.get(method, "#333333"),
                        linestyle=linestyles.get(method, "-"),
                        marker=markers.get(method, "o"),
                        markevery=max(len(epochs) // 10, 1),
                        alpha=0.85)

            ax.set_title(f"$T = {tp}$", fontsize=12)
            ax.set_xlabel("Epoch")
            ax.set_xlim(0, CURVE_MAX_EPOCH + 10)
            if ax_i == 0:
                ax.set_ylabel(metric_name)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        handles, labels = [], []
        for ax in axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)

        fig.legend(handles, labels,
                   loc="lower center", ncol=min(len(labels), 10),
                   bbox_to_anchor=(0.5, -0.10), frameon=True, edgecolor="#cccccc")

        fig.suptitle(f"Validation {metric_name} over Training Epochs", fontsize=13, y=1.02)
        fig.tight_layout()
        outpath = FIG_DIR / f"fig_curves_{metric_name.lower()}.pdf"
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {outpath}")


def _plot_ablation_bars_one(data: dict, tp: int, methods_list: List[str], outpath: Path,
                            baseline_set: Optional[set] = None,
                            suptitle: Optional[str] = None,
                            display_names: Optional[Dict[str, str]] = None) -> None:
    """绘制单张消融柱状图（指定方法列表），保存到 outpath。display_names 为方法名→图中显示名，仅改图例。"""
    if baseline_set is None:
        baseline_set = {"SVM", "RF", "LSTM", "Transformer"}
    methods_present = [m for m in methods_list if (m, tp) in data]
    if not methods_present:
        print(f"  [WARN] No data for tp={tp} with methods {methods_list}, skipping.")
        return

    labels = [display_names.get(m, m) for m in methods_present] if display_names else methods_present

    fig, axes = plt.subplots(1, 4, figsize=(max(10, len(methods_present) * 1.2), 4.5), sharey=False)
    x = np.arange(len(methods_present))
    width = 0.6

    for mi, (mname, ax) in enumerate(zip(METRICS, axes)):
        means = []
        stds = []
        bests = []
        for method in methods_present:
            d = data[(method, tp)]
            bests.append(d["final"][mi])
            curve = d.get("curve", [])
            if curve:
                stats = compute_stable_stats(curve)
                if mname in stats:
                    means.append(stats[mname]["mean"])
                    stds.append(stats[mname]["std"])
                else:
                    means.append(d["final"][mi])
                    stds.append(0)
            else:
                means.append(d["final"][mi])
                stds.append(0)

        colors_list = [COLORS.get(m, "#888") for m in methods_present]
        edge_styles = ["black" if m not in baseline_set else "#555"
                       for m in methods_present]
        hatches = ["//" if m in baseline_set else None
                   for m in methods_present]

        for j in range(len(methods_present)):
            ax.bar(x[j], means[j], width,
                   yerr=stds[j], capsize=3,
                   color=colors_list[j], edgecolor=edge_styles[j],
                   linewidth=0.5, hatch=hatches[j],
                   error_kw={"linewidth": 1, "ecolor": "#333"}, alpha=0.85)

        ax.scatter(x, bests, marker="*", s=60, color="red", zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_title(mname)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        all_vals = means + bests
        all_errs = stds if stds else [0]
        ymin = min(all_vals) - max(all_errs) - 0.03
        ymax = max(all_vals) + max(all_errs) + 0.03
        ax.set_ylim(max(0, ymin), min(1, ymax))

    star_legend = Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                         markersize=10, label='Best epoch')
    bar_legend = Line2D([0], [0], color='#333', linewidth=1,
                        label='Mean ± Std ')
    hatch_legend = matplotlib.patches.Patch(facecolor='#ddd', edgecolor='#555',
                                            hatch='//', label='Baseline methods')
    fig.legend(handles=[star_legend, bar_legend, hatch_legend], loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.06), frameon=True, edgecolor="#cccccc")

    title = suptitle if suptitle is not None else f"Method Comparison & Ablation ($T = {tp}$)"
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath}")


def plot_ablation_bars(data: dict) -> None:
    """Figure 2: 每个 tp 一张消融柱状图 (含 baselines + mean±std 误差棒)"""
    all_methods = ["SVM", "RF", "LSTM", "Transformer",
                   "Exp-A", "Exp-B", "Exp-C", "Exp-D", "Exp-E", "Exp-F",
                   "Exp-G", "Exp-H", "Exp-I", "Exp-J"]
    baseline_set = {"SVM", "RF", "LSTM", "Transformer"}
    tp_list = [2, 3, 4]

    for tp in tp_list:
        _plot_ablation_bars_one(data, tp, all_methods, FIG_DIR / f"fig_ablation_bars_tp{tp}.pdf",
                                baseline_set=baseline_set)

    # 额外两张图：仅 tp=3
    _plot_ablation_bars_one(
        data, 3,
        ["SVM", "LSTM", "Transformer", "Exp-F"],
        FIG_DIR / "fig_ablation_bars_tp3_baselines_expf.pdf",
        baseline_set=baseline_set,
        suptitle="Method Comparison ($T = 3$): SVM, LSTM, Transformer, Exp-F",
    )
    _plot_ablation_bars_one(
        data, 3,
        ["Exp-F", "Exp-G", "Exp-H", "Exp-I", "Exp-J"],
        FIG_DIR / "fig_ablation_bars_tp3_exp_f_to_j.pdf",
        baseline_set=baseline_set,
        suptitle="Method Comparison ($T = 3$): Exp-F to Exp-J",
    )

    # 科研版：仅改图中显示名，Exp-F→SWC-PSWM，Exp-G~J→Exp-A~D
    _plot_ablation_bars_one(
        data, 3,
        ["SVM", "LSTM", "Transformer", "Exp-F"],
        FIG_DIR / "fig_ablation_bars_tp3_baselines_expf_keyan.pdf",
        baseline_set=baseline_set,
        suptitle="Method Comparison ($T = 3$): SVM, LSTM, Transformer, SWC-PSWM",
        display_names={"Exp-F": "SWC-PSWM"},
    )
    _plot_ablation_bars_one(
        data, 3,
        ["Exp-F", "Exp-G", "Exp-H", "Exp-I", "Exp-J"],
        FIG_DIR / "fig_ablation_bars_tp3_exp_f_to_j_keyan.pdf",
        baseline_set=baseline_set,
        suptitle="Method Comparison ($T = 3$): SWC-PSWM, Exp-A to Exp-D",
        display_names={"Exp-F": "SWC-PSWM", "Exp-G": "Exp-A", "Exp-H": "Exp-B", "Exp-I": "Exp-C", "Exp-J": "Exp-D"},
    )

    # 与上述两张 keyan PDF 对应的表格
    _generate_ablation_bars_table(
        data, 3,
        ["SVM", "LSTM", "Transformer", "Exp-F"],
        {"Exp-F": "SWC-PSWM"},
        "ablation_bars_tp3_baselines_expf_keyan",
        caption="Performance table for fig\\_ablation\\_bars\\_tp3\\_baselines\\_expf\\_keyan.pdf ($T=3$: SVM, LSTM, Transformer, SWC-PSWM). Best values in \\textbf{bold}.",
        label="tab:ablation_bars_baselines_expf_keyan",
    )
    _generate_ablation_bars_table(
        data, 3,
        ["Exp-F", "Exp-G", "Exp-H", "Exp-I", "Exp-J"],
        {"Exp-F": "SWC-PSWM", "Exp-G": "Exp-A", "Exp-H": "Exp-B", "Exp-I": "Exp-C", "Exp-J": "Exp-D"},
        "ablation_bars_tp3_exp_f_to_j_keyan",
        caption="Performance table for fig\\_ablation\\_bars\\_tp3\\_exp\\_f\\_to\\_j\\_keyan.pdf ($T=3$: SWC-PSWM, Exp-A to Exp-D). Best values in \\textbf{bold}.",
        label="tab:ablation_bars_exp_f_to_j_keyan",
    )


def plot_tp_sensitivity(data: dict) -> None:
    """Figure 3: num_time_points 敏感性分析 (分方法画线)"""
    all_methods = ["SVM", "RF", "LSTM", "Transformer",
                   "Exp-A", "Exp-B", "Exp-C", "Exp-D", "Exp-E", "Exp-F",
                   "Exp-G", "Exp-H", "Exp-I", "Exp-J"]
    tp_list = [2, 3, 4]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    metric_indices = [(0, "Accuracy"), (3, "F1")]

    for ax, (mi, mname) in zip(axes, metric_indices):
        for method in all_methods:
            tps_present = []
            vals = []
            for tp in tp_list:
                if (method, tp) in data:
                    tps_present.append(tp)
                    vals.append(data[(method, tp)]["final"][mi])
            if not tps_present:
                continue

            is_baseline = method in ("SVM", "RF", "LSTM", "Transformer")
            ax.plot(tps_present, vals,
                    label=method,
                    color=COLORS.get(method, "#333"),
                    linestyle="--" if is_baseline else "-",
                    marker="s" if is_baseline else "o",
                    markersize=6, alpha=0.85)

        ax.set_xlabel("Number of Time Points ($T$)")
        ax.set_ylabel(mname)
        ax.set_xticks(tp_list)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.set_title(f"Best {mname} vs. Temporal Resolution")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(labels), 5), bbox_to_anchor=(0.5, -0.12),
               frameon=True, edgecolor="#cccccc")
    fig.tight_layout()
    outpath = FIG_DIR / "fig_tp_sensitivity.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath}")


def plot_method_comparison_grouped_bar(data: dict) -> None:
    """Figure 4: 所有方法在 tp=3 下的分组柱状图 (含 baselines)"""
    methods_order = ["SVM", "RF", "LSTM", "Transformer",
                     "Exp-A", "Exp-B", "Exp-C", "Exp-D", "Exp-E", "Exp-F",
                     "Exp-G", "Exp-H", "Exp-I", "Exp-J"]
    tp = 3

    methods_present = [m for m in methods_order if (m, tp) in data]
    if len(methods_present) < 2:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(methods_present))
    n_metrics = 4
    total_width = 0.75
    bar_w = total_width / n_metrics

    for mi, mname in enumerate(METRICS):
        vals = [data[(m, tp)]["final"][mi] for m in methods_present]
        offset = (mi - n_metrics / 2 + 0.5) * bar_w
        ax.bar(x + offset, vals, bar_w, label=mname,
               alpha=0.85, edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(methods_present, rotation=35, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Method Comparison ($T = 3$, Best Epoch)", fontsize=12)
    ax.legend(loc="upper left", frameon=True, edgecolor="#cccccc")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylim(0.4, 1.0)
    fig.tight_layout()
    outpath = FIG_DIR / "fig_method_comparison.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath}")


def generate_stability_table(data: dict) -> None:
    """Table 3: 稳定性分析表 (Mean±Std, 95% CI, Max)，各方法各 tp"""
    methods_order = ["LSTM", "Transformer",
                     "Exp-A", "Exp-B", "Exp-C", "Exp-D", "Exp-E", "Exp-F",
                     "Exp-G", "Exp-H", "Exp-I", "Exp-J"]
    tp_list = [2, 3, 4]

    rows = []
    csv_rows = []

    for tp in tp_list:
        for method in methods_order:
            key = (method, tp)
            if key not in data or not data[key]["curve"]:
                continue
            stats = compute_stable_stats(data[key]["curve"])
            if "F1" not in stats:
                continue

            s = stats["F1"]
            win_ep = stats.get("_window_epochs", ("?", "?"))
            win_n = stats.get("_window_size", "?")
            row_label = f"{method} (tp={tp})"
            latex_row = (
                f"  {row_label} & "
                f"${s['mean']:.4f} \\pm {s['std']:.4f}$ & "
                f"[{s['ci_lo']:.4f}, {s['ci_hi']:.4f}] & "
                f"{s['max']:.4f} & "
                f"ep {win_ep[0]}--{win_ep[1]} ($n$={win_n}) \\\\"
            )
            rows.append(latex_row)
            csv_rows.append({
                "Method": method, "TP": tp,
                "F1_Mean": s["mean"], "F1_Std": s["std"],
                "F1_CI_lo": s["ci_lo"], "F1_CI_hi": s["ci_hi"],
                "F1_Max": s["max"],
                "Window": f"{win_ep[0]}-{win_ep[1]} (n={win_n})",
            })

        rows.append("  \\midrule")

    if rows and rows[-1].strip() == "\\midrule":
        rows.pop()

    latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Training stability analysis (F1 score in a $\\pm$100-epoch window around peak, 95\\% Bootstrap CI).}\n"
        "\\label{tab:stability}\n"
        "\\begin{tabular}{l c c c c}\n"
        "\\toprule\n"
        "Method & F1 (Mean$\\pm$Std) & 95\\% CI & Best F1 & Window \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    (TABLE_DIR / "table_stability.tex").write_text(latex)
    pd.DataFrame(csv_rows).to_csv(TABLE_DIR / "table_stability.csv", index=False)
    print(f"  -> {TABLE_DIR / 'table_stability.tex'}")
    print(f"  -> {TABLE_DIR / 'table_stability.csv'}")


# ──────────────────────────────────────────────────────────────
# 5.  主入口
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  论文图表生成")
    print("=" * 60)

    print("\n[1/7] 收集日志数据...")
    data = collect_all_data()
    print(f"  共解析到 {len(data)} 个 (method, tp) 组合:")
    for k in sorted(data.keys()):
        d = data[k]
        n = len(d.get("curve", []))
        print(f"    {k[0]:15s} tp={k[1]}  |  eval_points={n:>3d}  |  best_F1={d['final'][3]:.4f}")

    print("\n[2/7] 生成方法对比总表...")
    generate_comparison_table(data)

    print("\n[3/7] 生成消融实验表...")
    generate_ablation_table(data)

    print("\n[4/7] 生成训练稳定性表...")
    generate_stability_table(data)

    print("\n[5/7] 绘制训练曲线...")
    plot_training_curves(data)

    print("\n[6/7] 绘制消融柱状图...")
    plot_ablation_bars(data)

    print("\n[7/7] 绘制时间点敏感性 + 方法对比图...")
    plot_tp_sensitivity(data)
    plot_method_comparison_grouped_bar(data)

    print("\n" + "=" * 60)
    print("  全部完成！")
    print(f"  图片: {FIG_DIR}")
    print(f"  表格: {TABLE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
