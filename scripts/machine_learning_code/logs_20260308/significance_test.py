"""
统计显著性检验脚本 —— McNemar 检验 + Bootstrap 置信区间

步骤:
  1. 从 baseline 日志解析逐样本预测
  2. 对 exp_new 模型: 加载 best checkpoint + val dataset → 推理获取逐样本预测
  3. McNemar 检验 (两两方法) + Bootstrap F1 差值检验

用法:
  cd /home/kudouzala/Application/ion_detect
  conda run -n ion_detect python scripts/machine_learning_code/logs/significance_test.py

输出:
  scripts/machine_learning_code/logs/tables/table_mcnemar.tex
  scripts/machine_learning_code/logs/tables/table_mcnemar.csv
  scripts/machine_learning_code/logs/tables/table_bootstrap_sig.tex
  scripts/machine_learning_code/logs/tables/table_bootstrap_sig.csv
"""

from __future__ import annotations
import sys, re, os, warnings, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import combinations

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_CODE_DIR  = PROJECT_ROOT / "scripts" / "machine_learning_code"
LOG_DIR      = ML_CODE_DIR / "logs"
TABLE_DIR    = LOG_DIR / "tables"
TABLE_DIR.mkdir(exist_ok=True)
PRED_CACHE_DIR = LOG_DIR / "predictions_cache"
PRED_CACHE_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ML_CODE_DIR))

BOOTSTRAP_N = 10000
CI_LEVEL = 0.95


# ──────────────────────────────────────────────────────────
# 1. Parse baseline per-sample predictions from logs
# ──────────────────────────────────────────────────────────

def _tp_from_filename(name: str) -> int:
    if "num_tp2" in name or "_tp2" in name or "_t2" in name:
        return 2
    if "num_tp4" in name or "_tp4" in name or "_t4" in name:
        return 4
    return 3


def parse_baseline_predictions(log_path: Path) -> List[Tuple[str, str, str]]:
    """
    从 baseline 日志解析逐样本预测。
    返回: [(sample_name, true_label, pred_label), ...]
    """
    lines = log_path.read_text(errors="replace").splitlines()
    results = []
    in_detail = False
    for line in lines:
        if "详细测试结果" in line:
            in_detail = True
            continue
        if in_detail:
            line = line.strip()
            if not line:
                continue
            if line.startswith("---") or line.startswith("准确率") or line.startswith("精确率"):
                break
            parts = line.split("\t")
            if len(parts) >= 3:
                sample = parts[0].strip()
                true_label = parts[1].strip()
                pred_label = parts[2].strip()
                results.append((sample, true_label, pred_label))
    return results


def collect_baseline_predictions() -> Dict[Tuple[str, int], List[Tuple[str, str, str]]]:
    """收集所有 baseline 逐样本预测, key = (method, tp)"""
    data = {}

    method_patterns = {
        "SVM": "log_main_svm_baseline_",
        "RF":  "log_main_rf_baseline_",
        "LSTM": "log_main_lstm_baseline_",
        "Transformer": "log_main_transformer_baseline_",
    }

    for method, prefix in method_patterns.items():
        for f in sorted(LOG_DIR.glob(f"{prefix}*.txt")):
            tp = _tp_from_filename(f.name)
            preds = parse_baseline_predictions(f)
            if preds:
                data[(method, tp)] = preds
                print(f"  {method} (tp={tp}): {len(preds)} samples from {f.name}")

    return data


# ──────────────────────────────────────────────────────────
# 2. Get exp_new per-sample predictions via model inference
# ──────────────────────────────────────────────────────────

def get_exp_new_predictions(cfg_name: str, tp: int) -> Optional[List[Tuple[str, str, str]]]:
    """
    加载 exp_new 的 best checkpoint, 在 val set 上逐样本推理, 返回逐样本预测。
    使用 base_val 数据集（非配对）直接遍历，确保获取真实文件名。
    """
    cache_file = PRED_CACHE_DIR / f"{cfg_name}_tp{tp}_predictions.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return [tuple(x) for x in json.load(f)]

    import torch
    import yaml
    from model_new_models import IonDetectModel
    from model_datasets import Dataset_2_Stable_plus

    yaml_name = cfg_name
    if tp == 2:
        yaml_name += "_num_tp2"
    elif tp == 4:
        yaml_name += "_num_tp4"
    yaml_path = ML_CODE_DIR / f"{yaml_name}.yaml"

    if not yaml_path.exists():
        print(f"  [SKIP] {yaml_path.name} not found")
        return None

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg_name if tp == 3 else f"{cfg_name}_num_tp{tp}"
    model_dir = PROJECT_ROOT / "output" / "trained_model_save" / exp_name
    best_ckpt = model_dir / "trained_model_epoch_best.pth"
    if not best_ckpt.exists():
        print(f"  [SKIP] {best_ckpt} not found")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = IonDetectModel(cfg).to(device)
    state = torch.load(str(best_ckpt), map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    d = cfg["data"]
    base_dir = PROJECT_ROOT
    test_dir = base_dir / "datasets" / "datasets_for_all_test_no_overlap" / exp_name
    stats_file = base_dir / "datasets" / "stats" / f"{exp_name}_stats.json"

    if not test_dir.exists():
        print(f"  [SKIP] test dir {test_dir} not found")
        return None

    if not stats_file.exists():
        stats_candidates = list((base_dir / "datasets" / "stats").glob(f"*{exp_name}*"))
        if not stats_candidates:
            stats_candidates = list((base_dir / "datasets" / "stats").glob("*.json"))
        if stats_candidates:
            stats_file = stats_candidates[0]

    base_val = Dataset_2_Stable_plus(
        data_folder=test_dir,
        stats_file=str(stats_file),
        save_stats=False,
        num_time_points=int(d["num_time_points"]),
    )

    label_mapping = cfg["data"].get("label_mapping", {})
    idx_to_name = {v: k for k, v in label_mapping.items()} if label_mapping else {}

    results = []
    with torch.no_grad():
        for i in range(len(base_val)):
            sample = base_val[i]
            # sample = (volt, impe, env, label, true_v, ep, conc)
            volt = sample[0].unsqueeze(0).to(device)
            impe = sample[1].unsqueeze(0).to(device)
            env  = sample[2].unsqueeze(0).to(device)
            ep   = sample[5].unsqueeze(0).to(device)
            conc = sample[6].unsqueeze(0).to(device) if len(sample) > 6 else None

            out = model(volt, impe, env, ep, conc)
            pred_idx = out["prob"].argmax(dim=1).item()
            true_idx = sample[3].item() if hasattr(sample[3], "item") else int(sample[3])

            fname = base_val.file_names[i]
            pred_str = idx_to_name.get(pred_idx, str(pred_idx))
            true_str = idx_to_name.get(true_idx, str(true_idx))
            results.append((fname, true_str, pred_str))

    with open(cache_file, "w") as f:
        json.dump(results, f)
    print(f"  {exp_name}: {len(results)} samples (cached to {cache_file.name})")
    return results


def collect_exp_new_predictions() -> Dict[Tuple[str, int], List[Tuple[str, str, str]]]:
    """收集所有 exp_new 逐样本预测"""
    data = {}
    exp_names = ["exp_new_a", "exp_new_b", "exp_new_c",
                 "exp_new_d", "exp_new_e", "exp_new_f"]
    tp_list = [2, 3, 4]

    for exp_name in exp_names:
        label = f"Exp-{exp_name[-1].upper()}"
        for tp in tp_list:
            try:
                preds = get_exp_new_predictions(exp_name, tp)
                if preds:
                    data[(label, tp)] = preds
            except Exception as e:
                print(f"  [ERROR] {exp_name} tp={tp}: {e}")
    return data


# ──────────────────────────────────────────────────────────
# 3. McNemar's Test
# ──────────────────────────────────────────────────────────

def mcnemar_test(preds_a: List[Tuple[str, str, str]],
                 preds_b: List[Tuple[str, str, str]]) -> dict:
    """
    McNemar 检验: 比较两个分类器在同一测试集上的差异。
    需要两组预测涉及相同的样本集。
    """
    from scipy.stats import chi2

    map_a = {s[0]: (s[1] == s[2]) for s in preds_a}
    map_b = {s[0]: (s[1] == s[2]) for s in preds_b}

    common = set(map_a.keys()) & set(map_b.keys())
    if len(common) < 10:
        return {"n_common": len(common), "p_value": None, "statistic": None,
                "b": 0, "c": 0, "note": "too few common samples"}

    b = sum(1 for s in common if map_a[s] and not map_b[s])
    c = sum(1 for s in common if not map_a[s] and map_b[s])

    if b + c == 0:
        return {"n_common": len(common), "p_value": 1.0, "statistic": 0.0,
                "b": b, "c": c, "note": "no discordant pairs"}

    if b + c < 25:
        from scipy.stats import binomtest
        result = binomtest(b, b + c, 0.5, alternative='two-sided')
        return {"n_common": len(common), "p_value": result.pvalue,
                "statistic": None, "b": b, "c": c, "note": "exact binomial (small n)"}

    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(stat, df=1)

    return {"n_common": len(common), "p_value": p_value, "statistic": stat,
            "b": b, "c": c, "note": "chi2 with continuity correction"}


# ──────────────────────────────────────────────────────────
# 4. Bootstrap Significance Test on F1 Difference
# ──────────────────────────────────────────────────────────

def bootstrap_f1_diff(preds_a: List[Tuple[str, str, str]],
                      preds_b: List[Tuple[str, str, str]],
                      n_boot: int = BOOTSTRAP_N) -> dict:
    """
    Bootstrap 检验 F1 差值是否显著。
    构建共同样本的 correct/incorrect 向量, 重采样计算 F1 差值分布。
    """
    map_a = {s[0]: (s[1], s[2]) for s in preds_a}
    map_b = {s[0]: (s[1], s[2]) for s in preds_b}
    common = sorted(set(map_a.keys()) & set(map_b.keys()))

    if len(common) < 10:
        return {"n_common": len(common), "f1_diff": None, "p_value": None,
                "ci_lo": None, "ci_hi": None}

    correct_a = np.array([1 if map_a[s][0] == map_a[s][1] else 0 for s in common])
    correct_b = np.array([1 if map_b[s][0] == map_b[s][1] else 0 for s in common])

    acc_a = correct_a.mean()
    acc_b = correct_b.mean()
    observed_diff = acc_a - acc_b

    rng = np.random.default_rng(42)
    n = len(common)
    boot_diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        d = correct_a[idx].mean() - correct_b[idx].mean()
        boot_diffs.append(d)
    boot_diffs = np.array(boot_diffs)

    p_value = (np.abs(boot_diffs) >= np.abs(observed_diff)).mean()
    if p_value == 0:
        p_value = 1.0 / n_boot

    ci_lo = np.percentile(boot_diffs, (1 - CI_LEVEL) / 2 * 100)
    ci_hi = np.percentile(boot_diffs, (1 + CI_LEVEL) / 2 * 100)

    return {
        "n_common": n,
        "acc_a": acc_a, "acc_b": acc_b,
        "acc_diff": observed_diff,
        "p_value": p_value,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
    }


# ──────────────────────────────────────────────────────────
# 5. Generate Tables
# ──────────────────────────────────────────────────────────

def _sig_symbol(p: Optional[float]) -> str:
    if p is None:
        return "-"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def generate_mcnemar_table(all_preds: dict, tp: int) -> pd.DataFrame:
    """对指定 tp 的所有方法两两做 McNemar 检验"""
    methods = sorted([m for (m, t) in all_preds if t == tp])
    rows = []

    for m_a, m_b in combinations(methods, 2):
        preds_a = all_preds.get((m_a, tp))
        preds_b = all_preds.get((m_b, tp))
        if not preds_a or not preds_b:
            continue

        mc = mcnemar_test(preds_a, preds_b)
        bs = bootstrap_f1_diff(preds_a, preds_b)

        acc_a = sum(1 for s in preds_a if s[1] == s[2]) / len(preds_a) if preds_a else 0
        acc_b = sum(1 for s in preds_b if s[1] == s[2]) / len(preds_b) if preds_b else 0

        rows.append({
            "Method A": m_a,
            "Method B": m_b,
            "TP": tp,
            "Acc A": f"{acc_a:.4f}",
            "Acc B": f"{acc_b:.4f}",
            "n_common": mc["n_common"],
            "McNemar b": mc["b"],
            "McNemar c": mc["c"],
            "McNemar p": mc["p_value"],
            "McNemar sig": _sig_symbol(mc["p_value"]),
            "Bootstrap p": bs.get("p_value"),
            "Bootstrap sig": _sig_symbol(bs.get("p_value")),
            "Bootstrap CI": f"[{bs.get('ci_lo', 0):.4f}, {bs.get('ci_hi', 0):.4f}]" if bs.get("ci_lo") is not None else "-",
        })

    return pd.DataFrame(rows)


def save_latex_significance_table(df: pd.DataFrame, tp: int):
    """输出 McNemar 检验 LaTeX 表"""
    if df.empty:
        return

    rows_tex = []
    for _, r in df.iterrows():
        mc_p = f"{r['McNemar p']:.4f}" if r["McNemar p"] is not None else "-"
        bs_p = f"{r['Bootstrap p']:.4f}" if r["Bootstrap p"] is not None else "-"
        row = (
            f"  {r['Method A']} vs {r['Method B']} & "
            f"{r['Acc A']} & {r['Acc B']} & "
            f"{r['n_common']} & "
            f"{r['McNemar b']} & {r['McNemar c']} & "
            f"{mc_p} & {r['McNemar sig']} & "
            f"{bs_p} & {r['Bootstrap sig']} & "
            f"{r['Bootstrap CI']} \\\\"
        )
        rows_tex.append(row)

    latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{Statistical significance tests ($T = {tp}$). "
        "McNemar's test and bootstrap test on accuracy difference. "
        "$^{{***}}p<0.001$, $^{{**}}p<0.01$, $^{{*}}p<0.05$, n.s. = not significant.}}\n"
        f"\\label{{tab:significance_tp{tp}}}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{l cc c cc cc cc c}\n"
        "\\toprule\n"
        "Comparison & Acc A & Acc B & $n$ & $b$ & $c$ & McNemar $p$ & Sig. & Bootstrap $p$ & Sig. & 95\\% CI \\\\\n"
        "\\midrule\n"
        + "\n".join(rows_tex) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\end{table}\n"
    )

    tex_path = TABLE_DIR / f"table_significance_tp{tp}.tex"
    tex_path.write_text(latex)
    csv_path = TABLE_DIR / f"table_significance_tp{tp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  -> {tex_path}")
    print(f"  -> {csv_path}")


# ──────────────────────────────────────────────────────────
# 6. Main
# ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  统计显著性检验")
    print("=" * 60)

    print("\n[1/4] 解析 baseline 逐样本预测...")
    baseline_preds = collect_baseline_predictions()

    print("\n[2/4] 获取 exp_new 逐样本预测 (加载模型推理)...")
    exp_preds = collect_exp_new_predictions()

    all_preds = {**baseline_preds, **exp_preds}
    print(f"\n  共 {len(all_preds)} 个 (method, tp) 有逐样本预测")

    print("\n[3/4] 运行 McNemar 检验 + Bootstrap 检验...")
    for tp in [2, 3, 4]:
        methods_at_tp = [m for (m, t) in all_preds if t == tp]
        if len(methods_at_tp) < 2:
            print(f"  tp={tp}: 不足 2 个方法, 跳过")
            continue
        print(f"\n  --- tp={tp} ({len(methods_at_tp)} methods) ---")
        df = generate_mcnemar_table(all_preds, tp)
        if not df.empty:
            save_latex_significance_table(df, tp)

            sig_pairs = df[df["McNemar sig"] != "n.s."]
            print(f"  显著差异对数: {len(sig_pairs)} / {len(df)}")
        else:
            print(f"  tp={tp}: 无法生成对比表")

    print("\n" + "=" * 60)
    print("  显著性检验完成！")
    print(f"  表格输出: {TABLE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
