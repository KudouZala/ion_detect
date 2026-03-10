#!/usr/bin/env python3
"""
Dataset statistics for manuscript (Comment 9).
Scans datasets/datasets_for_all and reports:
- Total number of experiments (unique experiment runs)
- Number of samples (sliding windows) per ion type
- Total number of sliding windows after augmentation
- Train/validation/test split ratios and splitting strategy
"""
from pathlib import Path
import re
import random
import json
from collections import defaultdict

# 与 main.py 一致：项目根目录
def resolve_base_dir():
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent

# 与 main.py 一致的解析正则
_NO_OVERLAP_REGEX = re.compile(
    r"^(?P<prefix>.*?_\[)(?P<times>\d+(?:\s*,\s*\d+)*)\]\s*(?:\.\w+)?$"
)

def parse_no_overlap_key(fname: str):
    """解析文件名，返回 (prefix, window_tuple) 或 None。"""
    m = _NO_OVERLAP_REGEX.match(fname.strip())
    if not m:
        return None
    times_str = m.group("times")
    try:
        window = tuple(sorted(int(t.strip()) for t in times_str.split(",")))
    except ValueError:
        return None
    return (m.group("prefix"), window)


def infer_ion_from_filename(fname: str) -> str:
    """从文件名推断离子类型（与 main.py 一致）；no_ion 细分为污染前 / H2SO4恢复后。"""
    keywords = [
        ("钙离子", "钙离子 (Ca2+)"),
        ("钠离子", "钠离子 (Na+)"),
        ("镍离子", "镍离子 (Ni2+)"),
        ("铬离子", "铬离子 (Cr3+)"),
        ("铜离子", "铜离子 (Cu2+)"),
        ("铁离子", "铁离子 (Fe3+)"),
    ]
    for kw, label in keywords:
        if kw in fname and "ion_column" not in fname:
            return label
    lower = fname.lower()
    if "blank" in lower or "纯水" in lower or "ion_column" in fname:
        # no_ion 细分：根据文件名区分为污染前 / H2SO4恢复后
        if "renew_h2so4" in lower or "ion_column_renew_h2so4" in fname:
            return "无污染 (no_ion) - H2SO4恢复后"
        return "无污染 (no_ion) - 污染前"
    return "其他/未知"


# 污染状态 = 6 种离子（不含 no_ion 污染前/H2SO4恢复后）
CONTAMINATION_ION_LABELS = {
    "钙离子 (Ca2+)",
    "钠离子 (Na+)",
    "镍离子 (Ni2+)",
    "铬离子 (Cr3+)",
    "铜离子 (Cu2+)",
    "铁离子 (Fe3+)",
}


def main():
    base_dir = resolve_base_dir()
    source_dir = base_dir / "datasets" / "datasets_for_all"
    if not source_dir.exists():
        print(f"错误: 目录不存在 {source_dir}")
        return

    all_xlsx = list(source_dir.glob("*.xlsx"))
    parsed = []
    for p in all_xlsx:
        key = parse_no_overlap_key(p.name)
        if key is None:
            continue
        prefix, window = key
        ion = infer_ion_from_filename(p.name)
        parsed.append({"path": p, "prefix": prefix, "window": window, "ion": ion})

    # ----- 1. Total experiments (unique prefixes) -----
    unique_prefixes = set(p["prefix"] for p in parsed)
    n_experiments = len(unique_prefixes)

    # ----- 2. Samples (sliding windows) per ion type -----
    ion_to_count = defaultdict(int)
    for p in parsed:
        ion_to_count[p["ion"]] += 1

    # ----- 2b. 污染状态（6 种离子）：按实验(前缀)分组，同一前缀内对 [t1,t2,...] 取并集，再对每种离子汇总 -----
    # 思路：前缀完全相同的才是同一实验；每个 (prefix, ion) 内对时间点取并集，该并集大小 = 该实验该离子的时间点数。每种离子的时间点总数 = 各实验(该离子)的并集大小之和。
    prefix_ion_to_time_points = defaultdict(set)
    for p in parsed:
        if p["ion"] in CONTAMINATION_ION_LABELS:
            prefix_ion_to_time_points[(p["prefix"], p["ion"])].update(p["window"])
    ion_to_total_time_points = defaultdict(int)
    for (prefix, ion), times in prefix_ion_to_time_points.items():
        ion_to_total_time_points[ion] += len(times)
    time_points_contamination_total = sum(ion_to_total_time_points.get(ion, 0) for ion in CONTAMINATION_ION_LABELS)

    # ----- 3. Total sliding windows -----
    n_total_windows = len(parsed)

    # ----- 4. Train/test split (no_overlap strategy, seed=42) -----
    n_test = 150
    seed = 42
    random.seed(seed)
    shuffled = list(parsed)
    random.shuffle(shuffled)
    test_selected = shuffled[:n_test]
    test_keys = {(x["prefix"], x["window"]) for x in test_selected}

    prefix_to_active_times = {}
    for x in test_selected:
        prefix, window = x["prefix"], x["window"]
        active = set(sorted(window)[:4])  # num_time_points=4
        if prefix not in prefix_to_active_times:
            prefix_to_active_times[prefix] = set(active)
        else:
            prefix_to_active_times[prefix] |= active

    def excluded_from_train(item):
        prefix, window = item["prefix"], item["window"]
        if (prefix, window) in test_keys:
            return True
        if prefix not in prefix_to_active_times:
            return False
        return bool(set(window) & prefix_to_active_times[prefix])

    train_selected = [x for x in parsed if not excluded_from_train(x)]
    n_train = len(train_selected)
    n_test_actual = len(test_selected)

    # Validation: 代码中 validation 使用的是 test_folder 中的样本，即与 test 相同集合
    # 若需单独 val 比例，通常从 train 中再划分；此处按实际用法：train / test (test 兼作 val)
    train_ratio = n_train / n_total_windows if n_total_windows else 0
    test_ratio = n_test_actual / n_total_windows if n_total_windows else 0

    # ----- 5. Experiments per ion type -----
    ion_to_experiments = defaultdict(set)
    for p in parsed:
        ion_to_experiments[p["ion"]].add(p["prefix"])
    ion_to_n_exp = {k: len(v) for k, v in ion_to_experiments.items()}

    # ----- 6. Time points per sample -----
    # 每个 sample 是一个 sliding window，通常含 4 个时间点 (0,2,4,6) 或 2 个等
    window_sizes = defaultdict(int)
    for p in parsed:
        window_sizes[len(p["window"])] += 1

    # ----- Output -----
    stats = {
        "total_experiments": n_experiments,
        "total_sliding_windows": n_total_windows,
        "samples_per_ion_type": dict(ion_to_count),
        "experiments_per_ion_type": ion_to_n_exp,
        "time_points_without_augmentation_per_ion": dict(ion_to_total_time_points),
        "time_points_contamination_only": {
            ion: ion_to_total_time_points[ion] for ion in CONTAMINATION_ION_LABELS if ion in ion_to_total_time_points
        },
        "time_points_contamination_total": time_points_contamination_total,
        "time_points_per_sample_distribution": dict(window_sizes),
        "split": {
            "train": n_train,
            "test": n_test_actual,
            "train_ratio": round(train_ratio, 4),
            "test_ratio": round(test_ratio, 4),
            "splitting_method": (
                "By sample (sliding window): random 150 samples as test; "
                "train = remainder excluding any sample that shares the same "
                "experiment (prefix) and overlaps in time with a test sample's "
                "first 4 time points (no temporal leakage)."
            ),
        },
    }

    out_path = base_dir / "datasets" / "stats" / "dataset_statistics_for_manuscript.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"已写入: {out_path}")

    # ----- 打印供论文/审稿人直接引用的摘要 -----
    print("\n" + "=" * 60)
    print("Dataset statistics for manuscript (Comment 9)")
    print("=" * 60)
    print(f"\n1. Total number of experiments conducted: {n_experiments}")
    print(f"\n2. Number of samples (sliding windows) per ion type:")
    for ion in sorted(ion_to_count.keys(), key=lambda x: -ion_to_count[x]):
        print(f"   - {ion}: {ion_to_count[ion]} samples, {ion_to_n_exp.get(ion, 0)} experiments")
    print(f"\n2b. 污染状态时间点总数（同一实验=同前缀内对 [t1,t2,...] 取并集，每种离子 = 各实验并集大小之和，不做滑动窗口）:")
    for ion in sorted(CONTAMINATION_ION_LABELS, key=lambda x: -ion_to_total_time_points.get(x, 0)):
        if ion not in ion_to_total_time_points:
            continue
        n_exp = ion_to_n_exp.get(ion, 0)
        n_samp = ion_to_count.get(ion, 0)
        n_tp = ion_to_total_time_points[ion]
        avg = n_tp / n_exp if n_exp else 0
        print(f"   - {ion}: {n_tp} time points ({n_exp} experiments, avg {avg:.1f} time points/experiment; {n_samp} samples)")
    print(f"   - 【合计】污染状态时间点总数: {time_points_contamination_total}")
    print(f"\n3. Total number of sliding windows (after augmentation): {n_total_windows}")
    print(f"   (Time points per sample: distribution {dict(window_sizes)})")
    print(f"\n4. Train/Test split (no-overlap strategy, seed={seed}):")
    print(f"   - Train: {n_train} samples ({train_ratio*100:.2f}%)")
    print(f"   - Test:  {n_test_actual} samples ({test_ratio*100:.2f}%)")
    print(f"   - Splitting: {stats['split']['splitting_method']}")
    print("=" * 60)

    # ----- 英文摘要（可直接用于稿件） -----
    en_path = base_dir / "datasets" / "stats" / "dataset_statistics_manuscript_en.txt"
    n_excluded = n_total_windows - n_train - n_test_actual
    # no_ion 子类在英文摘要中的表述
    ion_display_en = {
        "无污染 (no_ion) - 污染前": "no_ion (before contamination)",
        "无污染 (no_ion) - H2SO4恢复后": "no_ion (after H2SO4 recovery)",
    }
    en_lines = [
        "Dataset statistics for manuscript (Comment 9)",
        "==============================================",
        "",
        f"- Total number of experiments conducted: {n_experiments}",
        "",
        "- Number of samples (sliding windows) per ion type (no_ion split: before contamination / after H2SO4 recovery):",
    ]
    for ion in sorted(ion_to_count.keys(), key=lambda x: -ion_to_count[x]):
        label_en = ion_display_en.get(ion, ion)
        en_lines.append(f"  {label_en}: {ion_to_count[ion]} samples ({ion_to_n_exp.get(ion, 0)} experiments)")
    en_lines.append("")
    en_lines.append("- Number of time points per ion (contamination only): per experiment (same prefix), union of [t1,t2,...] in filenames; total = sum of union sizes over experiments (no sliding-window):")
    for ion in sorted(CONTAMINATION_ION_LABELS, key=lambda x: -ion_to_total_time_points.get(x, 0)):
        if ion not in ion_to_total_time_points:
            continue
        label_en = ion_display_en.get(ion, ion)
        n_exp = ion_to_n_exp.get(ion, 0)
        n_samp = ion_to_count.get(ion, 0)
        n_tp = ion_to_total_time_points[ion]
        avg = n_tp / n_exp if n_exp else 0
        en_lines.append(f"  {label_en}: {n_tp} time points ({n_exp} experiments, avg {avg:.1f} per experiment; {n_samp} samples)")
    en_lines.append(f"  Total (contamination states): {time_points_contamination_total} time points")
    en_lines.extend([
        "",
        f"- Total number of sliding windows after augmentation: {n_total_windows}",
        f"  (Each sample contains 4 time points; window size distribution: {dict(window_sizes)})",
        "",
        "- Train/validation/test split:",
        f"  Train: {n_train} samples ({100*train_ratio:.2f}% of total).",
        f"  Test:  {n_test_actual} samples ({100*test_ratio:.2f}% of total).",
        f"  Excluded from train (temporal overlap with test): {n_excluded} samples.",
        "",
        "- How splitting was performed: By sample (sliding window). We randomly selected 150 samples as the test set (fixed seed). "
        "The training set consists of the remainder, excluding any sample that shares the same experiment (same time-series run) "
        "and overlaps in time with a test sample's first 4 time points, to avoid temporal leakage. "
        "Validation during training uses the same test set.",
    ])
    with open(en_path, "w", encoding="utf-8") as f:
        f.write("\n".join(en_lines) + "\n")
    print(f"\n英文摘要已写入: {en_path}")


if __name__ == "__main__":
    main()
