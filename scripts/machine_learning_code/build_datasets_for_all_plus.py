import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


WINDOW_RE = re.compile(
    r"^(?P<prefix>.*?_)\[(?P<times>\d+(?:\s*,\s*\d+)*)\]\s*(?:\.\w+)?$"
)


def parse_window_from_name(fname: str):
    m = WINDOW_RE.match(fname.strip())
    if not m:
        return None
    prefix = m.group("prefix")
    times = [int(x.strip()) for x in m.group("times").split(",")]
    return prefix, times


def parse_int_list(text: str) -> List[int]:
    out = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not out:
        raise ValueError("window_sizes 不能为空")
    return out


def infer_base_step(times: List[int]) -> int:
    if len(times) < 2:
        return 1
    diffs = [b - a for a, b in zip(times[:-1], times[1:]) if (b - a) > 0]
    if not diffs:
        return 1
    # 以最小正间隔作为“连续窗口”步长，常见为 2h
    return min(diffs)


def build_windows(sorted_times: List[int], k: int, step: int) -> List[Tuple[int, ...]]:
    if k <= 0:
        return []
    out = []
    n = len(sorted_times)
    for i in range(0, n - k + 1):
        w = sorted_times[i : i + k]
        ok = True
        for a, b in zip(w[:-1], w[1:]):
            if (b - a) != step:
                ok = False
                break
        if ok:
            out.append(tuple(w))
    return out


def collect_prefix_groups(source_dir: Path):
    groups = defaultdict(list)
    unparsed = []
    for p in sorted(source_dir.glob("*.xlsx")):
        parsed = parse_window_from_name(p.name)
        if parsed is None:
            unparsed.append(p)
            continue
        prefix, times = parsed
        groups[prefix].append((p, times))
    return groups, unparsed


def _extract_slot_frames_by_filename_times(
    df: pd.DataFrame, fname_times: List[int], fpath: Path
):
    """
    关键逻辑：
    - 文件内 Time(h) 常是相对槽位（例如总是 0/2/4/6），不代表绝对时间。
    - 真实绝对时间用文件名中的 [a,b,c,d,...]。
    - 因此按“槽位顺序”把内部分组映射到 fname_times。
    """
    if "Time(h)" not in df.columns:
        return None, f"[WARN] 缺少 Time(h) 列，跳过: {fpath.name}"

    # 内部槽位（相对时间）按数值排序，作为第1/2/3/4...个槽
    slot_values = sorted(df["Time(h)"].dropna().unique().tolist())
    if len(slot_values) != len(fname_times):
        return (
            None,
            f"[WARN] 槽位数与文件名时间数不一致，跳过: {fpath.name} "
            f"(slot={len(slot_values)} vs name={len(fname_times)})",
        )

    abs_time_to_df = {}
    for i, slot_t in enumerate(slot_values):
        abs_t = int(fname_times[i])
        t_df = df[df["Time(h)"] == slot_t].copy()
        if "Freq" in t_df.columns:
            t_df = t_df.sort_values(by="Freq", ascending=False).reset_index(drop=True)
        # 为了后续排查，保留真实绝对时间标签
        t_df["Time(h)"] = abs_t
        abs_time_to_df[abs_t] = t_df
    return abs_time_to_df, None


def build_time_to_frame(files_with_times: List[Tuple[Path, List[int]]]):
    """
    对同一 prefix：
    - 合并所有文件中每个 Time(h) 对应的帧数据
    - 若同一时间点出现多次，优先保留第一次，并校验长度一致性
    """
    time_to_df: Dict[int, pd.DataFrame] = {}
    conflicts = 0
    for fpath, fname_times in files_with_times:
        try:
            df = pd.read_excel(fpath)
        except Exception:
            continue

        abs_map, warn = _extract_slot_frames_by_filename_times(df, fname_times, fpath)
        if warn is not None:
            print(warn)
            continue

        for t, t_df in abs_map.items():
            if t not in time_to_df:
                time_to_df[t] = t_df
            else:
                if len(time_to_df[t]) != len(t_df):
                    conflicts += 1
    return time_to_df, conflicts


def make_output_name(prefix: str, window: Tuple[int, ...]) -> str:
    times_str = ", ".join(str(x) for x in window)
    return f"{prefix}[{times_str}].xlsx"


def main():
    parser = argparse.ArgumentParser(
        description="从 datasets_for_all 生成多时间点窗口版本 datasets_for_all_plus"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="datasets/datasets_for_all",
        help="源数据目录（通常是 datasets_for_all）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/datasets_for_all_plus",
        help="输出目录（将生成 1~6 等多窗口样本）",
    )
    parser.add_argument(
        "--window-sizes",
        type=str,
        default="1,2,3,4,5,6",
        help="要生成的窗口长度，逗号分隔，如 1,2,3,4,5,6",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=0,
        help="时间步长；0 表示按每个 prefix 自动推断（推荐）",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="生成前清空输出目录",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若目标文件已存在则覆盖",
    )
    parser.add_argument(
        "--copy-unparsed",
        action="store_true",
        help="将无法解析窗口名的 xlsx 原样复制到输出目录",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    source_dir = (repo_root / args.source_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    window_sizes = parse_int_list(args.window_sizes)

    if not source_dir.exists():
        raise FileNotFoundError(f"源目录不存在: {source_dir}")

    if args.clear_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups, unparsed = collect_prefix_groups(source_dir)
    print(f"源目录: {source_dir}")
    print(f"输出目录: {output_dir}")
    print(f"prefix 组数: {len(groups)}")
    print(f"未解析命名文件数: {len(unparsed)}")

    total_created = 0
    total_skipped_exists = 0
    total_conflicts = 0

    for prefix, files_with_times in groups.items():
        time_to_df, conflicts = build_time_to_frame(files_with_times)
        total_conflicts += conflicts

        all_times = sorted(time_to_df.keys())
        if not all_times:
            continue

        step = int(args.step) if int(args.step) > 0 else infer_base_step(all_times)
        if step <= 0:
            step = 1

        for k in window_sizes:
            windows = build_windows(all_times, k, step)
            for w in windows:
                # 需要窗口内每个时间点都有对应帧
                if any(t not in time_to_df for t in w):
                    continue

                out_name = make_output_name(prefix, w)
                out_path = output_dir / out_name
                if out_path.exists() and not args.overwrite:
                    total_skipped_exists += 1
                    continue

                merged = pd.concat([time_to_df[t] for t in w], axis=0, ignore_index=True)
                merged.to_excel(out_path, index=False)
                total_created += 1

    if args.copy_unparsed:
        copied = 0
        for p in unparsed:
            dest = output_dir / p.name
            if dest.exists() and not args.overwrite:
                continue
            shutil.copy2(p, dest)
            copied += 1
        print(f"已复制未解析文件: {copied}")

    print("------ 完成 ------")
    print(f"新生成文件数: {total_created}")
    print(f"已存在未覆盖: {total_skipped_exists}")
    print(f"同时间点帧长度冲突次数(仅统计): {total_conflicts}")


if __name__ == "__main__":
    main()
