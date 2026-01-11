#该函数用于制造指定滑动窗口的步长step，从而删除该文件夹中的不符合这个step的xlsx文件
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path
from typing import List, Tuple

# ========= 配置 =========
root_dir = "/home/wangruyi3/Application/ion_detect/datasets/datasets_for_ion_train_step4"
exts = {"xlsx"}            # 需要处理的扩展名（大小写不敏感）
apply_delete = True       # 先 dry-run 看结果；确认无误后改 True 真的删除
stride = 4              # 目标步长：2 / 3 / 4 / ...（保留每 stride 个窗口）
win_len = 4                # 窗口长度固定为 4
origin = 0                 # 定义“第 0 个窗口”的起点（你的数据是从 0 开始）
debug = True               # 打印调试信息（推荐先开着）

# ========= 正则：抓取方括号内四个整数（空格随意）=========
BRACKET4 = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")

def extract_groups(fname: str) -> List[Tuple[int,int,int,int]]:
    """从文件名中提取所有 [a, b, c, d] 四整数的组。"""
    return [tuple(map(int, m)) for m in BRACKET4.findall(fname)]

def window_index_and_diff(group: Tuple[int,int,int,int]):
    """若是等差序列，返回 (窗口序号i, 公差d)；否则返回(None, None)。
       窗口序号 i = (起点 - origin) / d，要求整除。"""
    a, b, c, d = group
    diffs = (b - a, c - b, d - c)
    if not (diffs[0] == diffs[1] == diffs[2]):
        return None, None
    step = diffs[0]
    if step == 0:
        return None, None
    # 起点必须与 origin 同余，且能整除
    num = (a - origin)
    if num % step != 0:
        return None, None
    idx = num // step  # 第 idx 个滑窗（原始步长为1的定义下）
    return idx, step

def decide_keep_or_delete(fname: str):
    """返回 (should_delete, explain_str)。若文件包含符合规则的组，则按 stride 判定保留/删除。"""
    groups = extract_groups(fname)
    logs = [f"发现 {len(groups)} 个组" if groups else "未发现组"]
    for g in groups:
        idx, d = window_index_and_diff(g)
        logs.append(f"  组 {g} -> idx={idx}, diff={d}")
        if idx is None:
            continue  # 不是我们要处理的标准等差滑窗
        # 目标：只保留 idx % stride == 0 的窗口，其余删除
        if (idx % stride) != 0:
            logs.append(f"    ✗ idx % stride = {idx % stride} ≠ 0 → 删除")
            return True, "\n".join(logs)
        else:
            logs.append(f"    ✓ idx % stride = 0 → 保留")
            return False, "\n".join(logs)
    # 如果没有任何符合标准的组，就不动它
    return False, "\n".join(logs)

def main():
    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"目录不存在：{root}")

    exts_lower = {e.lower().lstrip(".") for e in exts}
    targets = []
    debug_logs = []

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower().lstrip(".") not in exts_lower:
            continue

        should_del, explain = decide_keep_or_delete(p.name)
        if debug:
            debug_logs.append(f"[DEBUG] {p.name}\n{explain}\n")
        if should_del:
            targets.append(p)

    if debug and debug_logs:
        print("========== 调试信息 ==========")
        print("\n".join(debug_logs))

    if not targets:
        print(f"未找到需要删除的文件（stride={stride}）。")
        return

    print(f"共找到 {len(targets)} 个需要删除的文件（stride={stride}）：")
    for t in targets:
        print(" -", t)

    if apply_delete:
        print("\n开始删除 ...")
        deleted = failed = 0
        for t in targets:
            try:
                os.remove(t)
                deleted += 1
            except Exception as e:
                failed += 1
                print(f"删除失败：{t} -> {e}")
        print(f"\n完成：成功删除 {deleted} 个，失败 {failed} 个。")
    else:
        print("\n（dry-run）未实际删除。确认无误后，将 apply_delete=True 再运行。")

if __name__ == "__main__":
    main()
