# paired_dataset.py (debug版)
import re
from collections import defaultdict
from torch.utils.data import Dataset
import torch
import os

class SlidingWindowPairDataset(Dataset):
    """
    将 base_dataset（单窗口样本）按同一前缀的相邻时间窗口配成 (A, B) 对。
    - 前缀：取文件名最后一个 `_[` 之前（含它本身）的全部。
    - 时间：取方括号里的数字序列，例如 0, 2, 4, 6（允许空格）。
    - keep_unpaired:
        "drop"：丢弃无后续窗口
        "self"：对无后续窗口生成 (A, A) 的 dummy 自配对（训练时屏蔽一致性）
    - debug: 开启后打印汇总与问题样本（可用 f 聚焦某一类前缀）
    """

    _regex = re.compile(r'^(?P<prefix>.*?_\[)(?P<times>\d+(?:\s*,\s*\d+)*)\]\s*(?:\.\w+)?$')

    def __init__(self, base_dataset, window_len=4, link_k=4,num_time_points=4,
                keep_unpaired="drop",
                debug=False, focus_prefix_contains=None, max_print=30):
        super().__init__()
        self.base = base_dataset
        self.num_time_points = int(num_time_points)
        self.window_len = int(window_len)   # 固定 4

        self.keep_unpaired = keep_unpaired
        self.debug = debug
        self.focus_prefix_contains = focus_prefix_contains
        self.max_print = max_print

        self.pairs = []           # list[(idxA, idxB, is_dummy)]
        self.metas = []           # list[dict] 每个解析成功的样本meta
        self.meta_by_idx = {}     # idx -> meta
        self.skipped = {          # 解析失败的统计
            "regex_fail": [],     # 正则不匹配
            "parse_fail": [],     # 解析 times 失败
            "empty_times": [],    # times为空
        }
        self.unadjacent = []      # 解析成功但未满足相邻的"间隙"对：[(metaA, metaB)]

        # 1) 解析
        self._parse_all_file_names()

        # 2) 分桶配对
        self._build_pairs()

        # 2.5) 打印前若干个配对结果
        if self.debug:
            self.print_example_pairs(num_examples=50)

        # 3) 打印调试信息
        if self.debug:
            self.summary()
            if self.focus_prefix_contains:
                self.explain_unpaired(self.focus_prefix_contains)

    
    def print_example_pairs(self, num_examples=5):
        """
        打印前 num_examples 个配对结果到日志中，便于人工检查。
        """
        print(f"\n[PairDataset][Example pairs] (show up to {num_examples})")

        if not self.pairs:
            print("  (no pairs built)")
            return

        for k, (idxA, idxB, is_dummy) in enumerate(self.pairs[:num_examples]):
            metaA = self.meta_by_idx.get(idxA)
            metaB = self.meta_by_idx.get(idxB)

            # 兜底处理：理论上 meta 都应该存在
            fnameA = metaA["fname"] if metaA is not None else f"(idx={idxA})"
            fnameB = metaB["fname"] if metaB is not None else f"(idx={idxB})"
            timesA = metaA["times"] if metaA is not None else "?"
            timesB = metaB["times"] if metaB is not None else "?"

            print(f"  [{k+1}] A(idx={idxA}) {fnameA}  times={timesA}")
            print(f"      → B(idx={idxB}) {fnameB}  times={timesB}  dummy={is_dummy}")


    # ---------- 公共调试方法 ----------
    def summary(self):
        total_files = len(getattr(self.base, "file_names", []))
        parsed = len(self.metas)
        print("\n[PairDataset][Summary]")
        print(f"- total files:        {total_files}")
        print(f"- parsed OK:          {parsed}")
        print(f"- regex fail:         {len(self.skipped['regex_fail'])}")
        print(f"- parse times fail:   {len(self.skipped['parse_fail'])}")
        print(f"- empty times:        {len(self.skipped['empty_times'])}")
        print(f"- total pairs:        {len(self.pairs)}  (keep_unpaired={self.keep_unpaired})")

        # 统计prefix级别信息
        buckets = defaultdict(list)
        for m in self.metas:
            buckets[m["prefix"]].append(m)
        print(f"- distinct prefixes:  {len(buckets)}")

        print("- prefixes coverage (first few):")
        c = 0
        for prefix, items in buckets.items():
            items = sorted(items, key=lambda x: (x["first"], x["last"]))
            by_first = {m["first"]: m for m in items}
            # 理论最大相邻对 = 每个窗口最多有一个 next（按 last→first 匹配）
            max_adj = len(items) - 1 if len(items) >= 2 else 0
            real_adj = sum(1 for a in items if by_first.get(a["last"]) is not None)
            print(f"  [{c+1}] {prefix}* -> windows={len(items)}, adj_pairs={real_adj}/{max_adj}")
            c += 1
            if c >= 10: break

        # 打印部分 regex 失败样本
        if self.skipped['regex_fail']:
            print("- regex fail examples:")
            for i, fname in enumerate(self.skipped['regex_fail'][:self.max_print]):
                print(f"  [R{i+1}] {fname}")
        if self.skipped['parse_fail']:
            print("- parse times fail examples:")
            for i, fname in enumerate(self.skipped['parse_fail'][:self.max_print]):
                print(f"  [P{i+1}] {fname}")
        if self.skipped['empty_times']:
            print("- empty times examples:")
            for i, fname in enumerate(self.skipped['empty_times'][:self.max_print]):
                print(f"  [E{i+1}] {fname}")

        # 打印一些未相邻的“间隙”例子
        if self.unadjacent:
            print(f"- unadjacent gaps samples: {len(self.unadjacent)} (showing first {min(self.max_print,len(self.unadjacent))})")
            for i, (a, b) in enumerate(self.unadjacent[:self.max_print]):
                print(f"  [G{i+1}] {a['fname']} (last={a['last']})  -->  {b['fname']} (first={b['first']})")

    def explain_unpaired(self, prefix_substr):
        """
        只看包含 prefix_substr 的前缀，逐窗口打印时间序列和相邻对的配对结果。
        便于核对像你给的例子：..._ion_firecloud_[0,2,4,6] & ..._[6,8,10,12]
        """
        print(f"\n[PairDataset][Explain] focus_prefix_contains='{prefix_substr}'")
        # 过滤相关前缀
        buckets = defaultdict(list)
        for m in self.metas:
            if prefix_substr in m["prefix"]:
                buckets[m["prefix"]].append(m)
        if not buckets:
            print("  (no prefix matched)")
            return
        for prefix, items in buckets.items():
            items = sorted(items, key=lambda x: (x["first"], x["last"]))
            print(f"  - PREFIX: {prefix}* windows={len(items)}")
            print("    windows (sorted by first,last):")
            for it in items:
                print(f"      · {it['fname']}  -> times={it['times']}  first={it['first']} last={it['last']}")
            # 标出是否相邻
            for a, b in zip(items, items[1:]):
                ok = (a["last"] == b["first"])
                mark = "OK " if ok else "MISS"
                print(f"      {mark}: {a['last']} -> {b['first']}   ({a['fname']}  →  {b['fname']})")

    # ---------- 内部：解析与配对 ----------
    def _parse_all_file_names(self):
        self.metas.clear()
        self.meta_by_idx.clear()
        self.skipped["regex_fail"].clear()
        self.skipped["parse_fail"].clear()
        self.skipped["empty_times"].clear()

        file_names = getattr(self.base, "file_names", [])
        for idx, fname in enumerate(file_names):
            # 先做一次规范化：去掉两端空格；保留扩展名
            fname_norm = fname.strip()
            m = self._regex.match(fname_norm)
            try_regex = True
            if m:
                prefix = m.group("prefix")
                times_str = m.group("times")
                try:
                    times = [int(t.strip()) for t in times_str.split(",")]
                except Exception:
                    self.skipped["parse_fail"].append(fname)
                    continue
            else:
                # 兜底：从右侧找 `_[` 与 `]`
                try_regex = False
                l = fname_norm.rfind("_[")
                r = fname_norm.rfind("]")
                if l == -1 or r == -1 or r <= l:
                    self.skipped["regex_fail"].append(fname)
                    continue
                prefix = fname_norm[:l+2]  # 保留到 `_[`
                times_str = fname_norm[l+2:r]
                try:
                    times = [int(t.strip()) for t in times_str.split(",")]
                except Exception:
                    self.skipped["parse_fail"].append(fname)
                    continue

            if not times:
                self.skipped["empty_times"].append(fname)
                continue

            meta = {
                "idx": idx,
                "prefix": prefix,
                "times": times,
                "first": times[0],
                "last": times[-1],
                "fname": fname_norm
            }
            self.metas.append(meta)
            self.meta_by_idx[idx] = meta
    def _build_pairs(self):
        self.pairs.clear()
        self.unadjacent.clear()

        buckets = defaultdict(list)
        for m in self.metas:
            buckets[m["prefix"]].append(m)

        link_k = int(self.num_time_points)  # 你定义的 link_k，就是 num_time_points（第 k 个数字）
        if link_k < 1:
            raise ValueError(f"num_time_points(link_k) must be >= 1, got {link_k}")

        for prefix, items in buckets.items():
            # 按时间排序（first,last）
            items.sort(key=lambda x: (x["first"], x["last"]))

            # 用 first 建索引
            by_first = defaultdict(list)
            for m in items:
                by_first[m["first"]].append(m)

            for a in items:
                # 这里的“窗口长度”来自文件自身（你的场景一般是 4）
                window_len = len(a["times"])

                # link_k 必须落在 A.times 内
                if link_k > window_len:
                    self.unadjacent.append((a, {"fname": f"(link_k={link_k} out of range for A.times len={window_len})"}))
                    if self.keep_unpaired == "self":
                        self.pairs.append((a["idx"], a["idx"], True))
                    continue

                # ★ 关键：A 的第 link_k 个数字，作为 B 的第 1 个数字
                desired_first = a["times"][link_k - 1]

                cand = by_first.get(desired_first, [])
                b = None
                for x in cand:
                    # 窗口长度一致（例如都为 4），避免把 link_k 误当窗口长度
                    if len(x["times"]) == window_len:
                        b = x
                        break

                if b is not None:
                    self.pairs.append((a["idx"], b["idx"], False))
                else:
                    self.unadjacent.append((a, {"fname": f"(no window with first={desired_first})"}))
                    if self.keep_unpaired == "self":
                        self.pairs.append((a["idx"], a["idx"], True))




    # ---------- Dataset 接口 ----------
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        idxA, idxB, is_dummy = self.pairs[i]
        sampleA = self.base[idxA]  # 期望 (volt, impe, env, label, true_v, ep, conc)
        sampleB = self.base[idxB]
        return sampleA, sampleB, (idxA, idxB, is_dummy)


def collate_pairs(batch):
    A_list, B_list, dummy_flags = [], [], []
    for a, b, ids in batch:
        A_list.append(a)
        B_list.append(b)
        dummy_flags.append(bool(ids[2]))

    def stack_side(side_list):
        cols = list(zip(*side_list))  # 7 列
        stacked = []
        for col in cols:
            if hasattr(col[0], "shape"):  # Tensor
                stacked.append(torch.stack(col))
            else:
                stacked.append(col)
        return stacked

    batchA = stack_side(A_list)
    batchB = stack_side(B_list)
    dummy_mask = torch.tensor(dummy_flags, dtype=torch.bool)
    return batchA, batchB, dummy_mask
