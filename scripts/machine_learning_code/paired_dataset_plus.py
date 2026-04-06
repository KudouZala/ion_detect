import re
from collections import defaultdict

import torch
from torch.utils.data import Dataset


class SlidingWindowPairDatasetPlus(Dataset):
    """
    支持混合 num_time_points 的配对数据集：
    - 仅在同一 tp 组内配对，避免不同时间长度样本互配。
    - 对每个样本 A，按 link_k=tp 取 A.times[tp-1] 作为 B 的 first 进行配对。
    """

    _regex = re.compile(
        r"^(?P<prefix>.*?_\[)(?P<times>\d+(?:\s*,\s*\d+)*)\]\s*(?:\.\w+)?$"
    )

    def __init__(
        self,
        base_dataset,
        keep_unpaired="drop",
        debug=False,
        focus_prefix_contains=None,
        max_print=30,
    ):
        super().__init__()
        self.base = base_dataset
        self.keep_unpaired = keep_unpaired
        self.debug = debug
        self.focus_prefix_contains = focus_prefix_contains
        self.max_print = int(max_print)

        self.pairs = []  # list[(idxA, idxB, is_dummy)]
        self.pair_tps = []  # 与 pairs 对齐，记录该对使用的 tp
        self.metas = []
        self.meta_by_idx = {}

        self._parse_all()
        self._build_pairs()

        if self.debug:
            print(
                f"[PairPlus] parsed={len(self.metas)} pairs={len(self.pairs)} "
                f"keep_unpaired={self.keep_unpaired}"
            )

    def _parse_all(self):
        self.metas.clear()
        self.meta_by_idx.clear()
        file_names = getattr(self.base, "file_names", [])
        for idx, fname in enumerate(file_names):
            if hasattr(self.base, "get_pair_meta"):
                meta = self.base.get_pair_meta(idx)
                prefix = meta.get("prefix")
                times = meta.get("times")
                tp = int(meta.get("tp"))
                if prefix is None or not times:
                    continue
            else:
                m = self._regex.match(fname.strip())
                if not m:
                    continue
                prefix = m.group("prefix")
                times = [int(t.strip()) for t in m.group("times").split(",")]
                tp = 4

            m = {
                "idx": idx,
                "fname": fname,
                "prefix": prefix,
                "times": times,
                "first": times[0],
                "last": times[-1],
                "tp": tp,
            }
            self.metas.append(m)
            self.meta_by_idx[idx] = m

    def _build_pairs(self):
        self.pairs.clear()
        self.pair_tps.clear()

        buckets = defaultdict(list)
        for m in self.metas:
            buckets[(m["prefix"], m["tp"])].append(m)

        for (prefix, tp), items in buckets.items():
            items.sort(key=lambda x: (x["first"], x["last"]))
            by_first = defaultdict(list)
            for m in items:
                by_first[m["first"]].append(m)

            for a in items:
                win_len = len(a["times"])
                if tp < 1 or tp > win_len:
                    if self.keep_unpaired == "self":
                        self.pairs.append((a["idx"], a["idx"], True))
                        self.pair_tps.append(tp)
                    continue

                desired_first = a["times"][tp - 1]
                cand = by_first.get(desired_first, [])
                b = None
                for x in cand:
                    if len(x["times"]) == win_len:
                        b = x
                        break

                if b is not None:
                    self.pairs.append((a["idx"], b["idx"], False))
                    self.pair_tps.append(tp)
                elif self.keep_unpaired == "self":
                    self.pairs.append((a["idx"], a["idx"], True))
                    self.pair_tps.append(tp)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        idxA, idxB, is_dummy = self.pairs[i]
        sampleA = self.base[idxA]
        sampleB = self.base[idxB]
        return sampleA, sampleB, (idxA, idxB, is_dummy)


def collate_pairs_plus(batch):
    A_list, B_list, dummy_flags = [], [], []
    for a, b, ids in batch:
        A_list.append(a)
        B_list.append(b)
        dummy_flags.append(bool(ids[2]))

    def stack_side(side_list):
        cols = list(zip(*side_list))
        stacked = []
        for col in cols:
            if hasattr(col[0], "shape"):
                stacked.append(torch.stack(col))
            else:
                stacked.append(col)
        return stacked

    batchA = stack_side(A_list)
    batchB = stack_side(B_list)
    dummy_mask = torch.tensor(dummy_flags, dtype=torch.bool)
    return batchA, batchB, dummy_mask
