import argparse
import copy
import csv
import os
import random
from collections import defaultdict
from pathlib import Path
import re
import shutil

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler

from runtime_core import (
    build_device,
    build_paths,
    load_config,
    load_label_mapping,
    _resolve_test_model_path,
    prepare_test_folder,
)
from model_datasets_plus import MultiTimePointsDatasetPlus
from model_new_models import IonDetectModel
from model_new_models_swctf import IonDetectModelSWCTF
from model_new_train import TrainerNew
from paired_dataset_plus import SlidingWindowPairDatasetPlus, collate_pairs_plus
from model_test_plus import test_single_xlsx_plus


def _parse_tp_options(tp_cfg):
    if isinstance(tp_cfg, int):
        out = [int(tp_cfg)]
    elif isinstance(tp_cfg, (list, tuple)):
        out = [int(x) for x in tp_cfg]
    else:
        raise ValueError(f"data.num_time_points 必须是 int 或 list[int]，当前为: {type(tp_cfg)}")
    out = sorted(set(out))
    for x in out:
        if x not in (1, 2, 3, 4, 5, 6):
            raise ValueError(f"仅支持 num_time_points in [1,2,3,4,5,6]，当前包含: {x}")
    return out


_WINDOW_RE = re.compile(r"^(?P<prefix>.*?_\[)(?P<times>\d+(?:\s*,\s*\d+)*)\]\s*(?:\.\w+)?$")


def _parse_window_meta(fname: str):
    match = _WINDOW_RE.match(fname.strip())
    if not match:
        return None
    times = [int(x.strip()) for x in match.group("times").split(",")]
    return match.group("prefix"), tuple(times)


def _prepare_no_overlap_split_plus(paths: dict, tp_options, seed: int) -> bool:
    data_folder: Path = paths["data_folder"]
    base_test_folder: Path = paths["test_folder_path"]
    if data_folder.name != "datasets_for_all_plus" or base_test_folder.name != "datasets_for_all_test_no_overlap":
        return False

    base_dir: Path = paths["base_dir"]
    exp_name: str = (paths["exp_name"] or "no_overlap_plus").strip() or "no_overlap_plus"
    source_dir = base_dir / "datasets" / "datasets_for_all_plus"
    if not source_dir.exists():
        print(f"⚠️ plus no_overlap 模式需要存在目录: {source_dir}")
        return False

    parsed_all = []
    for p in source_dir.glob("*.xlsx"):
        parsed = _parse_window_meta(p.name)
        if parsed is None:
            continue
        prefix, window = parsed
        parsed_all.append((p, prefix, window, len(window)))

    if not parsed_all:
        print(f"⚠️ 在 {source_dir} 未找到可解析窗口文件")
        return False

    candidates = [x for x in parsed_all if x[3] in set(int(t) for t in tp_options)]
    if not candidates:
        print(f"⚠️ 未找到 [] 长度属于 {tp_options} 的候选样本")
        return False

    random.seed(int(seed))
    random.shuffle(candidates)
    target_n_test = 200
    n_test = min(target_n_test, len(candidates))
    test_selected_all = candidates[:n_test]
    remain = candidates[n_test:]

    test_keys = {(x[1], x[2]) for x in test_selected_all}
    prefix_to_active_times = {}
    for _, prefix, window, _ in test_selected_all:
        prefix_to_active_times.setdefault(prefix, set()).update(set(window))

    def excluded_from_train(item):
        _, prefix, window, _ = item
        if (prefix, window) in test_keys:
            return True
        if prefix not in prefix_to_active_times:
            return False
        return bool(set(window) & prefix_to_active_times[prefix])

    train_selected_all = [x for x in remain if not excluded_from_train(x)]

    test_dist = defaultdict(int)
    train_dist = defaultdict(int)
    for x in test_selected_all:
        test_dist[int(x[3])] += 1
    for x in train_selected_all:
        train_dist[int(x[3])] += 1
    print(
        f"✅ 统一抽样: 候选 {len(candidates)}，测试 {len(test_selected_all)} "
        f"(目标 {target_n_test})，训练 {len(train_selected_all)}"
    )
    print(f"   测试集长度分布: {dict(sorted(test_dist.items()))}")
    print(f"   训练集长度分布: {dict(sorted(train_dist.items()))}")

    train_dir = base_dir / "datasets" / "datasets_for_all_train_no_overlap" / exp_name
    test_dir = base_dir / "datasets" / "datasets_for_all_test_no_overlap" / exp_name
    for folder in (train_dir, test_dir):
        if folder.exists():
            shutil.rmtree(folder)
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for p, _, _, _ in train_selected_all:
        shutil.copy2(str(p), str(train_dir / p.name))
    for p, _, _, _ in test_selected_all:
        shutil.copy2(str(p), str(test_dir / p.name))

    print(
        f"✅ plus no_overlap 划分完成：测试集 {len(test_selected_all)} -> {test_dir}，"
        f"训练集 {len(train_selected_all)} -> {train_dir}"
    )

    paths["data_folder"] = train_dir
    paths["test_folder_path"] = test_dir
    return True


class TimePointBucketBatchSampler(Sampler):
    def __init__(self, pair_tps, batch_size: int, shuffle: bool = True):
        self.pair_tps = list(pair_tps)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self._buckets = defaultdict(list)
        for i, tp in enumerate(self.pair_tps):
            self._buckets[int(tp)].append(i)

    def __iter__(self):
        all_batches = []
        rng = random.Random()
        for _, idxs in self._buckets.items():
            idxs = list(idxs)
            if self.shuffle:
                rng.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                chunk = idxs[i:i + self.batch_size]
                if chunk:
                    all_batches.append(chunk)
        if self.shuffle:
            rng.shuffle(all_batches)
        for batch in all_batches:
            yield batch

    def __len__(self):
        total = 0
        for _, idxs in self._buckets.items():
            total += (len(idxs) + self.batch_size - 1) // self.batch_size
        return total


def build_dataloaders_plus(paths: dict, cfg: dict, tp_options):
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    ds_cfg = cfg["dataset"]

    data_folder = paths["data_folder"]
    val_data_folder = paths["test_folder_path"]
    stats_file = paths["stats_file"]
    batch_size = int(tcfg["batch_size"])
    num_freq_points = int(dcfg["num_freq_points"])

    val_fnames = {p.name for p in Path(val_data_folder).glob("*.xlsx")}
    base_train = MultiTimePointsDatasetPlus(
        data_folder=data_folder,
        stats_file=str(stats_file),
        save_stats=True,
        num_time_points=tp_options,
        num_freq_points=num_freq_points,
        exclude_fnames=val_fnames,
    )
    base_val = MultiTimePointsDatasetPlus(
        data_folder=val_data_folder,
        stats_file=str(stats_file),
        save_stats=False,
        num_time_points=tp_options,
        num_freq_points=num_freq_points,
        exclude_fnames=None,
    )

    pair_train = SlidingWindowPairDatasetPlus(
        base_train,
        keep_unpaired=ds_cfg["keep_unpaired"],
        debug=ds_cfg["debug"],
        focus_prefix_contains=ds_cfg.get("focus_prefix_contains", None),
        max_print=int(ds_cfg.get("max_print", 0)),
    )
    pair_val = SlidingWindowPairDatasetPlus(
        base_val,
        keep_unpaired=ds_cfg["keep_unpaired"],
        debug=ds_cfg["debug"],
        focus_prefix_contains=None,
        max_print=int(ds_cfg.get("max_print", 0)),
    )

    print(f"[PLUS] tp_options = {tp_options}")
    print("num base train samples:", len(base_train))
    print("num base val   samples:", len(base_val))
    print("num pair train samples:", len(pair_train))
    print("num pair val   samples:", len(pair_val))

    train_loader = DataLoader(
        pair_train,
        batch_sampler=TimePointBucketBatchSampler(pair_train.pair_tps, batch_size, shuffle=True),
        collate_fn=collate_pairs_plus,
    )
    val_loader = DataLoader(
        pair_val,
        batch_sampler=TimePointBucketBatchSampler(pair_val.pair_tps, batch_size, shuffle=False),
        collate_fn=collate_pairs_plus,
    )
    return train_loader, val_loader


def build_model_ablation(device: torch.device, cfg: dict):
    classify_mode = cfg["new_model"].get("classify_mode", "clip_v2")
    arch = str(cfg["new_model"].get("arch", "legacy")).lower()
    if arch == "swc_tf":
        model = IonDetectModelSWCTF(cfg).to(device)
    else:
        model = IonDetectModel(cfg).to(device)
    model_tag = "CLIP v2" if classify_mode == "clip_v2" else classify_mode
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"🧩 {model_tag} 模型 [{arch}]: {total_params:,} 参数 ({trainable:,} 可训练)"
    )
    return model


def run_train_plus(device: torch.device, paths: dict, cfg: dict, tp_options):
    print("🚀 进入训练模式 (--train, plus)")
    train_loader, val_loader = build_dataloaders_plus(paths, cfg, tp_options)

    cfg_for_model = copy.deepcopy(cfg)
    cfg_for_model["data"]["num_time_points"] = int(max(tp_options))
    model = build_model_ablation(device, cfg_for_model)

    tcfg = cfg["train"]
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )

    trainer = TrainerNew(
        model=model,
        optimizer=optimizer,
        device=device,
        model_save_folder=paths["model_save_folder"],
        cfg=cfg_for_model,
    )

    use_cosine = bool(cfg["new_model"].get("use_cosine_lr", False))
    warmup_epochs = int(cfg["new_model"].get("warmup_epochs", 0))
    num_epochs = int(tcfg["num_epochs"])
    if use_cosine or warmup_epochs > 0:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        schedulers = []
        milestones = []
        if warmup_epochs > 0:
            warmup_sched = LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
            )
            schedulers.append(warmup_sched)
            milestones.append(warmup_epochs)
        if use_cosine:
            cosine_epochs = num_epochs - warmup_epochs
            cosine_sched = CosineAnnealingLR(
                optimizer, T_max=max(cosine_epochs, 1), eta_min=1e-6
            )
            schedulers.append(cosine_sched)
        if len(schedulers) > 1:
            trainer._scheduler = SequentialLR(
                optimizer, schedulers=schedulers, milestones=milestones
            )
        elif len(schedulers) == 1:
            trainer._scheduler = schedulers[0]
        else:
            trainer._scheduler = None
    else:
        trainer._scheduler = None

    tp = tcfg["train_pairs"]
    eval_every = int(tcfg.get("eval_every", 10))
    num_classes = cfg["new_model"].get("num_classes", 7)
    label_mapping = load_label_mapping(paths["json_path"])
    idx_to_name = {int(v): str(k) for k, v in label_mapping.items()} if label_mapping else {}
    class_names = [idx_to_name.get(i, f"class_{i}") for i in range(int(num_classes))]
    trainer.train_pairs(
        train_loader,
        num_epochs=int(tcfg["num_epochs"]),
        lambda_consistency=float(tp.get("lambda_consistency", 0.0)),
        eps=float(tp.get("eps", 1e-9)),
        use_log_space=bool(tp.get("use_log_space", True)),
        lambda_monodec=float(tp.get("lambda_monodec", 0.0)),
        lambda_polarity=float(tp.get("lambda_polarity", 0.0)),
        weight_ratio=float(tp.get("weight_ratio", 3.0)),
        val_loader=val_loader,
        eval_every=eval_every,
        num_classes=num_classes,
        class_names=class_names,
    )


def run_test_plus(paths: dict, cfg: dict, args):
    print("🔍 进入测试模式 (--test, plus)")
    test_model_path = _resolve_test_model_path(paths)
    test_folder_path = paths["test_folder_path"]
    stats_file = str(paths["stats_file"])
    device_str = cfg["test"].get("device_str", "cpu")

    cfg_for_model = copy.deepcopy(cfg)
    tp_cfg = cfg_for_model["data"]["num_time_points"]
    if isinstance(tp_cfg, (list, tuple)):
        cfg_for_model["data"]["num_time_points"] = int(max(tp_cfg))

    device = torch.device(device_str)
    model = build_model_ablation(device, cfg_for_model)
    state = torch.load(str(test_model_path), map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    xlsx_paths = sorted(str(test_folder_path / f) for f in os.listdir(test_folder_path) if f.endswith(".xlsx"))
    print(f"📊 共检测到 {len(xlsx_paths)} 个 .xlsx 文件，准备测试...")

    inference_run = args.inference_run.strip() if args.inference_run else paths["exp_name"]
    output_dir = paths["base_dir"] / "output" / "inference_results" / inference_run
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"🗂️ 测试结果输出目录: {output_dir}")

    correct_count = 0
    processed_count = 0
    failed_count = 0
    summary_rows = []
    for xlsx_path in xlsx_paths:
        try:
            correct, predict, truth = test_single_xlsx_plus(
                xlsx_path=xlsx_path,
                model=model,
                device=device,
                num_freq_points=int(cfg_for_model["data"]["num_freq_points"]),
                stats_file=stats_file,
                generate_explanations=not args.no_explanations,
                output_dir=str(output_dir),
            )
            processed_count += 1
            if correct:
                correct_count += 1
            summary_rows.append([os.path.basename(xlsx_path), truth, predict, "是" if correct else "否"])
        except Exception as e:
            print(f"❌ 文件处理失败: {xlsx_path} | {e}")
            failed_count += 1

    summary_csv = output_dir / "summary_predictions.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["样本名", "真实", "预测", "正确"])
        writer.writerows(summary_rows)
    print(f"✅ 汇总测试结果已保存: {summary_csv}")

    if processed_count > 0:
        acc = correct_count / processed_count
        print(f"\n✅ 总共测试样本数: {len(xlsx_paths)}")
        print(f"✅ 成功评测样本数: {processed_count}")
        print(f"⚠️ 失败样本数: {failed_count}")
        print(f"🎯 预测正确样本数: {correct_count}")
        print(f"📊 准确率: {acc:.2%}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="运行 ion_detect PLUS 实验（支持 clip_v2 / linear）"
    )
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--test-folder", type=str, default="", help="可选：指定测试数据文件夹，测试模式下优先使用")
    parser.add_argument("--inference-run", type=str, default="", help="可选：测试结果输出子目录名，默认使用 experiment.name")
    parser.add_argument("--no-explanations", action="store_true", help="测试时不导出 attention / IG / 结构化参数")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="运行训练模式")
    group.add_argument("--test", action="store_true", help="运行测试模式")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)

    seed = int(cfg["experiment"].get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    tp_options = _parse_tp_options(cfg["data"]["num_time_points"])
    max_tp = int(max(tp_options))

    device, _ = build_device()
    paths = build_paths(cfg)
    num_freq_points = int(cfg["data"]["num_freq_points"])

    label_mapping = load_label_mapping(paths["json_path"])
    if args.test and args.test_folder:
        paths["test_folder_path"] = Path(args.test_folder).expanduser().resolve()
        print(f"📌 使用手动指定测试目录: {paths['test_folder_path']}")
    else:
        if not _prepare_no_overlap_split_plus(paths, tp_options, seed):
            prepare_test_folder(paths, label_mapping, max_tp, num_freq_points, seed)

    print(f"✅ 当前使用设备: {device}")
    print(f"📂 训练集目录: {paths['data_folder']}")
    print(f"📂 测试/验证集目录: {paths['test_folder_path']}")
    print(f"🧪 experiment: {paths['exp_name']}")
    print(f"🧪 num_time_points 配置: {tp_options}")

    if args.train:
        run_train_plus(device, paths, cfg, tp_options)
    elif args.test:
        run_test_plus(paths, cfg, args)


if __name__ == "__main__":
    main()
