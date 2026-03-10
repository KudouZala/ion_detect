import os
import csv
import json
import re
import argparse
import traceback
import multiprocessing
import random
import shutil
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

# ==== 项目内部模块（保持不变） ====
from model_datasets import Dataset_2_Stable_plus
from model_models import Model_three_system_plus
from model_test import test_single_xlsx_and_generate_explanations_three_system_1117
from model_train import Trainer_ThreeSystem_plus
from paired_dataset import SlidingWindowPairDataset, collate_pairs

# ==== 新模块化模型 ====
from model_new_models import IonDetectModel
from model_new_train import TrainerNew


# --------------------------
# YAML load
# --------------------------
def load_config(cfg_path: Path) -> dict:
    try:
        import yaml  # PyYAML
    except Exception as e:
        raise RuntimeError(
            "缺少 PyYAML 依赖。请先安装：pip install pyyaml"
        ) from e

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件内容非法（应为 dict）：{cfg_path}")

    # minimal sanity checks
    required = ["experiment", "data", "train", "test", "paths", "dataset"]
    for k in required:
        if k not in cfg:
            raise KeyError(f"配置缺少字段: {k}")
    if "model" not in cfg and "new_model" not in cfg:
        raise KeyError("配置缺少字段: model 或 new_model（至少需要一个）")

    return cfg


def build_device() -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"


def resolve_base_dir() -> Path:
    # 与原脚本一致：脚本所在目录的 parent.parent.parent
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent


def build_paths(cfg: dict) -> dict:
    base_dir = resolve_base_dir()
    exp_name = cfg["experiment"]["name"]

    paths_cfg = cfg["paths"]

    data_folder = base_dir / paths_cfg["data_folder"]
    base_test_folder = base_dir / paths_cfg["test_folder"]

    model_save_folder = base_dir / paths_cfg["model_save_root"] / exp_name
    model_save_folder.mkdir(parents=True, exist_ok=True)

    test_model_path = model_save_folder / "trained_model_epoch_best.pth"

    stats_file = base_dir / paths_cfg["stats_file"]

    # label_mapping.json：优先使用 paths_cfg["label_mapping_json"]（相对 main.py 同目录 or base_dir）
    # 原脚本是 current_file.parent / "label_mapping.json"，这里更健壮：
    lm = Path(paths_cfg["label_mapping_json"])
    if lm.is_absolute():
        json_path = lm
    else:
        # 先尝试 main.py 同目录，再尝试 base_dir 下
        p1 = Path(__file__).resolve().parent / lm
        p2 = base_dir / lm
        json_path = p1 if p1.exists() else p2

    debug_log_root = base_dir / paths_cfg["debug_log_dir"]
    debug_log_root.mkdir(parents=True, exist_ok=True)
    debug_log_dir_test = debug_log_root / "test"
    debug_log_dir_search = debug_log_root / "search"
    debug_log_dir_test.mkdir(parents=True, exist_ok=True)
    debug_log_dir_search.mkdir(parents=True, exist_ok=True)

    tb_log_dir = base_dir / paths_cfg["tensorboard_root"] / exp_name
    tb_log_dir.mkdir(parents=True, exist_ok=True)

    return {
        "base_dir": base_dir,
        "exp_name": exp_name,
        "data_folder": data_folder,
        "test_folder_path": base_test_folder,
        "model_save_folder": model_save_folder,
        "test_model_path": test_model_path,
        "stats_file": stats_file,
        "json_path": json_path,
        "debug_log_dir": debug_log_root,
        "debug_log_dir_test": debug_log_dir_test,
        "debug_log_dir_search": debug_log_dir_search,
        "tb_log_dir": tb_log_dir,
    }


def load_label_mapping(json_path: Path) -> dict:
    if not json_path.exists():
        print(f"⚠️ 未找到 label_mapping.json: {json_path}，后续如果不需要可以忽略。")
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_label_from_filename(fname: str, label_mapping: dict) -> str | None:
    if label_mapping:
        for label in label_mapping.keys():
            if isinstance(label, str) and label in fname:
                return label

    keywords = ["钙离子", "钠离子", "镍离子", "铬离子", "铜离子", "铁离子", "无污染"]
    for kw in keywords:
        if (kw in fname) and ("ion_column" not in fname):
            return kw

    lower = fname.lower()
    if ("blank" in lower) or ("纯水" in lower) or ("ion_column" in fname):
        return "无污染"

    return None


def is_valid_xlsx_for_model(xlsx_path: Path, num_time_points: int, num_freq_points: int) -> bool:
    try:
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"[SKIP] 无法读取文件 {xlsx_path.name}: {e}")
        return False

    required_cols = ["Time(h)", "mean_voltage", "Zreal", "Zimag", "Freq"]
    for col in required_cols:
        if col not in df.columns:
            print(f"[SKIP] 文件 {xlsx_path.name} 缺少必要列: {col}")
            return False

    time_series = df["Time(h)"].dropna().unique().tolist()
    if len(time_series) < num_time_points:
        print(f"[SKIP] 文件 {xlsx_path.name}: 仅有 {len(time_series)} 个时间点 < 要求的 {num_time_points}")
        return False

    time_points = sorted(time_series)[:num_time_points]
    for t in time_points:
        time_data = df[df["Time(h)"] == t]
        if time_data.empty:
            print(f"[SKIP] 文件 {xlsx_path.name}: 缺少时间点 {t}h")
            return False

        time_data = time_data.sort_values(by="Freq")
        voltage = time_data["mean_voltage"].values[0]
        if pd.isna(voltage):
            print(f"[SKIP] 文件 {xlsx_path.name}: 时间点 {t}h 的 mean_voltage 为 NaN")
            return False

        impedance_np = time_data[["Zreal", "Zimag"]].values
        if impedance_np.shape[0] < num_freq_points:
            print(f"[SKIP] 文件 {xlsx_path.name}: 时间点 {t}h 的阻抗点数 {impedance_np.shape[0]} < 要求的 {num_freq_points}")
            return False

    print(f"[OK] 可作为测试样本: {xlsx_path.name}")
    return True


# 解析文件名中的 xxx 与时间窗口 []，用于 no_overlap 划分。(prefix, window_tuple) 唯一标识一个样本
_NO_OVERLAP_REGEX = re.compile(r"^(?P<prefix>.*?_\[)(?P<times>\d+(?:\s*,\s*\d+)*)\]\s*(?:\.\w+)?$")


def _parse_no_overlap_key(fname: str):
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


def _prepare_no_overlap_split(paths: dict, num_time_points: int, num_freq_points: int, seed: int) -> bool:
    """
    当 data_folder 为 datasets_for_all_no_overlap 且 test_folder 为 datasets_for_all_test_no_overlap 时：
    从 datasets/datasets_for_all 中随机选出 150 个做测试集，
    训练集为其余样本，且排除：与某测试样本前缀相同且 [] 中含该测试样本前 num_time_points 个时间点中任意一个的样本。
    不删磁盘文件，仅将最终训练/测试文件复制到对应目录，并更新 paths。返回 True 表示已处理。
    """
    data_folder: Path = paths["data_folder"]
    base_test_folder: Path = paths["test_folder_path"]
    if data_folder.name != "datasets_for_all_no_overlap" or base_test_folder.name != "datasets_for_all_test_no_overlap":
        return False

    base_dir: Path = paths["base_dir"]
    exp_name: str = (paths["exp_name"] or "no_overlap").strip() or "no_overlap"
    source_dir = base_dir / "datasets" / "datasets_for_all"
    if not source_dir.exists():
        print(f"⚠️ no_overlap 模式需要存在目录: {source_dir}")
        return False

    all_xlsx = list(source_dir.glob("*.xlsx"))
    parsed = []
    for p in all_xlsx:
        key = _parse_no_overlap_key(p.name)
        if key is None:
            continue
        parsed.append((p, key[0], key[1]))

    n_test = 150
    if len(parsed) < n_test:
        print(f"⚠️ no_overlap: 可解析样本数 {len(parsed)} < {n_test}，无法按 {n_test} 测试集划分")
        return False

    random.seed(int(seed))
    random.shuffle(parsed)
    test_selected = parsed[:n_test]
    test_keys = {(x[1], x[2]) for x in test_selected}

    # 每个测试样本取前 num_time_points 个时间点，用于排除训练集中同前缀且 [] 含任一时间点的样本
    prefix_to_active_times: dict = {}
    for (_, prefix, window) in test_selected:
        active = set(sorted(window)[:num_time_points])
        if prefix not in prefix_to_active_times:
            prefix_to_active_times[prefix] = set(active)
        else:
            prefix_to_active_times[prefix] |= active

    def excluded_from_train(item):
        (_, prefix, window) = item
        if (prefix, window) in test_keys:
            return True
        if prefix not in prefix_to_active_times:
            return False
        return bool(set(window) & prefix_to_active_times[prefix])

    train_selected = [x for x in parsed if not excluded_from_train(x)]

    train_dir = base_dir / "datasets" / "datasets_for_all_train_no_overlap" / exp_name
    test_dir = base_dir / "datasets" / "datasets_for_all_test_no_overlap" / exp_name
    for d in (train_dir, test_dir):
        if d.exists():
            shutil.rmtree(d)
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for (p, _, _) in train_selected:
        shutil.copy2(str(p), str(train_dir / p.name))
    for (p, _, _) in test_selected:
        shutil.copy2(str(p), str(test_dir / p.name))

    n_ion_in_test = sum(1 for x in test_selected if "ion_column" in x[0].name)
    print(f"✅ no_overlap 划分: 测试集 {len(test_selected)} 个（ion_column {n_ion_in_test} 个）-> {test_dir}，训练集 {len(train_selected)} 个 -> {train_dir}（已排除同前缀且窗口含测试前 {num_time_points} 个时间点的样本）")
    paths["data_folder"] = train_dir
    paths["test_folder_path"] = test_dir
    return True


def prepare_test_folder(paths: dict, label_mapping: dict, num_time_points: int, num_freq_points: int, seed: int):
    data_folder: Path = paths["data_folder"]
    base_test_folder: Path = paths["test_folder_path"]
    exp_name: str = paths["exp_name"]

    # no_overlap 模式：data_folder=datasets_for_all_no_overlap 且 test_folder=datasets_for_all_test_no_overlap
    if _prepare_no_overlap_split(paths, num_time_points, num_freq_points, seed):
        return

    # 固定测试目录模式：test_folder 的目录名不是 datasets_for_all_test
    if base_test_folder.name != "datasets_for_all_test":
        base_test_folder.mkdir(parents=True, exist_ok=True)
        existing_xlsx = list(base_test_folder.glob("*.xlsx"))
        if existing_xlsx:
            print(f"📂 使用用户指定的测试目录：{base_test_folder}（发现 {len(existing_xlsx)} 个 .xlsx 文件）")
        else:
            print(f"⚠️ 指定的测试目录 {base_test_folder} 中没有任何 .xlsx 文件，后续测试/验证将没有样本可用。")
        paths["test_folder_path"] = base_test_folder
        return

    # 自动划分模式：datasets_for_all_test/<exp_name>
    current_test_folder = base_test_folder / exp_name
    current_test_folder.mkdir(parents=True, exist_ok=True)

    existing_xlsx = list(current_test_folder.glob("*.xlsx"))
    if existing_xlsx:
        print(f"📂 检测到已有测试样本（共 {len(existing_xlsx)} 个），直接使用：{current_test_folder}")
        paths["test_folder_path"] = current_test_folder
        return

    all_xlsx = sorted(data_folder.glob("*.xlsx"))
    if not all_xlsx:
        print(f"⚠️ 在 {data_folder} 下未找到任何 .xlsx 文件，无法划分测试集。")
        paths["test_folder_path"] = current_test_folder
        return

    label_to_files: dict[str, list[Path]] = defaultdict(list)
    for f in all_xlsx:
        label = infer_label_from_filename(f.name, label_mapping)
        if label is None:
            continue
        if not is_valid_xlsx_for_model(f, num_time_points, num_freq_points):
            continue
        label_to_files[label].append(f)

    if not label_to_files:
        print("⚠️ 没有任何可用于划分测试集的合格样本。")
        paths["test_folder_path"] = current_test_folder
        return

    random.seed(int(seed))
    total_moved = 0
    for label, files in label_to_files.items():
        files_sorted = sorted(files, key=lambda p: p.name)
        selected = files_sorted[:3]  # 每类最多 3 个
        print(f"🧪 类别 [{label}] 选中 {len(selected)} 个样本作为测试集。")
        for f in selected:
            dest = current_test_folder / f.name
            print(f"  - 复制 {f} -> {dest}")
            shutil.copy2(str(f), str(dest))
            total_moved += 1

    print(f"✅ 测试集划分完成，共复制 {total_moved} 个样本到 {current_test_folder}")
    paths["test_folder_path"] = current_test_folder



def _is_new_model(cfg: dict) -> bool:
    return "new_model" in cfg


def build_model(device: torch.device, cfg: dict):
    if _is_new_model(cfg):
        model = IonDetectModel(cfg).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🧩 新模型: {total_params:,} 参数 ({trainable:,} 可训练)")
        return model

    d = cfg["data"]
    m = cfg["model"]

    model = Model_three_system_plus(
        volt_input_dim=m["volt_input_dim"],
        volt_mlp_hidden_dims=m["volt_mlp_hidden_dims"],
        mlp_output_dims=m["mlp_output_dims"],
        volt_mlp_num_layers=m["volt_mlp_num_layers"],

        impe_input_dim=m["impe_input_dim"],
        impe_mlp_hidden_dims=m["impe_mlp_hidden_dims"],
        impe_mlp_num_layers=m["impe_mlp_num_layers"],

        transformer_d_model=m["transformer_d_model"],
        nhead=m["nhead"],
        transformer_num_layers=m["transformer_num_layers"],
        param_transformer_num_layers=m["param_transformer_num_layers"],

        physic_mlp_hidden_dims=m["physic_mlp_hidden_dims"],
        physic_mlp_num_layers=m["physic_mlp_num_layers"],

        ion_attr_embed_hidden_dims=m["ion_attr_embed_hidden_dims"],
        ion_attr_embed_num_layers=m["ion_attr_embed_num_layers"],
        ion_encoder_num_layers=m["ion_encoder_num_layers"],
        ion_post_hidden_dims=m["ion_post_hidden_dims"],
        ion_post_num_layers=m["ion_post_num_layers"],

        probMLP_input_dims=m["probMLP_input_dims"],
        probMLP_hidden_dims=m["probMLP_hidden_dims"],
        probMLP_num_layers=m["probMLP_num_layers"],

        param_mlp_hidden_dims=m["param_mlp_hidden_dims"],
        param_mlp_num_layers=m["param_mlp_num_layers"],

        freq_encoder_hidden_dims=m["freq_encoder_hidden_dims"],
        freq_encoder_num_layers=m["freq_encoder_num_layers"],
        time_encoder_hidden_dims=m["time_encoder_hidden_dims"],
        time_encoder_num_layers=m["time_encoder_num_layers"],

        cross_transformer_num_layers=m["cross_transformer_num_layers"],

        param_embed_hidden_dims=m["param_embed_hidden_dims"],
        param_embed_num_layers=m["param_embed_num_layers"],
        physic_embed_hidden_dims=m["physic_embed_hidden_dims"],
        physic_embed_num_layers=m["physic_embed_num_layers"],

        envMLP_input_dim=m["envMLP_input_dim"],
        env_mlp_hidden_dims=m["env_mlp_hidden_dims"],
        env_mlp_num_layers=m["env_mlp_num_layers"],
        ep_input_dim=m["ep_input_dim"],
        ep_mlp_hidden_dims=m["ep_mlp_hidden_dims"],
        ep_mlp_num_layers=m["ep_mlp_num_layers"],

        Z_encoder_num_layers=m["Z_encoder_num_layers"],

        num_freq_points=d["num_freq_points"],
        num_time_points=d["num_time_points"],
    ).to(device)

    return model


def build_dataloaders(paths: dict, cfg: dict):
    d = cfg["data"]
    t = cfg["train"]
    ds = cfg["dataset"]

    data_folder = paths["data_folder"]
    val_data_folder = paths["test_folder_path"]
    # ✅ 收集测试集文件名（只取 name，用于与训练目录的 os.listdir 匹配）
    val_fnames = {p.name for p in Path(val_data_folder).glob("*.xlsx")}

    stats_file = paths["stats_file"]

    base_train = Dataset_2_Stable_plus(
        data_folder=data_folder,
        stats_file=str(stats_file),
        save_stats=True,
        num_time_points=d["num_time_points"],
        exclude_fnames=val_fnames,   # ✅ 新增

    )
    inter = set(base_train.file_names) & val_fnames
    print(f"[LEAK-CHECK] overlap(train, val) = {len(inter)}")
    if len(inter) > 0:
        print("[LEAK-CHECK] examples:", list(sorted(inter))[:10])

    print("[DEBUG] data_folder =", str(data_folder.resolve()))
    print("[DEBUG] base_train.file_names[:10] =")
    for x in base_train.file_names[:10]:
        print("  ", x)
    print("[DEBUG] exist check (first 10):")
    for x in base_train.file_names[:10]:
        p = (Path(data_folder) / x) if not str(x).startswith("/") else Path(x)
        print("  ", p, "exists=", p.exists())


    base_val = Dataset_2_Stable_plus(
        data_folder=val_data_folder,
        stats_file=str(stats_file),
        save_stats=False,
        num_time_points=d["num_time_points"],
    )

    pair_train = SlidingWindowPairDataset(
        base_train,
        keep_unpaired=ds["keep_unpaired"],
        debug=ds["debug"],
        focus_prefix_contains=ds.get("focus_prefix_contains", None),
        max_print=int(ds.get("max_print", 0)),
        num_time_points=d["num_time_points"],
    )

    pair_val = SlidingWindowPairDataset(
        base_val,
        keep_unpaired=ds["keep_unpaired"],
        debug=ds["debug"],
        focus_prefix_contains=None,
        max_print=int(ds.get("max_print", 0)),
        num_time_points=d["num_time_points"],
    )

    print("num base train samples:", len(base_train))
    print("num base val   samples:", len(base_val))
    print("num pair train samples:", len(pair_train))
    print("num pair val   samples:", len(pair_val))

    if len(pair_train) > 0:
        (A, B, dummy_mask) = collate_pairs([pair_train[0]])
        print("dummy_mask[0] =", bool(dummy_mask[0]))

    # WeightedRandomSampler (stage-based)
    base_weight = float(t["sampler"]["base_weight"])
    rapid_weight = float(t["sampler"]["rapid_weight"])

    sample_weights = []
    for i in range(len(pair_train)):
        sampleA, sampleB, *_ = pair_train[i]
        stageB = int(sampleB[-1].item())
        sample_weights.append(rapid_weight if stageB == 1 else base_weight)

    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True,
    )

    train_loader = DataLoader(
        pair_train,
        batch_size=int(t["batch_size"]),
        sampler=sampler,
        collate_fn=collate_pairs,
    )

    val_loader = DataLoader(
        pair_val,
        batch_size=int(t["batch_size"]),
        shuffle=False,
        collate_fn=collate_pairs,
    )

    return train_loader, val_loader


def process_single_file(
    xlsx_path_str: str,
    model_path: str,
    device_str: str,
    cfg: dict,
    exp_name: str,
    log_dir: Path,
    generate_explanations: bool = True,
):
    d = cfg["data"]
    xlsx_path = Path(xlsx_path_str)
    log_path = log_dir / f"{xlsx_path.stem}.log"

    with open(log_path, "w", encoding="utf-8") as logf:
        try:
            device = torch.device(device_str)
            model = build_model(device, cfg)
            state = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=False)
            model.eval()

            correct, predict, truth = test_single_xlsx_and_generate_explanations_three_system_1117(
                xlsx_path=xlsx_path_str,
                model=model,
                device=device,
                num_time_points=int(d["num_time_points"]),
                num_freq_points=int(d["num_freq_points"]),
                folder_name=exp_name,
                generate_explanations=generate_explanations,
            )
            return correct, predict, truth
        except Exception:
            print("❌ 文件处理出错:", file=logf)
            traceback.print_exc(file=logf)
            print("=== 处理失败 ===", file=logf)
            return None, None, None


def run_train(device: torch.device, paths: dict, cfg: dict):
    print("🚀 进入训练模式 (--train)")
    train_loader, val_loader = build_dataloaders(paths, cfg)

    model = build_model(device, cfg)

    tcfg = cfg["train"]
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )

    tb_log_dir = paths["tb_log_dir"]
    print(f"📝 TensorBoard 日志目录: {tb_log_dir}")

    # ====== 新模型分支 ======
    if _is_new_model(cfg):
        trainer = TrainerNew(
            model=model,
            optimizer=optimizer,
            device=device,
            model_save_folder=paths["model_save_folder"],
            cfg=cfg,
        )

        use_cosine = cfg["new_model"].get("use_cosine_lr", False)
        warmup_epochs = int(cfg["new_model"].get("warmup_epochs", 0))
        num_epochs = int(tcfg["num_epochs"])

        if use_cosine or warmup_epochs > 0:
            from torch.optim.lr_scheduler import (
                CosineAnnealingLR, LinearLR, SequentialLR
            )
            schedulers = []
            milestones = []

            if warmup_epochs > 0:
                warmup_sched = LinearLR(
                    optimizer, start_factor=0.01, end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                schedulers.append(warmup_sched)
                milestones.append(warmup_epochs)
                print(f"🔥 Warmup: {warmup_epochs} epochs (LR × 0.01 → 1.0)")

            if use_cosine:
                cosine_epochs = num_epochs - warmup_epochs
                cosine_sched = CosineAnnealingLR(
                    optimizer, T_max=max(cosine_epochs, 1), eta_min=1e-6,
                )
                schedulers.append(cosine_sched)
                print(f"📈 Cosine Annealing LR 调度 (T_max={cosine_epochs})")

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
        )
        return

    # ====== 旧模型分支 ======
    writer = SummaryWriter(log_dir=str(tb_log_dir))

    tr = tcfg["trainer"]
    trainer = Trainer_ThreeSystem_plus(
        model=model,
        optimizer=optimizer,
        device=device,
        model_save_folder=paths["model_save_folder"],
        alpha=float(tr["alpha"]),
        beta=float(tr["beta"]),
        gamma=float(tr["gamma"]),
        lambda_rule=float(tr["lambda_rule"]),
        lambda_group=float(tr["lambda_group"]),
        lambda_tree=float(tr["lambda_tree"]),
        lambda_band=float(tr["lambda_band"]),
        save_every=int(tr["save_every"]),
        label_smoothing=float(tr["label_smoothing"]),
    )

    tp = tcfg["train_pairs"]
    eval_every = int(tcfg.get("eval_every", 100))
    num_classes = 7
    trainer.train_pairs(
        train_loader,
        num_epochs=int(tcfg["num_epochs"]),
        lambda_consistency=float(tp["lambda_consistency"]),
        eps=float(tp["eps"]),
        use_log_space=bool(tp["use_log_space"]),
        lambda_monodec=float(tp["lambda_monodec"]),
        lambda_polarity=float(tp["lambda_polarity"]),
        weight_ratio=float(tp["weight_ratio"]),
        val_loader=val_loader,
        eval_every=eval_every,
        num_classes=num_classes,
    )

    writer.close()


def _parse_epoch_num(path: Path) -> int:
    m = re.search(r"trained_model_epoch_(\d+)\.pth$", path.name)
    return int(m.group(1)) if m else -1


def _get_max_workers(cfg: dict) -> int:
    cpu_count = multiprocessing.cpu_count()
    factor = float(cfg["test"].get("max_workers_factor", 0.5))
    return max(1, int(cpu_count * factor))


def _terminate_executor_workers(executor: ProcessPoolExecutor):
    """Ctrl+C 时尽量终止所有已启动的 worker 子进程。"""
    try:
        procs = getattr(executor, "_processes", {}) or {}
        for proc in procs.values():
            if proc is None:
                continue
            try:
                if proc.is_alive():
                    proc.terminate()
            except Exception:
                pass
        for proc in procs.values():
            if proc is None:
                continue
            try:
                proc.join(timeout=1.0)
            except Exception:
                pass
    except Exception:
        pass


def _resolve_test_model_path(paths: dict) -> Path:
    """确保 --test 可用的 best 权重存在；若缺失则自动回填。"""
    test_model_path = paths["test_model_path"]
    model_save_folder = paths["model_save_folder"]

    if test_model_path.exists():
        return test_model_path

    csv_path_acc = model_save_folder / "search_checkpoints_accuracy_sorted.csv"
    if csv_path_acc.exists():
        try:
            with open(csv_path_acc, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                best_name = rows[-1].get("ckpt_file", "").strip()
                best_path = model_save_folder / best_name
                if best_name and best_path.exists():
                    shutil.copy2(best_path, test_model_path)
                    print(f"ℹ️ 未找到 best 权重，已按 search 结果自动选择最佳 ckpt：{best_name}")
                    print(f"ℹ️ 已复制为: {test_model_path}")
                    return test_model_path
        except Exception as e:
            print(f"⚠️ 读取 {csv_path_acc} 失败，将尝试按 epoch 回退：{e}")

    ckpts = [
        p for p in model_save_folder.glob("trained_model_epoch_*.pth")
        if p.name != "trained_model_epoch_best.pth"
    ]
    ckpts = sorted(ckpts, key=_parse_epoch_num)
    if ckpts:
        fallback = ckpts[-1]
        shutil.copy2(fallback, test_model_path)
        print(f"ℹ️ 未找到 best 权重，也未能从 search CSV 选最佳；已回退到最高 epoch：{fallback.name}")
        print(f"ℹ️ 已复制为: {test_model_path}")
        return test_model_path

    raise FileNotFoundError(
        f"未找到可用于测试的权重文件：{test_model_path}，且目录 {model_save_folder} 中不存在 trained_model_epoch_*.pth"
    )


def run_test(paths: dict, cfg: dict):
    print("🔍 进入测试模式 (--test)")

    test_model_path = _resolve_test_model_path(paths)
    test_folder_path = paths["test_folder_path"]
    exp_name = paths["exp_name"]
    log_dir = paths["debug_log_dir_test"]

    device_str = cfg["test"].get("device_str", "cpu")
    max_workers = _get_max_workers(cfg)
    mp_ctx = multiprocessing.get_context("spawn")

    print(f"🔍 正在加载模型: {test_model_path}")
    xlsx_paths = [str(test_folder_path / f) for f in os.listdir(test_folder_path) if f.endswith(".xlsx")]
    print(f"📊 共检测到 {len(xlsx_paths)} 个 .xlsx 文件，准备并行处理...")

    correct_count = 0
    total = len(xlsx_paths)
    processed_count = 0
    failed_count = 0
    confusion_counter = defaultdict(lambda: defaultdict(int))

    executor = None
    try:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as executor:
            futures = [
                executor.submit(
                    process_single_file,
                    path,
                    str(test_model_path),
                    device_str,
                    cfg,
                    exp_name,
                    log_dir,
                    True,
                )
                for path in xlsx_paths
            ]

            for future in as_completed(futures):
                try:
                    correct, predict, truth = future.result()
                    if correct is None:
                        failed_count += 1
                        continue
                    processed_count += 1
                    if correct:
                        correct_count += 1
                    confusion_counter[truth][predict] += 1
                except Exception as e:
                    print(f"❌ 文件处理失败: {e}")
                    failed_count += 1
    except KeyboardInterrupt:
        print("\n⛔ 检测到 Ctrl+C，正在终止测试 worker 子进程...")
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            _terminate_executor_workers(executor)
        raise

    if processed_count > 0:
        acc = correct_count / processed_count
        print(f"\n✅ 总共测试样本数: {total}")
        print(f"✅ 成功评测样本数: {processed_count}")
        print(f"⚠️ 失败样本数: {failed_count}")
        print(f"🎯 预测正确样本数: {correct_count}")
        print(f"📊 准确率: {acc:.2%}")
    else:
        if total == 0:
            print("⚠️ 未找到任何 .xlsx 测试文件")
        else:
            print(f"⚠️ 共 {total} 个样本，但全部评测失败（失败数 {failed_count}）")

    print("\n📉 错误分析（真实标签 → 预测标签 → 个数）:")
    for truth_label, pred_dict in confusion_counter.items():
        for predicted_label, count in pred_dict.items():
            print(f"  真实: {truth_label} → 预测: {predicted_label} : {count} 个")


def run_search(paths: dict, cfg: dict):
    print("🔎 进入搜索模式 (--search)")

    model_save_folder = paths["model_save_folder"]
    test_folder_path = paths["test_folder_path"]
    exp_name = paths["exp_name"]
    log_dir = paths["debug_log_dir_search"]

    device_str = cfg["test"].get("device_str", "cpu")
    max_workers = _get_max_workers(cfg)
    mp_ctx = multiprocessing.get_context("spawn")

    ckpts = [
        p for p in model_save_folder.glob("trained_model_epoch_*.pth")
        if p.name not in {"trained_model_epoch_best.pth"}
    ]
    ckpts = sorted(ckpts, key=_parse_epoch_num)

    if not ckpts:
        print(f"⚠️ 未在 {model_save_folder} 找到任何 trained_model_epoch_*.pth")
        return

    xlsx_paths = [str(test_folder_path / f) for f in os.listdir(test_folder_path) if f.endswith(".xlsx")]
    if not xlsx_paths:
        print(f"⚠️ 未在 {test_folder_path} 找到任何 .xlsx 测试文件")
        return

    print(f"🔎 共发现 {len(ckpts)} 个 checkpoint，将逐个评测；测试集样本数：{len(xlsx_paths)}")
    results = []

    for i, ckpt_path in enumerate(ckpts, 1):
        epoch_num = _parse_epoch_num(ckpt_path)
        if epoch_num < 0:
            print(f"跳过无法识别 epoch 的文件：{ckpt_path.name}")
            continue

        print(f"\n[{i}/{len(ckpts)}] 🔍 评测 checkpoint: {ckpt_path.name}  (epoch={epoch_num})")
        correct_count, total = 0, len(xlsx_paths)
        processed_count, failed_count = 0, 0
        confusion_counter = defaultdict(lambda: defaultdict(int))

        executor = None
        try:
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as executor:
                futures = [
                    executor.submit(
                        process_single_file,
                        path,
                        str(ckpt_path),
                        device_str,
                        cfg,
                        exp_name,
                        log_dir,
                        False,
                    )
                    for path in xlsx_paths
                ]

                for future in as_completed(futures):
                    try:
                        correct, predict, truth = future.result()
                        if correct is None:
                            failed_count += 1
                            continue
                        processed_count += 1
                        if correct:
                            correct_count += 1
                        confusion_counter[truth][predict] += 1
                    except Exception as e:
                        print(f"❌ 文件处理失败: {e}")
                        failed_count += 1
        except KeyboardInterrupt:
            print("\n⛔ 检测到 Ctrl+C，正在终止 search worker 子进程...")
            if executor is not None:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                _terminate_executor_workers(executor)
            raise

        acc = (correct_count / processed_count) if processed_count > 0 else 0.0
        print(
            f"🎯 epoch={epoch_num} | 正确 {correct_count}/{processed_count} "
            f"| 失败 {failed_count} | 准确率={acc:.2%}"
        )
        results.append((epoch_num, acc, correct_count, processed_count, ckpt_path.name))

    results_by_epoch = sorted(results, key=lambda x: x[0])
    csv_path_epoch = model_save_folder / "search_checkpoints_epoch_sorted.csv"
    with open(csv_path_epoch, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank_by_epoch", "epoch", "accuracy", "correct", "total", "ckpt_file"])
        for rank, (ep, acc, cor, tot, name) in enumerate(results_by_epoch, 1):
            w.writerow([rank, ep, f"{acc:.6f}", cor, tot, name])
    print(f"\n✅ 已保存 CSV（按 epoch 升序）：{csv_path_epoch}")

    results_by_acc = sorted(results, key=lambda x: x[1])
    csv_path_acc = model_save_folder / "search_checkpoints_accuracy_sorted.csv"
    with open(csv_path_acc, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank_by_acc", "epoch", "accuracy", "correct", "total", "ckpt_file"])
        for rank, (ep, acc, cor, tot, name) in enumerate(results_by_acc, 1):
            w.writerow([rank, ep, f"{acc:.6f}", cor, tot, name])
    print(f"✅ 已保存 CSV（按准确率升序）：{csv_path_acc}")

    try:
        eps = [r[0] for r in results_by_epoch]
        acc_e = [r[1] for r in results_by_epoch]
        plt.figure()
        plt.plot(eps, acc_e, marker="o")
        plt.xlabel("Epoch (ascending)")
        plt.ylabel("Accuracy")
        plt.title("Checkpoint Accuracy vs Epoch (epoch-ascending)")
        plt.grid(True)
        plt.tight_layout()
        png_path_by_epoch = model_save_folder / "search_checkpoints_accuracy_by_epoch.png"
        plt.savefig(png_path_by_epoch, dpi=150)
        plt.close()
        print(f"🖼️ 已保存曲线图（按 epoch 升序）：{png_path_by_epoch}")

        ranks = list(range(1, len(results_by_acc) + 1))
        accs = [r[1] for r in results_by_acc]
        epochs_sorted_for_acc = [r[0] for r in results_by_acc]
        plt.figure()
        plt.plot(ranks, accs, marker="o")
        plt.xlabel("Rank by Accuracy (ascending)")
        plt.ylabel("Accuracy")
        plt.title("Checkpoint Accuracy (sorted by accuracy ascending)")
        if len(ranks) <= 20:
            plt.xticks(ranks, [f"ep{e}" for e in epochs_sorted_for_acc], rotation=45, ha="right")
        plt.grid(True)
        plt.tight_layout()
        png_path_sorted = model_save_folder / "search_checkpoints_accuracy_sorted.png"
        plt.savefig(png_path_sorted, dpi=150)
        plt.close()
        print(f"🖼️ 已保存曲线图（按准确率升序）：{png_path_sorted}")
    except Exception as e:
        print(f"⚠️ 画图失败：{e}")


def parse_args():
    parser = argparse.ArgumentParser(description="运行 ion_detect 模型训练 / 测试 / 搜索（YAML 配置版）")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径，例如 20251208a.yaml")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="运行训练模式")
    group.add_argument("--test", action="store_true", help="运行测试模式")
    group.add_argument("--search", action="store_true", help="遍历目录中的所有 epoch 权重并评测")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)

    # seed
    seed = int(cfg["experiment"].get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    device, device_str = build_device()
    paths = build_paths(cfg)

    d = cfg["data"]
    num_time_points = int(d["num_time_points"])
    num_freq_points = int(d["num_freq_points"])

    label_mapping = load_label_mapping(paths["json_path"])
    # 自动划分测试样本/或使用固定测试目录
    prepare_test_folder(paths, label_mapping, num_time_points, num_freq_points, seed)

    print(f"✅ 当前使用设备: {device}")
    print(f"📂 训练集目录: {paths['data_folder']}")
    print(f"📂 测试/验证集目录: {paths['test_folder_path']}")
    print(f"🧪 experiment: {paths['exp_name']}")

    if args.train:
        run_train(device, paths, cfg)
    elif args.test:
        run_test(paths, cfg)
    elif args.search:
        run_search(paths, cfg)
    else:
        raise ValueError("必须指定 --train / --test / --search 之一")


if __name__ == "__main__":
    main()
