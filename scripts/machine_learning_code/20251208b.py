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
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

# ==== 项目内部模块 ====
from model_datasets import Dataset_2_Stable_plus
from model_models_schemeB import Model_three_system_1117
from model_test import test_single_xlsx_and_generate_explanations_three_system_1117
from model_train import Trainer_ThreeSystem_1117
from paired_dataset import SlidingWindowPairDataset, collate_pairs
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler


# ==========================
# 全局基础配置（后续可以迁移到 YAML）
# ==========================

# 频点数 / 时间点数（跟数据集和模型保持一致）
NUM_FREQ_POINTS = 63
NUM_TIME_POINTS = 3

# 训练相关（小数据集推荐）
BATCH_SIZE = 8          # 样本很少时，减小 batch，提升梯度多样性
NUM_EPOCHS = 2000       # 小数据集下不需要太多 epoch，避免严重过拟合
LEARNING_RATE = 1e-4    # 比 1e-5 略大一些，让模型能更快收敛


# 多进程测试相关：最多使用 CPU 一半
CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = max(1, CPU_COUNT // 2)
test_device_str = "cpu"


def build_device():
    """获取设备与 device 字符串，保证多进程里统一使用同样的 device_str。"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_str = "cuda"
    else:
        device = torch.device("cpu")
        device_str = "cpu"
    return device, device_str


def build_paths():
    """集中管理本脚本依赖的所有路径。"""
    current_file = Path(__file__).resolve()
    folder_name = current_file.stem             # 当前脚本文件名（不含后缀）
    base_dir = current_file.parent.parent.parent

    # 训练集 / 测试集
    data_folder = base_dir / "datasets" / "datasets_for_all"
    # 注意：真正使用时会在此基础上加上 folder_name 作为子目录
    test_folder_path = base_dir / "datasets" / "datasets_for_all_test"
    # test_folder_path = base_dir / "datasets" / "datasets_for_range_ion_0_6_2ppm"

    # 模型保存目录
    model_save_folder = base_dir / "output" / "trained_model_save" / folder_name
    model_save_folder.mkdir(parents=True, exist_ok=True)

    # 默认测试模型（最终模型）
    test_model_path = model_save_folder / "trained_model_epoch_final.pth"

    # 统计量与标签映射
    stats_file = base_dir / "datasets" / "stats_dataset.json"
    json_path = current_file.parent / "label_mapping.json"

    # debug 日志目录（测试时写入单文件 log）
    debug_log_dir = base_dir / "output" / "debug_logs"
    debug_log_dir.mkdir(parents=True, exist_ok=True)

    # 新增：TensorBoard 日志目录
    tb_log_dir = base_dir / "output" / "tensorboard" / folder_name
    tb_log_dir.mkdir(parents=True, exist_ok=True)

    return {
        "current_file": current_file,
        "folder_name": folder_name,
        "base_dir": base_dir,
        "data_folder": data_folder,
        "test_folder_path": test_folder_path,   # 后续会被细化为带 folder_name 的子目录
        "model_save_folder": model_save_folder,
        "test_model_path": test_model_path,
        "stats_file": stats_file,
        "json_path": json_path,
        "debug_log_dir": debug_log_dir,
        "tb_log_dir": tb_log_dir,
    }


def load_label_mapping(json_path: Path):
    """读取 label_mapping.json，方便后续需要时使用。"""
    if not json_path.exists():
        print(f"⚠️ 未找到 label_mapping.json: {json_path}，后续如果不需要可以忽略。")
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
    return label_mapping


# ==========================
# 新增：根据文件名推断离子类别 & 切分测试集
# ==========================

def infer_label_from_filename(fname: str, label_mapping: dict) -> str | None:
    """
    尝试从文件名推断离子类别：
    1. 优先使用 label_mapping.json 中的键（如果是中文标签，会直接匹配）
    2. 退而求其次，使用固定的中文关键词匹配
    返回：匹配到的“类别名”（字符串）；若无法识别则返回 None。
    """
    # 先尝试用 label_mapping 的 key 做子串匹配
    if label_mapping:
        for label in label_mapping.keys():
            try:
                if isinstance(label, str) and label in fname:
                    return label
            except Exception:
                continue

    # 关键词兜底（按你当前任务里的 7 类来写）
    keywords = [
        "钙离子",
        "钠离子",
        "镍离子",
        "铬离子",
        "铜离子",
        "铁离子",
        "无污染",
    ]
    for kw in keywords:
        if (kw in fname) and  ("ion_column" not in fname):
            return kw

    # 常见“无污染”别名兜底
    lower = fname.lower()
    if ("blank" in lower) or ("纯水" in lower) or ("ion_column" in fname):
        return "无污染"

    return None
def is_valid_xlsx_for_model(xlsx_path: Path,
                            num_time_points: int,
                            num_freq_points: int) -> bool:
    """
    判断一个 xlsx 是否“可用”，用于划分测试集时过滤坏样本。

    判定规则（与你给的逻辑保持一致）：
    1) 能成功读取为 DataFrame
    2) 必须包含列：Time(h), mean_voltage, Zreal, Zimag, Freq(Hz)
    3) 至少有 num_time_points 个不同的 Time(h)
    4) 选定 num_time_points 个时间点（升序前 num_time_points），
       对于每个时间点：
       - 行数 >= num_freq_points
       - mean_voltage 非 NaN
    """
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

    # 去掉 NaN 后统计时间点
    time_series = df["Time(h)"].dropna().unique().tolist()
    if len(time_series) < num_time_points:
        print(
            f"[SKIP] 文件 {xlsx_path.name}: 仅有 {len(time_series)} 个时间点 "
            f"< 要求的 {num_time_points}"
        )
        return False

    # 选定需要检查的时间点：升序取前 num_time_points 个
    time_points = sorted(time_series)[:num_time_points]

    for t in time_points:
        time_data = df[df["Time(h)"] == t]
        if time_data.empty:
            print(f"[SKIP] 文件 {xlsx_path.name}: 缺少时间点 {t}h")
            return False

        # 按频率排序
        time_data = time_data.sort_values(by="Freq")

        voltage = time_data["mean_voltage"].values[0]
        if pd.isna(voltage):
            print(f"[SKIP] 文件 {xlsx_path.name}: 时间点 {t}h 的 mean_voltage 为 NaN")
            return False

        impedance_np = time_data[["Zreal", "Zimag"]].values
        if impedance_np.shape[0] < num_freq_points:
            print(
                f"[SKIP] 文件 {xlsx_path.name}: 时间点 {t}h 的阻抗点数 "
                f"{impedance_np.shape[0]} < 要求的 {num_freq_points}"
            )
            return False

    # 所有检查通过
    print(f"[OK] 可作为测试样本: {xlsx_path.name}")
    return True

def prepare_test_folder(paths: dict, label_mapping: dict,num_time_points, num_freq_points):
    """
    目标：
    - 如果 test_folder_path == .../datasets_for_all_test：
        从 data_folder 中为每种离子抽取 3 个“可用”样本，移动到
        test_folder_path / folder_name 对应的子目录，作为自动划分的测试集；
    - 否则（用户指定了其它测试目录）：
        不做任何划分/移动，直接使用该目录下已有的 .xlsx 作为测试/验证集。
    """
    data_folder: Path = paths["data_folder"]
    base_test_folder: Path = paths["test_folder_path"]
    folder_name: str = paths["folder_name"]

    # ========= 新增：区分“自动划分模式”和“固定测试目录模式” =========
    if base_test_folder.name != "datasets_for_all_test":
        # 固定测试目录模式：不再创建子目录，也不做划分/移动
        base_test_folder.mkdir(parents=True, exist_ok=True)
        existing_xlsx = list(base_test_folder.glob("*.xlsx"))

        if existing_xlsx:
            print(
                f"📂 使用用户指定的测试目录：{base_test_folder} "
                f"（发现 {len(existing_xlsx)} 个 .xlsx 文件）"
            )
        else:
            print(
                f"⚠️ 指定的测试目录 {base_test_folder} 中没有任何 .xlsx 文件，"
                f"后续测试/验证将没有样本可用。"
            )

        # 直接把 test_folder_path 固定为这个目录
        paths["test_folder_path"] = base_test_folder
        return

    # ========= 以下是原来的“自动划分测试集”逻辑，仅在
    #          test_folder_path == .../datasets_for_all_test 时生效 =========
    current_test_folder = base_test_folder / folder_name
    current_test_folder.mkdir(parents=True, exist_ok=True)

    # 若该目录已存在测试样本，则不再重新划分
    existing_xlsx = list(current_test_folder.glob("*.xlsx"))
    if existing_xlsx:
        print(
            f"📂 检测到已有测试样本（共 {len(existing_xlsx)} 个），"
            f"直接使用：{current_test_folder}"
        )
        paths["test_folder_path"] = current_test_folder
        return

    # ----------------- 自动划分测试集 -----------------
    all_xlsx = sorted(data_folder.glob("*.xlsx"))

    if not all_xlsx:
        print(f"⚠️ 在 {data_folder} 下未找到任何 .xlsx 文件，无法划分测试集。")
        paths["test_folder_path"] = current_test_folder
        return

    label_to_files: dict[str, list[Path]] = defaultdict(list)

    for f in all_xlsx:
        label = infer_label_from_filename(f.name, label_mapping)
        if label is None:
            # 不识别的样本暂时忽略，不参与“每类 3 个”的划分
            continue

        # 在划分阶段就做“可用性检查”，只把合格样本放入候选池
        if not is_valid_xlsx_for_model(f,num_time_points, num_freq_points):
            continue

        label_to_files[label].append(f)

    if not label_to_files:
        print("⚠️ 没有任何可用于划分测试集的合格样本。")
        paths["test_folder_path"] = current_test_folder
        return

    # 为保证可复现，这里不打乱，只按文件名排序后取前 3 个
    random.seed(42)

    total_moved = 0
    for label, files in label_to_files.items():
        if not files:
            continue

        files_sorted = sorted(files, key=lambda p: p.name)
        # 各类最多 3 个，如果不足 3 个，就全拿
        selected = files_sorted[:3]

        print(f"🧪 类别 [{label}] 选中 {len(selected)} 个样本作为测试集。")
        for f in selected:
            dest = current_test_folder / f.name
            print(f"  - 移动 {f} -> {dest}")
            shutil.move(str(f), str(dest))
            total_moved += 1

    print(
        f"✅ 测试集划分完成，共移动 {total_moved} 个样本到 {current_test_folder}"
    )
    paths["test_folder_path"] = current_test_folder


# ==========================
# 模型构建
# ==========================

def build_model(device: torch.device):
    """
    构造 Model_three_system_1117 模型。

    当前参数仍然是硬编码的，可以在未来迁移到 config 文件中。
    """
    model = Model_three_system_1117(
        # ---- 电压分支 ----
        volt_input_dim=1,
        volt_mlp_hidden_dims=[64],
        mlp_output_dims=64,
        volt_mlp_num_layers=1,

        # ---- 阻抗分支 ----
        impe_input_dim=2,
        impe_mlp_hidden_dims=[64],
        impe_mlp_num_layers=1,

        # ---- Transformer 配置 ----
        transformer_d_model=64,
        nhead=4,
        transformer_num_layers=1,
        param_transformer_num_layers=1,

        # ---- 物理 MLP ----
        physic_mlp_hidden_dims=[64],
        physic_mlp_num_layers=1,

        # ---- Ion 属性 ----
        ion_attr_embed_hidden_dims=[64],
        ion_attr_embed_num_layers=1,
        ion_encoder_num_layers=1,
        ion_post_hidden_dims=[64],
        ion_post_num_layers=1,

        # ---- 分类头 ----
        probMLP_input_dims=64,
        probMLP_hidden_dims=[64],
        probMLP_num_layers=1,

        # ---- 物性参数 MLP ----
        param_mlp_hidden_dims=[64],
        param_mlp_num_layers=1,

        # ---- 编码器 ----
        freq_encoder_hidden_dims=[64],
        freq_encoder_num_layers=1,
        time_encoder_hidden_dims=[64],
        time_encoder_num_layers=1,

        # ---- Cross Transformer ----
        cross_transformer_num_layers=1,

        # ---- 参数 / 物理 embedding ----
        param_embed_hidden_dims=[64],
        param_embed_num_layers=0,
        physic_embed_hidden_dims=[64],
        physic_embed_num_layers=0,

        # ---- 环境参数 / EP 分支 ----
        envMLP_input_dim=3,
        env_mlp_hidden_dims=[64],
        env_mlp_num_layers=1,
        ep_input_dim=8,
        ep_mlp_hidden_dims=[64],
        ep_mlp_num_layers=1,

        # ---- Z 编码器 ----
        Z_encoder_num_layers=1,

        num_freq_points=NUM_FREQ_POINTS,
        num_time_points=NUM_TIME_POINTS,
    ).to(device)

    return model


def build_dataloaders(paths: dict):
    """
    构建 Dataset 和 Dataloader（训练/验证）。

    - 基础数据：Dataset_2_Stable_plus
    - Pair 数据：SlidingWindowPairDataset
    - 训练集：paths["data_folder"]
    - 验证/测试集：paths["test_folder_path"]（当前脚本对应的子目录）
    """
    data_folder = paths["data_folder"]
    val_data_folder = paths["test_folder_path"]
    stats_file = paths["stats_file"]

    # 1) 构建基础训练数据集（带归一化, 并写入 stats）
    base_train = Dataset_2_Stable_plus(
        data_folder=data_folder,
        stats_file=str(stats_file),   # json 路径用 str
        save_stats=True,
        num_time_points=NUM_TIME_POINTS
    )

    # 2) 构建验证数据集（使用同一份 stats_file，只读不再写）
    base_val = Dataset_2_Stable_plus(
        data_folder=val_data_folder,
        stats_file=str(stats_file),
        save_stats=False,             # 验证集只复用统计量，避免覆盖
        num_time_points=NUM_TIME_POINTS
    )

    # 3) 构建滑动窗口 Pair 数据集
    pair_train = SlidingWindowPairDataset(
        base_train,
        keep_unpaired="drop",
        debug=True,
        focus_prefix_contains="20240915_2ppm铜离子污染测试_旧版电解槽_ion_firecloud_",
        max_print=0,
        num_time_points=NUM_TIME_POINTS
    )

    # 验证集不做前缀过滤，完整使用指定文件夹中的样本
    pair_val = SlidingWindowPairDataset(
        base_val,
        keep_unpaired="drop",
        debug=False,
        focus_prefix_contains=None,
        max_print=0,
        num_time_points=NUM_TIME_POINTS
    )

    print("num base train samples:", len(base_train))
    print("num base val   samples:", len(base_val))
    print("num pair train samples:", len(pair_train))
    print("num pair val   samples:", len(pair_val))

    if len(pair_train) > 0:
        (A, B, dummy_mask) = collate_pairs([pair_train[0]])
        print("dummy_mask[0] =", bool(dummy_mask[0]))

    # 4) 训练集采样权重（按阶段 upweight 剧烈阶段）
    train_dataset = pair_train
    val_dataset = pair_val

    base_weight = 1.0        # stage = 0 时的权重
    rapid_weight = 3.0       # stage = 1 时的权重（可以根据需要调大/调小）

    sample_weights = []
    for i in range(len(train_dataset)):
        sampleA, sampleB, *others = train_dataset[i]
        stageA = int(sampleA[-1].item())
        stageB = int(sampleB[-1].item())

        # 只要 B 是剧烈阶段，就认为这个 pair 属于 "rapid" pair
        if stageB == 1:
            sample_weights.append(rapid_weight)
        else:
            sample_weights.append(base_weight)

    # 转成 tensor 供 WeightedRandomSampler 使用
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),  # 每个 epoch 采样这么多 pair
        replacement=True,                        # 允许重复采样
    )

    # 训练集：使用加权采样，不再用 shuffle=True
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        collate_fn=collate_pairs,
    )

    # 验证集：保持原来的均匀顺序
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_pairs,
    )

    return train_loader, val_loader


# ==========================
# 测试/搜索用的单文件处理函数
# ==========================

def process_single_file(
    xlsx_path_str: str,
    model_path: str,
    device_str: str,
    num_freq_points: int,
    num_time_points: int,
    folder_name: str,
    log_dir: Path,
):
    """
    单个 xlsx 文件的完整测试流程：
    - 加载模型
    - 调用 test_single_xlsx_and_generate_explanations_three_system_1117
    - 返回 (correct, predict, truth)

    ⚠️ 当前实现：每个进程 / 文件都会重新加载模型，逻辑较简单但效率略低。
       如果后续测试文件非常多，可以用“进程初始化时加载模型”的方式进行优化。
    """
    xlsx_path = Path(xlsx_path_str)
    log_path = log_dir / f"{xlsx_path.stem}.log"

    with open(log_path, "w", encoding="utf-8") as logf:
        try:
            device = torch.device(device_str)
            model = build_model(device)
            state = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=False)
            model.eval()

            correct, predict, truth = test_single_xlsx_and_generate_explanations_three_system_1117(
                xlsx_path=xlsx_path_str,
                model=model,
                device=device,
                num_time_points=num_time_points,
                num_freq_points=num_freq_points,
                folder_name=folder_name,
            )
            return correct, predict, truth
        except Exception:
            print("❌ 文件处理出错:", file=logf)
            traceback.print_exc(file=logf)
            print("=== 处理失败 ===", file=logf)
            return None, None, None


# ==========================
# 三种模式：train / test / search
# ==========================

def run_train(device, paths):
    print("🚀 进入训练模式 (--train)")
    train_loader, val_loader = build_dataloaders(paths)

    model = build_model(device)
    # 使用适度的 weight_decay，缓解小数据集下的过拟合
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    # ✅ 新增：创建 TensorBoard writer
    tb_log_dir = paths["tb_log_dir"]
    print(f"📝 TensorBoard 日志目录: {tb_log_dir}")
    writer = SummaryWriter(log_dir=str(tb_log_dir))

    trainer = Trainer_ThreeSystem_1117(
        model=model,
        optimizer=optimizer,
        device=device,
        model_save_folder=paths["model_save_folder"],

        # ---- 主任务：分类仍然绝对主导 ----
        alpha=0.05,     # 电压做一点点辅助，不要太大
        beta=1.0,       # 分类主任务
        gamma=6e-5,      # 浓度暂时不开启，避免目标太多

        # ---- 规则 / 分组 / 决策树 / 频段：全部作为弱正则 ----
        lambda_rule=5e-4,   # rule 原始 ~40 → 加权 ~0.02
        lambda_group=5e-4,  # group 原始 ~1.3 → 加权 ~0.0007
        lambda_tree=5e-4,   # tree 原始 ~3.2 → 加权 ~0.0016
        lambda_band=2e-3,   # band 原始 ~0.13 → 加权 ~0.00026

        save_every=10,
        label_smoothing=0.05   # 可以保留 0.05 或者用你在阶段一中表现最好的值
    )

    trainer.train_pairs(
        train_loader,
        num_epochs=NUM_EPOCHS,
        lambda_consistency=5e-4,  # consistency 原始 ~0.48 → 加权 ~0.00024
        eps=1e-9,
        use_log_space=True,
        lambda_monodec=5e-4,      # monodec 原始 ~3 左右 → 加权 ~0.0015
        lambda_polarity=2e-4,     # polarity 原始 ~2 左右 → 加权 ~0.0004
        weight_ratio=3.0          # 让 stage=1 稍微更重一点，但不要太极端
    )

    # ✅ 训练结束记得关掉
    writer.close()


def run_test(device_str: str, paths: dict):
    """测试模式：使用最终final模型 test_model_path，对 test_folder_path 下所有 .xlsx 做并行测试。"""
    print("🔍 进入测试模式 (--test)")

    test_model_path = paths["test_model_path"]
    test_folder_path = paths["test_folder_path"]
    folder_name = paths["folder_name"]
    log_dir = paths["debug_log_dir"]

    print(f"🔍 正在加载模型: {test_model_path}")
    xlsx_paths = [
        str(test_folder_path / f)
        for f in os.listdir(test_folder_path)
        if f.endswith(".xlsx")
    ]
    print(f"📊 共检测到 {len(xlsx_paths)} 个 .xlsx 文件，准备并行处理...")

    correct_count = 0
    total = len(xlsx_paths)

    # confusion_counter[true_label][predicted_label] = count
    confusion_counter = defaultdict(lambda: defaultdict(int))

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(
                process_single_file,
                path,
                str(test_model_path),
                test_device_str,   # ← 这里使用 "cpu"
                NUM_FREQ_POINTS,
                NUM_TIME_POINTS,
                folder_name,
                log_dir,
            )
            for path in xlsx_paths
        ]

        for future in as_completed(futures):
            try:
                correct, predict, truth = future.result()
                if correct is None:
                    continue
                if correct:
                    correct_count += 1
                confusion_counter[truth][predict] += 1
            except Exception as e:
                print(f"❌ 文件处理失败: {e}")

    if total > 0:
        accuracy = correct_count / total
        print(f"\n✅ 总共测试样本数: {total}")
        print(f"🎯 预测正确样本数: {correct_count}")
        print(f"📊 准确率: {accuracy:.2%}")
    else:
        print("⚠️ 未找到任何 .xlsx 测试文件")

    # 打印详细的错误分布（confusion matrix）
    print("\n📉 错误分析（真实标签 → 预测标签 → 个数）:")
    for truth_label, pred_dict in confusion_counter.items():
        for predicted_label, count in pred_dict.items():
            print(f"  真实: {truth_label} → 预测: {predicted_label} : {count} 个")


def _parse_epoch_num(path: Path) -> int:
    """从 checkpoint 文件名中解析 epoch 数字。形如 trained_model_epoch_123.pth。"""
    m = re.search(r"trained_model_epoch_(\d+)\.pth$", path.name)
    return int(m.group(1)) if m else -1


def run_search(device_str: str, paths: dict):
    """
    搜索模式：遍历某个目录下所有 checkpoint，对测试集做完整评估，导出 CSV + 曲线图。
    """
    print("🔎 进入搜索模式 (--search)")

    model_save_folder = paths["model_save_folder"]
    test_folder_path = paths["test_folder_path"]
    folder_name = paths["folder_name"]
    log_dir = paths["debug_log_dir"]

    # 1) 搜索所有 checkpoint（不含 final）
    ckpts = [
        p for p in model_save_folder.glob("trained_model_epoch_*.pth")
        if p.name != "trained_model_epoch_final.pth"
    ]
    ckpts = sorted(ckpts, key=_parse_epoch_num)

    if not ckpts:
        print(f"⚠️ 未在 {model_save_folder} 找到任何 trained_model_epoch_*.pth")
        return

    # 2) 准备测试文件列表
    xlsx_paths = [
        str(test_folder_path / f)
        for f in os.listdir(test_folder_path)
        if f.endswith(".xlsx")
    ]
    if not xlsx_paths:
        print(f"⚠️ 未在 {test_folder_path} 找到任何 .xlsx 测试文件")
        return

    print(f"🔎 共发现 {len(ckpts)} 个 checkpoint，将逐个评测；测试集样本数：{len(xlsx_paths)}")

    # 结果列表：[(epoch, acc, correct, total, ckpt_file), ...]
    results = []

    # 3) 逐个 checkpoint 做评估
    for i, ckpt_path in enumerate(ckpts, 1):
        epoch_num = _parse_epoch_num(ckpt_path)
        if epoch_num < 0:
            print(f"跳过无法识别 epoch 的文件：{ckpt_path.name}")
            continue

        print(f"\n[{i}/{len(ckpts)}] 🔍 评测 checkpoint: {ckpt_path.name}  (epoch={epoch_num})")
        correct_count, total = 0, len(xlsx_paths)
        confusion_counter = defaultdict(lambda: defaultdict(int))

        # 对当前 ckpt 跑一遍完整测试
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(
                    process_single_file,
                    path,
                    str(ckpt_path),   # 注意这里使用当前 epoch 对应的权重
                    test_device_str,  # ← 这里使用 "cpu"
                    NUM_FREQ_POINTS,
                    NUM_TIME_POINTS,
                    folder_name,
                    log_dir,
                )
                for path in xlsx_paths
            ]

            for future in as_completed(futures):
                try:
                    correct, predict, truth = future.result()
                    if correct is None:
                        continue
                    if correct:
                        correct_count += 1
                    confusion_counter[truth][predict] += 1
                except Exception as e:
                    print(f"❌ 文件处理失败: {e}")

        acc = (correct_count / total) if total > 0 else 0.0
        print(f"🎯 epoch={epoch_num} | 正确 {correct_count}/{total} | 准确率={acc:.2%}")
        results.append((epoch_num, acc, correct_count, total, ckpt_path.name))

    # 4) 导出 CSV（按 epoch 升序）
    results_by_epoch = sorted(results, key=lambda x: x[0])
    csv_path_epoch = model_save_folder / "search_checkpoints_epoch_sorted.csv"
    with open(csv_path_epoch, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank_by_epoch", "epoch", "accuracy", "correct", "total", "ckpt_file"])
        for rank, (ep, acc, cor, tot, name) in enumerate(results_by_epoch, 1):
            w.writerow([rank, ep, f"{acc:.6f}", cor, tot, name])
    print(f"\n✅ 已保存 CSV（按 epoch 升序）：{csv_path_epoch}")

    # 5) 另存一份“按准确率升序”的 CSV
    results_by_acc = sorted(results, key=lambda x: x[1])
    csv_path_acc = model_save_folder / "search_checkpoints_accuracy_sorted.csv"
    with open(csv_path_acc, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank_by_acc", "epoch", "accuracy", "correct", "total", "ckpt_file"])
        for rank, (ep, acc, cor, tot, name) in enumerate(results_by_acc, 1):
            w.writerow([rank, ep, f"{acc:.6f}", cor, tot, name])
    print(f"✅ 已保存 CSV（按准确率升序）：{csv_path_acc}")

    # 6) 画图（主图按 epoch 升序）
    try:
        # 图1：Accuracy vs Epoch
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

        # 图2：按准确率升序的折线图
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


# ==========================
# main 入口
# ==========================

def parse_args():
    parser = argparse.ArgumentParser(description="运行 ion_detect 模型训练 / 测试 / 搜索脚本")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="运行训练模式")
    group.add_argument("--test", action="store_true", help="运行测试模式")
    group.add_argument("--search", action="store_true", help="遍历目录中的所有 epoch 权重并评测")
    return parser.parse_args()


def main():
    args = parse_args()
    device, device_str = build_device()
    paths = build_paths()
    num_time_points = NUM_TIME_POINTS
    num_freq_points = NUM_FREQ_POINTS

    # 读取标签映射，并据此 + 文件名规则，自动划分测试样本
    label_mapping = load_label_mapping(paths["json_path"])
    prepare_test_folder(paths, label_mapping,num_time_points, num_freq_points)

    print(f"✅ 当前使用设备: {device}")
    print(f"📂 训练集目录: {paths['data_folder']}")
    print(f"📂 测试/验证集目录: {paths['test_folder_path']}")

    if args.train:
        run_train(device, paths)
    elif args.test:
        run_test(device_str, paths)
    elif args.search:
        run_search(device_str, paths)
    else:
        # 理论上不会到这里，因为互斥组 required=True
        raise ValueError("必须指定 --train / --test / --search 之一")


if __name__ == "__main__":
    main()
