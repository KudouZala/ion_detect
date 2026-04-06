import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch


def load_config(cfg_path: Path) -> dict:
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("缺少 PyYAML 依赖。请先安装：pip install pyyaml") from e

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件内容非法（应为 dict）：{cfg_path}")

    required = ["experiment", "data", "train", "test", "paths", "dataset"]
    for key in required:
        if key not in cfg:
            raise KeyError(f"配置缺少字段: {key}")
    if "model" not in cfg and "new_model" not in cfg:
        raise KeyError("配置缺少字段: model 或 new_model（至少需要一个）")
    return cfg


def build_device() -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"


def resolve_base_dir() -> Path:
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

    lm = Path(paths_cfg["label_mapping_json"])
    if lm.is_absolute():
        json_path = lm
    else:
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
            print(
                f"[SKIP] 文件 {xlsx_path.name}: 时间点 {t}h 的阻抗点数 {impedance_np.shape[0]} < 要求的 {num_freq_points}"
            )
            return False
    print(f"[OK] 可作为测试样本: {xlsx_path.name}")
    return True


_NO_OVERLAP_REGEX = re.compile(r"^(?P<prefix>.*?_\[)(?P<times>\d+(?:\s*,\s*\d+)*)\]\s*(?:\.\w+)?$")


def _parse_no_overlap_key(fname: str):
    match = _NO_OVERLAP_REGEX.match(fname.strip())
    if not match:
        return None
    times_str = match.group("times")
    try:
        window = tuple(sorted(int(t.strip()) for t in times_str.split(",")))
    except ValueError:
        return None
    return match.group("prefix"), window


def _prepare_no_overlap_split(paths: dict, num_time_points: int, num_freq_points: int, seed: int) -> bool:
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

    parsed = []
    for p in source_dir.glob("*.xlsx"):
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

    prefix_to_active_times = {}
    for _, prefix, window in test_selected:
        active = set(sorted(window)[:num_time_points])
        prefix_to_active_times.setdefault(prefix, set()).update(active)

    def excluded_from_train(item):
        _, prefix, window = item
        if (prefix, window) in test_keys:
            return True
        if prefix not in prefix_to_active_times:
            return False
        return bool(set(window) & prefix_to_active_times[prefix])

    train_selected = [x for x in parsed if not excluded_from_train(x)]
    train_dir = base_dir / "datasets" / "datasets_for_all_train_no_overlap" / exp_name
    test_dir = base_dir / "datasets" / "datasets_for_all_test_no_overlap" / exp_name
    for folder in (train_dir, test_dir):
        if folder.exists():
            shutil.rmtree(folder)
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for p, _, _ in train_selected:
        shutil.copy2(str(p), str(train_dir / p.name))
    for p, _, _ in test_selected:
        shutil.copy2(str(p), str(test_dir / p.name))

    n_ion_in_test = sum(1 for x in test_selected if "ion_column" in x[0].name)
    print(
        f"✅ no_overlap 划分: 测试集 {len(test_selected)} 个（ion_column {n_ion_in_test} 个）-> {test_dir}，"
        f"训练集 {len(train_selected)} 个 -> {train_dir}（已排除同前缀且窗口含测试前 {num_time_points} 个时间点的样本）"
    )
    paths["data_folder"] = train_dir
    paths["test_folder_path"] = test_dir
    return True


def prepare_test_folder(paths: dict, label_mapping: dict, num_time_points: int, num_freq_points: int, seed: int):
    data_folder: Path = paths["data_folder"]
    base_test_folder: Path = paths["test_folder_path"]
    exp_name: str = paths["exp_name"]

    if _prepare_no_overlap_split(paths, num_time_points, num_freq_points, seed):
        return

    if base_test_folder.name != "datasets_for_all_test":
        base_test_folder.mkdir(parents=True, exist_ok=True)
        existing_xlsx = list(base_test_folder.glob("*.xlsx"))
        if existing_xlsx:
            print(f"📂 使用用户指定的测试目录：{base_test_folder}（发现 {len(existing_xlsx)} 个 .xlsx 文件）")
        else:
            print(f"⚠️ 指定的测试目录 {base_test_folder} 中没有任何 .xlsx 文件，后续测试/验证将没有样本可用。")
        paths["test_folder_path"] = base_test_folder
        return

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
        selected = files_sorted[:3]
        print(f"🧪 类别 [{label}] 选中 {len(selected)} 个样本作为测试集。")
        for f in selected:
            dest = current_test_folder / f.name
            print(f"  - 复制 {f} -> {dest}")
            shutil.copy2(str(f), str(dest))
            total_moved += 1

    print(f"✅ 测试集划分完成，共复制 {total_moved} 个样本到 {current_test_folder}")
    paths["test_folder_path"] = current_test_folder


def _resolve_test_model_path(paths: dict) -> Path:
    return paths["test_model_path"]
