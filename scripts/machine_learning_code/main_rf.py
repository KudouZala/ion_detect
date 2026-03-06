# -*- coding: utf-8 -*-
"""
随机森林基线：使用 model_datasets.Dataset_2_Stable_plus 的数据（与 main.py 相同的数据与划分），
从 EIS + 电压中构造工程特征后训练随机森林，输出准确率、精确率、召回率、F1。
"""
import argparse
import numpy as np
from pathlib import Path

from main import load_config, build_paths, prepare_test_folder, load_label_mapping
from model_datasets import Dataset_2_Stable_plus
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def sample_to_feature(volt, impe, env):
    """将 Dataset_2_Stable_plus 返回的单个样本转为一条特征向量（用于 RF）。"""
    v = volt.numpy().ravel()
    i = impe.numpy().ravel()
    e = env.numpy().ravel()
    return np.concatenate([v, i, e]).astype(np.float32)


def build_xy(dataset):
    """遍历 dataset，收集 X, y。"""
    X_list, y_list = [], []
    for idx in range(len(dataset)):
        out = dataset[idx]
        volt, impe, env_param = out[0], out[1], out[2]
        label = out[3]
        x = sample_to_feature(volt, impe, env_param)
        X_list.append(x)
        y_list.append(label.item())
    return np.stack(X_list), np.array(y_list, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="随机森林基线（使用 model_datasets 数据）")
    parser.add_argument("--config", type=str, required=True, help="与 main.py 相同的 YAML 配置路径")
    parser.add_argument("--n_estimators", type=int, default=100, help="树的数量")
    parser.add_argument("--max_depth", type=int, default=None, help="最大深度")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    paths = build_paths(cfg)
    label_mapping = load_label_mapping(paths["json_path"])
    d = cfg["data"]
    num_t, num_f = int(d["num_time_points"]), int(d.get("num_freq_points", 63))
    seed = int(cfg["experiment"].get("seed", 42))
    prepare_test_folder(paths, label_mapping, num_t, num_f, seed)

    test_fnames = {p.name for p in Path(paths["test_folder_path"]).glob("*.xlsx")}
    stats_file = str(paths["stats_file"])

    train_ds = Dataset_2_Stable_plus(
        data_folder=paths["data_folder"],
        stats_file=stats_file,
        save_stats=True,
        num_time_points=num_t,
        exclude_fnames=test_fnames,
    )
    val_ds = Dataset_2_Stable_plus(
        data_folder=paths["test_folder_path"],
        stats_file=stats_file,
        save_stats=False,
        num_time_points=num_t,
    )

    X_train, y_train = build_xy(train_ds)
    X_val, y_val = build_xy(val_ds)
    num_classes = len(label_mapping) if label_mapping else 7

    if len(X_train) == 0 or len(X_val) == 0:
        print("训练集或测试集为空，退出")
        return

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, labels=range(num_classes), average="macro", zero_division=0
    )
    print("\n--- Random Forest 基线 (EIS+电压 工程特征) ---")
    print(f"  准确率 (Accuracy):  {acc:.4f}")
    print(f"  精确率 (Precision): {prec:.4f}")
    print(f"  召回率 (Recall):    {rec:.4f}")
    print(f"  F1 分数 (F1):       {f1:.4f}")


if __name__ == "__main__":
    main()
