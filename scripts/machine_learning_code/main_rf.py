# -*- coding: utf-8 -*-
"""
随机森林基线：使用 model_datasets.Dataset_2_Stable_plus 的数据（与 main.py 相同的数据与划分），
从 EIS + 电压中构造工程特征后训练随机森林，输出准确率、精确率、召回率、F1。

训练集/测试集：由 --config 指定的 YAML 中 paths.data_folder 与 paths.test_folder 决定（与 main.py 一致）。
当 test_folder 为 datasets_for_all_test 时：从 data_folder 中按规则选出若干样本【复制】到
datasets_for_all_test/<experiment_name>/ 作为测试集；原 data_folder 中的文件【不删除】；
训练时通过 exclude_fnames 排除这批文件名，使这些样本仅用于测试、不参与训练。
"""
import argparse
import csv
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
    parser.add_argument("--config", type=str, required=True, help="与 main.py 相同的 YAML 配置路径（训练/测试集由其中 paths 指定）")
    parser.add_argument("--n_estimators", type=int, default=100, help="树的数量")
    parser.add_argument("--max_depth", type=int, default=None, help="最大深度")
    parser.add_argument("--out_csv", type=str, default="", help="可选：详细测试结果保存的 CSV 路径（列：样本名, 真实, 预测, 正确）")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    paths = build_paths(cfg)
    label_mapping = load_label_mapping(paths["json_path"])
    d = cfg["data"]
    num_t, num_f = int(d["num_time_points"]), int(d.get("num_freq_points", 63))
    seed = int(cfg["experiment"].get("seed", 42))
    # 与 main.py 一致：test_folder=datasets_for_all_test 时，将选中样本【复制】到
    # datasets_for_all_test/<exp_name>/，不删除 data_folder 中的原文件；训练时用 exclude_fnames 排除这些文件名。
    prepare_test_folder(paths, label_mapping, num_t, num_f, seed)

    test_fnames = {p.name for p in Path(paths["test_folder_path"]).glob("*.xlsx")}
    stats_file = str(paths["stats_file"])

    train_ds = Dataset_2_Stable_plus(
        data_folder=paths["data_folder"],  # 训练集目录（YAML paths.data_folder）
        stats_file=stats_file,
        save_stats=True,
        num_time_points=num_t,
        exclude_fnames=test_fnames,  # 排除测试集文件，避免泄漏
    )
    val_ds = Dataset_2_Stable_plus(
        data_folder=paths["test_folder_path"],  # 测试集目录（YAML paths.test_folder 解析结果）
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

    # 标签索引 -> 名称（用于详细结果）
    idx_to_name = {v: k for k, v in label_mapping.items()} if label_mapping else {}
    if not idx_to_name:
        idx_to_name = {i: str(i) for i in range(num_classes)}

    # 详细测试结果：样本名, 真实, 预测, 是否正确
    sample_names = getattr(val_ds, "file_names", [f"sample_{i}" for i in range(len(y_val))])
    print("\n--- 详细测试结果 (样本名, 真实, 预测, 正确) ---")
    rows = []
    for i in range(len(y_val)):
        name = sample_names[i] if i < len(sample_names) else f"sample_{i}"
        true_name = idx_to_name.get(int(y_val[i]), str(y_val[i]))
        pred_name = idx_to_name.get(int(y_pred[i]), str(y_pred[i]))
        correct = "是" if y_val[i] == y_pred[i] else "否"
        rows.append((name, true_name, pred_name, correct))
        print(f"  {name}\t{true_name}\t{pred_name}\t{correct}")

    acc = accuracy_score(y_val, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, labels=range(num_classes), average="macro", zero_division=0
    )
    print("\n--- Random Forest 基线 (EIS+电压 工程特征) ---")
    print(f"  准确率 (Accuracy):  {acc:.4f}")
    print(f"  精确率 (Precision): {prec:.4f}")
    print(f"  召回率 (Recall):    {rec:.4f}")
    print(f"  F1 分数 (F1):       {f1:.4f}")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["样本名", "真实", "预测", "正确"])
            w.writerows(rows)
        print(f"\n  详细结果已保存: {out_path}")


if __name__ == "__main__":
    main()
