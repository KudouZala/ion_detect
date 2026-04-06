# -*- coding: utf-8 -*-
"""
LSTM 基线：使用 model_datasets.Dataset_2_Stable_plus 的数据（与 main.py 相同的数据与划分），
将电压+阻抗按时序输入 LSTM 编码，再经分类头得到类别。编码后的分类方式可在下方 LSTM 模型中修改。

训练集/测试集：由 --config 指定的 YAML 中 paths.data_folder 与 paths.test_folder 决定（与 main.py 一致）。
当 test_folder 为 datasets_for_all_test 时：从 data_folder 中按规则选出若干样本【复制】到
datasets_for_all_test/<experiment_name>/ 作为测试集；原 data_folder 中的文件【不删除】；
训练时通过 exclude_fnames 排除这批文件名，使这些样本仅用于测试、不参与训练。
"""
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from runtime_core import load_config, build_paths, prepare_test_folder, load_label_mapping
from model_datasets import Dataset_2_Stable_plus
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def sample_to_sequence(volt, impe):
    """将 (volt, impe) 转为 (T, D) 序列：每时间步 [volt_t; impe_t 展平]。"""
    T = volt.shape[0]
    v = volt.numpy()
    i = impe.numpy().reshape(T, -1)
    return np.concatenate([v, i], axis=1).astype(np.float32)


class LSTMForClassification(nn.Module):
    """
    LSTM 编码 + 分类头。
    编码后的表示如何用于分类可在此修改（例如改为 attention pooling、多加几层 MLP 等）。
    """
    def __init__(self, input_dim, hidden_size=64, num_layers=1, num_classes=7, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_h = h_n[-1]
        return self.fc(self.drop(last_h))


class SeqDataset(torch.utils.data.Dataset):
    """包装 Dataset_2_Stable_plus，__getitem__ 返回 (sequence, label)。"""
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        out = self.base[idx]
        volt, impe = out[0], out[1]
        label = out[3]
        seq = sample_to_sequence(volt, impe)
        return torch.from_numpy(seq), label


def main():
    parser = argparse.ArgumentParser(description="LSTM 基线（使用 model_datasets 数据）")
    parser.add_argument("--config", type=str, required=True, help="与 main.py 相同的 YAML 配置路径（训练/测试集由其中 paths 指定）")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--eval_every", type=int, default=100, help="每 N 个 epoch 在测试集上评估并打印一次准确率/精确率/召回率/F1")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--hidden", type=int, default=64, help="LSTM hidden size")
    parser.add_argument("--layers", type=int, default=1, help="LSTM 层数")
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

    seq_train = SeqDataset(train_ds)
    seq_val = SeqDataset(val_ds)
    num_classes = len(label_mapping) if label_mapping else 7
    input_dim = 1 + num_f * 2

    if len(seq_train) == 0 or len(seq_val) == 0:
        print("训练集或测试集为空，退出")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(seq_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(seq_val, batch_size=args.batch_size, shuffle=False)

    model = LSTMForClassification(
        input_dim=input_dim,
        hidden_size=args.hidden,
        num_layers=args.layers,
        num_classes=num_classes,
        dropout=0.2,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    eval_every = max(1, int(args.eval_every))

    for ep in range(args.epochs):
        model.train()
        for seq, lab in train_loader:
            seq, lab = seq.to(device), lab.to(device)
            opt.zero_grad()
            logits = model(seq)
            loss = criterion(logits, lab)
            loss.backward()
            opt.step()

        if (ep + 1) % eval_every == 0:
            model.eval()
            all_p, all_t = [], []
            with torch.no_grad():
                for seq, lab in val_loader:
                    logits = model(seq.to(device))
                    pred = logits.argmax(dim=1).cpu().numpy()
                    all_p.extend(pred)
                    all_t.extend(lab.numpy().tolist())
            y_p = np.array(all_p)
            y_t = np.array(all_t)
            acc = accuracy_score(y_t, y_p)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_t, y_p, labels=range(num_classes), average="macro", zero_division=0
            )
            print(f"\n--- LSTM 基线 (EIS+电压 时序) [epoch {ep + 1}] ---")
            print(f"  准确率 (Accuracy):  {acc:.4f}")
            print(f"  精确率 (Precision): {prec:.4f}")
            print(f"  召回率 (Recall):    {rec:.4f}")
            print(f"  F1 分数 (F1):       {f1:.4f}")

    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for seq, lab in val_loader:
            logits = model(seq.to(device))
            pred = logits.argmax(dim=1).cpu().numpy()
            all_pred.extend(pred)
            all_true.extend(lab.numpy().tolist())
    y_pred = np.array(all_pred)
    y_val = np.array(all_true)

    idx_to_name = {v: k for k, v in label_mapping.items()} if label_mapping else {}
    if not idx_to_name:
        idx_to_name = {i: str(i) for i in range(num_classes)}
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
    print("\n--- LSTM 基线 (EIS+电压 时序) ---")
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
