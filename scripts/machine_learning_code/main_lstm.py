# -*- coding: utf-8 -*-
"""
LSTM 基线：使用 model_datasets.Dataset_2_Stable_plus 的数据（与 main.py 相同的数据与划分），
将电压+阻抗按时序输入 LSTM 编码，再经分类头得到类别。编码后的分类方式可在下方 LSTM 模型中修改。
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from main import load_config, build_paths, prepare_test_folder, load_label_mapping
from model_datasets import Dataset_2_Stable_plus
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def sample_to_sequence(volt, impe):
    """将 (volt, impe) 转为 (T, D) 序列：每时间步 [volt_t; impe_t 展平]。"""
    # volt: (T, 1), impe: (T, 63, 2) -> 每步 1+63*2
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
        # 当前用“最后时间步的隐状态”接线性层分类；你可改为 attention / 多层 MLP 等
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, T, D)
        _, (h_n, _) = self.lstm(x)
        last_h = h_n[-1]  # (B, hidden_size)
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
    parser.add_argument("--config", type=str, required=True, help="与 main.py 相同的 YAML 配置路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--hidden", type=int, default=64, help="LSTM hidden size")
    parser.add_argument("--layers", type=int, default=1, help="LSTM 层数")
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

    seq_train = SeqDataset(train_ds)
    seq_val = SeqDataset(val_ds)
    num_classes = len(label_mapping) if label_mapping else 7
    # 输入维度：每时间步 1 + 63*2
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

    for ep in range(args.epochs):
        model.train()
        for seq, lab in train_loader:
            seq, lab = seq.to(device), lab.to(device)
            opt.zero_grad()
            logits = model(seq)
            loss = criterion(logits, lab)
            loss.backward()
            opt.step()

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

    acc = accuracy_score(y_val, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, labels=range(num_classes), average="macro", zero_division=0
    )
    print("\n--- LSTM 基线 (EIS+电压 时序) ---")
    print(f"  准确率 (Accuracy):  {acc:.4f}")
    print(f"  精确率 (Precision): {prec:.4f}")
    print(f"  召回率 (Recall):    {rec:.4f}")
    print(f"  F1 分数 (F1):       {f1:.4f}")
    print("  （编码后的分类方式可在 LSTMForClassification 中修改）")


if __name__ == "__main__":
    main()
