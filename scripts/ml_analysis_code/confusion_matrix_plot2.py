import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# 1) 数据：按你给的表
# -------------------------
labels = ["Ca2+", "Na+", "Ni2+", "Cr3+", "Fe3+", "Cu2+"]

cm = np.array([
    [0.67, 0.33, 0.00, 0.00, 0.00, 0.00],  # Truth Ca2+
    [0.00, 1.00, 0.00, 0.00, 0.00, 0.00],  # Truth Na+
    [0.00, 0.67, 0.33, 0.00, 0.00, 0.00],  # Truth Ni2+
    [0.00, 0.00, 0.00, 0.67, 0.33, 0.00],  # Truth Cr3+
    [0.00, 0.00, 0.00, 0.00, 0.67, 0.33],  # Truth Fe3+
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.67],  # Truth Cu2+
], dtype=float)

df = pd.DataFrame(cm, index=labels, columns=labels)

# -------------------------
# 2) 画图风格（论文友好）
# -------------------------
sns.set_theme(style="white", context="paper")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

def plot_confusion_matrix(df, title, out_prefix, annotate=True):
    fig, ax = plt.subplots(figsize=(5.6, 4.8), constrained_layout=True)

    # heatmap
    sns.heatmap(
        df,
        ax=ax,
        annot=annotate,
        fmt=".2f",
        square=True,
        linewidths=0.8,
        linecolor="white",
        vmin=0.0,
        vmax=1.0,
        cbar=True,
        cbar_kws={"label": "Proportion"},
    )

    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)


    # 让 x 轴标签更适合论文（不挤）
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # 导出：PDF（矢量）+ PNG（600 dpi）
    fig.savefig(f"{out_prefix}.pdf")
    fig.savefig(f"{out_prefix}.png", dpi=600)
    return fig, ax

# -------------------------
# 3) 生成图
# -------------------------
plot_confusion_matrix(df, "Normalized Confusion Matrix", "confusion_matrix_norm", annotate=True)

# 如果你想要一个“不带数字”的版本（有些期刊版式更喜欢）
plot_confusion_matrix(df, "Normalized Confusion Matrix", "confusion_matrix_norm_no_text", annotate=False)

plt.show()
