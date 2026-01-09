import pandas as pd
import matplotlib.pyplot as plt
import re
from io import StringIO

# -----------------------------
# 1) 数据（4列：R0, R_LF, R_MF, R_HF）
# -----------------------------
data_str = """ion\tR0\tR_LF\tR_MF\tR_HF
Ca2+\t0.006092609\t0.126954279\t0.279879385\t0.096754256
Na+\t0.007986564\t0.0740825\t0.253155858\t0.233171628
Ni2+\t0.00687388\t0.080898746\t0.139794694\t0.063039526
Cr3+\t0.003978433\t0.017678727\t0.029296702\t0.041731213
Cu2+\t0.002400738\t0.007751463\t0.018019338\t-0.002973565
Fe3+\t0.002373692\t0.01436017\t0.019444374\t0.017958874
"""
df = pd.read_csv(StringIO(data_str), sep=r"\s+")

# -----------------------------
# 2) 顺序与颜色（与你之前一致）
# -----------------------------
ION_ORDER = ["Al", "Ca", "Na", "Ni", "Cr", "Fe", "Cu"]
ION_RANK = {k: i for i, k in enumerate(ION_ORDER)}

ion_colors_raw = {
    "Ca2+": "blue",
    "Na+":  "gold",
    "Ni2+": "green",
    "Cr3+": "red",
    "Cu2+": "purple",
    "Fe3+": "brown",
    "Al3+": "gray",
}

def norm_ion(s: str) -> str:
    return str(s).replace(" ", "").replace("＋", "+").replace("－", "-")

ion_colors = {norm_ion(k): v for k, v in ion_colors_raw.items()}

def extract_elem(ion_label: str) -> str:
    m = re.match(r'^([A-Z][a-z]?)', str(ion_label))
    return m.group(1) if m else str(ion_label)

def ion_to_mathtext(ion: str) -> str:
    ion = str(ion).strip()
    m = re.match(r'^([A-Za-z]+)(\d+)\+$', ion)
    if m:
        elem, n = m.group(1), m.group(2)
        return rf"{elem}$^{{{n}+}}$"
    m = re.match(r'^([A-Za-z]+)\+$', ion)
    if m:
        elem = m.group(1)
        return rf"{elem}$^{{+}}$"
    m = re.match(r'^([A-Za-z]+)(\d+)\-$', ion)
    if m:
        elem, n = m.group(1), m.group(2)
        return rf"{elem}$^{{{n}-}}$"
    m = re.match(r'^([A-Za-z]+)\-$', ion)
    if m:
        elem = m.group(1)
        return rf"{elem}$^{{-}}$"
    return ion

# -----------------------------
# 3) 通用绘图函数：N列 -> N个 subplot
# -----------------------------
def plot_metrics_bars(
    df: pd.DataFrame,
    metrics=("R0", "R_LF", "R_MF", "R_HF"),
    figsize=(18, 4.8),
    title_map=None,
    ylabel_map=None,
    suptitle=None,
    bar_width=0.75,

    # 字体与刻度可调
    title_fontsize=22,
    suptitle_fontsize=24,
    label_fontsize=18,
    xtick_fontsize=24,     # 离子字符大小
    ytick_fontsize=20,
    xtick_rotation=0,

    tick_length=8,
    tick_width=2.0,

    # 轴风格
    show_grid=False,
    zero_line=True,
    sharey=False,
):
    if title_map is None:
        title_map = {m: m for m in metrics}
    if ylabel_map is None:
        ylabel_map = {m: "Value (Ohm)" for m in metrics}

    d = df.copy()
    d["ion"] = d["ion"].map(norm_ion)
    d["elem"] = d["ion"].apply(extract_elem)
    d["rank"] = d["elem"].map(lambda x: ION_RANK.get(x, 10**9))
    d = d.sort_values(["rank", "ion"]).reset_index(drop=True)

    x = list(range(len(d)))
    xticklabels = [ion_to_mathtext(v) for v in d["ion"].tolist()]
    colors = [ion_colors.get(norm_ion(v), "black") for v in d["ion"].tolist()]

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, sharex=True, sharey=sharey)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ax.bar(x, d[metric].astype(float).values, width=bar_width, color=colors)

        ax.set_title(title_map.get(metric, metric), fontsize=title_fontsize)
        ax.set_ylabel(ylabel_map.get(metric, "Value (Ohm)"), fontsize=label_fontsize)

        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=xtick_rotation)
        ax.tick_params(axis="x", labelsize=xtick_fontsize, length=tick_length, width=tick_width)
        ax.tick_params(axis="y", labelsize=ytick_fontsize, length=tick_length, width=tick_width)

        if zero_line:
            ax.axhline(0.0, linewidth=1.0)

        ax.grid(show_grid)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")

    if suptitle:
        fig.suptitle(suptitle, fontsize=suptitle_fontsize)

    fig.tight_layout()
    return fig, axes

# -----------------------------
# 4) 画图：4个 subplot
# -----------------------------
fig, axes = plot_metrics_bars(
    df,
    metrics=("R0", "R_LF", "R_MF", "R_HF"),

    # 0 / LF / MF / HF 都作为下标（右下角）
    title_map={
        "R0":   r"$\Delta R_{0}$",
        "R_LF": r"$\Delta R_{LF}$",
        "R_MF": r"$\Delta R_{MF}$",
        "R_HF": r"$\Delta R_{HF}$",
    },

    # 只在第一个子图给 y label，其他留空（你也可以都写成 Value (Ohm)）
    ylabel_map={"R0": "Value (Ohm)", "R_LF": " ", "R_MF": " ", "R_HF": " "},

    title_fontsize=22,
    label_fontsize=22,
    xtick_fontsize=24,   # <<< 调离子字符大小
    ytick_fontsize=20,
    xtick_rotation=0,    # 挤的话改 30/45

    tick_length=8,
    tick_width=2.0,

    show_grid=False,
    zero_line=True,
    sharey=False,
)

plt.show()
fig.savefig("four_metrics_bars.png", dpi=300, bbox_inches="tight", pad_inches=0.10)
