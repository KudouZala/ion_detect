import pandas as pd
import matplotlib.pyplot as plt
import re
from io import StringIO

# -----------------------------
# 1) 你的数据（直接粘贴即可）
# -----------------------------
data_str = """ion\tR0\tR_LF\tR_HF
Ca2+\t0.009793461\t0.403268836\t0.142370283
Na+\t0.007963878\t0.136002949\t0.279189513
Ni2+\t0.00331882\t0.154729536\t0.082315453
Cr3+\t0.006893279\t0.053855654\t0.039134921
Cu2+\t0.00200151\t0.025816238\t-0.001799881
Fe3+\t0.002420376\t0.026122025\t0.006759337
"""
df = pd.read_csv(StringIO(data_str), sep=r"\s+")

# -----------------------------
# 2) 顺序与颜色（与你“刚才”的一致）
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
# 3) 绘图函数：3列 -> 3个 subplot
# -----------------------------
def plot_three_metrics_bars(
    df: pd.DataFrame,
    metrics=("R0", "R_LF", "R_HF"),
    figsize=(14, 4.8),
    title_map=None,
    xlabel=" ",
    ylabel_map=None,
    suptitle=None,
    bar_width=0.75,

    # 字体与刻度可调
    title_fontsize=20,
    suptitle_fontsize=22,
    label_fontsize=18,

    xtick_fontsize=20,              # <<< 关键：离子字符大小（x轴tick label）
    ytick_fontsize=20,
    xtick_rotation=0,               # 0 / 30 / 45 等

    tick_length=7,
    tick_width=1.8,

    # 轴风格
    show_grid=False,
    zero_line=True,
    sharey=False,
):
    if title_map is None:
        title_map = {m: m for m in metrics}
    if ylabel_map is None:
        ylabel_map = {m: m for m in metrics}

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
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
        ax.set_ylabel(ylabel_map.get(metric, metric), fontsize=label_fontsize)

        # x ticks（离子字符）
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=xtick_rotation)
        ax.tick_params(axis="x", labelsize=xtick_fontsize, length=tick_length, width=tick_width)

        # y ticks
        ax.tick_params(axis="y", labelsize=ytick_fontsize, length=tick_length, width=tick_width)

        # 0线（尤其 R_HF 可能有负值）
        if zero_line:
            ax.axhline(0.0, linewidth=1.0)

        # 网格
        ax.grid(show_grid)

        # 只保留左/下轴
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")

    if suptitle:
        fig.suptitle(suptitle, fontsize=suptitle_fontsize)

    fig.tight_layout()
    return fig, axes

# -----------------------------
# 4) 直接画图（你可以改fontsize等参数）
# -----------------------------
fig, axes = plot_three_metrics_bars(
    df,
    metrics=("R0", "R_LF", "R_HF"),

    # 关键：0 / LF / HF 作为下标
    title_map={
        "R0":   r"$\Delta R_{0}$",
        "R_LF": r"$\Delta R_{LF}$",
        "R_HF": r"$\Delta R_{HF}$",
    },

    ylabel_map={"R0": "Value (Ohm)", "R_LF": " ", "R_HF": " "},

    title_fontsize=22,
    label_fontsize=22,

    # 你要调的就在这里：
    xtick_fontsize=20,      # <<< x轴离子字符大小
    ytick_fontsize=20,
    xtick_rotation=0,       # 如果挤，可以改 30/45

    tick_length=8,
    tick_width=2.0,

    show_grid=False,
    zero_line=True,
    sharey=False,
)

plt.show()

# 如需保存：
fig.savefig("three_metrics_bars.png", dpi=300, bbox_inches="tight", pad_inches=0.10)
