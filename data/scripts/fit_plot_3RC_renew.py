import pandas as pd
import matplotlib.pyplot as plt
import re
from io import StringIO

# -----------------------------
# 1) 你的数据（直接粘贴）
# -----------------------------
data_str = """Ion\tR0_2minus1\tR_LF_2minus1\tR_MF_2minus1\tR_HF_2minus1\tR0_3minus1\tR_LF_3minus1\tR_MF_3minus1\tR_HF_3minus1
Na⁺\t0.008020296\t0.075070244\t0.275186383\t0.244636302\t0.00069417\t-0.002982378\t-0.000112274\t-0.000407222
Ca²⁺\t0.006092609\t0.126954279\t0.279879385\t0.096754256\t0.000483351\t0.001033106\t-0.001708355\t0.004184478
Cr³⁺\t0.003978433\t0.017678727\t0.029296702\t0.041731213\t0.001317531\t0.004218989\t-0.001824205\t0.00884688
Ni²⁺\t0.00687388\t0.080898746\t0.139794694\t0.063039526\t0.004121861\t0.006613781\t-0.002135262\t0.045028036
Cu²⁺\t0.002400738\t0.007751463\t0.018019338\t-0.002973565\t0.000970328\t-0.00254414\t-0.003852922\t-0.001405932
Fe³⁺\t0.002373692\t0.01436017\t0.019444374\t0.017958874\t-0.000464556\t-0.025031026\t-0.00462111\t-0.00351341
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

def norm_ion_ascii(s: str) -> str:
    """把 Na⁺/Ca²⁺ 这类 unicode 上标，规整成 Na+/Ca2+ 这类 ASCII。"""
    if s is None:
        return ""
    s = str(s).strip()
    # unicode 上标到普通字符
    sup_map = {
        "⁰":"0","¹":"1","²":"2","³":"3","⁴":"4","⁵":"5","⁶":"6","⁷":"7","⁸":"8","⁹":"9",
        "⁺":"+","⁻":"-",
    }
    for k, v in sup_map.items():
        s = s.replace(k, v)
    # 清理全角符号
    s = s.replace("＋", "+").replace("－", "-").replace(" ", "")
    return s

ion_colors = {norm_ion_ascii(k): v for k, v in ion_colors_raw.items()}

def extract_elem(ion_label: str) -> str:
    m = re.match(r'^([A-Z][a-z]?)', str(ion_label))
    return m.group(1) if m else str(ion_label)

def ion_to_mathtext(ion_ascii: str) -> str:
    """Na+ / Ca2+ / Cr3+ / Cu2+ 等 -> Na^{+} / Ca^{2+}（显示为上标）"""
    s = norm_ion_ascii(ion_ascii)

    m = re.match(r'^([A-Za-z]+)(\d+)\+$', s)
    if m:
        elem, n = m.group(1), m.group(2)
        return rf"{elem}$^{{{n}+}}$"

    m = re.match(r'^([A-Za-z]+)\+$', s)
    if m:
        elem = m.group(1)
        return rf"{elem}$^{{+}}$"

    m = re.match(r'^([A-Za-z]+)(\d+)\-$', s)
    if m:
        elem, n = m.group(1), m.group(2)
        return rf"{elem}$^{{{n}-}}$"

    m = re.match(r'^([A-Za-z]+)\-$', s)
    if m:
        elem = m.group(1)
        return rf"{elem}$^{{-}}$"

    return s

# -----------------------------
# 3) 绘图：4个subplot，每个离子两根柱（深色污染/浅色恢复）
# -----------------------------
def plot_contam_vs_recover_bars(
    df: pd.DataFrame,
    metrics=("R0", "R_LF", "R_MF", "R_HF"),
    contam_suffix="2minus1",   # 深色：污染后
    recover_suffix="3minus1",  # 浅色：恢复后
    figsize=(20, 5.2),

    # 标题（0/LF/MF/HF 都作为下标）
    title_map=None,

    # 字体可调
    title_fontsize=22,
    xlabel_fontsize=18,
    ylabel_fontsize=20,
    xtick_fontsize=24,   # <<< 离子字符大小就在这里调
    ytick_fontsize=20,
    legend_fontsize=18,

    xtick_rotation=0,

    # tick 外观
    tick_length=8,
    tick_width=2.0,

    # 柱子样式
    group_width=0.75,        # 每个离子“组”的总宽度
    recover_alpha=0.45,      # 恢复后的浅色程度
    edgecolor=None,
    linewidth=0.0,

    # 轴风格
    show_grid=False,
    zero_line=True,
    sharey=False,
):
    d = df.copy()

    # 规整 Ion 字段（兼容 Na⁺/Ca²⁺）
    d["ion_raw"] = d["Ion"]
    d["ion"] = d["Ion"].map(norm_ion_ascii)

    # 排序：按你指定的元素顺序
    d["elem"] = d["ion"].apply(extract_elem)
    d["rank"] = d["elem"].map(lambda x: ION_RANK.get(x, 10**9))
    d = d.sort_values(["rank", "ion"]).reset_index(drop=True)

    # x 轴
    ions = d["ion"].tolist()
    x = list(range(len(ions)))
    xticklabels = [ion_to_mathtext(v) for v in ions]

    # 默认 title_map
    if title_map is None:
        title_map = {
            "R0":   r"$\Delta R_{0}$",
            "R_LF": r"$\Delta R_{LF}$",
            "R_MF": r"$\Delta R_{MF}$",
            "R_HF": r"$\Delta R_{HF}$",
        }

    # 两根柱的偏移
    bar_w = group_width / 2.0
    off = bar_w / 2.0

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, sharex=True, sharey=sharey)
    if len(metrics) == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics):
        col_contam = f"{m}_{contam_suffix}"
        col_recover = f"{m}_{recover_suffix}"

        y_contam = d[col_contam].astype(float).values
        y_recover = d[col_recover].astype(float).values

        # 颜色：污染=原色，恢复=同色+alpha
        base_colors = [ion_colors.get(norm_ion_ascii(v), "black") for v in ions]

        # 污染后（深色）
        ax.bar(
            [xi - off for xi in x],
            y_contam,
            width=bar_w,
            color=base_colors,
            edgecolor=edgecolor,
            linewidth=linewidth,
            label="After contamination" if m == metrics[0] else None,
        )

        # 恢复后（浅色，同色 alpha）
        ax.bar(
            [xi + off for xi in x],
            y_recover,
            width=bar_w,
            color=base_colors,
            alpha=recover_alpha,
            edgecolor=edgecolor,
            linewidth=linewidth,
            label="After recovery" if m == metrics[0] else None,
        )

        ax.set_title(title_map.get(m, m), fontsize=title_fontsize)
        ax.set_xlabel(" ", fontsize=xlabel_fontsize)
        ax.set_ylabel("Value (Ohm)" if m == metrics[0] else " ", fontsize=ylabel_fontsize)

        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=xtick_rotation)
        ax.tick_params(axis="x", labelsize=xtick_fontsize, length=tick_length, width=tick_width)
        ax.tick_params(axis="y", labelsize=ytick_fontsize, length=tick_length, width=tick_width)

        if zero_line:
            ax.axhline(0.0, linewidth=1.0)

        ax.grid(show_grid)

        # 只保留左/下轴
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")

    # 图例（只做一次）
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=2,
        fontsize=legend_fontsize,
        frameon=True,
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)  # 给底部图例留空间
    return fig, axes

# -----------------------------
# 4) 直接画图（可调 fontsize）
# -----------------------------
fig, axes = plot_contam_vs_recover_bars(
    df,
    metrics=("R0", "R_LF", "R_MF", "R_HF"),
    title_fontsize=24,
    ylabel_fontsize=22,
    xtick_fontsize=26,   # <<< 这里调离子字符大小（x轴字体）
    ytick_fontsize=20,
    legend_fontsize=20,
    tick_length=8,
    tick_width=2.0,
    recover_alpha=0.45,  # 恢复后的浅色程度
    show_grid=False,
    zero_line=True,
    sharey=False,
)

plt.show()
fig.savefig("contam_vs_recover_4metrics.png", dpi=300, bbox_inches="tight", pad_inches=0.10)
