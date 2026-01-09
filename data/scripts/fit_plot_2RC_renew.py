import pandas as pd
import matplotlib.pyplot as plt
import re
from io import StringIO

# -----------------------------
# 1) 你的数据（3列版本：R0 / R_LF / R_HF）
# -----------------------------
data_str = """Ion\tR0_2minus1\tR_LF_2minus1\tR_HF_2minus1\tR0_3minus1\tR_LF_3minus1\tR_HF_3minus1
Na⁺\t0.007951509\t0.135511569\t0.279128648\t0.000665123\t-0.002114292\t-0.000482939
Ca²⁺\t0.009793461\t0.403268836\t0.142370283\t0.00047858\t-0.000964973\t0.000688412
Cr³⁺\t0.006893279\t0.053855654\t0.039134921\t0.00136745\t0.003220283\t-0.00015628
Ni²⁺\t0.00331882\t0.154729536\t0.082315453\t0.005080751\t0.010587894\t0.068773338
Cu²⁺\t0.00200151\t0.025816238\t-0.001799881\t0.001013996\t-0.002889218\t0.001489736
Fe³⁺\t0.002420376\t0.026122025\t0.006759337\t-0.000303869\t0.00203768\t-0.000755933
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
    """Na⁺/Ca²⁺/Fe³⁺ -> Na+/Ca2+/Fe3+"""
    if s is None:
        return ""
    s = str(s).strip()
    sup_map = {
        "⁰":"0","¹":"1","²":"2","³":"3","⁴":"4","⁵":"5","⁶":"6","⁷":"7","⁸":"8","⁹":"9",
        "⁺":"+","⁻":"-",
    }
    for k, v in sup_map.items():
        s = s.replace(k, v)
    s = s.replace("＋", "+").replace("－", "-").replace(" ", "")
    return s

ion_colors = {norm_ion_ascii(k): v for k, v in ion_colors_raw.items()}

def extract_elem(ion_label: str) -> str:
    m = re.match(r'^([A-Z][a-z]?)', str(ion_label))
    return m.group(1) if m else str(ion_label)

def ion_to_mathtext(ion_ascii: str) -> str:
    """Na+ / Ca2+ / Cr3+ -> Na^{+} / Ca^{2+} / Cr^{3+}"""
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
# 3) 绘图：3个subplot，两根柱（污染深色 / 恢复浅色）
# -----------------------------
def plot_contam_vs_recover_bars(
    df: pd.DataFrame,
    metrics=("R0", "R_LF", "R_HF"),
    contam_suffix="2minus1",
    recover_suffix="3minus1",
    figsize=(16, 5.0),
    title_map=None,

    title_fontsize=22,
    xlabel_fontsize=18,
    ylabel_fontsize=20,
    xtick_fontsize=24,   # <<< 离子字符大小在这里调
    ytick_fontsize=20,
    legend_fontsize=18,

    tick_length=8,
    tick_width=2.0,

    group_width=0.75,
    recover_alpha=0.45,
    show_grid=False,
    zero_line=True,
    sharey=False,
):
    d = df.copy()
    d["ion"] = d["Ion"].map(norm_ion_ascii)

    d["elem"] = d["ion"].apply(extract_elem)
    d["rank"] = d["elem"].map(lambda x: ION_RANK.get(x, 10**9))
    d = d.sort_values(["rank", "ion"]).reset_index(drop=True)

    ions = d["ion"].tolist()
    x = list(range(len(ions)))
    xticklabels = [ion_to_mathtext(v) for v in ions]
    base_colors = [ion_colors.get(norm_ion_ascii(v), "black") for v in ions]

    if title_map is None:
        title_map = {
            "R0":   r"$\Delta R_{0}$",
            "R_LF": r"$\Delta R_{LF}$",
            "R_HF": r"$\Delta R_{HF}$",
        }

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

        ax.bar(
            [xi - off for xi in x],
            y_contam,
            width=bar_w,
            color=base_colors,
            label="After contamination" if m == metrics[0] else None,
        )
        ax.bar(
            [xi + off for xi in x],
            y_recover,
            width=bar_w,
            color=base_colors,
            alpha=recover_alpha,
            label="After recovery" if m == metrics[0] else None,
        )

        ax.set_title(title_map.get(m, m), fontsize=title_fontsize)
        ax.set_xlabel(" ", fontsize=xlabel_fontsize)
        ax.set_ylabel("Value (Ohm)" if m == metrics[0] else " ", fontsize=ylabel_fontsize)

        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
        ax.tick_params(axis="x", labelsize=xtick_fontsize, length=tick_length, width=tick_width)
        ax.tick_params(axis="y", labelsize=ytick_fontsize, length=tick_length, width=tick_width)

        if zero_line:
            ax.axhline(0.0, linewidth=1.0)

        ax.grid(show_grid)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")

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
    fig.subplots_adjust(bottom=0.22)
    return fig, axes

# -----------------------------
# 4) 画图（这里调字体即可）
# -----------------------------
fig, axes = plot_contam_vs_recover_bars(
    df,
    metrics=("R0", "R_LF", "R_HF"),
    title_fontsize=24,
    ylabel_fontsize=22,
    xtick_fontsize=26,   # <<< 调离子字符大小
    ytick_fontsize=20,
    legend_fontsize=20,
    tick_length=8,
    tick_width=2.0,
    recover_alpha=0.45,
    show_grid=False,
    zero_line=True,
    sharey=False,
)

plt.show()
fig.savefig("contam_vs_recover_3metrics.png", dpi=300, bbox_inches="tight", pad_inches=0.10)
