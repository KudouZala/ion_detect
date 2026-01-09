import pandas as pd
import matplotlib.pyplot as plt
import re
from io import StringIO

# -----------------------------
# 1) 你的数据（直接粘贴即可）
# -----------------------------
data_str = """ion\tsigma_mem@delta_polluted\talpha_ca@delta_polluted\talpha_an@delta_polluted\ti_0ca@delta_polluted\ti_0an@delta_polluted\tsigma_mem@delta_recovery\talpha_ca@delta_recovery\talpha_an@delta_recovery\ti_0ca@delta_recovery\ti_0an@delta_recovery
Ca\t-0.039805710315704346\t-0.020426571369171143\t-0.04422160983085632\t-0.01998924370855093\t-0.0030669274274259806\t0.004739642143249512\t0.0004826784133911133\t0.0007726550102233887\t-0.0004661343991756439\t-0.00020449049770832062
Na\t-0.024790391325950623\t-0.017872028052806854\t-0.03200342506170273\t-0.02075899625197053\t-0.0032935210765572265\t0.0010685523351033528\t5.834301312764486e-05\t-0.0002708534399668376\t-0.0004764093707005183\t-0.000246538237358133
Ni\t-0.0455629825592041\t-0.010053416093190512\t-0.02924791971842448\t-0.01954858253399531\t-0.002544463573334118\t0.0020865201950073242\t-0.0002421736717224121\t-2.8967857360839844e-05\t-0.0013584531843662262\t-0.0004670233465731144
Cr\t-0.028003652890523274\t-0.008291582266489664\t-0.016672650973002117\t-0.01135465626915296\t-0.0017520228866487741\t-0.0018187761306762695\t0.00026977062225341797\t-0.000632166862487793\t0.0003060363233089447\t6.61918893456459e-05
Fe\t-0.01065400242805481\t-0.0036892741918563843\t-0.00905601680278778\t-0.005721852649003267\t-0.0009447697666473687\t0.002979576587677002\t3.2335519790649414e-05\t-0.00026303529739379883\t0.0018198639154434204\t0.00011147093027830124
Cu\t-0.010809898376464844\t-0.0025555590788523355\t-0.004448533058166504\t-0.005274854289988677\t-0.0006791352449605862\t0.00030618906021118164\t1.7970800399780273e-05\t0.0013829320669174194\t-0.0019623935222625732\t-0.0001754743279889226
"""
df = pd.read_csv(StringIO(data_str), sep=r"\s+")
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

def extract_elem(ion_label: str) -> str:
    m = re.match(r'^([A-Z][a-z]?)', str(ion_label))
    return m.group(1) if m else str(ion_label)

df["elem"] = df["ion"].apply(extract_elem)
df["rank"] = df["elem"].map(lambda x: ION_RANK.get(x, 10**9))
df = df.sort_values(["rank", "elem"]).reset_index(drop=True)

# elem -> base color（保持与你之前的颜色映射一致）
elem_to_color = {
    "Ca": ion_colors_raw["Ca2+"],
    "Na": ion_colors_raw["Na+"],
    "Ni": ion_colors_raw["Ni2+"],
    "Cr": ion_colors_raw["Cr3+"],
    "Fe": ion_colors_raw["Fe3+"],
    "Cu": ion_colors_raw["Cu2+"],
    "Al": ion_colors_raw["Al3+"],
}
colors = [elem_to_color.get(e, "black") for e in df["elem"].tolist()]

# -----------------------------
# 3) 参数名标题：希腊/罗马 + 下角标（你之前那套）
# -----------------------------
title_map = {
    "sigma_mem": r"$\Delta \sigma_{\mathrm{mem}}$",
    "alpha_ca":  r"$\Delta \alpha_{\mathrm{ca}}$",
    "alpha_an":  r"$\Delta \alpha_{\mathrm{an}}$",
    "i_0ca":     r"$\Delta i_{0,\mathrm{ca}}$",
    "i_0an":     r"$\Delta i_{0,\mathrm{an}}$",
}

# -----------------------------
# 4) x轴离子：补齐右上角电荷数（上标）
# -----------------------------
elem_to_charge = {
    "Ca": "2+",
    "Na": "+",
    "Ni": "2+",
    "Cr": "3+",
    "Fe": "3+",
    "Cu": "2+",
    "Al": "3+",
}

def ion_elem_to_mathtext(elem: str) -> str:
    """
    Ca -> Ca^{2+}, Na -> Na^{+} ...
    """
    c = elem_to_charge.get(elem, "")
    if c == "":
        return elem
    return rf"{elem}$^{{{c}}}$"

xticklabels = [ion_elem_to_mathtext(e) for e in df["elem"].tolist()]

# -----------------------------
# 5) 绘图：5个subplot；每个离子两根柱（污染深色/恢复浅色）
#    新增：整体取相反数（乘以 -1）
# -----------------------------
def plot_5params_contam_vs_recover_flip_sign(
    df: pd.DataFrame,
    params=("sigma_mem","alpha_ca","alpha_an","i_0ca","i_0an"),
    figsize=(22, 5.2),

    # 字体可调
    title_fontsize=22,
    ylabel_fontsize=18,
    xtick_fontsize=22,   # <<< 离子字符大小在这里调
    ytick_fontsize=18,
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
    ions = d["elem"].tolist()
    x = list(range(len(ions)))

    fig, axes = plt.subplots(1, len(params), figsize=figsize, sharex=True, sharey=sharey)
    if len(params) == 1:
        axes = [axes]

    bar_w = group_width / 2.0
    off = bar_w / 2.0

    for ax, p in zip(axes, params):
        col_poll = f"{p}@delta_polluted"
        col_recv = f"{p}@delta_recovery"

        # ★ 关键修改：整体取相反数（正->负，负->正）
        y_poll = -d[col_poll].astype(float).values
        y_recv = -d[col_recv].astype(float).values

        ax.bar([xi - off for xi in x], y_poll, width=bar_w, color=colors,
               label="After contamination" if p == params[0] else None)
        ax.bar([xi + off for xi in x], y_recv, width=bar_w, color=colors, alpha=recover_alpha,
               label="After recovery" if p == params[0] else None)

        ax.set_title(title_map.get(p, p), fontsize=title_fontsize)
        ax.set_ylabel("Δ value" if p == params[0] else " ", fontsize=ylabel_fontsize)

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

    # 统一图例（放底部）
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               fontsize=legend_fontsize, frameon=True, bbox_to_anchor=(0.5, -0.06))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    return fig, axes

# -----------------------------
# 6) 画图（这里调字号）
# -----------------------------
fig, axes = plot_5params_contam_vs_recover_flip_sign(
    df,
    params=("sigma_mem","alpha_ca","alpha_an","i_0ca","i_0an"),
    title_fontsize=24,
    ylabel_fontsize=20,
    xtick_fontsize=24,   # <<< x轴离子字符大小
    ytick_fontsize=18,
    legend_fontsize=20,
    tick_length=8,
    tick_width=2.0,
    recover_alpha=0.45,
    show_grid=False,
    zero_line=True,
    sharey=False,
)

plt.show()
fig.savefig("params_5subplots_contam_vs_recover_flipped_sign.png", dpi=300, bbox_inches="tight", pad_inches=0.10)
