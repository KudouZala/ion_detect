import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Axes definition
# -----------------------------
window_lengths = [1, 2, 3, 4]
steps = [3, 2, 1]

# -----------------------------
# Data matrix
# IMPORTANT: rows must match steps order [3,2,1]
# -----------------------------
vals = np.array([

    [23.8, 38.1, 42.8, 47.6],   # step = 1
            [28.5, 42.8, 57.0, 47.6],   # step = 2
    [33.3, 42.8, 47.0, 52.0],   # step = 3

], dtype=float)

# -----------------------------
# Pair mask (same row order as steps: [3,2,1])
# -----------------------------
is_pair = np.array([
    [False, False, False, False],  # step=3
    [False, False, True,  False],  # step=2 (L=3 is pair)
    [False, True,  False, True ],  # step=1 (L=2 and L=4 are pair)
], dtype=bool)

df = pd.DataFrame(vals, index=steps, columns=window_lengths)

# -----------------------------
# Build annotation: value + optional star
# -----------------------------
annot = df.copy().astype(object)
for i, s in enumerate(steps):
    for j, w in enumerate(window_lengths):
        txt = f"{df.loc[s, w]:.1f}"
        if is_pair[i, j]:
            txt += r"$^{\star}$"
        annot.loc[s, w] = txt

# -----------------------------
# Plot style (paper-level) - keep your style
# -----------------------------
sns.set_theme(style="white", context="paper")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 10,
})

fig, ax = plt.subplots(figsize=(4.6, 3.0), dpi=300)

sns.heatmap(
    df,
    ax=ax,
    cmap="YlGnBu",
    annot=annot.values,   # use our string annotations (with stars)
    fmt="",               # important: strings, so fmt must be empty
    annot_kws={"color": "black", "fontsize": 10},
    linewidths=0.6,
    linecolor=(0, 0, 0, 0.10),
    cbar=True,
    cbar_kws={"label": "Accuracy (%)", "shrink": 0.9},
    square=True
)

ax.set_xlabel("Sliding-window length")
ax.set_ylabel("Step size")

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig("Fig_window_step_L1_L4.png", bbox_inches="tight")
plt.savefig("Fig_window_step_L1_L4.pdf", bbox_inches="tight")
plt.show()

print("Saved: Fig_window_step_L1_L4.pdf and .png")
print("★ indicates paired setting.")
