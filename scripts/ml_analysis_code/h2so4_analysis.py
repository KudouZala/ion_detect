#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
h2so4_analysis.py — AI-assisted H2SO4 recovery analysis (initial-state 5-factor deltas)

Usage:
    cd ~/.../ion_detect/data/scripts/excel_code/
    python h2so4_analysis.py --load_run=20250827c

What it does (per your spec):
- Looks under: <repo_root>/output/inference_results/<load_run>
- Scans only files whose names contain "_phys_params_structured.csv"
- For each "project", **group by the first three underscore-separated fields in the filename stem**
  (e.g., "20241017_2ppm铬离子污染和恢复测试_新版电解槽_..."):
    1) POLLUTION-BEFORE (baseline): *_ion_{gamry|firecloud}_[0, 2, 4, 6]_phys_params_structured.csv
    2) POLLUTION-AFTER:            *_ion_{gamry|firecloud}_[6, 8, 10, 12]_phys_params_structured.csv
    3) RECOVERY-AFTER (H2SO4):     *_ion_column_renew_H2SO4_{gamry|firecloud}_[0, 2, 4, 6]_phys_params_structured.csv
- From each csv, read the 5 initial-state parameters by locating the cell with that key
  and taking the **next column on the same row**:
    sigma_mem, alpha_ca, alpha_an, i_0ca, i_0an
- For each project, compute per-parameter deltas RELATIVE TO the baseline (treat baseline as 0):
    delta_polluted = (POLLUTION-AFTER)  - (POLLUTION-BEFORE)
    delta_recovery = (RECOVERY-AFTER)   - (POLLUTION-BEFORE)
- Group by ion (钙/钠/铁/镍/铬/铜), average the deltas across projects of the same ion.
- Save a single CSV summary and one figure (two bars per ion per parameter).
"""

from __future__ import annotations
import argparse
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Fonts ----------
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = True

# --------------------------- Config ---------------------------
ION_KEYWORDS = {
    "钙离子": "Ca",
    "钠离子": "Na",
    "铁离子": "Fe",
    "镍离子": "Ni",
    "铬离子": "Cr",
    "铜离子": "Cu",
}
ION_ORDER = ["Ca", "Na", "Ni","Cr","Fe","Cu"]

# Ion colors (aligned with csv_plot_param.py)
ION_COLORS = {
    "Na": "yellow",
    "Ca": "purple",
    "Fe": "orange",
    "Ni": "cyan",
    "Cr": "green",
    "Cu": "blue",
}

def lighten_color(color, factor=0.5):
    """Return a lighter variant of the given Matplotlib color by blending toward white.
    factor in [0,1]: 0=no change, 1=white.
    """
    import matplotlib.colors as mcolors
    r, g, b, a = mcolors.to_rgba(color)
    r = r + (1 - r) * factor
    g = g + (1 - g) * factor
    b = b + (1 - b) * factor
    return (r, g, b, a)

INIT_PARAM_KEYS = ["sigma_mem", "alpha_ca", "alpha_an", "i_0ca", "i_0an"]

# LaTeX-like pretty labels for parameters (to mirror csv_plot_param.py style)
INIT_PARAM_KEYS_LATEX = {
    "sigma_mem": r"$\sigma_{mem}$",
    "alpha_ca": r"$\alpha_{ca}$",
    "alpha_an": r"$\alpha_{an}$",
    "i_0ca": r"$i_{0,ca}$",
    "i_0an": r"$i_{0,an}$",
}

# time-window fragments
PAT_0H  = re.compile(r"\[\s*0\s*,\s*2\s*,\s*4\s*,\s*6\s*\]")
PAT_6H  = re.compile(r"\[\s*6\s*,\s*8\s*,\s*10\s*,\s*12\s*\]")

# roles
ROLE_BASE  = "base"     # pollution-before
ROLE_POLL  = "polluted" # pollution-after
ROLE_RECV  = "recovery" # recovery-after H2SO4

_FLOAT_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="AI-assisted H2SO4 recovery analysis (baseline vs polluted vs recovered deltas).")
    ap.add_argument("--load_run", required=True, help="e.g., 20250827c")
    return ap.parse_args()

def find_target_dir(load_run: str) -> Path:
    script_path = Path(__file__).resolve()
    base_dir = script_path.parent.parent.parent
    target = base_dir / "output" / "inference_results" / load_run
    if not target.exists():
        print(f"[ERROR] directory not found: {target}")
        sys.exit(1)
    return target

def robust_read_csv(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "gbk", "gb2312"]:
        try:
            return pd.read_csv(path, header=None, dtype=str, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path, header=None, dtype=str, engine="python")

def first_float_in_cell(cell) -> Optional[float]:
    if cell is None:
        return None
    if isinstance(cell, float) and np.isnan(cell):
        return None
    s = str(cell).strip().replace(",", "")
    m = _FLOAT_PATTERN.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def extract_params_from_df(df: pd.DataFrame, keys: List[str]) -> Dict[str, List[float]]:
    """Find each key (case-insensitive exact string) in any cell, take the next column's first float."""
    res = {k: [] for k in keys}
    lower_df = df.applymap(lambda x: str(x).strip().lower() if pd.notna(x) else x)
    wanted = {k: k.lower() for k in keys}
    R, C = df.shape
    for r in range(R):
        for c in range(C):
            cell = lower_df.iat[r, c]
            if cell is None or (isinstance(cell, float) and np.isnan(cell)):
                continue
            for p, p_low in wanted.items():
                if cell == p_low:
                    if c + 1 < C:
                        v = first_float_in_cell(df.iat[r, c + 1])
                        if v is not None:
                            res[p].append(v)
    return res

def detect_ion(name: str) -> Optional[str]:
    for zh, abbr in ION_KEYWORDS.items():
        if zh in name:
            return abbr
    return None

def is_role_of_interest(name: str) -> Optional[str]:
    """Return ROLE_BASE / ROLE_POLL / ROLE_RECV if matches, otherwise None."""
    if ("ion_column_renew_H2SO4" in name) and PAT_0H.search(name):
        return ROLE_RECV
    if ("ion_" in name) and PAT_0H.search(name) and ("ion_column" not in name):
        return ROLE_BASE
    if ("ion_" in name) and PAT_6H.search(name) and ("ion_column" not in name):
        return ROLE_POLL
    return None

def project_base_key(name: str) -> str:
    """
    Group by the first three underscore-separated fields in the filename stem.
    For example:
      '20241017_2ppm铬离子污染和恢复测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_phys_params_structured.csv'
    -> project key: '20241017_2ppm铬离子污染和恢复测试_新版电解槽_'
    """
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3]) + "_"
    return stem

def mean_or_nan(lst: List[float]) -> float:
    arr = np.array(lst, dtype=float) if lst else np.array([], dtype=float)
    return float(np.nanmean(arr)) if arr.size else np.nan

def main():
    args = parse_args()
    target_dir = find_target_dir(args.load_run)

    # 1) scan CSVs
    csv_files = sorted(target_dir.glob("*_phys_params_structured.csv"))
    if not csv_files:
        print("[WARN] no *_phys_params_structured.csv found.")
        sys.exit(0)

    # group found files into projects & roles
    projects: Dict[str, Dict[str, List[Path]]] = {}
    project_ion: Dict[str, str] = {}

    for f in csv_files:
        name = f.name
        role = is_role_of_interest(name)
        if role is None:
            continue
        ion = detect_ion(name)
        if ion is None:
            continue
        bk = project_base_key(name)
        projects.setdefault(bk, {}).setdefault(role, []).append(f)
        project_ion.setdefault(bk, ion)

    if not projects:
        print("[WARN] No matching (base/polluted/recovery) CSV triplets found.")
        sys.exit(0)

    per_project_rows = []
    per_ion_pool = {ion: {"delta_polluted": {k: [] for k in INIT_PARAM_KEYS},
                          "delta_recovery": {k: [] for k in INIT_PARAM_KEYS}} for ion in ION_ORDER}

    for bk, role_map in projects.items():
        ion = project_ion.get(bk, None)
        if ion is None:
            continue

        role_means: Dict[str, Dict[str, float]] = {}
        for role in [ROLE_BASE, ROLE_POLL, ROLE_RECV]:
            files = role_map.get(role, [])
            if not files:
                continue

            accum = {k: [] for k in INIT_PARAM_KEYS}
            for p in files:
                df = robust_read_csv(p)
                vals = extract_params_from_df(df, INIT_PARAM_KEYS)
                for k in INIT_PARAM_KEYS:
                    accum[k].extend(vals.get(k, []))

            role_means[role] = {k: mean_or_nan(accum[k]) for k in INIT_PARAM_KEYS}

        if ROLE_BASE not in role_means:
            continue

        base_vec = role_means[ROLE_BASE]
        if ROLE_POLL in role_means:
            poll_vec = role_means[ROLE_POLL]
            for k in INIT_PARAM_KEYS:
                d = (poll_vec[k] - base_vec[k]) if (not np.isnan(poll_vec[k]) and not np.isnan(base_vec[k])) else np.nan
                if not np.isnan(d):
                    per_ion_pool[ion]["delta_polluted"][k].append(float(d))

        if ROLE_RECV in role_means:
            recv_vec = role_means[ROLE_RECV]
            for k in INIT_PARAM_KEYS:
                d = (recv_vec[k] - base_vec[k]) if (not np.isnan(recv_vec[k]) and not np.isnan(base_vec[k])) else np.nan
                if not np.isnan(d):
                    per_ion_pool[ion]["delta_recovery"][k].append(float(d))

        per_project_rows.append({
            "project": bk,
            "ion": ion,
            **{f"BASE_{k}": base_vec.get(k, np.nan) for k in INIT_PARAM_KEYS},
            **{f"POLL_{k}": (role_means.get(ROLE_POLL, {}).get(k, np.nan)) for k in INIT_PARAM_KEYS},
            **{f"RECV_{k}": (role_means.get(ROLE_RECV, {}).get(k, np.nan)) for k in INIT_PARAM_KEYS},
        })

    summary_rows = []
    for ion in ION_ORDER:
        row = {"ion": ion}
        for k in INIT_PARAM_KEYS:
            row[f"{k}@delta_polluted"] = mean_or_nan(per_ion_pool[ion]["delta_polluted"][k])
            row[f"{k}@delta_recovery"] = mean_or_nan(per_ion_pool[ion]["delta_recovery"][k])
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows, columns=["ion"] + [f"{k}@delta_polluted" for k in INIT_PARAM_KEYS] + [f"{k}@delta_recovery" for k in INIT_PARAM_KEYS])

    out_csv = target_dir / f"{args.load_run}_H2SO4_recovery_summary.csv"
    summary_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] saved summary CSV: {out_csv}")

    if per_project_rows:
        diag_df = pd.DataFrame(per_project_rows)
        out_diag = target_dir / f"{args.load_run}_H2SO4_recovery_per_project.csv"
        diag_df.to_csv(out_diag, index=False, encoding="utf-8-sig")
        print(f"[OK] saved per-project detail CSV: {out_diag}")

    ions = ION_ORDER
    idx = np.arange(len(ions))
    W = 0.38

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
    axes = axes.flatten()

    for i, k in enumerate(INIT_PARAM_KEYS):
        y_poll = [summary_df.loc[summary_df["ion"] == ion, f"{k}@delta_polluted"].values[0] if not summary_df.loc[summary_df["ion"] == ion, f"{k}@delta_polluted"].empty else np.nan for ion in ions]
        y_recv = [summary_df.loc[summary_df["ion"] == ion, f"{k}@delta_recovery"].values[0] if not summary_df.loc[summary_df["ion"] == ion, f"{k}@delta_recovery"].empty else np.nan for ion in ions]

        ax = axes[i]
        poll_colors = [ION_COLORS.get(ion, "gray") for ion in ions]
        recv_colors = [lighten_color(ION_COLORS.get(ion, "gray"), factor=0.5) for ion in ions]
        ax.bar(idx - W/2, y_poll, width=W, label="Polluted", color=poll_colors)
        ax.bar(idx + W/2, y_recv, width=W, label="Recovered", color=recv_colors)
        ax.set_xticks(idx, ions)
        pretty = INIT_PARAM_KEYS_LATEX.get(k, k)
        ax.set_title(fr"Δ vs Before — {pretty}")
        ax.set_ylabel("Delta")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if i == 0:
            ax.legend()

    if len(INIT_PARAM_KEYS) < len(axes):
        for j in range(len(INIT_PARAM_KEYS), len(axes)):
            axes[j].axis("off")

    fig.suptitle(f"{args.load_run} | H2SO4 recovery: Per-ion averaged deltas (relative to baseline)")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_png = target_dir / f"{args.load_run}_H2SO4_recovery_bars.png"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[OK] saved figure: {out_png}")

    print("Done.")

if __name__ == "__main__":
    main()