import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objs as go
import plotly.offline as pyo
import numpy as np

import os
import sys

# 动态添加上两级目录到系统路径
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print('base_dir :', base_dir)
ion_color_path = os.path.join(base_dir, 'ion_color.py')

# 如果 ion_color.py 在该路径下，添加该目录到 sys.path
if os.path.exists(ion_color_path):
    sys.path.append(os.path.dirname(ion_color_path))

# 现在你可以导入 ion_color 中的函数了
from ion_color import get_ion_color


# Matplotlib标记符号和Plotly标记符号的映射
marker_symbols = {
    'o': 'circle',
    's': 'square',
    '+': 'cross',
    'x': 'x',
    '^': 'triangle-up',
    'v': 'triangle-down',
    '<': 'triangle-left',
    '>': 'triangle-right',
    'd': 'diamond',
    '*': 'star',
}
import re

def format_ion_label(label: str):
    """
    在任意字符串中，将离子电荷（如 Ni2+, Cl-, SO4^2-）显示为右上角上标。
    兼容 '2ppm Ni2+ 0h' 这类：只替换其中的 Ni2+ 片段，其余不变。
    返回: (mpl_label, plotly_label)
    """
    if label is None:
        return label, label
    s = str(label).strip()

    # 1) 优先匹配 SO4^2- 这种带 ^ 的写法
    pattern_hat = re.compile(r"(?P<ion>[A-Za-z][A-Za-z0-9()_]*?)\^(?P<num>\d+)(?P<sign>[+-])")
    m = pattern_hat.search(s)
    if m:
        ion = m.group("ion")
        charge = f"{m.group('num')}{m.group('sign')}"
        mpl_repl = rf"$\mathrm{{{ion}}}^{{{charge}}}$"
        ply_repl = f"{ion}<sup>{charge}</sup>"
        return s[:m.start()] + mpl_repl + s[m.end():], s[:m.start()] + ply_repl + s[m.end():]

    # 2) 匹配 Ni2+ / Cl- / Na+ 这种末尾电荷
    pattern_plain = re.compile(r"(?P<ion>[A-Za-z][A-Za-z0-9()_]*?)(?P<num>\d+)?(?P<sign>[+-])")
    m = pattern_plain.search(s)
    if m:
        ion = m.group("ion")
        num = m.group("num")
        sign = m.group("sign")
        charge = f"{num}{sign}" if num else sign
        mpl_repl = rf"$\mathrm{{{ion}}}^{{{charge}}}$"
        ply_repl = f"{ion}<sup>{charge}</sup>"
        return s[:m.start()] + mpl_repl + s[m.end():], s[:m.start()] + ply_repl + s[m.end():]

    return s, s


# 数据提取函数不变
def extract_eis_data_from_csv(file_path, remove_first_n_points=0):
    df = pd.read_csv(file_path)
    if 'Freq/Hz' not in df.columns or 'Re(Z)/Ohm' not in df.columns or 'Im(Z)/Ohm' not in df.columns:
        raise ValueError(f"CSV文件 {file_path} 中没有找到正确的列名")

    # 移除前n个数据点
    df = df.iloc[remove_first_n_points:]

    data = {
        'Freq': df['Freq/Hz'].values,
        'Zreal': df['Re(Z)/Ohm'].values,
        'Zimag': df['Im(Z)/Ohm'].values
    }
    return data


# RGB颜色转换函数
def rgb_to_hex(rgb_tuple):
    return mcolors.to_hex([x / 255.0 for x in rgb_tuple])


# 绘制Bode图：幅值和相角
def plot_bode(ax_mag, ax_phase, freq, zreal, zimag, label, color, marker, markersize=4):
    # 计算幅值和相位
    magnitude = np.abs(zreal + 1j * zimag)
    phase = np.angle(zreal + 1j * zimag, deg=True)

    mpl_label, _ = format_ion_label(label)

    # 绘制幅值
    ax_mag.semilogx(
        freq,
        20 * np.log10(magnitude),
        label=mpl_label,
        color=color,
        marker=marker,
        linewidth=2,
        markersize=markersize
    )

    # 绘制相位
    ax_phase.semilogx(
        freq,
        phase,
        label=mpl_label,
        color=color,
        marker=marker,
        linewidth=2,
        markersize=markersize
    )

def find_data_starting_indices(lines, keywords):
    indices = {key: None for key in keywords}
    for i, line in enumerate(lines):
        columns = line.strip().split('\t')
        for j, column in enumerate(columns):
            if column in indices:
                indices[column] = (i, j)
    return indices


def extract_eis_data_from_lines(lines, indices):
    data = {'Freq': [], 'Zreal': [], 'Zimag': []}
    start_row = max(index[0] for index in indices.values()) + 2
    for line in lines[start_row:]:
        columns = line.strip().split('\t')
        if len(columns) > max(index[1] for index in indices.values()):
            data['Freq'].append(float(columns[indices['Freq'][1]]))
            data['Zreal'].append(float(columns[indices['Zreal'][1]]))
            data['Zimag'].append(float(columns[indices['Zimag'][1]]))
    return data
# 数据转换和绘制
def convert_and_plot_eis(file_specifications, source_type, marker_size=6):
    plotly_traces_mag = []
    plotly_traces_phase = []

    for file_spec in file_specifications:
        if len(file_spec) == 4:
            file_path, rgb_color, marker, remove_first_n_points = file_spec
            display_name = os.path.basename(file_path)
        elif len(file_spec) == 5:
            file_path, rgb_color, marker, display_name, remove_first_n_points = file_spec
        else:
            continue

        color = rgb_to_hex(rgb_color)

        if os.path.isfile(file_path):
            if source_type == 'zhiyun':
                data = extract_eis_data_from_csv(file_path, remove_first_n_points)
            elif source_type == 'gamry':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    lines = file.readlines()
                indices = find_data_starting_indices(lines, ['Freq', 'Zreal', 'Zimag'])
                if None in indices.values():
                    print(f"Could not find required columns in {file_path}")
                    continue
                data = extract_eis_data_from_lines(lines, indices)
                for key in data:
                    data[key] = data[key][remove_first_n_points:]
            else:
                continue

            df = pd.DataFrame(data)

            # === label 上标格式化（Matplotlib + Plotly 各一份）===
            mpl_label, plotly_label = format_ion_label(display_name)

            # Matplotlib：用 mpl_label
            plot_bode(
                ax_mag, ax_phase,
                df['Freq'], df['Zreal'], df['Zimag'],
                label=mpl_label, color=color, marker=marker,
                markersize=4
            )

            # Plotly：用 plotly_label
            plotly_marker = marker_symbols.get(marker, 'circle')

            trace_mag = go.Scatter(
                x=df['Freq'],
                y=20 * np.log10(np.abs(df['Zreal'] + 1j * df['Zimag'])),
                mode='lines+markers',
                name=plotly_label,
                line=dict(color=color),
                marker=dict(symbol=plotly_marker, size=marker_size),
            )
            plotly_traces_mag.append(trace_mag)

            trace_phase = go.Scatter(
                x=df['Freq'],
                y=np.angle(df['Zreal'] + 1j * df['Zimag'], deg=True),
                mode='lines+markers',
                name=plotly_label,
                line=dict(color=color),
                marker=dict(symbol=plotly_marker, size=marker_size),
            )
            plotly_traces_phase.append(trace_phase)

        else:
            print(f"File not found: {file_path}")

    return plotly_traces_mag, plotly_traces_phase

from pathlib import Path, PureWindowsPath, PurePosixPath

def _normalize_rel(p: str) -> Path:
    """
    将传入的相对路径字符串（可能含有 Windows 的 '\' 或 POSIX 的 '/'）
    解析为路径部件并用本机 Path 重组；不改变中文或全角字符。
    """
    s = str(p).strip().strip('"').strip("'").replace('\u00a0', ' ')
    pure = PureWindowsPath(s) if "\\" in s else PurePosixPath(s)
    return Path(*pure.parts)


def plot_impedance_bode(output_folder, file_specifications_zhiyun_old, file_specifications_gamry_old, file_name,marker_size=6):
    global ax_mag, ax_phase
    fig_mag, ax_mag = plt.subplots(figsize=(12, 6))
    fig_phase, ax_phase = plt.subplots(figsize=(12, 6))

    # 脚本目录上三级 + “校内测试”
    base_dir = (Path(__file__).resolve().parent / ".." / ".." / ".." / "校内测试").resolve()
    print(f"[DEBUG] base_dir = {base_dir}, exists: {base_dir.exists()}")

    # 输出目录
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    # —— 规范 zhiyun 规格：绝对路径 + 颜色默认化 ——
    file_specifications_zhiyun = []
    for item in file_specifications_zhiyun_old:
        rel_path, color, marker, label, remove_n = item
        if color == 'default':
            color = get_ion_color(rel_path)
        abs_path = (base_dir / _normalize_rel(rel_path)).resolve()
        file_specifications_zhiyun.append((str(abs_path), color, marker, label, remove_n))

    # —— 规范 gamry 规格：绝对路径 + 颜色默认化 ——
    file_specifications_gamry = []
    for item in file_specifications_gamry_old:
        rel_path, color, marker, label, remove_n = item
        if color == 'default':
            color = get_ion_color(rel_path)
        abs_path = (base_dir / _normalize_rel(rel_path)).resolve()
        file_specifications_gamry.append((str(abs_path), color, marker, label, remove_n))

    # === 绘制（你的函数需返回两个列表：幅值trace列表、相位trace列表）===
    plotly_traces_mag_zhiyun, plotly_traces_phase_zhiyun = convert_and_plot_eis(file_specifications_zhiyun, 'zhiyun', marker_size=marker_size)
    plotly_traces_mag_gamry,  plotly_traces_phase_gamry  = convert_and_plot_eis(file_specifications_gamry,  'gamry', marker_size=marker_size)

    # === Matplotlib 外观 ===
    ax_mag.set_xlabel('Frequency (Hz)', fontsize=22)
    ax_mag.set_ylabel('Magnitude (Ohm)', fontsize=22)
    ax_mag.set_title('Bode Plot - Magnitude', fontsize=22)
    ax_mag.grid(True)
    ax_mag.legend(fontsize=22, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)

    ax_phase.set_xlabel('Frequency (Hz)', fontsize=22)
    ax_phase.set_ylabel('Phase (degrees)', fontsize=22)
    ax_phase.set_title('Bode Plot - Phase', fontsize=22)
    ax_phase.grid(True)
    ax_mag.grid(False)
    ax_phase.grid(False)

    # 只保留 x/y 轴（去掉上/右边框）
    for ax in (ax_mag, ax_phase):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_phase.legend(fontsize=22, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    
    for ax in (ax_mag, ax_phase):
        ax.tick_params(axis='both', which='both', labelsize=18)

    fig_mag.tight_layout(rect=[0, 0, 1, 0.95])
    fig_phase.tight_layout(rect=[0, 0, 1, 0.95])

    # === 保存图片 ===
    magnitude_png = output_dir / f"bode_magnitude_combined_{file_name}.png"
    phase_png     = output_dir / f"bode_phase_combined_{file_name}.png"
    fig_mag.savefig(magnitude_png, dpi=300, bbox_inches='tight')
    fig_phase.savefig(phase_png, dpi=300, bbox_inches='tight')
    print(f"PNG 已保存: {magnitude_png}")
    print(f"PNG 已保存: {phase_png}")

    # === 组装并导出 CSV（长表格式）===
    # 约定：trace.x 为频率 Hz，trace.y 为幅值/相位的 y 值
    rows_mag, rows_phase = [], []

    def _collect(traces, source_name, kind, rows_sink):
        for tr in traces:
            x = getattr(tr, "x", None)
            y = getattr(tr, "y", None)
            if x is None or y is None:
                continue
            label = getattr(tr, "name", source_name)
            for xi, yi in zip(x, y):
                rows_sink.append({
                    "source": source_name,
                    "label": label,
                    "freq_Hz": xi,
                    "value": yi,
                    "kind": kind,  # "magnitude_Ohm" 或 "phase_deg"
                })

    _collect(plotly_traces_mag_zhiyun, "zhiyun", "magnitude_Ohm", rows_mag)
    _collect(plotly_traces_mag_gamry,  "gamry",  "magnitude_Ohm", rows_mag)
    _collect(plotly_traces_phase_zhiyun, "zhiyun", "phase_deg", rows_phase)
    _collect(plotly_traces_phase_gamry,  "gamry",  "phase_deg", rows_phase)

    df_mag   = pd.DataFrame(rows_mag)
    df_phase = pd.DataFrame(rows_phase)

    magnitude_csv = magnitude_png.with_suffix(".csv")
    phase_csv     = phase_png.with_suffix(".csv")
    df_mag.to_csv(magnitude_csv, index=False, encoding="utf-8-sig")
    df_phase.to_csv(phase_csv,   index=False, encoding="utf-8-sig")
    print(f"CSV 已保存: {magnitude_csv}")
    print(f"CSV 已保存: {phase_csv}")

    # plt.show()

    # # 使用Plotly绘制交互式图形
    # layout_mag = go.Layout(
    #     xaxis=dict(title='Frequency (Hz)'),
    #     yaxis=dict(title='Magnitude (Ohm)'),
    #     title='Bode Plot - Magnitude',
    #     legend=dict(orientation="h", y=-0.2)
    # )
    # layout_phase = go.Layout(
    #     xaxis=dict(title='Frequency (Hz)'),
    #     yaxis=dict(title='Phase (degrees)'),
    #     title='Bode Plot - Phase',
    #     legend=dict(orientation="h", y=-0.2)
    # )

    # fig_mag = go.Figure(data=plotly_traces_mag_zhiyun + plotly_traces_mag_gamry, layout=layout_mag)
    # fig_phase = go.Figure(data=plotly_traces_phase_zhiyun + plotly_traces_phase_gamry, layout=layout_phase)

    # pyo.plot(fig_mag, filename=os.path.join(output_folder, 'bode_magnitude_combined.html'))
    # pyo.plot(fig_phase, filename=os.path.join(output_folder, 'bode_phase_combined.html'))




if __name__ == "__main__":
    # 这里指定输出文件夹
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    output_folder = f"{base_dir}/图片输出文件夹"
    print("output_folder:", output_folder)



    file_specifications_zhiyun = [
        # 你可以继续添加其他文件，并为每个文件指定remove_first_n_points
        # (file_path, color, marker, remove_first_n_points)
        # 或者 (file_path, color, marker, display_name, remove_first_n_points)

        (r"20241101_2ppm钙离子污染和恢复测试\新版电解槽_firecloud\20241101_ion\循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv",  (255, 200, 200), '+', '2ppm Ca2+ 0h_1101_firecloud',1),  # 更多文件...
        (r"20241101_2ppm钙离子污染和恢复测试\新版电解槽_firecloud\20241101_ion\循环1／1_工步组2(工步组)(2／80)_工步3(阻抗).csv",  (255, 140, 140), 's', '2ppm Ca2+ 2h_1101_firecloud',1),  # 更多文件...
        # (r"20241101_2ppm钙离子污染和恢复测试\新版电解槽_firecloud\20241101_ion\循环1／1_工步组2(工步组)(6／80)_工步3(阻抗).csv",  (255, 80, 80), 'o', '2ppm Ca2+ 12h_1101_firecloud',1),  # 更多文件...
        # (r"20241101_2ppm钙离子污染和恢复测试\新版电解槽_firecloud\20241101_ion\循环1／1_工步组2(工步组)(12／80)_工步3(阻抗).csv",  (255, 20, 20), '*', '2ppm Ca2+ 24h_1101_firecloud',1),  # 更多文件...
        
        # (r"20241107_0.1ppm钙离子污染及恢复测试\新版电解槽_firecloud\20241107_ion\循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv",  'default', '+', '0.1ppm Ca2+ 0h_1107_firecloud',1),  # 更多文件...
        # (r"20241107_0.1ppm钙离子污染及恢复测试\新版电解槽_firecloud\20241107_ion\循环1／1_工步组2(工步组)(1／80)_工步3(阻抗).csv",  'default', 's', '0.1ppm Ca2+ 2h_1107_firecloud',1),  # 更多文件...
        # (r"20241107_0.1ppm钙离子污染及恢复测试\新版电解槽_firecloud\20241107_ion\循环1／1_工步组2(工步组)(6／80)_工步3(阻抗).csv",  'default', 'o', '0.1ppm Ca2+ 12h_1107_firecloud',1),  # 更多文件...
        # (r"20241107_0.1ppm钙离子污染及恢复测试\新版电解槽_firecloud\20241107_ion\循环1／1_工步组2(工步组)(12／80)_工步3(阻抗).csv",  'default', 'o', '0.1ppm Ca2+ 24h_1107_firecloud',1),  # 更多文件...
    ]
    file_specifications_gamry = [

        # (r"20240910_10ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240908_ion_0.DTA",'default', '+', '10ppm Ca2+ 0h_0908_gamry', 0),
        # (r"20240910_10ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240908_ion_2.DTA",'default', 's', '10ppm Ca2+ 2h_0908_gamry', 0),
        # (r"20240910_10ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240908_ion_5.DTA", (204, 153, 255), 'o', '10ppm Ca2+ 12h_0908_gamry',0),
        # (r"20240910_10ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240908_ion_10.DTA", (204, 153, 255), '*', '10ppm Ca2+ 24h_0908_gamry',0),


        (r"20240918_2ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240917_ion_0.DTA", (200, 255, 200), '+', '2ppm Ca2+ 0h_0917_gamry',0),
        (r"20240918_2ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240917_ion_1.DTA",(140, 255, 140), 's', '2ppm Ca2+ 2h_0917_gamry',0),
        # (r"20240918_2ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240917_ion_5.DTA", (80, 255, 80), 'o', '2ppm Ca2+ 12h_0917_gamry',0),
        # (r"20240918_2ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240917_ion_10.DTA", (20, 255, 20), '*', '2ppm Ca2+ 24h_0917_gamry',0),
    

    ]


    plot_impedance_bode(output_folder, file_specifications_zhiyun,file_specifications_gamry,file_name='')