import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objs as go
import plotly.offline as pyo


import os
import sys

# 动态添加上两级目录到系统路径
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print('base_dir :',base_dir )
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
    示例:
      '2ppm Ni2+ 0h' -> Matplotlib: r'2ppm $\mathrm{Ni}^{2+}$ 0h'
                       Plotly:     '2ppm Ni<sup>2+</sup> 0h'
    若找不到可识别的离子电荷片段，则原样返回。
    """
    if label is None:
        return label, label

    s = str(label).strip()

    # 1) 先处理形如 SO4^2- 的片段（带 ^）
    #    允许离子部分包含字母/数字/括号/下划线等常见化学表达
    pattern_hat = re.compile(r"(?P<ion>[A-Za-z][A-Za-z0-9()_]*?)\^(?P<num>\d+)(?P<sign>[+-])")
    m = pattern_hat.search(s)
    if m:
        ion = m.group("ion")
        charge = f"{m.group('num')}{m.group('sign')}"
        # 用替换方式把命中的片段替换掉
        mpl_repl = rf"$\mathrm{{{ion}}}^{{{charge}}}$"
        ply_repl = f"{ion}<sup>{charge}</sup>"
        s_mpl = s[:m.start()] + mpl_repl + s[m.end():]
        s_ply = s[:m.start()] + ply_repl + s[m.end():]
        return s_mpl, s_ply

    # 2) 再处理形如 Ni2+ / Fe3- / Cl- / Na+ 的片段（末尾带电荷）
    #    这里用“离子主体 + 可选数字 + 正负号”的模式做 search（而不是全串 match）
    pattern_plain = re.compile(r"(?P<ion>[A-Za-z][A-Za-z0-9()_]*?)(?P<num>\d+)?(?P<sign>[+-])")
    m = pattern_plain.search(s)
    if m:
        ion = m.group("ion")
        num = m.group("num")
        sign = m.group("sign")
        charge = f"{num}{sign}" if num else sign

        mpl_repl = rf"$\mathrm{{{ion}}}^{{{charge}}}$"
        ply_repl = f"{ion}<sup>{charge}</sup>"
        s_mpl = s[:m.start()] + mpl_repl + s[m.end():]
        s_ply = s[:m.start()] + ply_repl + s[m.end():]
        return s_mpl, s_ply

    return s, s

# 炙云数据提取
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


# Gamry数据提取
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


# RGB颜色转换函数，将RGB三元组转换为Matplotlib可用的格式
def rgb_to_hex(rgb_tuple):
    return mcolors.to_hex([x/255.0 for x in rgb_tuple])

# 绘制奈奎斯特图
def plot_nyquist(ax, zreal, zimag, label, color, marker,markersize=6):
    ax.plot(zreal, -zimag, label=label, color=color, marker=marker,linewidth=2, markersize=markersize)

# 数据转换和绘制
def convert_and_plot_eis(file_specifications, source_type, marker_size=6):
    """
    file_specifications: list of tuples
        支持两种格式：
        (file_path, rgb_color, marker, remove_first_n_points)
        (file_path, rgb_color, marker, display_name, remove_first_n_points)
    source_type: 'zhiyun' | 'gamry'
    marker_size: 散点大小（同时作用于 Matplotlib 和 Plotly）
    """
    plotly_traces = []

    for file_spec in file_specifications:
        # 解析规格
        if len(file_spec) == 4:
            file_path, rgb_color, marker, remove_first_n_points = file_spec
            display_name = os.path.basename(file_path)
        elif len(file_spec) == 5:
            file_path, rgb_color, marker, display_name, remove_first_n_points = file_spec
        else:
            # 规格不符合预期，跳过
            continue
        mpl_label, plotly_label = format_ion_label(display_name)


        # 颜色转换为 hex
        color = rgb_to_hex(rgb_color)

        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            continue

        # 按来源读取数据
        if source_type == 'zhiyun':
            data = extract_eis_data_from_csv(file_path, remove_first_n_points)
        elif source_type == 'gamry':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            indices = find_data_starting_indices(lines, ['Freq', 'Zreal', 'Zimag'])
            if None in indices.values():
                print(f"Could not find required columns in {file_path}")
                continue
            data = extract_eis_data_from_lines(lines, indices)
            # 移除每个列表前 n 个点
            for k in data:
                data[k] = data[k][remove_first_n_points:]
        else:
            continue

        df = pd.DataFrame(data)

        # ——— Matplotlib 绘制（传入 marker_size）———
        # 你的 plot_nyquist 需要支持 marker_size 参数（见下方说明）
        plot_nyquist(ax_nyquist, df['Zreal'], df['Zimag'],
                    label=mpl_label, color=color, marker=marker,
                    markersize=marker_size)


        # ——— Plotly trace（携带 size）———
        plotly_marker = marker_symbols.get(marker, 'circle')
        trace = go.Scatter(
            x=df['Zreal'],
            y=-df['Zimag'],  # 图里绘制 -Zimag
            mode='lines+markers',
            name=plotly_label,
            line=dict(color=color),
            marker=dict(symbol=plotly_marker, size=marker_size)  # <<< 设置点大小
        )
        plotly_traces.append(trace)

    return plotly_traces

# 自动修复路径的函数
def fix_path(path):
    # 判断操作系统类型
    if sys.platform == "win32":  # 如果是 Windows 系统
        # 将所有正斜杠替换为反斜杠
        return path.replace("/", "\\")
    elif sys.platform == "linux" or sys.platform == "linux2":  # 如果是 Linux 系统
        # 将所有反斜杠替换为正斜杠
        return path.replace("\\", "/")
    else:
        # 如果是其他操作系统（例如 MacOS），也使用正斜杠
        return path.replace("\\", "/")

# 处理 file_specifications_zhiyun 的函数
def process_file_specifications(file_specifications):
    file_specifications_new = []

    for item in file_specifications:
        # 修复路径
        fixed_path = fix_path(item[0])
        # 将修复后的路径添加到新的列表中
        file_specifications_new.append((fixed_path, item[1], item[2], item[3], item[4]))

    return file_specifications_new

from pathlib import Path, PureWindowsPath, PurePosixPath
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as pyo

def _normalize_rel(p: str) -> Path:
    """把相对路径字符串按 \ 或 / 切分为部件，再用本机 Path 重组；不改动中文和全角字符。"""
    s = str(p).strip().strip('"').strip("'").replace('\u00a0', ' ')
    pure = PureWindowsPath(s) if "\\" in s else PurePosixPath(s)
    return Path(*pure.parts)
def plot_impedance_nyquist(
    output_folder,
    file_specifications_zhiyun_old,
    file_specifications_gamry_old,
    file_name,
    csv_path=None,
    marker_size=6,
    # === 新增：刻度控制参数 ===
    tick_labelsize=18,
    tick_major_len=18,
    tick_major_w=1.2,
    tick_minor_len=3,
    tick_minor_w=1.0,
):
    """
    画 Nyquist 图，并在保存 PNG/HTML 的同时，导出 CSV（长表：source, label, Zreal, Zimag[, freq]）。
    file_specifications_*：列表，每项形如 (rel_folder, color, pattern, marker, label)
    """

    # 1) 目录准备（当前脚本上三级目录下的“校内测试”）
    base_dir = (Path(__file__).resolve().parent / ".." / ".." / ".." / "校内测试").resolve()
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2) 规范化输入规格，得到绝对路径 + 颜色等样式
    def _process_specs(specs_old):
        specs = []
        for item in specs_old:
            rel_folder, color, pattern, marker, label = item
            if color == 'default':
                color = get_ion_color(rel_folder)  # 你现有的取色函数
            abs_folder = (base_dir / _normalize_rel(rel_folder)).resolve()
            specs.append((str(abs_folder), color, pattern, marker, label))
        return process_file_specifications(specs)  # 你已有的函数

    file_specifications_zhiyun = _process_specs(file_specifications_zhiyun_old)
    file_specifications_gamry  = _process_specs(file_specifications_gamry_old)

    # 3) 绘制（matplotlib）
    global ax_nyquist
    fig_nyquist, ax_nyquist = plt.subplots(figsize=(12, 6))

    # 4) 生成 plotly traces（你现有的函数）
    plotly_traces_zhiyun = convert_and_plot_eis(file_specifications_zhiyun, 'zhiyun', marker_size=marker_size)
    plotly_traces_gamry  = convert_and_plot_eis(file_specifications_gamry,  'gamry',  marker_size=marker_size)

    # 5) Matplotlib 外观
    ax_nyquist.set_xlabel('Zreal (Ohm)', fontsize=22)
    ax_nyquist.set_ylabel('-Zimag (Ohm)', fontsize=22)

    # --- 已有：关闭网格 ---
    ax_nyquist.grid(False)

    # --- 已有：只保留 x/y 轴（左、下轴线）---
    ax_nyquist.spines['top'].set_visible(False)
    ax_nyquist.spines['right'].set_visible(False)

    ax_nyquist.tick_params(axis='x', bottom=True, top=False, labelbottom=True)
    ax_nyquist.tick_params(axis='y', left=True, right=False, labelleft=True)

    ax_nyquist.xaxis.set_ticks_position('bottom')
    ax_nyquist.yaxis.set_ticks_position('left')

    # === 新增：调整刻度“大小”（文字字号 + 刻度线长度/粗细）===
    ax_nyquist.tick_params(
        axis='both', which='major',
        labelsize=tick_labelsize,
        length=tick_major_len,
        width=tick_major_w
    )
    ax_nyquist.tick_params(
        axis='both', which='minor',
        labelsize=max(tick_labelsize - 2, 1),
        length=tick_minor_len,
        width=tick_minor_w
    )

    ax_nyquist.legend(fontsize=22, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
    fig_nyquist.tight_layout(rect=[0, 0, 1, 0.95])

    # 6) 保存静态图
    png_path = Path(output_folder) / file_name
    fig_nyquist.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"PNG 已保存到: {png_path}")

    plt.show()

    # 7) 交互 HTML
    layout = go.Layout(
        xaxis=dict(title='Zreal (Ohm)'),
        yaxis=dict(title='-Zimag (Ohm)'),
        title='Nyquist Plot (Combined)',
        legend=dict(orientation="h", y=-0.2)
    )
    fig = go.Figure(data=plotly_traces_zhiyun + plotly_traces_gamry, layout=layout)
    html_file_path = Path(output_folder) / "nyquist_interactive.html"
    pyo.plot(fig, filename=str(html_file_path), auto_open=False)
    print(f"交互 HTML 已保存到: {html_file_path}")

    # 8) 导出 CSV（从 plotly traces 抽取数据）
    rows = []
    def _collect(traces, source_name):
        for tr in traces:
            x = getattr(tr, "x", None)
            y = getattr(tr, "y", None)
            if x is None or y is None:
                continue
            label = getattr(tr, "name", source_name)

            freq_list = None
            if hasattr(tr, "customdata") and tr.customdata is not None:
                try:
                    cd = tr.customdata
                    if len(cd) == len(x):
                        if isinstance(cd[0], (list, tuple)) and len(cd[0]) >= 1:
                            freq_list = [row[0] for row in cd]
                        else:
                            freq_list = list(cd)
                except Exception:
                    pass

            for i in range(min(len(x), len(y))):
                row = {
                    "source": source_name,
                    "label": label,
                    "Zreal": x[i],
                    "Zimag": -y[i],
                }
                if freq_list is not None:
                    row["freq_Hz"] = freq_list[i]
                rows.append(row)

    _collect(plotly_traces_zhiyun, "zhiyun")
    _collect(plotly_traces_gamry,  "gamry")

    df = pd.DataFrame(rows)

    if csv_path is None:
        csv_path = png_path.with_suffix(".csv")
    else:
        csv_path = Path(csv_path)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV 已保存到: {csv_path}")


if __name__ == "__main__":
    # 这里指定输出文件夹
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    output_folder = f"{base_dir}/eis_t_plot"
    print("output_folder:",output_folder)



    file_specifications_zhiyun = [
        # 你可以继续添加其他文件，并为每个文件指定remove_first_n_points
        # (file_path, color, marker, remove_first_n_points)
        # 或者 (file_path, color, marker, display_name, remove_first_n_points)

        # (r"20240915_2ppm铜离子污染测试\旧版电解槽_firecloud\20240914_ion\循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv", 'default', '+', '2ppm Cu2+ 0h_0914_firecloud',1),
        # (r"20240915_2ppm铜离子污染测试\旧版电解槽_firecloud\20240914_ion\循环1／1_工步组2(工步组)(1／80)_工步3(阻抗).csv",'default', 's', '2ppm Cu2+ 2h_0914_firecloud',1),        # 更多文件...
        # (r"20240915_2ppm铜离子污染测试\旧版电解槽_firecloud\20240914_ion\循环1／1_工步组2(工步组)(6／80)_工步3(阻抗).csv",'default', 'o', '2ppm Cu2+ 12h',1),        # 更多文件...
        # (r"20240915_2ppm铜离子污染测试\旧版电解槽_firecloud\20240914_ion\循环1／1_工步组2(工步组)(12／80)_工步3(阻抗).csv",'default', '*', '2ppm Cu2+ 24h',1),  # 更多文件...
        # #
        # # # #
        # (r"20241001_2ppm铁离子污染测试\旧版电解槽_firecloud\20240931_ion-20241001-151034-默认1727882278521\循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv",  (255, 230, 204), '+', '2ppm Fe3+ 0h_0931_firecloud',1),
        # (r"20241001_2ppm铁离子污染测试\旧版电解槽_firecloud\20240931_ion-20241001-151034-默认1727882278521\循环1／1_工步组2(工步组)(1／80)_工步3(阻抗).csv", (255, 230, 204), 's', '2ppm Fe3+ 2h_0931_firecloud',1),        # 更多文件...
        # # # # (r"20241001_2ppm铁离子污染测试\旧版电解槽_firecloud\20240931_ion-20241001-151034-默认1727882278521\循环1／1_工步组2(工步组)(6／80)_工步3(阻抗).csv", (255, 230, 204), 'o', '2ppm Fe3+ 12h',1),        # 更多文件...
        # # (r"20241001_2ppm铁离子污染测试\旧版电解槽_firecloud\20240931_ion-20241001-151034-默认1727882278521\循环1／1_工步组2(工步组)(12／80)_工步3(阻抗).csv", (255, 230, 204), '*', '2ppm Fe3+ 24h',1),  # 更多文件...
        #
        # # # #
        # (r"20241003_2ppm镍离子污染测试\旧版电解槽_firecloud\20241003_ion-20241003-025748-默认1728123214071\循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv",  (173, 255, 204), '+', '2ppm Ni2+ 0h_1003_firecloud',1),
        # (r"20241003_2ppm镍离子污染测试\旧版电解槽_firecloud\20241003_ion-20241003-025748-默认1728123214071\循环1／1_工步组2(工步组)(1／80)_工步3(阻抗).csv", (173, 255, 204), 's', '2ppm Ni2+ 2h_1003_firecloud',1),        # 更多文件...
        # # # # (r"20241003_2ppm镍离子污染测试\旧版电解槽_firecloud\20241003_ion-20241003-025748-默认1728123214071\循环1／1_工步组2(工步组)(6／80)_工步3(阻抗).csv", (173, 255, 204), 'o', '2ppm Ni2+ 12h',1),        # 更多文件...
        # # (r"20241003_2ppm镍离子污染测试\旧版电解槽_firecloud\20241003_ion-20241003-025748-默认1728123214071\循环1／1_工步组2(工步组)(12／80)_工步3(阻抗).csv", (173, 255, 204), '*', '2ppm Ni2+ 24h',1),  # 更多文件...
        #
        # # #
        # (r"20241006_2ppm铬离子污染测试\旧版电解槽_firecloud\20241006_ion-20241006-122253-默认1728392350779\循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv", (200, 255, 200), '+', '2ppm Cr3+ 0h_1006_firecloud',1),
        # (r"20241006_2ppm铬离子污染测试\旧版电解槽_firecloud\20241006_ion-20241006-122253-默认1728392350779\循环1／1_工步组2(工步组)(1／80)_工步3(阻抗).csv",(200, 255, 200), 's', '2ppm Cr3+ 2h_1006_firecloud',1),        # 更多文件...
        # # # # (r"20241006_2ppm铬离子污染测试\旧版电解槽_firecloud\20241006_ion-20241006-122253-默认1728392350779\循环1／1_工步组2(工步组)(6／80)_工步3(阻抗).csv",(200, 255, 200), 'o', '2ppm Cr3+ 12h',1),        # 更多文件...
        # # (r"20241006_2ppm铬离子污染测试\旧版电解槽_firecloud\20241006_ion-20241006-122253-默认1728392350779\循环1／1_工步组2(工步组)(12／80)_工步3(阻抗).csv",(200, 255, 200), '*', '2ppm Cr3+ 24h',1),  # 更多文件...
        # #

        # #
        # (r"20241008_无离子污染测试\新版电解槽_firecloud\20241008_ion_column-20241008-145037-默认1728569517911\循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv",(192, 192, 192), '+', 'no_ion 0h_1008_firecloud', 1),
        # (r"20241008_无离子污染测试\新版电解槽_firecloud\20241008_ion_column-20241008-145037-默认1728569517911\循环1／1_工步组2(工步组)(1／80)_工步3(阻抗).csv",(192, 192, 192), 's', 'no_ion 2h_1008_firecloud', 1),  # 更多文件...
        # # # (r"20241008_无离子污染测试\新版电解槽_firecloud\20241008_ion_column-20241008-145037-默认1728569517911\循环1／1_工步组2(工步组)(6／80)_工步3(阻抗).csv",(192, 192, 192), 'o', 'no_ion 12h_1008_firecloud',1),        # 更多文件...
        # (r"20241008_无离子污染测试\新版电解槽_firecloud\20241008_ion_column-20241008-145037-默认1728569517911\循环1／1_工步组2(工步组)(12／80)_工步3(阻抗).csv",(192, 192, 192), '*', 'no_ion 24h_1008_firecloud',1),

        # #
        # (r"20241010_2ppm钠离子污染测试\新版电解槽_firecloud\20241011_ion-20241011-131456-默认1728834048305\循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv",  (255, 255, 200), '+', '2ppm Na+ 0h_1011_firecloud',1),
        # (r"20241010_2ppm钠离子污染测试\新版电解槽_firecloud\20241011_ion-20241011-131456-默认1728834048305\循环1／1_工步组2(工步组)(1／80)_工步3(阻抗).csv", (255, 255, 200), 's', '2ppm Na+ 2h_1011_firecloud',1),        # 更多文件...
        # # # (r"20241010_2ppm钠离子污染测试\新版电解槽_firecloud\20241011_ion-20241011-131456-默认1728834048305\循环1／1_工步组2(工步组)(6／80)_工步3(阻抗).csv", (255, 255, 200, 'o', '2ppm Na+ 12h',1),        # 更多文件...
        # # (r"20241010_2ppm钠离子污染测试\新版电解槽_firecloud\20241011_ion-20241011-131456-默认1728834048305\循环1／1_工步组2(工步组)(12／80)_工步3(阻抗).csv", (255, 255, 200, '*', '2ppm Na+ 24h',1),  # 更多文件...
        #
        # # #
        # (r"20241013_2ppm铝离子污染测试\新版电解槽_firecloud\20241015_ion-20241015-035648-默认1728987741540\循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv",  (180, 210, 230), '+', '2ppm Al+ 0h_1015_firecloud',1),  # 更多文件...
        # (r"20241013_2ppm铝离子污染测试\新版电解槽_firecloud\20241015_ion-20241015-035648-默认1728987741540\循环1／1_工步组2(工步组)(1／80)_工步3(阻抗).csv",  (180, 210, 230), 's', '2ppm Al+ 2h_1015_firecloud', 1),  # 更多文件...
        # # (r"20241013_2ppm铝离子污染测试\新版电解槽_firecloud\20241015_ion-20241015-035648-默认1728987741540\循环1／1_工步组2(工步组)(2／80)_工步3(阻抗).csv",  (180, 210, 230), 'o', '2ppm Al+ 4h',1),  # 更多文件...

        # (r"/home/cagalii/Application/autoeis/AutoEIS/examples/校内测试/20241020_2ppm镍离子污染和恢复测试/新版电解槽_firecloud/20241020_ion\循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv",  (180, 210, 230), '+', '2ppm Al+ 0h_1015_firecloud',1),  # 更多文件...
        # (r"/home/cagalii/Application/autoeis/AutoEIS/examples/校内测试/20241020_2ppm镍离子污染和恢复测试/新版电解槽_firecloud/20241020_ion\循环1／1_工步组2(工步组)(1／80)_工步3(阻抗).csv",  (180, 210, 230), 's', '2ppm Al+ 2h_1015_firecloud', 1),  # 更多文件...
        # # (r"20241013_2ppm铝离子污染测试\新版电解槽_firecloud\20241015_ion-20241015-035648-默认1728987741540\循环1／1_工步组2(工步组)(2／80)_工步3(阻抗).csv",  (180, 210, 230), 'o', '2ppm Al+ 4h',1),  # 更多文件...

        # (r"20241013_2ppm铝离子污染测试\新版电解槽_firecloud\20241015_ion-20241015-035648-默认1728987741540\循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv",  (180, 210, 230), '+', '2ppm Al+ 0h_1015_firecloud',1),  # 更多文件...
        # (r"20241013_2ppm铝离子污染测试\新版电解槽_firecloud\20241015_ion-20241015-035648-默认1728987741540\循环1／1_工步组2(工步组)(1／80)_工步3(阻抗).csv",  (180, 210, 230), 's', '2ppm Al+ 2h_1015_firecloud', 1),  # 更多文件...
        # # (r"20241013_2ppm铝离子污染测试\新版电解槽_firecloud\20241015_ion-20241015-035648-默认1728987741540\循环1／1_工步组2(工步组)(2／80)_工步3(阻抗).csv",  (180, 210, 230), 'o', '2ppm Al+ 4h',1),  # 更多文件...

        # (r"20241101_2ppm钙离子污染和恢复测试\新版电解槽_firecloud\20241101_ion\循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv",  'default', '+', '2ppm Ca2+ 0h_1101_firecloud',1),  # 更多文件...
        # (r"20241101_2ppm钙离子污染和恢复测试\新版电解槽_firecloud\20241101_ion\循环1／1_工步组2(工步组)(1／80)_工步3(阻抗).csv",  'default', 's', '2ppm Ca2+ 2h_1101_firecloud',1),  # 更多文件...
        # (r"20241101_2ppm钙离子污染和恢复测试\新版电解槽_firecloud\20241101_ion\循环1／1_工步组2(工步组)(6／80)_工步3(阻抗).csv",  'default', 'o', '2ppm Ca2+ 12h_1101_firecloud',1),  # 更多文件...
        # (r"20241101_2ppm钙离子污染和恢复测试\新版电解槽_firecloud\20241101_ion\循环1／1_工步组2(工步组)(12／80)_工步3(阻抗).csv",  'default', '*', '2ppm Ca2+ 24h_1101_firecloud',1),  # 更多文件...
        
        # (r"20241107_0.1ppm钙离子污染及恢复测试\新版电解槽_firecloud\20241107_ion\循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv",  'default', '+', '0.1ppm Ca2+ 0h_1107_firecloud',1),  # 更多文件...
        # (r"20241107_0.1ppm钙离子污染及恢复测试\新版电解槽_firecloud\20241107_ion\循环1／1_工步组2(工步组)(1／80)_工步3(阻抗).csv",  'default', 's', '0.1ppm Ca2+ 2h_1107_firecloud',1),  # 更多文件...
        # (r"20241107_0.1ppm钙离子污染及恢复测试\新版电解槽_firecloud\20241107_ion\循环1／1_工步组2(工步组)(6／80)_工步3(阻抗).csv",  'default', 'o', '0.1ppm Ca2+ 12h_1107_firecloud',1),  # 更多文件...
        # (r"20241107_0.1ppm钙离子污染及恢复测试\新版电解槽_firecloud\20241107_ion\循环1／1_工步组2(工步组)(12／80)_工步3(阻抗).csv",  'default', 'o', '0.1ppm Ca2+ 24h_1107_firecloud',1),  # 更多文件...
        
       
    ]

    file_specifications_zhiyun_special =[]

    file_specifications_gamry = [
        #
        # (r"20240813-0814_10ppm铜离子污染和恢复测试（拔去离子柱跑误差较大）\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240813_no_ion_column_34.DTA", (0, 128, 255), '+', '10ppm Cu2+ 0h_0813_gamry', 0),
        # (r"20240813-0814_10ppm铜离子污染和恢复测试（拔去离子柱跑误差较大）\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240814_ion_8.DTA", (0, 128, 255), 's', '10ppm Cu2+ 2h_0813_gamry', 0),
        # (r"20240813-0814_10ppm铜离子污染和恢复测试（拔去离子柱跑误差较大）\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240814_ion_32.DTA", (0, 128, 255), 'o', '10ppm Cu2+ 8h_0813_gamry', 0),
        #
        #
        # (r"20240815-0817_10ppm铁离子污染和恢复测试（拔去离子柱跑误差较大）\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240815_no_ion_column_39.DTA", (0, 128, 255), '+', '10ppm Fe3+ 0h_0815_gamry', 0),
        # (r"20240815-0817_10ppm铁离子污染和恢复测试（拔去离子柱跑误差较大）\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240815_ion_4.DTA", (204, 153, 102), 's', '10ppm Fe3+ 2h_0815_gamry', 0),
        # (r"20240815-0817_10ppm铁离子污染和恢复测试（拔去离子柱跑误差较大）\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240815_ion_32.DTA", (0, 128, 255), 'o', '10ppm Fe3+ 8h_0815_gamry', 0),
        #
        # (r"20240820_100ppm铜离子污染和恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240819_ion_column_38.DTA", (0, 128, 255), '+', '100ppm Cu2+ 0h_0820_gamry', 0),
        # (r"20240820_100ppm铜离子污染和恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240820_ion_4.DTA", (0, 128, 255), 's', '100ppm Cu2+ 2h_0820_gamry', 0),
        # (r"20240820_100ppm铜离子污染和恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240820_ion_32.DTA", (0, 128, 255), 'o', '100ppm Cu2+ 8h_0820_gamry', 0),

        #
        # r"20240822_10ppm铜离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240822_ion_0.DTA",
        # (102, 204, 255), '+', '10ppm Cu2+ 0h_0822_gamry', 0),
        # (
        # r"20240822_10ppm铜离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240822_ion_2.DTA",
        # (102, 204, 255), 's', '10ppm Cu2+ 2h_0822_gamry', 0),
        # (r"20240822_10ppm铜离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240822_ion_5.DTA",(102, 204, 255), 'o', '10ppm Cu2+ 12h_0822_gamry', 0),
        # (r"20240822_10ppm铜离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240822_ion_10.DTA",(102, 204, 255), '*', '10ppm Cu2+ 24h_0822_gamry', 0),

        #
        # (r"20240827_10ppm铬离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240828_ion_0.DTA", (102, 204, 102), '+', '10ppm Cr3+ 0h_0828_gamry', 0),
        # (r"20240827_10ppm铬离子污染和恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240828_ion_2.DTA", (102, 204, 102), 's', '10ppm Cr3+ 2h_0828_gamry', 0),
        # (r"20240827_10ppm铬离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240828_ion_5.DTA", (102, 204, 102), 'o', '10ppm Cr3+ 12h_0828_gamry', 0),
        # (r"20240827_10ppm铬离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240828_ion_10.DTA", (102, 204, 102), '*', '10ppm Cr3+ 24h_0828_gamry', 0),

        #
        # (r"20240831_10ppm镍离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240831_ion_0.DTA", (102, 204, 153), '+', '10ppm Ni2+ 0h_0831_gamry', 0),
        # (r"20240831_10ppm镍离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240831_ion_2.DTA", (102, 204, 153), 's', '10ppm Ni2+ 2h_0831_gamry', 0),
        # (r"20240831_10ppm镍离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240831_ion_5.DTA", (102, 204, 153), 'o', '10ppm Ni2+ 12h_0831_gamry', 0),
        # (r"20240831_10ppm镍离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240831_ion_10.DTA", (102, 204, 153), '*', '10ppm Ni2+ 24h_0831_gamry', 0),

        #
        # (r"20240907_10ppm铁离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240905_ion_0.DTA",(204, 153, 102), '+', '10ppm Fe3+ 0h_0905_gamry', 0),
        # (r"20240907_10ppm铁离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240905_ion_2.DTA",(204, 153, 102), 's', '10ppm Fe3+ 2h_0905_gamry', 0),
        # (r"20240907_10ppm铁离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240905_ion_5.DTA",(204, 153, 102), 'o', '10ppm Fe3+ 12h_0905_gamry', 0),
        # (r"20240907_10ppm铁离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240905_ion_10.DTA",(204, 153, 102), '*', '10ppm Fe3+ 24h_0905_gamry', 0),

        # (r"20240910_10ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240908_ion_0.DTA",'default', '+', '10ppm Ca2+ 0h_0908_gamry', 0),
        # (r"20240910_10ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240908_ion_2.DTA",'default', 's', '10ppm Ca2+ 2h_0908_gamry', 0),
        # (r"20240910_10ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240908_ion_5.DTA", (204, 153, 255), 'o', '10ppm Ca2+ 12h_0908_gamry',0),
        # (r"20240910_10ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240908_ion_10.DTA", (204, 153, 255), '*', '10ppm Ca2+ 24h_0908_gamry',0),

        # (r"20240915_2ppm铜离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240914_ion_0.DTA", (204, 255, 255), '+', '2ppm Cu2+ 0h_0914_gamry',0),
        # (r"20240915_2ppm铜离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240914_ion_1.DTA", (204, 255, 255), 's', '2ppm Cu2+ 2h_0914_gamry',0),
        # (r"20240915_2ppm铜离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240914_ion_5.DTA", (204, 255, 255), 'o', '2ppm Cu2+ 12h_0914_gamry',0),
        # (r"20240915_2ppm铜离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240914_ion_10.DTA", (204, 255, 255), '*', '2ppm Cu2+ 24h_0914_gamry',0),
        # # #
        # (r"20240918_2ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240917_ion_0.DTA", 'default', '+', '2ppm Ca2+ 0h_0917_gamry',0),
        # (r"20240918_2ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240917_ion_1.DTA",'default', 's', '2ppm Ca2+ 2h_0917_gamry',0),
        # (r"20240918_2ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240917_ion_5.DTA", (230, 200, 255), 'o', '2ppm Ca2+ 12h_0917_gamry',0),
        # (r"20240918_2ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240917_ion_10.DTA", (230, 200, 255), '*', '2ppm Ca2+ 24h_0917_gamry',0),
        # # #
        # (r"20241001_2ppm铁离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241001_ion_0.DTA",(255, 230, 204), '+', '2ppm Fe3+ 0h_1001_gamry',0),
        # (r"20241001_2ppm铁离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241001_ion_1.DTA",(255, 230, 204), 's', '2ppm Fe3+ 2h_1001_gamry',0),
        # # # # (r"20241001_2ppm铁离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241001_ion_5.DTA",(255, 230, 204), 'o', '2ppm Fe3+ 12h_1001_gamry',0),
        # # (r"20241001_2ppm铁离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241001_ion_10.DTA",(255, 230, 204), '*', '2ppm Fe3+ 24h_1001_gamry',0),
        # # #
        # (r"20241003_2ppm镍离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241003_ion_0.DTA", (173, 255, 204), '+', '2ppm Ni2+ 0h_1003_gamry',0),
        # (r"20241003_2ppm镍离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241003_ion_1.DTA", (173, 255, 204), 's', '2ppm Ni2+ 2h_1003_gamry',0),
        # # # (r"20241003_2ppm镍离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241003_ion_5.DTA", (173, 255, 204), 'o', '2ppm Ni2+ 12h_1003_gamry',0),
        # # (r"20241003_2ppm镍离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241003_ion_10.DTA", (173, 255, 204), '*', '2ppm Ni2+ 24h_1003_gamry',0),
        # #
        # (r"20241006_2ppm铬离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241006_ion_0.DTA",(200, 255, 200), '+', '2ppm Cr3+ 0h_1006_gamry',0),
        # (r"20241006_2ppm铬离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241006_ion_1.DTA",(200, 255, 200), 's', '2ppm Cr3+ 2h_1006_gamry',0),
        # # # (r"20241006_2ppm铬离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241006_ion_5.DTA",(200, 255, 200), 'o', '2ppm Cr3+ 12h_1006_gamry',0),
        # # (r"20241006_2ppm铬离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241006_ion_10.DTA",(200, 255, 200), '*', '2ppm Cr3+ 24h_1006_gamry',0),
        #
        # #
        # (r"20241017_2ppm铬离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241018_ion_0.DTA",(200, 255, 200), '+', '2ppm Cr3+ 0h_1017_gamry', 0),
        # (r"20241017_2ppm铬离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241018_ion_1.DTA",(200, 255, 200), 's', '2ppm Cr3+ 2h_1017_gamry',0),
        # # (r"20241017_2ppm铬离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241018_ion_5.DTA",(200, 255, 200), 'o', '2ppm Cr3+ 12h_1017_gamry',0),
        # # (r"20241017_2ppm铬离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241018_ion_10.DTA",(200, 255, 200), '*', '2ppm Cr3+ 24h_1017_gamry',0),
    
        # (r"20241028_2ppm钠离子污染和恢复测试/旧版电解槽_gamry/EISGALV_60℃_150ml_1A/cm2_20241029_ion_0.DTA",'default', '+', '2ppm Na+ 0h_1028_gamry', 0),
        # (r"20241028_2ppm钠离子污染和恢复测试/旧版电解槽_gamry/EISGALV_60℃_150ml_1A/cm2_20241029_ion_1.DTA",'default', 's', '2ppm Na+ 2h_1028_gamry',0),
        # # (r"20241028_2ppm钠离子污染和恢复测试/旧版电解槽_gamry/EISGALV_60℃_150ml_1A/cm2_20241029_ion_5.DTA",'default', 'o', '2ppm Na+ 12h_1028_gamry',0),
        # # (r"20241028_2ppm钠离子污染和恢复测试/旧版电解槽_gamry/EISGALV_60℃_150ml_1A/cm2_20241029_ion_10.DTA",'default', '*', '2ppm Na+ 24h_1028_gamry',0),
    
        
        
        #  (r"20241107_0.1ppm铬离子污染及恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241107_ion_0.DTA",'default', '+', '0.1ppm Cr3+ 0h_1107_gamry', 10),
        #  (r"20241107_0.1ppm铬离子污染及恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241107_ion_1.DTA",'default', '+', '0.1ppm Cr3+ 2h_1107_gamry', 10),
        #  (r"20241107_0.1ppm铬离子污染及恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241107_ion_5.DTA",'default', '+', '0.1ppm Cr3+ 12h_1107_gamry', 10),
        #  (r"20241107_0.1ppm铬离子污染及恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241107_ion_10.DTA",'default', '+', '0.1ppm Cr3+ 24h_1107_gamry', 10),
        #常用颜色变化：
        #  (r"20241112_2ppm钠离子污染和恢复测试80摄氏度/新版电解槽_gamry/EISGALV_80℃_150ml_1A\cm2_20241113_ion_0.DTA",'default', '+', '2ppm Na+ 80C 0h_1113_gamry', 0),
        #  (r"20241112_2ppm钠离子污染和恢复测试80摄氏度//新版电解槽_gamry/EISGALV_80℃_150ml_1A/cm2_20241113_ion_1.DTA",'default', '+', '2ppm Na+ 80C 2h_1113_gamry', 0),
        #  (r"20241112_2ppm钠离子污染和恢复测试80摄氏度/\新版电解槽_gamry\EISGALV_80℃_150ml_1A\cm2_20241113_ion_5.DTA",'default', '+', '2ppm Na+ 80C 12h_1113_gamry', 0),
        #  (r"20241112_2ppm钠离子污染和恢复测试80摄氏度/\新版电解槽_gamry\EISGALV_80℃_150ml_1A\cm2_20241113_ion_10.DTA",'default', '+', '2ppm Na+ 80C 24h_1113_gamry', 0),

        #  (r"20241112_2ppm镍离子污染及恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241113_ion_0.DTA",'default', '+', '2ppm Ni2+ 0h_1113_gamry', 10),
        #  (r"20241112_2ppm镍离子污染及恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241113_ion_1.DTA",'default', '+', '2ppm Ni2+ 0h_1113_gamry', 10),
        #  (r"20241112_2ppm镍离子污染及恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241113_ion_5.DTA",'default', '+', '2ppm Ni2+ 0h_1113_gamry', 10),
        #  (r"20241112_2ppm镍离子污染及恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241113_ion_10.DTA",'default', '+', '2ppm Ni2+ 0h_1113_gamry', 10),

        # (r"20241117_2ppm钠离子污染及恢复测试10mlmin/旧版电解槽_gamry/EISGALV_60℃_10ml_1A/cm2_20241119_ion_0.DTA",'default', '+', '2ppm Na+ 10ml 0h_1117_gamry', 0),
        # (r"20241117_2ppm钠离子污染及恢复测试10mlmin/旧版电解槽_gamry/EISGALV_60℃_10ml_1A/cm2_20241119_ion_1.DTA",'default', '+', '2ppm Na+ 10ml 2h_1117_gamry', 0),
        # (r"20241117_2ppm钠离子污染及恢复测试10mlmin/旧版电解槽_gamry/EISGALV_60℃_10ml_1A/cm2_20241119_ion_5.DTA",'default', '+', '2ppm Na+ 10ml 12h_1117_gamry', 0),
        # (r"20241117_2ppm钠离子污染及恢复测试10mlmin/旧版电解槽_gamry/EISGALV_60℃_10ml_1A/cm2_20241119_ion_10.DTA",'default', '+', '2ppm Na+ 10ml 24h_1117_gamry', 0),


        (r"20241117_2ppm钠离子污染及恢复测试40摄氏度/新版电解槽_gamry/EISGALV_40℃_150ml_1A/cm2_20241119_ion_0.DTA",'default', '+', '2ppm Na+ 40C 0h_1117_gamry', 0),
        (r"20241117_2ppm钠离子污染及恢复测试40摄氏度/新版电解槽_gamry/EISGALV_40℃_150ml_1A/cm2_20241119_ion_1.DTA",'default', '+', '2ppm Na+ 40C 2h_1117_gamry', 0),
        (r"20241117_2ppm钠离子污染及恢复测试40摄氏度/新版电解槽_gamry/EISGALV_40℃_150ml_1A/cm2_20241119_ion_5.DTA",'default', '+', '2ppm Na+ 40C 12h_1117_gamry', 0),
        (r"20241117_2ppm钠离子污染及恢复测试40摄氏度/新版电解槽_gamry/EISGALV_40℃_150ml_1A/cm2_20241119_ion_10.DTA",'default', '+', '2ppm Na+ 40C 24h_1117_gamry', 0),


        # (r"20241122_2ppm钠离子污染及恢复测试300mlmin/旧版电解槽_gamry/EISGALV_60℃_300ml_1A/cm2_20241123_ion_0.DTA",'default', '+', '2ppm Na+ 300ml 0h_1123_gamry', 0),
        # (r"20241122_2ppm钠离子污染及恢复测试300mlmin/旧版电解槽_gamry/EISGALV_60℃_300ml_1A/cm2_20241123_ion_1.DTA",'default', '+', '2ppm Na+ 300ml 2h_1123_gamry', 0),
        # (r"20241122_2ppm钠离子污染及恢复测试300mlmin/旧版电解槽_gamry/EISGALV_60℃_300ml_1A/cm2_20241123_ion_5.DTA",'default', '+', '2ppm Na+ 300ml 12h_1123_gamry', 0),
        # (r"20241122_2ppm钠离子污染及恢复测试300mlmin/旧版电解槽_gamry/EISGALV_60℃_300ml_1A/cm2_20241123_ion_10.DTA",'default', '+', '2ppm Na+ 300ml 24h_1123_gamry', 0),

        (r"20241122_2ppm铜离子污染及恢复测试40摄氏度\新版电解槽_gamry\EISGALV_40℃_150ml_1A/cm2_20241123_ion_0.DTA",'default', '+', '2ppm Cu2+ 40C 0h_1122_gamry', 10),
        (r"20241122_2ppm铜离子污染及恢复测试40摄氏度\新版电解槽_gamry\EISGALV_40℃_150ml_1A/cm2_20241123_ion_1.DTA",'default', '+', '2ppm Cu2+ 40C 2h_1122_gamry', 10),
        (r"20241122_2ppm铜离子污染及恢复测试40摄氏度\新版电解槽_gamry\EISGALV_40℃_150ml_1A/cm2_20241123_ion_5.DTA",'default', '+', '2ppm Cu2+ 40C 12h_1122_gamry', 10),
        (r"20241122_2ppm铜离子污染及恢复测试40摄氏度\新版电解槽_gamry\EISGALV_40℃_150ml_1A/cm2_20241123_ion_10.DTA",'default', '+', '2ppm Cu2+ 40C 24h_1122_gamry', 10),

    ]

    
    file_specifications_gamry_special =[

        #  (r"20241028_2ppm钠离子污染和恢复测试/旧版电解槽_gamry/EISGALV_60℃_150ml_1A/cm2_20241029_ion_0.DTA",(255, 165, 0), '+', '2ppm Na+ 0h_1028_gamry', 0),
        # (r"20241028_2ppm钠离子污染和恢复测试/旧版电解槽_gamry/EISGALV_60℃_150ml_1A/cm2_20241029_ion_1.DTA",(255, 165, 0), 's', '2ppm Na+ 2h_1028_gamry',0),
        # (r"20241028_2ppm钠离子污染和恢复测试/旧版电解槽_gamry/EISGALV_60℃_150ml_1A/cm2_20241029_ion_5.DTA",(255, 165, 0), 'o', '2ppm Na+ 12h_1028_gamry',0),
        # (r"20241028_2ppm钠离子污染和恢复测试/旧版电解槽_gamry/EISGALV_60℃_150ml_1A/cm2_20241029_ion_10.DTA",(255, 165, 0), '*', '2ppm Na+ 24h_1028_gamry',0),


        #  (r"20241112_2ppm钠离子污染和恢复测试80摄氏度/新版电解槽_gamry/EISGALV_80℃_150ml_1A\cm2_20241113_ion_0.DTA",(255, 0, 0), '+', '2ppm Na+ 80C 0h_1113_gamry', 0),
        #  (r"20241112_2ppm钠离子污染和恢复测试80摄氏度//新版电解槽_gamry/EISGALV_80℃_150ml_1A/cm2_20241113_ion_1.DTA",(255, 0, 0), 's', '2ppm Na+ 80C 2h_1113_gamry', 0),
        #  (r"20241112_2ppm钠离子污染和恢复测试80摄氏度/\新版电解槽_gamry\EISGALV_80℃_150ml_1A\cm2_20241113_ion_5.DTA",(255, 0, 0), 'o', '2ppm Na+ 80C 12h_1113_gamry', 0),
        #  (r"20241112_2ppm钠离子污染和恢复测试80摄氏度/\新版电解槽_gamry\EISGALV_80℃_150ml_1A\cm2_20241113_ion_10.DTA",(255, 0, 0), '*', '2ppm Na+ 80C 24h_1113_gamry', 0),


        # # (r"20241117_2ppm钠离子污染及恢复测试10mlmin/旧版电解槽_gamry/EISGALV_60℃_10ml_1A/cm2_20241119_ion_0.DTA",(255, 255, 0), '+', '2ppm Na+ 10ml 0h_1117_gamry', 0),
        # (r"20241117_2ppm钠离子污染及恢复测试10mlmin/旧版电解槽_gamry/EISGALV_60℃_10ml_1A/cm2_20241119_ion_1.DTA",(255, 255, 0), 's', '2ppm Na+ 10ml 2h_1117_gamry', 0),
        # # (r"20241117_2ppm钠离子污染及恢复测试10mlmin/旧版电解槽_gamry/EISGALV_60℃_10ml_1A/cm2_20241119_ion_5.DTA",(255, 255, 0), 'o', '2ppm Na+ 10ml 12h_1117_gamry', 0),
        # # (r"20241117_2ppm钠离子污染及恢复测试10mlmin/旧版电解槽_gamry/EISGALV_60℃_10ml_1A/cm2_20241119_ion_10.DTA",(255, 255, 0), '*', '2ppm Na+ 10ml 24h_1117_gamry', 0),


        (r"20241117_2ppm钠离子污染及恢复测试40摄氏度/新版电解槽_gamry/EISGALV_40℃_150ml_1A/cm2_20241119_ion_0.DTA",(0, 255, 0), '+', '2ppm Na+ 40C 0h_1117_gamry', 0),
        (r"20241117_2ppm钠离子污染及恢复测试40摄氏度/新版电解槽_gamry/EISGALV_40℃_150ml_1A/cm2_20241119_ion_1.DTA",(0, 255, 0), 's', '2ppm Na+ 40C 2h_1117_gamry', 0),
        (r"20241117_2ppm钠离子污染及恢复测试40摄氏度/新版电解槽_gamry/EISGALV_40℃_150ml_1A/cm2_20241119_ion_5.DTA",(0, 255, 0), 'o', '2ppm Na+ 40C 12h_1117_gamry', 0),
        (r"20241117_2ppm钠离子污染及恢复测试40摄氏度/新版电解槽_gamry/EISGALV_40℃_150ml_1A/cm2_20241119_ion_10.DTA",(0, 255, 0), '*', '2ppm Na+ 40C 24h_1117_gamry', 0),


        # (r"20241122_2ppm钠离子污染及恢复测试300mlmin/旧版电解槽_gamry/EISGALV_60℃_300ml_1A/cm2_20241123_ion_0.DTA",(0, 0, 255), '+', '2ppm Na+ 300ml 0h_1123_gamry', 0),
        # (r"20241122_2ppm钠离子污染及恢复测试300mlmin/旧版电解槽_gamry/EISGALV_60℃_300ml_1A/cm2_20241123_ion_1.DTA",(0, 0, 255), 's', '2ppm Na+ 300ml 2h_1123_gamry', 0),
        # (r"20241122_2ppm钠离子污染及恢复测试300mlmin/旧版电解槽_gamry/EISGALV_60℃_300ml_1A/cm2_20241123_ion_5.DTA",(0, 0, 255), 'o', '2ppm Na+ 300ml 12h_1123_gamry', 0),
        # (r"20241122_2ppm钠离子污染及恢复测试300mlmin/旧版电解槽_gamry/EISGALV_60℃_300ml_1A/cm2_20241123_ion_10.DTA",(0, 0, 255), '*', '2ppm Na+ 300ml 24h_1123_gamry', 0),

        (r"20241122_2ppm铜离子污染及恢复测试40摄氏度\新版电解槽_gamry\EISGALV_40℃_150ml_1A/cm2_20241123_ion_0.DTA",(0, 0, 255), '+', '2ppm Cu2+ 40C 0h_1122_gamry', 10),
        (r"20241122_2ppm铜离子污染及恢复测试40摄氏度\新版电解槽_gamry\EISGALV_40℃_150ml_1A/cm2_20241123_ion_1.DTA",(0, 0, 255), '+', '2ppm Cu2+ 40C 2h_1122_gamry', 10),
        (r"20241122_2ppm铜离子污染及恢复测试40摄氏度\新版电解槽_gamry\EISGALV_40℃_150ml_1A/cm2_20241123_ion_5.DTA",(0, 0, 255), '+', '2ppm Cu2+ 40C 12h_1122_gamry', 10),
        (r"20241122_2ppm铜离子污染及恢复测试40摄氏度\新版电解槽_gamry\EISGALV_40℃_150ml_1A/cm2_20241123_ion_10.DTA",(0, 0, 255), '+', '2ppm Cu2+ 40C 24h_1122_gamry', 10),
        ]
    
    
    # plot_impedance_nyquist(output_folder,file_specifications_zhiyun,file_specifications_gamry)

    plot_impedance_nyquist(output_folder,file_specifications_zhiyun_special,file_specifications_gamry_special)




