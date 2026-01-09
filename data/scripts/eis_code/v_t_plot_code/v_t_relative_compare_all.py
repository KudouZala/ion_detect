#本代码用于输入dta或excel文件，然后获取平均电压的变化
#绘制不同电压的相对变化大小比较
import os
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import os
import plotly.graph_objs as go
import plotly.offline as pyo
import os
import sys
import os
import platform
import pandas as pd
# 动态添加上两级目录到系统路径
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print('base_dir :',base_dir )
ion_color_path = os.path.join(base_dir, 'ion_color.py')

# 如果 ion_color.py 在该路径下，添加该目录到 sys.path
if os.path.exists(ion_color_path):
    sys.path.append(os.path.dirname(ion_color_path))

# 现在你可以导入 ion_color 中的函数了
from ion_color import get_ion_color

# RGB 转换为 matplotlib 颜色格式（归一化为0-1范围）
def rgb_to_mpl_color(rgb_tuple):
    return tuple([x / 255.0 for x in rgb_tuple])
from pathlib import Path, PureWindowsPath, PurePosixPath
import pandas as pd
import os
import pandas as pd
import matplotlib.pyplot as plt

# 颜色归一化函数
def normalize_color(color):
    """将 (255, 255, 100) 格式的 RGB 转换为 Matplotlib 可用的 [0, 1] 格式"""
    if isinstance(color, tuple) and len(color) == 3:
        return tuple(c / 255.0 for c in color)
    return color  # 如果不是 tuple，直接返回
baseline_voltage = None
relative_voltages = []

# 检测系统类型
is_windows = platform.system().lower() == "windows"
from pathlib import Path, PureWindowsPath, PurePosixPath
import pandas as pd

def _normalize_rel(p: str) -> Path:
    # 把相对路径字符串按其包含的分隔符解析成“部件”，再用本机的 Path 重组
    if "\\" in p:
        pure = PureWindowsPath(p)
    else:
        pure = PurePosixPath(p)
    return Path(*pure.parts)

def plot_relative_average_voltage_firecloud(group_folders_firecloud, time_interval=2):
    base_path = (Path(__file__).resolve().parent / ".." / ".." / ".." / "校内测试").resolve()

    results = {}
    time_hours = {}
    styles = {}

    for group_name, group in group_folders_firecloud.items():
        # 1) 规范 group['folder']，得到正确的目录
        folder_path = (base_path / _normalize_rel(group['folder'])).resolve()

        csv_files = group['csv_files']
        relative_voltages = []
        baseline_voltage = None

        for i, file in enumerate(csv_files):
            try:
                # 2) 规范 file，并与 folder_path 拼接
                full_path = (folder_path / _normalize_rel(file)).resolve()
                print(f"正在读取文件: {full_path}, index为:{i}")

                if not full_path.exists():
                    raise FileNotFoundError(str(full_path))

                # 如遇编码问题可尝试 encoding='utf-8' 或 'gbk'
                df = pd.read_csv(full_path)

                if 'Voltage/V' in df.columns:
                    avg_voltage = df['Voltage/V'].mean()
                    print(f"平均电压：{avg_voltage}")

                    if baseline_voltage is None:
                        baseline_voltage = avg_voltage
                        relative_voltages.append(0.0)
                    else:
                        relative_voltages.append(avg_voltage - baseline_voltage)
                else:
                    print("⚠️ 缺少列 'Voltage/V'，记为 None")
                    relative_voltages.append(None)

            except Exception as e:
                print(f"读取文件 {repr(file)} 时出错: {e}")
                relative_voltages.append(None)

        results[group_name] = relative_voltages
        time_hours[group_name] = [i * time_interval for i in range(len(csv_files))]

        # 提取绘图样式
        if group.get('color', None) == 'default':
            styles[group_name] = {
                'marker': group.get('marker', 'o'),
                'linestyle': group.get('linestyle', '-'),
                'color': normalize_color(get_ion_color(str(folder_path))),
                'label': group.get('label', f"{group_name}")
            }
        else:
            styles[group_name] = {
                'marker': group.get('marker', 'o'),
                'linestyle': group.get('linestyle', '-'),
                'color': normalize_color(group.get('color', None)),
                'label': group.get('label', f"{group_name}")
            }

    return results, time_hours, styles

def extract_voltage_data(lines, voltage_key='V', current_key='A', column_indices=None):
    """
    从文件行中提取电压数据，动态定位关键行和列索引。

    参数:
        lines (list): 文件的所有行。
        voltage_key (str): 表示电压的列关键字 (默认 'V')。
        current_key (str): 表示电流的列关键字 (默认 'A')。
        column_indices (dict): 列的索引映射，例如 {'D': 2, 'E': 3}。

    返回:
        list: 提取的电压数据。
    """
    if column_indices is None:
        column_indices = {'D': 2, 'E': 3}  # 默认列索引

    # 查找满足条件的起始行
    start_row = None
    for i, line in enumerate(lines):
        columns = line.strip().split()
        if len(columns) > max(column_indices.values()) and \
                columns[column_indices['D']] == voltage_key and \
                columns[column_indices['E']] == current_key:
            start_row = i + 1
            break

    if start_row is None:
        raise ValueError(f"未找到满足条件的行 (D列={voltage_key}, E列={current_key})")

    # 提取数据
    data = []
    for line in lines[start_row:]:
        columns = line.strip().split()
        try:
            # 假设数据在 D 列
            data.append(float(columns[column_indices['D']]))
        except (ValueError, IndexError):
            # 跳过无效行
            continue

    return data

from pathlib import Path, PureWindowsPath, PurePosixPath
import pandas as pd

def _normalize_rel(p: str) -> Path:
    """把相对路径字符串按其包含的分隔符解析为部件，再用本机 Path 重组。"""
    if p is None:
        return Path()
    s = str(p).strip().strip('"').strip("'").replace('\u00a0', ' ')  # 去首尾引号/空白，清理不换行空格
    pure = PureWindowsPath(s) if "\\" in s else PurePosixPath(s)
    return Path(*pure.parts)

def _read_lines_with_fallback(path: Path):
    """文本读取编码兜底：utf-8 -> gbk -> latin-1"""
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.readlines()
        except Exception:
            continue
    # 最后再忽略非法字符
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()

def plot_relative_average_voltage_gamry(group_folders_gamry, time_interval=2):
    # 当前脚本上三级目录下的“校内测试”
    base_path = (Path(__file__).resolve().parent / ".." / ".." / ".." / "校内测试").resolve()

    results = {}
    time_hours = {}
    styles = {}

    for group_name, group in group_folders_gamry.items():
        # 规范子目录
        folder_path = (base_path / _normalize_rel(group["folder"])).resolve()
        dta_files = group["dta_files"]

        relative_voltages = []
        baseline_voltage = None

        for i, file in enumerate(dta_files):
            try:
                # 规范文件名并拼接
                full_path = (folder_path / _normalize_rel(file)).resolve()
                print(f"正在读取文件: {full_path}")

                if not full_path.exists():
                    raise FileNotFoundError(str(full_path))

                # 读取文件行（带编码兜底）
                lines = _read_lines_with_fallback(full_path)

                # 提取电压并计算均值
                voltage_data = extract_voltage_data(lines)
                if not voltage_data:
                    raise ValueError("未能提取电压数据！")

                avg_voltage = sum(voltage_data) / len(voltage_data)
                print(f"文件 {file} 平均电压: {avg_voltage}")

                # 用第一个“成功读取”的文件作为基线
                if baseline_voltage is None:
                    baseline_voltage = avg_voltage
                    relative_voltages.append(0.0)
                else:
                    relative_voltages.append(avg_voltage - baseline_voltage)

            except Exception as e:
                print(f"读取文件 {repr(file)} 时出错: {e}")
                relative_voltages.append(None)

        results[group_name] = relative_voltages
        time_hours[group_name] = [i * time_interval for i in range(len(dta_files))]

        # 绘图样式（与 firecloud 保持一致）
        if group.get("color") == "default":
            styles[group_name] = {
                "marker": group.get("marker", "o"),
                "linestyle": group.get("linestyle", "-"),
                "color": normalize_color(get_ion_color(str(folder_path))),
                "label": group.get("label", f"{group_name}"),
            }
        else:
            styles[group_name] = {
                "marker": group.get("marker", "o"),
                "linestyle": group.get("linestyle", "-"),
                "color": normalize_color(group.get("color")),
                "label": group.get("label", f"{group_name}"),
            }

    return results, time_hours, styles


        

from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt
import re

import re

def format_ion_in_label(s: str) -> str:
    if not isinstance(s, str):
        return s

    # 1️⃣ 先处理 Al3+ / Ca2+ / Mg2+ 这种“有数字”的
    def repl_with_num(m: re.Match) -> str:
        elem = m.group(1)
        charge = m.group(2)
        return rf"{elem}$^{{{charge}+}}$"

    s = re.sub(r'([A-Za-z]{1,2})(\d+)\+', repl_with_num, s)

    # 2️⃣ 再处理 Na+ / K+ / H+ 这种“无数字，默认 +1”
    def repl_no_num(m: re.Match) -> str:
        elem = m.group(1)
        return rf"{elem}$^{{+}}$"

    s = re.sub(r'([A-Za-z]{1,2})\+', repl_no_num, s)

    return s

def plot_combined_results(
    firecloud_results, gamry_results,
    firecloud_times, gamry_times,
    firecloud_styles, gamry_styles,
    save_path=None, csv_path=None,
    xtick_fontsize: int = 22,
    ytick_fontsize: int = 22,
    tick_length: float = 6,
    tick_width: float = 1.5,
    minor_tick: bool = False,
):

    """
    绘制 Firecloud 和 Gamry 的相对平均电压曲线，并将用于绘图的数据保存为 CSV（长表格式）。

    参数:
        firecloud_results (dict[str, list[float|None]])
        gamry_results     (dict[str, list[float|None]])
        firecloud_times   (dict[str, list[float]])
        gamry_times       (dict[str, list[float]])
        firecloud_styles  (dict[str, dict])
        gamry_styles      (dict[str, dict])
        save_path         (str|Path|None): 图片保存路径（可选）
        csv_path          (str|Path|None): CSV 保存路径（可选）。若未提供但给了 save_path，则用同名 .csv
    """
    # ——— 1) 收集并导出 CSV（长表 tidy 格式） ———
    rows = []

    # Firecloud
    for group_name, values in firecloud_results.items():
        times = firecloud_times.get(group_name, [])
        # 对齐长度（取两者最短），避免越界
        n = min(len(times), len(values))
        for t, v in zip(times[:n], values[:n]):
            rows.append({
                "source": "Firecloud",
                "group": group_name,
                "time_hours": t,
                "relative_voltage": v
            })

    # Gamry
    for group_name, values in gamry_results.items():
        times = gamry_times.get(group_name, [])
        n = min(len(times), len(values))
        for t, v in zip(times[:n], values[:n]):
            rows.append({
                "source": "Gamry",
                "group": group_name,
                "time_hours": t,
                "relative_voltage": v
            })

    df = pd.DataFrame(rows)

    # 决定 CSV 输出路径
    csv_out = None
    if csv_path is not None:
        csv_out = Path(csv_path)
    elif save_path is not None:
        csv_out = Path(save_path).with_suffix(".csv")

    if csv_out is not None:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        # utf-8-sig 便于 Excel 直接打开中文不乱码
        df.to_csv(csv_out, index=False, encoding="utf-8-sig")
        print(f"CSV 已保存到: {csv_out}")

    # ——— 2) 绘图 ———
    fig, ax = plt.subplots(figsize=(10, 8))

    # Firecloud 曲线
    for group_name, values in firecloud_results.items():
        style = firecloud_styles[group_name]
        ax.plot(
            firecloud_times[group_name], values,
            marker=style.get('marker', 'o'),
            linestyle=style.get('linestyle', '-'),
            color=style.get('color', None),
            label=format_ion_in_label(style.get('label', group_name))

        )

    # Gamry 曲线
    for group_name, values in gamry_results.items():
        style = gamry_styles[group_name]
        ax.plot(
            gamry_times[group_name], values,
            marker=style.get('marker', 'o'),
            linestyle=style.get('linestyle', '-'),
            color=style.get('color', None),
            label=format_ion_in_label(style.get('label', group_name))

        )
    ax.set_xlabel('Time (hours)', fontsize=22)
    ax.set_ylabel('Relative Average Voltage (V)', fontsize=22)
    # ax.set_title('Relative Average Voltage vs Time', fontsize=22)

    # ax.set_xlabel('时间 (h)', fontsize=14)
    # ax.set_ylabel('相对电压变化 (V)', fontsize=14)
    # ax.set_title('相对电压变化 vs 时间', fontsize=16)
    # 不要网格
    ax.grid(False)

    # 只保留左、下两条坐标轴线（去掉方框）
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    # 统一控制刻度（字号、长度、粗细）
    ax.tick_params(axis="x", which="major", labelsize=xtick_fontsize, length=tick_length, width=tick_width)
    ax.tick_params(axis="y", which="major", labelsize=ytick_fontsize, length=tick_length, width=tick_width)

    # 如果你希望有 minor tick（可选）
    if minor_tick:
        ax.minorticks_on()
        ax.tick_params(axis="both", which="minor", length=tick_length * 0.6, width=max(1.0, tick_width * 0.8))




    # 图例列数：根据总曲线数自适应
    total_labels = len(firecloud_styles) + len(gamry_styles)
    ncol = max(1, math.ceil(total_labels / 2))  # 一行太挤就分两行
    # fig.legend(
    #     loc='lower center',
    #     fontsize=10,
    #     frameon=False,
    #     ncol=ncol,
    #     bbox_to_anchor=(0.5, -0.1)
    # )

    ION_ORDER = ["Al", "Ca", "Na", "Ni", "Cr", "Fe", "Cu"]
    ION_RANK = {ion: i for i, ion in enumerate(ION_ORDER)}

    handles, labels = ax.get_legend_handles_labels()

    def label_to_rank(lbl: str) -> int:
        # 1) 优先匹配 label 中的离子写法：Al3+ / Na+ / Ca2+（可在字符串任意位置）
        m = re.search(r'([A-Z][a-z]?)(?:\d+)?\+', lbl)
        if m:
            elem = m.group(1)  # Al / Na / Ca ...
            return ION_RANK.get(elem, 10**9)

        # 2) 兜底：匹配任意元素符号（避免 label 没有 '+'）
        m = re.search(r'([A-Z][a-z]?)', lbl)
        if m:
            elem = m.group(1)
            return ION_RANK.get(elem, 10**9)

        return 10**9  # 完全匹配不到就放最后

    # 稳定排序：同一 rank 的保持原出现顺序
    idx_sorted = sorted(range(len(labels)), key=lambda i: (label_to_rank(labels[i]), i))

    handles_sorted = [handles[i] for i in idx_sorted]
    labels_sorted  = [labels[i] for i in idx_sorted]

    ax.legend(
        handles_sorted, labels_sorted,
        loc="upper right",
        fontsize=24,
        frameon=True,
        borderaxespad=0.6,
    )


    fig.tight_layout()
    fig.subplots_adjust(bottom=0.05)  # 多给图例一些空间

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.10)
        print(f"图片已保存到: {save_path}")

    plt.show()

# 示例调用
if __name__ == "__main__":
    # Firecloud配置
    group_folders_firecloud = {
        # '2ppm Cu2+ 20240915': {
        #     'folder': r'20240915_2ppm铜离子污染测试\旧版电解槽_firecloud\20240914_ion',
        #     'csv_files': [
        #         r'循环1／1_工步组1(工步组)(1／1)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(1／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(2／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(3／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(4／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(5／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(6／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(7／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(8／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(9／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(10／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(11／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(12／80)_工步1(CC).csv',
        #         # r'循环1／1_工步组2(工步组)(13／80)_工步1(CC).csv',

        #     ],
        #     'marker': 'x', 
        #     'linestyle': '--', 
        #     'color': 'default', 
        #     'label': '2ppm Cu2+ 20240915'
        # },
        # '2ppm Fe3+ 20241001': {
        #     'folder': r'20241001_2ppm铁离子污染测试\旧版电解槽_firecloud\20240931_ion',
        #     'csv_files': [
        #         r'循环1／1_工步组1(工步组)(1／1)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(1／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(2／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(3／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(4／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(5／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(6／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(7／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(8／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(9／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(10／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(11／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(12／80)_工步1(CC).csv',
        #         # r'循环1／1_工步组2(工步组)(13／80)_工步1(CC).csv',

        #     ],
        #     'marker': 'x', 
        #     'linestyle': '--', 
        #     'color': 'default', 
        #     'label': '2ppm Fe3+ 20241001'
        # },
        # '2ppm Ni2+ 20241003': {
        #     'folder': r'20241003_2ppm镍离子污染测试\旧版电解槽_firecloud\20241003_ion',
        #     'csv_files': [
        #         r'循环1／1_工步组1(工步组)(1／1)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(1／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(2／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(3／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(4／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(5／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(6／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(7／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(8／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(9／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(10／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(11／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(12／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(13／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(14／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(15／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(16／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(17／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(18／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(19／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(20／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(21／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(22／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(23／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(24／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(25／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(26／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(27／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(28／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(29／80)_工步1(CC).csv',
        #         # r'循环1／1_工步组2(工步组)(30／80)_工步1(CC).csv',
        #     ],
        #     'marker': 'x', 
        #     'linestyle': '--', 
        #     'color': 'default', 
        #     'label': '2ppm Ni2+ 20241003'
        # },
        # '2ppm Cr3+ 20241006': {
        #     'folder': r'20241006_2ppm铬离子污染测试\旧版电解槽_firecloud\20241006_ion',
        #     'csv_files': [
        #         r'循环1／1_工步组1(工步组)(1／1)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(1／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(2／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(3／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(4／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(5／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(6／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(7／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(8／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(9／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(10／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(11／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(12／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(13／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(14／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(15／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(16／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(17／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(18／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(19／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(20／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(21／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(22／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(23／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(24／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(25／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(26／80)_工步1(CC).csv',
        #         # r'循环1／1_工步组2(工步组)(27／80)_工步1(CC).csv',

   
        #     ],
        #     'marker': 'x', 
        #     'linestyle': '--', 
        #     'color': 'default', 
        #     'label': '2ppm Cr3+ 20241006'
        # },

        # 'No ion 20241008': {
        #     'folder': r'20241008_无离子污染测试\新版电解槽_firecloud\20241008_ion_column',
        #     'csv_files': [
        #         r'循环1／1_工步组1(工步组)(1／1)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(1／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(2／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(3／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(4／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(5／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(6／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(7／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(8／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(9／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(10／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(11／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(12／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(13／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(14／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(15／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(16／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(17／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(18／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(19／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(20／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(21／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(22／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(23／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(24／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(25／80)_工步1(CC).csv',
        #         # r'循环1／1_工步组2(工步组)(26／80)_工步1(CC).csv',

   
        #     ],
        #     'marker': 'x', 
        #     'linestyle': '--', 
        #     'color': 'default', 
        #     'label': 'No ion 20241008'
        # },
        '2ppm Na+ 20241010': {
            'folder': r'20241010_2ppm钠离子污染测试\新版电解槽_firecloud\20241011_ion',
            'csv_files': [
                r'循环1／1_工步组1(工步组)(1／1)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(1／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(2／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(3／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(4／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(5／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(6／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(7／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(8／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(9／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(10／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(11／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(12／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(13／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(14／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(15／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(16／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(17／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(18／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(19／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(20／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(21／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(22／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(23／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(24／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(25／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(26／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(27／80)_工步1(CC).csv',
                # r'循环1／1_工步组2(工步组)(28／80)_工步1(CC).csv',
   
            ],
            'marker': 'x', 
            'linestyle': '--', 
            'color': 'default', 
            'label': '2ppm Na+ 20241010'
        },
        # '2ppm Al3+ 20241013': {
        #     'folder': r'20241013_2ppm铝离子污染测试\新版电解槽_firecloud\20241015_ion',
        #     'csv_files': [
        #         r'循环1／1_工步组1(工步组)(1／1)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(1／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(2／80)_工步1(CC).csv',
        #         # r'循环1／1_工步组2(工步组)(3／80)_工步1(CC).csv',
        #     ],
        #     'marker': 'x', 
        #     'linestyle': '--', 
        #     'color': 'default', 
        #     'label': '2ppm Al3+ 20241013'
        # },
        # '2ppm Ni2+ 20241020': {
        #     'folder': r'20241020_2ppm镍离子污染和恢复测试\新版电解槽_firecloud\20241020_ion',
        #     'csv_files': [
        #         r'循环1／1_工步组1(工步组)(1／1)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(1／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(2／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(3／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(4／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(5／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(6／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(7／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(8／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(9／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(10／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(11／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(12／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(13／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(14／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(15／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(16／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(17／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(18／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(19／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(20／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(21／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(22／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(23／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(24／80)_工步1(CC).csv',
        #         # r'循环1／1_工步组2(工步组)(25／80)_工步1(CC).csv',
   
        #     ],
        #     'marker': 'x', 
        #     'linestyle': '--', 
        #     'color': 'default', 
        #     'label': '2ppm Ni2+ 20241020'
        # },
        # '2ppm Fe3+ 20241024': {
        #     'folder': r'20241024_2ppm铁离子污染和恢复测试\新版电解槽_firecloud\20241025_ion',
        #     'csv_files': [
        #         r'循环1／1_工步组1(工步组)(1／1)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(1／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(2／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(3／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(4／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(5／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(6／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(7／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(8／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(9／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(10／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(11／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(12／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(13／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(14／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(15／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(16／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(17／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(18／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(19／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(20／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(21／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(22／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(23／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(24／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(25／80)_工步1(CC).csv',
        #         # r'循环1／1_工步组2(工步组)(26／80)_工步1(CC).csv',
   
        #     ],
        #     'marker': 'x', 
        #     'linestyle': '--', 
        #     'color': 'default', 
        #     'label': '2ppm Fe3+ 20241024'
        # },
        # '2ppm Fe3+ 20241029': {
        #     'folder': r'20241029_2ppm铁离子污染和恢复测试\新版电解槽_firecloud\20241029_ion',
        #     'csv_files': [
        #         r'循环1／1_工步组1(工步组)(1／1)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(1／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(2／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(3／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(4／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(5／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(6／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(7／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(8／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(9／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(10／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(11／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(12／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(13／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(14／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(15／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(16／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(17／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(18／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(19／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(20／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(21／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(22／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(23／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(24／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(25／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(26／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(27／80)_工步1(CC).csv',
        #         # r'循环1／1_工步组2(工步组)(28／80)_工步1(CC).csv',
        #     ],
        #     'marker': 'x', 
        #     'linestyle': '--', 
        #     'color': 'default', 
        #     'label': '2ppm Fe3+ 20241029'
        # },
        # '2ppm Ca2+ 20241101': {
        #     'folder': r'20241101_2ppm钙离子污染和恢复测试\新版电解槽_firecloud\20241101_ion',
        #     'csv_files': [
        #         r'循环1／1_工步组1(工步组)(1／1)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(1／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(2／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(3／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(4／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(5／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(6／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(7／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(8／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(9／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(10／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(11／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(12／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(13／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(14／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(15／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(16／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(17／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(18／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(19／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(20／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(21／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(22／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(23／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(24／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(25／80)_工步1(CC).csv',
        #         # r'循环1／1_工步组2(工步组)(26／80)_工步1(CC).csv',
        #     ],
        #     'marker': 'x', 
        #     'linestyle': '--', 
        #     'color': 'default', 
        #     'label': '2ppm Ca2+ 20241101'
        # },
        # '2ppm Cu2+ 20241101': {
        #     'folder': r'20241101_2ppm铜离子污染和恢复测试\旧版电解槽_firecloud\20241101_ion',
        #     'csv_files': [
        #         r'循环1／1_工步组1(工步组)(1／1)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(1／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(2／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(3／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(4／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(5／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(6／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(7／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(8／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(9／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(10／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(11／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(12／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(13／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(14／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(15／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(16／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(17／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(18／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(19／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(20／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(21／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(22／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(23／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(24／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(25／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(26／80)_工步1(CC).csv',
        #         # r'循环1／1_工步组2(工步组)(27／80)_工步1(CC).csv',
        #     ],
        #     'marker': 'x', 
        #     'linestyle': '--', 
        #     'color': 'default', 
        #     'label': '2ppm Cu2+ 20241101'
        # },
        # '0.1ppm Ca2+ 20241107': {
        #     'folder': r'20241107_0.1ppm钙离子污染及恢复测试\新版电解槽_firecloud\20241107_ion',
        #     'csv_files': [
        #         r'循环1／1_工步组1(工步组)(1／1)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(1／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(2／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(3／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(4／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(5／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(6／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(7／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(8／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(9／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(10／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(11／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(12／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(13／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(14／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(15／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(16／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(17／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(18／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(19／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(20／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(21／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(22／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(23／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(24／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(25／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(26／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(27／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(28／80)_工步1(CC).csv',
        #         # r'循环1／1_工步组2(工步组)(29／80)_工步1(CC).csv',
        #     ],
        #     'marker': 'o', 
        #     'linestyle': '-', 
        #     'color': 'default', 
        #     'label': '0.1ppm Ca2+ 20241107'
        # },
        # '2ppm Cu2+ 20241201': {
        #     'folder': r'20241201_2ppm铜离子污染及恢复测试\新版电解槽_firecloud\20241202_ion',
        #     'csv_files': [
        #         r'循环1／1_工步组1(工步组)(1／1)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(1／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(2／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(3／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(4／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(5／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(6／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(7／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(8／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(9／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(10／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(11／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(12／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(13／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(14／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(15／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(16／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(17／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(18／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(19／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(20／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(21／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(22／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(23／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(24／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(25／80)_工步1(CC).csv',
        #         # r'循环1／1_工步组2(工步组)(26／80)_工步1(CC).csv',
        #     ],
        #     'marker': 'o', 
        #     'linestyle': '-', 
        #     'color': 'default', 
        #     'label': '2ppm Cu2+ 20241201'
        # },
        
    }
##################################################################后面是gamry##################################################################
    # Gamry配置
    group_folders_gamry = {
        # '10ppm Cu2+ 20240822': {
        #     'folder': r'20240822_10ppm铜离子污染测试\新版电解槽_gamry\PWRGALVANOSTATIC_60℃_150ml_1A',
        #     'dta_files': [
        #         r'cm2_20240822_ion_0.DTA',
        #         r'cm2_20240822_ion_1.DTA',
        #         r'cm2_20240822_ion_2.DTA',
        #         r'cm2_20240822_ion_3.DTA',
        #         r'cm2_20240822_ion_4.DTA',
        #         r'cm2_20240822_ion_5.DTA',
        #         r'cm2_20240822_ion_6.DTA',
        #         r'cm2_20240822_ion_7.DTA',
        #         r'cm2_20240822_ion_8.DTA',
        #         r'cm2_20240822_ion_9.DTA',
        #         r'cm2_20240822_ion_10.DTA',
        #         # r'cm2_20240822_ion_11.DTA',
        #     ],
        #     'marker': 's', 
        #     'linestyle': ':', 
        #     'color': 'default', 
        #     'label': '10ppm Cu2+ 20240822'
        # },
        # # '10ppm Ca2+ 20240823': {
        # #     'folder': r'20240823_10ppm钙离子污染和恢复测试\新版电解槽_gamry\PWRGALVANOSTATIC_60℃_150ml_1A',
        # #     'dta_files': [
        # #         r'cm2_20240824_ion_0.DTA',
        # #         r'cm2_20240824_ion_1.DTA',
        # #         r'cm2_20240824_ion_2.DTA',
        # #         r'cm2_20240824_ion_3.DTA',
        # #         r'cm2_20240824_ion_4.DTA',
        # #         r'cm2_20240824_ion_5.DTA',
        # #         r'cm2_20240824_ion_6.DTA',
        # #         r'cm2_20240824_ion_7.DTA',
        # #         r'cm2_20240824_ion_8.DTA',
        # #         r'cm2_20240824_ion_9.DTA',
        # #         r'cm2_20240824_ion_10.DTA',
        # #         r'cm2_20240824_ion_11.DTA',
        # #         r'cm2_20240824_ion_12.DTA',
        # #         # r'cm2_20240824_ion_13.DTA',
        # #     ],
        # #     'marker': 's', 
        # #     'linestyle': ':', 
        # #     'color': 'default', 
        # #     'label': '10ppm Ca2+ 20240823'
        # # },
        # '10ppm Cr3+ 20240827': {
        #     'folder': r'20240827_10ppm铬离子污染和恢复测试\新版电解槽_gamry\PWRGALVANOSTATIC_60℃_150ml_1A',
        #     'dta_files': [
        #         r'cm2_20240828_ion_0.DTA',
        #         r'cm2_20240828_ion_1.DTA',
        #         r'cm2_20240828_ion_2.DTA',
        #         r'cm2_20240828_ion_3.DTA',
        #         r'cm2_20240828_ion_4.DTA',
        #         r'cm2_20240828_ion_5.DTA',
        #         r'cm2_20240828_ion_6.DTA',
        #         r'cm2_20240828_ion_7.DTA',
        #         r'cm2_20240828_ion_8.DTA',
        #         r'cm2_20240828_ion_9.DTA',
        #         r'cm2_20240828_ion_10.DTA',
        #         r'cm2_20240828_ion_11.DTA',
        #         # r'cm2_20240828_ion_12.DTA',
        #     ],
        #     'marker': 's', 
        #     'linestyle': ':', 
        #     'color': 'default', 
        #     'label': '10ppm Cr3+ 20240827'
        # },
        # '10ppm Ni2+ 20240831': {
        #     'folder': r'20240831_10ppm镍离子污染测试\新版电解槽_gamry\PWRGALVANOSTATIC_60℃_150ml_1A',
        #     'dta_files': [
        #         r'cm2_20240831_ion_0.DTA',
        #         r'cm2_20240831_ion_1.DTA',
        #         r'cm2_20240831_ion_2.DTA',
        #         r'cm2_20240831_ion_3.DTA',
        #         r'cm2_20240831_ion_4.DTA',
        #         r'cm2_20240831_ion_5.DTA',
        #         r'cm2_20240831_ion_6.DTA',
        #         r'cm2_20240831_ion_7.DTA',
        #         r'cm2_20240831_ion_8.DTA',
        #         r'cm2_20240831_ion_9.DTA',
        #         r'cm2_20240831_ion_10.DTA',
        #         r'cm2_20240831_ion_11.DTA',
        #         r'cm2_20240831_ion_12.DTA',
        #         r'cm2_20240831_ion_13.DTA',
        #         r'cm2_20240831_ion_14.DTA',
        #         r'cm2_20240831_ion_15.DTA',
        #         r'cm2_20240831_ion_16.DTA',
        #         r'cm2_20240831_ion_17.DTA',
        #         r'cm2_20240831_ion_18.DTA',
        #         r'cm2_20240831_ion_19.DTA',
        #         r'cm2_20240831_ion_20.DTA',
        #         r'cm2_20240831_ion_21.DTA',
        #         # r'cm2_20240831_ion_22.DTA',
        #     ],
        #     'marker': 's', 
        #     'linestyle': ':', 
        #     'color': 'default', 
        #     'label': '10ppm Ni2+ 20240831'
        # },
        # '10ppm Fe3+ 20240907': {
        #     'folder': r'20240907_10ppm铁离子污染测试\新版电解槽_gamry\PWRGALVANOSTATIC_60℃_150ml_1A',
        #     'dta_files': [
        #         r'cm2_20240905_ion_0.DTA',
        #         r'cm2_20240905_ion_1.DTA',
        #         r'cm2_20240905_ion_2.DTA',
        #         r'cm2_20240905_ion_3.DTA',
        #         r'cm2_20240905_ion_4.DTA',
        #         r'cm2_20240905_ion_5.DTA',
        #         r'cm2_20240905_ion_6.DTA',
        #         r'cm2_20240905_ion_7.DTA',
        #         r'cm2_20240905_ion_8.DTA',
        #         r'cm2_20240905_ion_9.DTA',
        #         r'cm2_20240905_ion_10.DTA',
        #         r'cm2_20240905_ion_11.DTA',
        #         r'cm2_20240905_ion_12.DTA',
        #         r'cm2_20240905_ion_13.DTA',
        #         r'cm2_20240905_ion_14.DTA',
        #         r'cm2_20240905_ion_15.DTA',
        #         r'cm2_20240905_ion_16.DTA',
        #         r'cm2_20240905_ion_17.DTA',
        #         r'cm2_20240905_ion_18.DTA',
        #         r'cm2_20240905_ion_19.DTA',
        #         r'cm2_20240905_ion_20.DTA',
        #         r'cm2_20240905_ion_21.DTA',
        #         # r'cm2_20240905_ion_22.DTA',
        #     ],
        #     'marker': 's', 
        #     'linestyle': ':', 
        #     'color': 'default', 
        #     'label': '10ppm Fe3+ 20240907'
        # },
        # '10ppm Ca2+ 20240910': {
        #     'folder': r'20240910_10ppm钙离子污染测试\新版电解槽_gamry\PWRGALVANOSTATIC_60℃_150ml_1A',
        #     'dta_files': [
        #         r'cm2_20240908_ion_0.DTA',
        #         r'cm2_20240908_ion_1.DTA',
        #         r'cm2_20240908_ion_2.DTA',
        #         r'cm2_20240908_ion_3.DTA',
        #         r'cm2_20240908_ion_4.DTA',
        #         r'cm2_20240908_ion_5.DTA',
        #         r'cm2_20240908_ion_6.DTA',
        #         r'cm2_20240908_ion_7.DTA',
        #         r'cm2_20240908_ion_8.DTA',
        #         r'cm2_20240908_ion_9.DTA',
        #         r'cm2_20240908_ion_10.DTA',
        #         r'cm2_20240908_ion_11.DTA',
        #         r'cm2_20240908_ion_12.DTA',
        #         r'cm2_20240908_ion_13.DTA',
        #         r'cm2_20240908_ion_14.DTA',
        #         r'cm2_20240908_ion_15.DTA',
        #         r'cm2_20240908_ion_16.DTA',
        #         r'cm2_20240908_ion_17.DTA',
        #         r'cm2_20240908_ion_18.DTA',
        #         r'cm2_20240908_ion_19.DTA',
        #         r'cm2_20240908_ion_20.DTA',
        #         r'cm2_20240908_ion_21.DTA',
        #         r'cm2_20240908_ion_22.DTA',
        #         r'cm2_20240908_ion_23.DTA',
        #         r'cm2_20240908_ion_24.DTA',
        #         r'cm2_20240908_ion_25.DTA',
        #         # r'cm2_20240908_ion_26.DTA',
        #     ],
        #     'marker': 's', 
        #     'linestyle': ':', 
        #     'color': 'default', 
        #     'label': '10ppm Ca2+ 20240910'
        # },
        # '2ppm Cu2+ 20240915': {
        #     'folder': r'20240915_2ppm铜离子污染测试\新版电解槽_gamry\PWRGALVANOSTATIC_60℃_150ml_1A',
        #     'dta_files': [
        #         r'cm2_20240914_ion_0.DTA',
        #         r'cm2_20240914_ion_1.DTA',
        #         r'cm2_20240914_ion_2.DTA',
        #         r'cm2_20240914_ion_3.DTA',
        #         r'cm2_20240914_ion_4.DTA',
        #         r'cm2_20240914_ion_5.DTA',
        #         r'cm2_20240914_ion_6.DTA',
        #         r'cm2_20240914_ion_7.DTA',
        #         r'cm2_20240914_ion_8.DTA',
        #         r'cm2_20240914_ion_9.DTA',
        #         r'cm2_20240914_ion_10.DTA',
        #         r'cm2_20240914_ion_11.DTA',
        #         # r'cm2_20240914_ion_12.DTA',
        #     ],
        #     'marker': 's', 
        #     'linestyle': ':', 
        #     'color': 'default', 
        #     'label': '2ppm Cu2+ 20240915'
        # },
        # '2ppm Ca2+ 20240918': {
        #     'folder': r'20240918_2ppm钙离子污染测试\新版电解槽_gamry\PWRGALVANOSTATIC_60℃_150ml_1A',
        #     'dta_files': [
        #         r'cm2_20240917_ion_0.DTA',
        #         r'cm2_20240917_ion_1.DTA',
        #         r'cm2_20240917_ion_2.DTA',
        #         r'cm2_20240917_ion_3.DTA',
        #         r'cm2_20240917_ion_4.DTA',
        #         r'cm2_20240917_ion_5.DTA',
        #         r'cm2_20240917_ion_6.DTA',
        #         r'cm2_20240917_ion_7.DTA',
        #         r'cm2_20240917_ion_8.DTA',
        #         r'cm2_20240917_ion_9.DTA',
        #         r'cm2_20240917_ion_10.DTA',
        #         r'cm2_20240917_ion_11.DTA',
        #         r'cm2_20240917_ion_12.DTA',
        #         r'cm2_20240917_ion_13.DTA',
        #         r'cm2_20240917_ion_14.DTA',
        #         r'cm2_20240917_ion_15.DTA',
        #         r'cm2_20240917_ion_16.DTA',
        #         r'cm2_20240917_ion_17.DTA',
        #         r'cm2_20240917_ion_18.DTA',
        #         r'cm2_20240917_ion_19.DTA',
        #         r'cm2_20240917_ion_20.DTA',
        #         # r'cm2_20240917_ion_21.DTA',
                
        #     ],
        #     'marker': 's', 
        #     'linestyle': ':', 
        #     'color': 'default', 
        #     'label': '2ppm Ca2+ 20240918'
        # },
        # '2ppm Fe3+ 20241001': {
        #     'folder': r'20241001_2ppm铁离子污染测试\新版电解槽_gamry\PWRGALVANOSTATIC_60℃_150ml_1A',
        #     'dta_files': [
        #         r'cm2_20241001_ion_0.DTA',
        #         r'cm2_20241001_ion_1.DTA',
        #         r'cm2_20241001_ion_2.DTA',
        #         r'cm2_20241001_ion_3.DTA',
        #         r'cm2_20241001_ion_4.DTA',
        #         r'cm2_20241001_ion_5.DTA',
        #         r'cm2_20241001_ion_6.DTA',
        #         r'cm2_20241001_ion_7.DTA',
        #         r'cm2_20241001_ion_8.DTA',
        #         r'cm2_20241001_ion_9.DTA',
        #         r'cm2_20241001_ion_10.DTA',
        #         r'cm2_20241001_ion_11.DTA',
        #         # r'cm2_20241001_ion_12.DTA',
                
        #     ],
        #     'marker': 's', 
        #     'linestyle': ':', 
        #     'color': 'default', 
        #     'label': '2ppm Fe3+ 20241001'
        # },
        # '2ppm Ni2+ 20241003': {
        #     'folder': r'20241003_2ppm镍离子污染测试\新版电解槽_gamry\PWRGALVANOSTATIC_60℃_150ml_1A',
        #     'dta_files': [
        #         r'cm2_20241003_ion_0.DTA',
        #         r'cm2_20241003_ion_1.DTA',
        #         r'cm2_20241003_ion_2.DTA',
        #         r'cm2_20241003_ion_3.DTA',
        #         r'cm2_20241003_ion_4.DTA',
        #         r'cm2_20241003_ion_5.DTA',
        #         r'cm2_20241003_ion_6.DTA',
        #         r'cm2_20241003_ion_7.DTA',
        #         r'cm2_20241003_ion_8.DTA',
        #         r'cm2_20241003_ion_9.DTA',
        #         r'cm2_20241003_ion_10.DTA',
        #         r'cm2_20241003_ion_11.DTA',
        #         r'cm2_20241003_ion_12.DTA',
        #         r'cm2_20241003_ion_13.DTA',
        #         r'cm2_20241003_ion_14.DTA',
        #         r'cm2_20241003_ion_15.DTA',
        #         r'cm2_20241003_ion_16.DTA',
        #         r'cm2_20241003_ion_17.DTA',
        #         r'cm2_20241003_ion_18.DTA',
        #         r'cm2_20241003_ion_19.DTA',
        #         r'cm2_20241003_ion_20.DTA',
        #         r'cm2_20241003_ion_21.DTA',
        #         r'cm2_20241003_ion_22.DTA',
        #         r'cm2_20241003_ion_23.DTA',
        #         r'cm2_20241003_ion_24.DTA',
        #         r'cm2_20241003_ion_25.DTA',
        #         # r'cm2_20241003_ion_26.DTA',
        #     ],
        #     'marker': 's', 
        #     'linestyle': ':', 
        #     'color': 'default', 
        #     'label': '2ppm Ni2+ 20241003'
        # },
        # # '2ppm Cr3+ 20241006': {
        # #     'folder': r'20241006_2ppm铬离子污染测试\新版电解槽_gamry\PWRGALVANOSTATIC_60℃_150ml_1A',
        # #     'dta_files': [
        # #         r'cm2_20241006_ion_0.DTA',
        # #         r'cm2_20241006_ion_1.DTA',
        # #         r'cm2_20241006_ion_2.DTA',
        # #         r'cm2_20241006_ion_3.DTA',
        # #         r'cm2_20241006_ion_4.DTA',
        # #         r'cm2_20241006_ion_5.DTA',
        # #         r'cm2_20241006_ion_6.DTA',
        # #         r'cm2_20241006_ion_7.DTA',
        # #         r'cm2_20241006_ion_8.DTA',
        # #         r'cm2_20241006_ion_9.DTA',
        # #         r'cm2_20241006_ion_10.DTA',
        # #         r'cm2_20241006_ion_11.DTA',
        # #         r'cm2_20241006_ion_12.DTA',
        # #         r'cm2_20241006_ion_13.DTA',
        # #         r'cm2_20241006_ion_14.DTA',
        # #         r'cm2_20241006_ion_15.DTA',
        # #         r'cm2_20241006_ion_16.DTA',
        # #         r'cm2_20241006_ion_17.DTA',
        # #         r'cm2_20241006_ion_18.DTA',
        # #         r'cm2_20241006_ion_19.DTA',
        # #         r'cm2_20241006_ion_20.DTA',
        # #         r'cm2_20241006_ion_21.DTA',
        # #         r'cm2_20241006_ion_22.DTA',
        # #         # r'cm2_20241006_ion_23.DTA',
        # #     ],
        # #     'marker': 's', 
        # #     'linestyle': ':', 
        # #     'color': 'default', 
        # #     'label': '2ppm Cr3+ 20241006'
        # # },
        # '2ppm Cr3+ 20241017': {
        #     'folder': r'20241017_2ppm铬离子污染和恢复测试\新版电解槽_gamry\PWRGALVANOSTATIC_60℃_150ml_1A',
        #     'dta_files': [
        #         r'cm2_20241018_ion_0.DTA',
        #         r'cm2_20241018_ion_1.DTA',
        #         r'cm2_20241018_ion_2.DTA',
        #         r'cm2_20241018_ion_3.DTA',
        #         r'cm2_20241018_ion_4.DTA',
        #         r'cm2_20241018_ion_5.DTA',
        #         r'cm2_20241018_ion_6.DTA',
        #         r'cm2_20241018_ion_7.DTA',
        #         r'cm2_20241018_ion_8.DTA',
        #         r'cm2_20241018_ion_9.DTA',
        #         r'cm2_20241018_ion_10.DTA',
        #         r'cm2_20241018_ion_11.DTA',
        #         r'cm2_20241018_ion_12.DTA',
        #         r'cm2_20241018_ion_13.DTA',
        #         r'cm2_20241018_ion_14.DTA',
        #         r'cm2_20241018_ion_15.DTA',
        #         r'cm2_20241018_ion_16.DTA',
        #         r'cm2_20241018_ion_17.DTA',
        #         r'cm2_20241018_ion_18.DTA',
        #         r'cm2_20241018_ion_19.DTA',
        #         r'cm2_20241018_ion_20.DTA',
        #         r'cm2_20241018_ion_21.DTA',
        #         # r'cm2_20241018_ion_22.DTA',
        #     ],
        #     'marker': 's', 
        #     'linestyle': ':', 
        #     'color': 'default', 
        #     'label': '2ppm Cr3+ 20241017'
        # },
        '2ppm Na+ 20241028': {
            'folder': r'20241028_2ppm钠离子污染和恢复测试\旧版电解槽_gamry\PWRGALVANOSTATIC_60℃_150ml_1A',
            'dta_files': [
                r'cm2_20241029_ion_0.DTA',
                r'cm2_20241029_ion_1.DTA',
                r'cm2_20241029_ion_2.DTA',
                r'cm2_20241029_ion_3.DTA',
                r'cm2_20241029_ion_4.DTA',
                r'cm2_20241029_ion_5.DTA',
                r'cm2_20241029_ion_6.DTA',
                r'cm2_20241029_ion_7.DTA',
                r'cm2_20241029_ion_8.DTA',
                r'cm2_20241029_ion_9.DTA',
                r'cm2_20241029_ion_10.DTA',
                r'cm2_20241029_ion_11.DTA',
                r'cm2_20241029_ion_12.DTA',
                r'cm2_20241029_ion_13.DTA',
                r'cm2_20241029_ion_14.DTA',
                r'cm2_20241029_ion_15.DTA',
                r'cm2_20241029_ion_16.DTA',
                r'cm2_20241029_ion_17.DTA',
                r'cm2_20241029_ion_18.DTA',
                r'cm2_20241029_ion_19.DTA',
                r'cm2_20241029_ion_20.DTA',
                r'cm2_20241029_ion_21.DTA',
                r'cm2_20241029_ion_22.DTA',
                r'cm2_20241029_ion_23.DTA',
                # r'cm2_20241029_ion_24.DTA',
            ],
            'marker': 's', 
            'linestyle': ':', 
            'color': 'default', 
            'label': '2ppm Na+ 20241028'
        },
        # # '0.1ppm Cr3+ 202411107': {
        # #     'folder': r'20241107_0.1ppm铬离子污染及恢复测试\旧版电解槽_gamry\PWRGALVANOSTATIC_60℃_150ml_1A',
        # #     'dta_files': [
        # #         r'cm2_20241107_ion_0.DTA',
        # #         r'cm2_20241107_ion_1.DTA',
        # #         r'cm2_20241107_ion_2.DTA',
        # #         r'cm2_20241107_ion_3.DTA',
        # #         r'cm2_20241107_ion_4.DTA',
        # #         r'cm2_20241107_ion_5.DTA',
        # #         r'cm2_20241107_ion_6.DTA',
        # #         r'cm2_20241107_ion_7.DTA',
        # #         r'cm2_20241107_ion_8.DTA',
        # #         r'cm2_20241107_ion_9.DTA',
        # #         r'cm2_20241107_ion_10.DTA',
        # #         r'cm2_20241107_ion_11.DTA',
        # #         r'cm2_20241107_ion_12.DTA',
        # #         r'cm2_20241107_ion_13.DTA',
        # #         r'cm2_20241107_ion_14.DTA',
        # #         r'cm2_20241107_ion_15.DTA',
        # #         r'cm2_20241107_ion_16.DTA',
        # #         r'cm2_20241107_ion_17.DTA',
        # #         r'cm2_20241107_ion_18.DTA',
        # #         r'cm2_20241107_ion_19.DTA',
        # #         r'cm2_20241107_ion_20.DTA',
        # #         r'cm2_20241107_ion_21.DTA',
        # #         r'cm2_20241107_ion_22.DTA',
        # #         r'cm2_20241107_ion_23.DTA',
        # #         r'cm2_20241107_ion_24.DTA',
        # #         # r'cm2_20241107_ion_25.DTA',
        # #     ],
        # #     'marker': 's', 
        # #     'linestyle': ':', 
        # #     'color': 'default', 
        # #     'label': '0.1ppm Cr3+ 202411107'
        # # },
        '2ppm Na+ 20241112 ': {
            'folder': r'20241112_2ppm钠离子污染和恢复测试80摄氏度\新版电解槽_gamry\PWRGALVANOSTATIC_80℃_150ml_1A',
            'dta_files': [
                r'cm2_20241113_ion_0.DTA',
                r'cm2_20241113_ion_1.DTA',
                r'cm2_20241113_ion_2.DTA',
                r'cm2_20241113_ion_3.DTA',
                r'cm2_20241113_ion_4.DTA',
                r'cm2_20241113_ion_5.DTA',
                r'cm2_20241113_ion_6.DTA',
                r'cm2_20241113_ion_7.DTA',
                r'cm2_20241113_ion_8.DTA',
                r'cm2_20241113_ion_9.DTA',
                r'cm2_20241113_ion_10.DTA',
                r'cm2_20241113_ion_11.DTA',
                r'cm2_20241113_ion_12.DTA',
                r'cm2_20241113_ion_13.DTA',
                r'cm2_20241113_ion_14.DTA',
                r'cm2_20241113_ion_15.DTA',
                r'cm2_20241113_ion_16.DTA',
                r'cm2_20241113_ion_17.DTA',
                r'cm2_20241113_ion_18.DTA',
                r'cm2_20241113_ion_19.DTA',
                r'cm2_20241113_ion_20.DTA',
                r'cm2_20241113_ion_21.DTA',
                r'cm2_20241113_ion_22.DTA',
                # r'cm2_20241113_ion_23.DTA',
            ],
            'marker': 's', 
            'linestyle': ':', 
            'color': 'default', 
            'label': '2ppm Na+ 20241112'
        },
        # # '2ppm Ni2+ 20241112 ': {
        # #     'folder': r'20241112_2ppm镍离子污染及恢复测试\旧版电解槽_gamry\PWRGALVANOSTATIC_60℃_150ml_1A',
        # #     'dta_files': [
        # #         r'cm2_20241113_ion_0.DTA',
        # #         r'cm2_20241113_ion_1.DTA',
        # #         r'cm2_20241113_ion_2.DTA',
        # #         r'cm2_20241113_ion_3.DTA',
        # #         r'cm2_20241113_ion_4.DTA',
        # #         r'cm2_20241113_ion_5.DTA',
        # #         r'cm2_20241113_ion_6.DTA',
        # #         r'cm2_20241113_ion_7.DTA',
        # #         r'cm2_20241113_ion_8.DTA',
        # #         r'cm2_20241113_ion_9.DTA',
        # #         r'cm2_20241113_ion_10.DTA',
        # #         r'cm2_20241113_ion_11.DTA',
        # #         r'cm2_20241113_ion_12.DTA',
        # #         r'cm2_20241113_ion_13.DTA',
        # #         r'cm2_20241113_ion_14.DTA',
        # #         r'cm2_20241113_ion_15.DTA',
        # #         r'cm2_20241113_ion_16.DTA',
        # #         r'cm2_20241113_ion_17.DTA',
        # #         r'cm2_20241113_ion_18.DTA',
        # #         r'cm2_20241113_ion_19.DTA',
        # #         r'cm2_20241113_ion_20.DTA',
        # #         r'cm2_20241113_ion_21.DTA',
        # #         r'cm2_20241113_ion_22.DTA',
        # #         # r'cm2_20241113_ion_23.DTA',
        # #     ],
        # #     'marker': 's', 
        # #     'linestyle': ':', 
        # #     'color': 'default', 
        # #     'label': '2ppm Ni2+ 20241112'
        # # },
        # # '2ppm Na+ 20241117 10ml/min': {
        # #     'folder': r'20241117_2ppm钠离子污染及恢复测试10mlmin\旧版电解槽_gamry\PWRGALVANOSTATIC_60℃_10ml_1A',
        # #     'dta_files': [
        # #         r'cm2_20241119_ion_0.DTA',
        # #         r'cm2_20241119_ion_1.DTA',
        # #         r'cm2_20241119_ion_2.DTA',
        # #         r'cm2_20241119_ion_3.DTA',
        # #         r'cm2_20241119_ion_4.DTA',
        # #         r'cm2_20241119_ion_5.DTA',
        # #         r'cm2_20241119_ion_6.DTA',
        # #         r'cm2_20241119_ion_7.DTA',
        # #         r'cm2_20241119_ion_8.DTA',
        # #         r'cm2_20241119_ion_9.DTA',
        # #         r'cm2_20241119_ion_10.DTA',
        # #         r'cm2_20241119_ion_11.DTA',
        # #         r'cm2_20241119_ion_12.DTA',
        # #         r'cm2_20241119_ion_13.DTA',
        # #         r'cm2_20241119_ion_14.DTA',
        # #         r'cm2_20241119_ion_15.DTA',
        # #         r'cm2_20241119_ion_16.DTA',
        # #         r'cm2_20241119_ion_17.DTA',
        # #         r'cm2_20241119_ion_18.DTA',
        # #         r'cm2_20241119_ion_19.DTA',
        # #         r'cm2_20241119_ion_20.DTA',
        # #         r'cm2_20241119_ion_21.DTA',
        # #         r'cm2_20241119_ion_22.DTA',
        # #         # r'cm2_20241119_ion_23.DTA',
        # #     ],
        # #     'marker': 's', 
        # #     'linestyle': ':', 
        # #     'color': 'default', 
        # #     'label': '2ppm Na+ 20241117 10ml/min'
        # # },
        # # '2ppm Na+ 20241117 40℃': {
        # #     'folder': r'20241117_2ppm钠离子污染及恢复测试40摄氏度\新版电解槽_gamry\PWRGALVANOSTATIC_40℃_150ml_1A',
        # #     'dta_files': [
        # #         r'cm2_20241119_ion_0.DTA',
        # #         r'cm2_20241119_ion_1.DTA',
        # #         r'cm2_20241119_ion_2.DTA',
        # #         r'cm2_20241119_ion_3.DTA',
        # #         r'cm2_20241119_ion_4.DTA',
        # #         r'cm2_20241119_ion_5.DTA',
        # #         r'cm2_20241119_ion_6.DTA',
        # #         r'cm2_20241119_ion_7.DTA',
        # #         r'cm2_20241119_ion_8.DTA',
        # #         r'cm2_20241119_ion_9.DTA',
        # #         r'cm2_20241119_ion_10.DTA',
        # #         r'cm2_20241119_ion_11.DTA',
        # #         r'cm2_20241119_ion_12.DTA',
        # #         r'cm2_20241119_ion_13.DTA',
        # #         r'cm2_20241119_ion_14.DTA',
        # #         r'cm2_20241119_ion_15.DTA',
        # #         r'cm2_20241119_ion_16.DTA',
        # #         r'cm2_20241119_ion_17.DTA',
        # #         r'cm2_20241119_ion_18.DTA',
        # #         r'cm2_20241119_ion_19.DTA',
        # #         r'cm2_20241119_ion_20.DTA',
        # #         r'cm2_20241119_ion_21.DTA',
        # #         r'cm2_20241119_ion_22.DTA',
        # #         # r'cm2_20241119_ion_23.DTA',
        # #     ],
        # #     'marker': 's', 
        # #     'linestyle': ':', 
        # #     'color': 'default', 
        # #     'label': '2ppm Na+ 20241117 40℃'
        # # },
        # # '2ppm Na+ 20241122 300ml/min': {
        # #     'folder': r'20241122_2ppm钠离子污染及恢复测试300mlmin\旧版电解槽_gamry\PWRGALVANOSTATIC_60℃_300ml_1A',
        # #     'dta_files': [
        # #         r'cm2_20241123_ion_0.DTA',
        # #         r'cm2_20241123_ion_1.DTA',
        # #         r'cm2_20241123_ion_2.DTA',
        # #         r'cm2_20241123_ion_3.DTA',
        # #         r'cm2_20241123_ion_4.DTA',
        # #         r'cm2_20241123_ion_5.DTA',
        # #         r'cm2_20241123_ion_6.DTA',
        # #         r'cm2_20241123_ion_7.DTA',
        # #         r'cm2_20241123_ion_8.DTA',
        # #         r'cm2_20241123_ion_9.DTA',
        # #         r'cm2_20241123_ion_10.DTA',
        # #         r'cm2_20241123_ion_11.DTA',
        # #         r'cm2_20241123_ion_12.DTA',
        # #         r'cm2_20241123_ion_13.DTA',
        # #         r'cm2_20241123_ion_14.DTA',
        # #         r'cm2_20241123_ion_15.DTA',
        # #         r'cm2_20241123_ion_16.DTA',
        # #         r'cm2_20241123_ion_17.DTA',
        # #         r'cm2_20241123_ion_18.DTA',
        # #         r'cm2_20241123_ion_19.DTA',
        # #         r'cm2_20241123_ion_20.DTA',
        # #         r'cm2_20241123_ion_21.DTA',
        # #         r'cm2_20241123_ion_22.DTA',
        # #         # r'cm2_20241123_ion_23.DTA',
        # #     ],
        # #     'marker': 's', 
        # #     'linestyle': ':', 
        # #     'color': 'default', 
        # #     'label': '2ppm Na+ 20241122 300ml/min'
        # # },
        # # '2ppm Cu2+ 20241122 40℃': {
        # #     'folder': r'20241122_2ppm铜离子污染及恢复测试40摄氏度\新版电解槽_gamry\PWRGALVANOSTATIC_40℃_150ml_1A',
        # #     'dta_files': [
        # #         r'cm2_20241123_ion_0.DTA',
        # #         r'cm2_20241123_ion_1.DTA',
        # #         r'cm2_20241123_ion_2.DTA',
        # #         r'cm2_20241123_ion_3.DTA',
        # #         r'cm2_20241123_ion_4.DTA',
        # #         r'cm2_20241123_ion_5.DTA',
        # #         r'cm2_20241123_ion_6.DTA',
        # #         r'cm2_20241123_ion_7.DTA',
        # #         r'cm2_20241123_ion_8.DTA',
        # #         r'cm2_20241123_ion_9.DTA',
        # #         r'cm2_20241123_ion_10.DTA',
        # #         r'cm2_20241123_ion_11.DTA',
        # #         r'cm2_20241123_ion_12.DTA',
        # #         r'cm2_20241123_ion_13.DTA',
        # #         r'cm2_20241123_ion_14.DTA',
        # #         r'cm2_20241123_ion_15.DTA',
        # #         r'cm2_20241123_ion_16.DTA',
        # #         r'cm2_20241123_ion_17.DTA',
        # #         r'cm2_20241123_ion_18.DTA',
        # #         r'cm2_20241123_ion_19.DTA',
        # #         r'cm2_20241123_ion_20.DTA',
        # #         r'cm2_20241123_ion_21.DTA',
        # #         r'cm2_20241123_ion_22.DTA',
        # #         # r'cm2_20241123_ion_23.DTA',
        # #     ],
        # #     'marker': 's', 
        # #     'linestyle': ':', 
        # #     'color': 'default', 
        # #     'label': '2ppm Cu2+ 20241122 40℃'
        # # },
    }

    # 处理Firecloud和Gamry数据
    firecloud_results, firecloud_times, firecloud_styles = plot_relative_average_voltage_firecloud(group_folders_firecloud)
    gamry_results, gamry_times, gamry_styles = plot_relative_average_voltage_gamry(group_folders_gamry)

    output_folder = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')),"图片输出文件夹")
    output_name = "voltage_relative_compare.png"#输出图片的文件名
    jpg_save_path = os.path.join(output_folder, output_name)
    # 绘制综合图
    plot_combined_results(firecloud_results, gamry_results, firecloud_times, gamry_times, firecloud_styles, gamry_styles,jpg_save_path)






