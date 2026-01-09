from v_t_relative_compare_all import plot_relative_average_voltage_firecloud,plot_combined_results,plot_relative_average_voltage_gamry
import os
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

# 动态添加上两级目录到系统路径
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print('base_dir :',base_dir )
ion_color_path = os.path.join(base_dir, 'ion_color.py')

# 如果 ion_color.py 在该路径下，添加该目录到 sys.path
if os.path.exists(ion_color_path):
    sys.path.append(os.path.dirname(ion_color_path))

# 现在你可以导入 ion_color 中的函数了
from ion_color import get_ion_color

import os
import pandas as pd
import matplotlib.pyplot as plt

def export_plot_data_to_csv(
    results: dict,
    times: dict,
    styles: dict,
    csv_path: str,
    source: str,
) -> None:
    """将用于绘图的曲线数据导出为 long-format CSV。

    期望 results/times/styles 的 key 均为 group_name（如 '2ppm Al3+ 20241013'）。
    - results[group] : 1D 序列（相对电压或电压均值曲线）
    - times[group]   : 1D 序列（与 results 等长的时间轴；若缺失则用 0..N-1）
    - styles[group]  : dict（可选，用于记录 marker/linestyle/color/label 等）
    """
    rows = []
    for group, y in (results or {}).items():
        # 允许 y 是 list/np.ndarray/pd.Series
        y_list = list(y) if y is not None else []
        t = (times or {}).get(group, None)
        t_list = list(t) if t is not None else list(range(len(y_list)))

        # 对齐长度（以最短为准，避免越界）
        n = min(len(y_list), len(t_list))
        style = (styles or {}).get(group, {}) or {}

        for i in range(n):
            rows.append(
                {
                    "source": source,
                    "group": group,
                    "label": style.get("label", group),
                    "index": i,
                    "time": t_list[i],
                    "relative_voltage": y_list[i],
                    "marker": style.get("marker", None),
                    "linestyle": style.get("linestyle", None),
                    "color": style.get("color", None),
                }
            )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[CSV] 已保存绘图数据: {csv_path} (rows={len(df)})")
# 示例调用
if __name__ == "__main__":
    # Firecloud配置
    group_folders_firecloud = {

        'No ion 20241008': {
            'folder': r'20241008_无离子污染测试\新版电解槽_firecloud\20241008_ion_column',
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
                # r'循环1／1_工步组2(工步组)(26／80)_工步1(CC).csv',

   
            ],
            'marker': 'o', 
            'linestyle': '-', 
            'color': 'default', 
            'label': 'No ion '
        },
        '2ppm Al3+ 20241013': {
            'folder': r'20241013_2ppm铝离子污染测试\新版电解槽_firecloud\20241015_ion',
            'csv_files': [
                r'循环1／1_工步组1(工步组)(1／1)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(1／80)_工步1(CC).csv',
                r'循环1／1_工步组2(工步组)(2／80)_工步1(CC).csv',
                # r'循环1／1_工步组2(工步组)(3／80)_工步1(CC).csv',
            ],
            'marker': 'x', 
            'linestyle': '-', 
            'color': 'default', 
            'label': 'Al3+ '
        },
        '2ppm Ca2+ 20241101': {
            'folder': r'20241101_2ppm钙离子污染和恢复测试\新版电解槽_firecloud\20241101_ion',
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
                # r'循环1／1_工步组2(工步组)(26／80)_工步1(CC).csv',
            ],
            'marker': '+', 
            'linestyle': '-', 
            'color': 'default', 
            'label': 'Ca2+'
        },
        '2ppm Cr3+ 20241006': {
            'folder': r'20241006_2ppm铬离子污染测试\旧版电解槽_firecloud\20241006_ion',
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
                # r'循环1／1_工步组2(工步组)(27／80)_工步1(CC).csv',

   
            ],
            'marker': 'o', 
            'linestyle': '-', 
            'color': 'default', 
            'label': 'Cr3+'
        },

        '2ppm Cu2+ 20241101': {
            'folder': r'20241101_2ppm铜离子污染和恢复测试\旧版电解槽_firecloud\20241101_ion',
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
                # r'循环1／1_工步组2(工步组)(27／80)_工步1(CC).csv',
            ],
            'marker': 'o', 
            'linestyle': '-', 
            'color': 'default', 
            'label': 'Cu2+'
        },
        '2ppm Fe3+ 20241029': {
            'folder': r'20241029_2ppm铁离子污染和恢复测试\新版电解槽_firecloud\20241029_ion',
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
            'marker': '*', 
            'linestyle': '-', 
            'color': 'default', 
            'label': 'Fe3+'
        },
        '2ppm Ni2+ 20241020': {
            'folder': r'20241020_2ppm镍离子污染和恢复测试\新版电解槽_firecloud\20241020_ion',
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
                # r'循环1／1_工步组2(工步组)(25／80)_工步1(CC).csv',
   
            ],
            'marker': '+', 
            'linestyle': '-', 
            'color': 'default', 
            'label': 'Ni2+'
        },
        
        
    }
##################################################################后面是gamry##################################################################
    # Gamry配置
    group_folders_gamry = {
        
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
            'marker': '+', 
            'linestyle': '-', 
            'color': 'default', 
            'label': 'Na+'
        },
        
       
        
    }

    # 处理Firecloud和Gamry数据
    firecloud_results, firecloud_times, firecloud_styles = plot_relative_average_voltage_firecloud(group_folders_firecloud)
    gamry_results, gamry_times, gamry_styles = plot_relative_average_voltage_gamry(group_folders_gamry)

    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 定位到上三级目录
    parent_3 = os.path.abspath(os.path.join(current_dir, "../../.."))
    # 拼接 volt_t_plot 文件夹路径
    output_folder = os.path.join(parent_3, "volt_t_plot")
    print("输出目录:", output_folder)

        # 合并导出一份总表，便于后续统计/画图
    export_plot_data_to_csv(
        {**(firecloud_results or {}), **(gamry_results or {})},
        {**(firecloud_times or {}), **(gamry_times or {})},
        {**(firecloud_styles or {}), **(gamry_styles or {})},
        os.path.join(output_folder, "voltage_relative_compare_all.csv"),
        source="mixed",
    )

    output_name = "voltage_relative_compare_plot.png"#输出图片的文件名
    jpg_save_path = os.path.join(output_folder, output_name)
    # 绘制综合图
    plot_combined_results(firecloud_results, gamry_results, firecloud_times, gamry_times, firecloud_styles, gamry_styles,jpg_save_path)