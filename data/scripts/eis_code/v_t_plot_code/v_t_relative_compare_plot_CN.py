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
import matplotlib
from matplotlib import rcParams

import matplotlib
from matplotlib import rcParams

# --- 强制注册系统里的 Noto CJK 字体，并设置为中文字体 ---
import os, glob
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager

# 1) 动态注册所有 NotoSansCJK*.ttc（你 fc-list 打印的这些路径就包含它们）
candidates = []
candidates += glob.glob("/usr/share/fonts/opentype/noto/NotoSansCJK-*.ttc")
candidates += glob.glob("/usr/share/fonts/truetype/noto/NotoSansCJK-*.ttc")
for p in candidates:
    try:
        font_manager.fontManager.addfont(p)
    except Exception as e:
        print("注册失败:", p, e)

# 2) 看看 Matplotlib 现在认识的字体里，有没有 Noto Sans CJK SC
families = sorted(set(f.name for f in font_manager.fontManager.ttflist))
print("可用字体族数量：", len(families))
hit = [f for f in families if "Noto Sans CJK SC" in f]
print("匹配到的 'Noto Sans CJK SC'：", hit[:5])

# 3) 指定字体（先用 SC；若没找到，就退而求其次用 JP/KR/TC，至少能显示 CJK）
target = None
for prefer in ["Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK KR", "Noto Sans CJK TC"]:
    if any(prefer == f for f in families):
        target = prefer
        break

if target is None:
    # 兜底：让 DejaVu 之外的 sans-serif 也能被搜索到
    target = "Noto Sans CJK SC"  # 仍然设置名字，下面配合 FontProperties 也能生效

rcParams["font.family"] = target
rcParams["axes.unicode_minus"] = False

print("最终使用字体：", target)

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
        # '2ppm Al3+ 20241013': {
        #     'folder': r'20241013_2ppm铝离子污染测试\新版电解槽_firecloud\20241015_ion',
        #     'csv_files': [
        #         r'循环1／1_工步组1(工步组)(1／1)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(1／80)_工步1(CC).csv',
        #         r'循环1／1_工步组2(工步组)(2／80)_工步1(CC).csv',
        #         # r'循环1／1_工步组2(工步组)(3／80)_工步1(CC).csv',
        #     ],
        #     'marker': 'x', 
        #     'linestyle': '-', 
        #     'color': 'default', 
        #     'label': 'Al3+ '
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
        #     'marker': '+', 
        #     'linestyle': '-', 
        #     'color': 'default', 
        #     'label': 'Ca2+'
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
        #     'marker': 'o', 
        #     'linestyle': '-', 
        #     'color': 'default', 
        #     'label': 'Cr3+'
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
        #     'marker': 'o', 
        #     'linestyle': '-', 
        #     'color': 'default', 
        #     'label': 'Cu2+'
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
        #     'marker': '*', 
        #     'linestyle': '-', 
        #     'color': 'default', 
        #     'label': 'Fe3+'
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
        #     'marker': '+', 
        #     'linestyle': '-', 
        #     'color': 'default', 
        #     'label': 'Ni2+'
        # },
        
        
    }
##################################################################后面是gamry##################################################################
    # Gamry配置
    group_folders_gamry = {
        
        # '2ppm Na+ 20241028': {
        #     'folder': r'20241028_2ppm钠离子污染和恢复测试\旧版电解槽_gamry\PWRGALVANOSTATIC_60℃_150ml_1A',
        #     'dta_files': [
        #         r'cm2_20241029_ion_0.DTA',
        #         r'cm2_20241029_ion_1.DTA',
        #         r'cm2_20241029_ion_2.DTA',
        #         r'cm2_20241029_ion_3.DTA',
        #         r'cm2_20241029_ion_4.DTA',
        #         r'cm2_20241029_ion_5.DTA',
        #         r'cm2_20241029_ion_6.DTA',
        #         r'cm2_20241029_ion_7.DTA',
        #         r'cm2_20241029_ion_8.DTA',
        #         r'cm2_20241029_ion_9.DTA',
        #         r'cm2_20241029_ion_10.DTA',
        #         r'cm2_20241029_ion_11.DTA',
        #         r'cm2_20241029_ion_12.DTA',
        #         r'cm2_20241029_ion_13.DTA',
        #         r'cm2_20241029_ion_14.DTA',
        #         r'cm2_20241029_ion_15.DTA',
        #         r'cm2_20241029_ion_16.DTA',
        #         r'cm2_20241029_ion_17.DTA',
        #         r'cm2_20241029_ion_18.DTA',
        #         r'cm2_20241029_ion_19.DTA',
        #         r'cm2_20241029_ion_20.DTA',
        #         r'cm2_20241029_ion_21.DTA',
        #         r'cm2_20241029_ion_22.DTA',
        #         r'cm2_20241029_ion_23.DTA',
        #         # r'cm2_20241029_ion_24.DTA',
        #     ],
        #     'marker': '+', 
        #     'linestyle': '-', 
        #     'color': 'default', 
        #     'label': 'Na+'
        # },
        
       
        
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

    output_name = "voltage_relative_compare_plot_CN.png"#输出图片的文件名
    jpg_save_path = os.path.join(output_folder, output_name)
    # 绘制综合图
    plot_combined_results(firecloud_results, gamry_results, firecloud_times, gamry_times, firecloud_styles, gamry_styles,jpg_save_path)