import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objs as go
import plotly.offline as pyo
import numpy as np

import os
import sys
from bode_plot_zhiyun_and_gamry import plot_impedance_bode

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


    file_name = "Fe"
    plot_impedance_bode(output_folder, file_specifications_zhiyun,file_specifications_gamry,file_name=file_name)