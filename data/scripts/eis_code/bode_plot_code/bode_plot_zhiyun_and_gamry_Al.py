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
        # (file_path, color, marker, remove_first_n_points)
        (r"20241013_2ppm铝离子污染测试\新版电解槽_firecloud\20241015_ion\循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv",  (173, 216, 230), 'o', '2ppm Al3+ 0h',1),  # 更多文件...
        (r"20241013_2ppm铝离子污染测试\新版电解槽_firecloud\20241015_ion\循环1／1_工步组2(工步组)(1／80)_工步3(阻抗).csv",  (70, 130, 180), 's', '2ppm Al3+ 2h', 1),  # 更多文件...
        (r"20241013_2ppm铝离子污染测试\新版电解槽_firecloud\20241015_ion\循环1／1_工步组2(工步组)(2／80)_工步3(阻抗).csv",  (0, 0, 139), '+', '2ppm Al3+ 4h',1),  # 更多文件...
        
         ]

    file_specifications_gamry = [
        # 可以在这里添加Gamry的数据
    ]


    file_name = "Al"
    plot_impedance_bode(output_folder, file_specifications_zhiyun,file_specifications_gamry,file_name=file_name)
