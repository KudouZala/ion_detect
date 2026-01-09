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
    output_folder = f"{base_dir}/eis_t_plot"
    print("output_folder:", output_folder)


    file_specifications_zhiyun = [
        # 你可以继续添加其他文件，并为每个文件指定remove_first_n_points
        # (file_path, color, marker, remove_first_n_points)
        # 或者 (file_path, color, marker, display_name, remove_first_n_points)

        # #
        # (r"20241013_2ppm铝离子污染测试\新版电解槽_firecloud\20241015_ion\循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv",  (160,160,160), 'x', '2ppm Al3+ 0h',1),  # 更多文件...
        # (r"20241013_2ppm铝离子污染测试\新版电解槽_firecloud\20241015_ion\循环1／1_工步组2(工步组)(1／80)_工步3(阻抗).csv",  (180,210,255), 'x', '2ppm Al3+ 2h', 1),  # 更多文件...
        # (r"20241013_2ppm铝离子污染测试\新版电解槽_firecloud\20241015_ion\循环1／1_工步组2(工步组)(2／80)_工步3(阻抗).csv",  (100,160,230), 'x', '2ppm Al3+ 4h',1),  # 更多文件...


    ]

    file_specifications_gamry = [ 
        # # === 铝离子 Al3+ (蓝色) ===
        (r"20250321_2ppm铝离子污染测试/旧版电解槽_gamry/EISGALV_60℃_150ml_1A/cm2_20250321_ion_0.DTA",(160,160,160),'x','2ppm Al3+ 0h',0),
        (r"20250321_2ppm铝离子污染测试/旧版电解槽_gamry/EISGALV_60℃_150ml_1A/cm2_20250321_ion_1.DTA",(180,210,255),'x','2ppm Al3+ 2h',0),
        (r"20250321_2ppm铝离子污染测试/旧版电解槽_gamry/EISGALV_60℃_150ml_1A/cm2_20250321_ion_2.DTA",(100,160,230),'x','2ppm Al3+ 4h',0),

        # === 钙离子 Ca2+ (蓝色) ===
        # (r"20240918_2ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240917_ion_0.DTA",(160,160,160),'+','2ppm Ca2+ 0h',0),
        # (r"20240918_2ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240917_ion_1.DTA",(180,210,255),'+','2ppm Ca2+ 2h',0),
        # (r"20240918_2ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240917_ion_2.DTA",(100,160,230),'+','2ppm Ca2+ 4h',0),
        # (r"20240918_2ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240917_ion_3.DTA",(40,80,180),'+','2ppm Ca2+ 6h',0),
        #     # === 钠离子 Na+ (黄色) ===
        # (r"20241117_2ppm钠离子污染及恢复测试40摄氏度/新版电解槽_gamry/EISGALV_40℃_150ml_1A/cm2_20241119_ion_0.DTA",(160,160,160),'+','2ppm Na+ 0h',0),
        # (r"20241117_2ppm钠离子污染及恢复测试40摄氏度/新版电解槽_gamry/EISGALV_40℃_150ml_1A/cm2_20241119_ion_1.DTA",(255,240,180),'+','2ppm Na+ 2h',0),
        # (r"20241117_2ppm钠离子污染及恢复测试40摄氏度/新版电解槽_gamry/EISGALV_40℃_150ml_1A/cm2_20241119_ion_2.DTA",(230,200,100),'+','2ppm Na+ 4h',0),
        # (r"20241117_2ppm钠离子污染及恢复测试40摄氏度/新版电解槽_gamry/EISGALV_40℃_150ml_1A/cm2_20241119_ion_3.DTA",(180,160,40),'+','2ppm Na+ 6h',0),

    #     # === 镍离子 Ni2+ (绿色) ===
        # (r"20241003_2ppm镍离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241003_ion_0.DTA",(160,160,160),'+','2ppm Ni2+ 0h',0),
        # (r"20241003_2ppm镍离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241003_ion_1.DTA",(200,255,200),'+','2ppm Ni2+ 2h',0),
        # (r"20241003_2ppm镍离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241003_ion_2.DTA",(100,230,100),'+','2ppm Ni2+ 4h',0),
        # (r"20241003_2ppm镍离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241003_ion_3.DTA",(40,160,40),'+','2ppm Ni2+ 6h',0),
    # ]

    #     # === 铬离子 Cr3+ (红色) ===
        # (r"20241017_2ppm铬离子污染和恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241018_ion_0.DTA",(160,160,160),'o','2ppm Cr3+ 0h',0),
        # (r"20241017_2ppm铬离子污染和恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241018_ion_1.DTA",(255,180,180),'o','2ppm Cr3+ 2h',0),
        # (r"20241017_2ppm铬离子污染和恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241018_ion_2.DTA",(230,100,100),'o','2ppm Cr3+ 4h',0),
        # (r"20241017_2ppm铬离子污染和恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241018_ion_3.DTA",(180,40,40),'o','2ppm Cr3+ 6h',0),

    #     # === 铜离子 Cu2+ (紫色) ===
        # (r"20241122_2ppm铜离子污染及恢复测试40摄氏度\新版电解槽_gamry\EISGALV_40℃_150ml_1A/cm2_20241123_ion_0.DTA",(160,160,160),'o','2ppm Cu2+ 0h',0),
        # (r"20241122_2ppm铜离子污染及恢复测试40摄氏度\新版电解槽_gamry\EISGALV_40℃_150ml_1A/cm2_20241123_ion_1.DTA",(220,180,255),'o','2ppm Cu2+ 2h',0),
        # (r"20241122_2ppm铜离子污染及恢复测试40摄氏度\新版电解槽_gamry\EISGALV_40℃_150ml_1A/cm2_20241123_ion_2.DTA",(180,100,230),'o','2ppm Cu2+ 4h',0),
        # (r"20241122_2ppm铜离子污染及恢复测试40摄氏度\新版电解槽_gamry\EISGALV_40℃_150ml_1A/cm2_20241123_ion_3.DTA",(120,40,160),'o','2ppm Cu2+ 6h',0),

    #     # === 铁离子 Fe3+ (棕色) ===
        # (r"20241001_2ppm铁离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241001_ion_0.DTA",(160,160,160),'o','2ppm Fe3+ 0h',0),
        # (r"20241001_2ppm铁离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241001_ion_1.DTA",(210,180,140),'o','2ppm Fe3+ 2h',0),
        # (r"20241001_2ppm铁离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241001_ion_2.DTA",(160,120,60),'o','2ppm Fe3+ 4h',0),
        # (r"20241001_2ppm铁离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241001_ion_3.DTA",(100,60,20),'o','2ppm Fe3+ 6h',0),

    ]



    file_name = "Al"
    plot_impedance_bode(output_folder, file_specifications_zhiyun,file_specifications_gamry,file_name=file_name)
