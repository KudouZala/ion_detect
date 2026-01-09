import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objs as go
import plotly.offline as pyo


import os
import sys
from nyquist_plot_zhiyun_and_gamry_all import plot_impedance_nyquist
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



if __name__ == "__main__":
    # 这里指定输出文件夹
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    output_folder = f"{base_dir}/eis_t_plot"
    print("output_folder:",output_folder)



    file_specifications_zhiyun = [
        
        # (r"20241001_2ppm铁离子污染测试\旧版电解槽_firecloud\20240931_ion-20241001-151034-默认1727882278521\循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv",  (255, 230, 204), '+', '2ppm Fe3+ 0h_0931_firecloud',1),
        # (r"20241001_2ppm铁离子污染测试\旧版电解槽_firecloud\20240931_ion-20241001-151034-默认1727882278521\循环1／1_工步组2(工步组)(1／80)_工步3(阻抗).csv", (255, 230, 204), 's', '2ppm Fe3+ 2h_0931_firecloud',1),        # 更多文件...
        # # # # (r"20241001_2ppm铁离子污染测试\旧版电解槽_firecloud\20240931_ion-20241001-151034-默认1727882278521\循环1／1_工步组2(工步组)(6／80)_工步3(阻抗).csv", (255, 230, 204), 'o', '2ppm Fe3+ 12h',1),        # 更多文件...
        # # (r"20241001_2ppm铁离子污染测试\旧版电解槽_firecloud\20240931_ion-20241001-151034-默认1727882278521\循环1／1_工步组2(工步组)(12／80)_工步3(阻抗).csv", (255, 230, 204), '*', '2ppm Fe3+ 24h',1),  # 更多文件...
        
        # #
        # (r"20241008_无离子污染测试\新版电解槽_firecloud\20241008_ion_column-20241008-145037-默认1728569517911\循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv",(192, 192, 192), '+', 'no_ion 0h_1008_firecloud', 1),
        # (r"20241008_无离子污染测试\新版电解槽_firecloud\20241008_ion_column-20241008-145037-默认1728569517911\循环1／1_工步组2(工步组)(1／80)_工步3(阻抗).csv",(192, 192, 192), 's', 'no_ion 2h_1008_firecloud', 1),  # 更多文件...
        # # # (r"20241008_无离子污染测试\新版电解槽_firecloud\20241008_ion_column-20241008-145037-默认1728569517911\循环1／1_工步组2(工步组)(6／80)_工步3(阻抗).csv",(192, 192, 192), 'o', 'no_ion 12h_1008_firecloud',1),        # 更多文件...
        # (r"20241008_无离子污染测试\新版电解槽_firecloud\20241008_ion_column-20241008-145037-默认1728569517911\循环1／1_工步组2(工步组)(12／80)_工步3(阻抗).csv",(192, 192, 192), '*', 'no_ion 24h_1008_firecloud',1),

        
    ]


    file_specifications_gamry = [
        
        # (r"20240815-0817_10ppm铁离子污染和恢复测试（拔去离子柱跑误差较大）\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240815_no_ion_column_39.DTA", (0, 128, 255), '+', '10ppm Fe3+ 0h_0815_gamry', 0),
        # (r"20240815-0817_10ppm铁离子污染和恢复测试（拔去离子柱跑误差较大）\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240815_ion_4.DTA", (204, 153, 102), 's', '10ppm Fe3+ 2h_0815_gamry', 0),
        # (r"20240815-0817_10ppm铁离子污染和恢复测试（拔去离子柱跑误差较大）\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240815_ion_32.DTA", (0, 128, 255), 'o', '10ppm Fe3+ 8h_0815_gamry', 0),
       
        # (r"20240907_10ppm铁离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240905_ion_0.DTA",(204, 153, 102), '+', '10ppm Fe3+ 0h_0905_gamry', 0),
        # (r"20240907_10ppm铁离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240905_ion_2.DTA",(204, 153, 102), 's', '10ppm Fe3+ 2h_0905_gamry', 0),
        # (r"20240907_10ppm铁离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240905_ion_5.DTA",(204, 153, 102), 'o', '10ppm Fe3+ 12h_0905_gamry', 0),
        # (r"20240907_10ppm铁离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240905_ion_10.DTA",(204, 153, 102), '*', '10ppm Fe3+ 24h_0905_gamry', 0),

        (r"20241001_2ppm铁离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241001_ion_0.DTA",(255, 230, 204), '+', '2ppm Fe3+ 0h_1001_gamry',0),
        (r"20241001_2ppm铁离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241001_ion_1.DTA",(255, 230, 204), 's', '2ppm Fe3+ 2h_1001_gamry',0),
        # # # (r"20241001_2ppm铁离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241001_ion_5.DTA",(255, 230, 204), 'o', '2ppm Fe3+ 12h_1001_gamry',0),
        # (r"20241001_2ppm铁离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241001_ion_10.DTA",(255, 230, 204), '*', '2ppm Fe3+ 24h_1001_gamry',0),
       
    ]

    
    
    file_name = "nyquist_combined_Fe.png"

    plot_impedance_nyquist(output_folder,file_specifications_zhiyun,file_specifications_gamry,file_name,marker_size=4)







