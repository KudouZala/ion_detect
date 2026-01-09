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
        # 你可以继续添加其他文件，并为每个文件指定remove_first_n_points
        # (file_path, color, marker, remove_first_n_points)
        # 或者 (file_path, color, marker, display_name, remove_first_n_points)

        # (r"20240915_2ppm铜离子污染测试\旧版电解槽_firecloud\20240914_ion\循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv", 'default', '+', '2ppm Cu2+ 0h_0914_firecloud',1),
        # (r"20240915_2ppm铜离子污染测试\旧版电解槽_firecloud\20240914_ion\循环1／1_工步组2(工步组)(1／80)_工步3(阻抗).csv",'default', 's', '2ppm Cu2+ 2h_0914_firecloud',1),        # 更多文件...
        # (r"20240915_2ppm铜离子污染测试\旧版电解槽_firecloud\20240914_ion\循环1／1_工步组2(工步组)(6／80)_工步3(阻抗).csv",'default', 'o', '2ppm Cu2+ 12h',1),        # 更多文件...
        # (r"20240915_2ppm铜离子污染测试\旧版电解槽_firecloud\20240914_ion\循环1／1_工步组2(工步组)(12／80)_工步3(阻抗).csv",'default', '*', '2ppm Cu2+ 24h',1),  # 更多文件...
        # #
        
        # #
        # (r"20241008_无离子污染测试\新版电解槽_firecloud\20241008_ion_column-20241008-145037-默认1728569517911\循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv",(192, 192, 192), '+', 'no_ion 0h_1008_firecloud', 1),
        # (r"20241008_无离子污染测试\新版电解槽_firecloud\20241008_ion_column-20241008-145037-默认1728569517911\循环1／1_工步组2(工步组)(1／80)_工步3(阻抗).csv",(192, 192, 192), 's', 'no_ion 2h_1008_firecloud', 1),  # 更多文件...
        # # # (r"20241008_无离子污染测试\新版电解槽_firecloud\20241008_ion_column-20241008-145037-默认1728569517911\循环1／1_工步组2(工步组)(6／80)_工步3(阻抗).csv",(192, 192, 192), 'o', 'no_ion 12h_1008_firecloud',1),        # 更多文件...
        # (r"20241008_无离子污染测试\新版电解槽_firecloud\20241008_ion_column-20241008-145037-默认1728569517911\循环1／1_工步组2(工步组)(12／80)_工步3(阻抗).csv",(192, 192, 192), '*', 'no_ion 24h_1008_firecloud',1),

        
    ]

    file_specifications_zhiyun_special =[]

    file_specifications_gamry = [
        #
        # (r"20240813-0814_10ppm铜离子污染和恢复测试（拔去离子柱跑误差较大）\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240813_no_ion_column_34.DTA", (0, 128, 255), '+', '10ppm Cu2+ 0h_0813_gamry', 0),
        # (r"20240813-0814_10ppm铜离子污染和恢复测试（拔去离子柱跑误差较大）\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240814_ion_8.DTA", (0, 128, 255), 's', '10ppm Cu2+ 2h_0813_gamry', 0),
        # (r"20240813-0814_10ppm铜离子污染和恢复测试（拔去离子柱跑误差较大）\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240814_ion_32.DTA", (0, 128, 255), 'o', '10ppm Cu2+ 8h_0813_gamry', 0),
        #
        #
       
        # (r"20240820_100ppm铜离子污染和恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240819_ion_column_38.DTA", (0, 128, 255), '+', '100ppm Cu2+ 0h_0820_gamry', 0),
        # (r"20240820_100ppm铜离子污染和恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240820_ion_4.DTA", (0, 128, 255), 's', '100ppm Cu2+ 2h_0820_gamry', 0),
        # (r"20240820_100ppm铜离子污染和恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240820_ion_32.DTA", (0, 128, 255), 'o', '100ppm Cu2+ 8h_0820_gamry', 0),

        #
        # r"20240822_10ppm铜离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240822_ion_0.DTA",  (102, 204, 255), '+', '10ppm Cu2+ 0h_0822_gamry', 0),
        # (r"20240822_10ppm铜离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240822_ion_2.DTA", (102, 204, 255), 's', '10ppm Cu2+ 2h_0822_gamry', 0),
        # (r"20240822_10ppm铜离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240822_ion_5.DTA",(102, 204, 255), 'o', '10ppm Cu2+ 12h_0822_gamry', 0),
        # (r"20240822_10ppm铜离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240822_ion_10.DTA",(102, 204, 255), '*', '10ppm Cu2+ 24h_0822_gamry', 0),

        #
        
        # (r"20240915_2ppm铜离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240914_ion_0.DTA", (204, 255, 255), '+', '2ppm Cu2+ 0h_0914_gamry',0),
        # (r"20240915_2ppm铜离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240914_ion_1.DTA", (204, 255, 255), 's', '2ppm Cu2+ 2h_0914_gamry',0),
        # (r"20240915_2ppm铜离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240914_ion_5.DTA", (204, 255, 255), 'o', '2ppm Cu2+ 12h_0914_gamry',0),
        # (r"20240915_2ppm铜离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240914_ion_10.DTA", (204, 255, 255), '*', '2ppm Cu2+ 24h_0914_gamry',0),
        # # #
        
        (r"20241122_2ppm铜离子污染及恢复测试40摄氏度\新版电解槽_gamry\EISGALV_40℃_150ml_1A/cm2_20241123_ion_0.DTA",'default', '+', '2ppm Cu2+ 40C 0h_1122_gamry', 10),
        (r"20241122_2ppm铜离子污染及恢复测试40摄氏度\新版电解槽_gamry\EISGALV_40℃_150ml_1A/cm2_20241123_ion_1.DTA",'default', '+', '2ppm Cu2+ 40C 2h_1122_gamry', 10),
        (r"20241122_2ppm铜离子污染及恢复测试40摄氏度\新版电解槽_gamry\EISGALV_40℃_150ml_1A/cm2_20241123_ion_5.DTA",'default', '+', '2ppm Cu2+ 40C 12h_1122_gamry', 10),
        (r"20241122_2ppm铜离子污染及恢复测试40摄氏度\新版电解槽_gamry\EISGALV_40℃_150ml_1A/cm2_20241123_ion_10.DTA",'default', '+', '2ppm Cu2+ 40C 24h_1122_gamry', 10),


    ]

    
    
    file_name = "nyquist_combined_Cu.png"

    plot_impedance_nyquist(output_folder,file_specifications_zhiyun,file_specifications_gamry,file_name,marker_size=4)






