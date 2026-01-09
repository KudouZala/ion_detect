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

       
    ]

    file_specifications_zhiyun_special =[]

    file_specifications_gamry_special = [
        #
       
        #
        # (r"20240827_10ppm铬离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240828_ion_0.DTA", (102, 204, 102), '+', '10ppm Cr3+ 0h_0828_gamry', 0),
        # (r"20240827_10ppm铬离子污染和恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240828_ion_2.DTA", (102, 204, 102), 's', '10ppm Cr3+ 2h_0828_gamry', 0),
        # (r"20240827_10ppm铬离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240828_ion_5.DTA", (102, 204, 102), 'o', '10ppm Cr3+ 12h_0828_gamry', 0),
        # (r"20240827_10ppm铬离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20240828_ion_10.DTA", (102, 204, 102), '*', '10ppm Cr3+ 24h_0828_gamry', 0),

       
        # (r"20241006_2ppm铬离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241006_ion_0.DTA",(200, 255, 200), '+', '2ppm Cr3+ 0h_1006_gamry',0),
        # (r"20241006_2ppm铬离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241006_ion_1.DTA",(200, 255, 200), 's', '2ppm Cr3+ 2h_1006_gamry',0),
        # # # (r"20241006_2ppm铬离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241006_ion_5.DTA",(200, 255, 200), 'o', '2ppm Cr3+ 12h_1006_gamry',0),
        # # (r"20241006_2ppm铬离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241006_ion_10.DTA",(200, 255, 200), '*', '2ppm Cr3+ 24h_1006_gamry',0),
        #
        # #
        (r"20241017_2ppm铬离子污染和恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241018_ion_0.DTA",(200, 255, 200), '+', '2ppm Cr3+ 0h_1017_gamry', 0),
        (r"20241017_2ppm铬离子污染和恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241018_ion_1.DTA",(200, 255, 200), 's', '2ppm Cr3+ 2h_1017_gamry',0),
        (r"20241017_2ppm铬离子污染和恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241018_ion_5.DTA",(200, 255, 200), 'o', '2ppm Cr3+ 12h_1017_gamry',0),
        (r"20241017_2ppm铬离子污染和恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241018_ion_10.DTA",(200, 255, 200), '*', '2ppm Cr3+ 24h_1017_gamry',0),
    
       
        
        #  (r"20241107_0.1ppm铬离子污染及恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241107_ion_0.DTA",'default', '+', '0.1ppm Cr3+ 0h_1107_gamry', 10),
        #  (r"20241107_0.1ppm铬离子污染及恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241107_ion_1.DTA",'default', '+', '0.1ppm Cr3+ 2h_1107_gamry', 10),
        #  (r"20241107_0.1ppm铬离子污染及恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241107_ion_5.DTA",'default', '+', '0.1ppm Cr3+ 12h_1107_gamry', 10),
        #  (r"20241107_0.1ppm铬离子污染及恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A\cm2_20241107_ion_10.DTA",'default', '+', '0.1ppm Cr3+ 24h_1107_gamry', 10),
     
    ]

    
    
    # plot_impedance_nyquist(output_folder,file_specifications_zhiyun,file_specifications_gamry)

    file_name = "nyquist_combined_Cr.png"

    plot_impedance_nyquist(output_folder,file_specifications_zhiyun,file_specifications_gamry,file_name,marker_size=4)




