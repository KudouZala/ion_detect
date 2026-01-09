import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objs as go
import plotly.offline as pyo

import sys
import os
import os
import glob

# 将库的绝对路径添加到系统路径中
lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '最新_绘图和输出文件代码_更新位置', 'eis_code', 'nyquist_plot_code')


sys.path.append(lib_path)

from nyquist_plot_zhiyun_and_gamry import plot_impedance_nyquist

if __name__ == "__main__":
    # 这里指定输出文件夹
    current_folder = os.path.dirname(__file__)
    parent_folder = os.path.dirname(current_folder)  # 获取当前文件夹的父级文件夹
    gamry_folder = [d for d in os.listdir(parent_folder) if d.endswith('_gamry')][0]

    output_folder= f"{parent_folder}/{gamry_folder}/output_plot_nyquist"
    file_specifications_zhiyun = [
        # 你可以继续添加其他文件，并为每个文件指定remove_first_n_points
        # (file_path, color, marker, remove_first_n_points)
        # 或者 (file_path, color, marker, display_name, remove_first_n_points)

        
        # # # #

    ]


        # 获取文件夹路径
    ion_numbers = [0, 1, 6, 12]
    file_specifications_gamry = []

    for ion in ion_numbers:
        path = glob.glob(f"{parent_folder}/{gamry_folder}/EISGALV_60℃_150ml_1A/*_ion_{ion}.DTA")
        if path:  # 确保找到了匹配的文件
            time=ion*2
            file_specifications_gamry.append(
                (path[0], (255, 255, 200), '+', f'2ppm Na+ {time}h_1028_gamry', 0)
            )


    plot_impedance_nyquist(output_folder, file_specifications_zhiyun, file_specifications_gamry)



