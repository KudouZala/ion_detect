import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objs as go
import plotly.offline as pyo

import sys
import os

# 将库的绝对路径添加到系统路径中
lib_path = r'/最新_绘图和输出文件代码_更新位置/eis_code/nyquist_plot_code'
sys.path.append(lib_path)

from nyquist_plot_zhiyun_and_gamry import plot_impedance_nyquist

if __name__ == "__main__":
    # 这里指定输出文件夹
    output_folder= r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241029_2ppm铁离子污染和恢复测试\新版电解槽_firecloud\output_plot_nyquist"
    file_specifications_zhiyun = [
        # 你可以继续添加其他文件，并为每个文件指定remove_first_n_points
        # (file_path, color, marker, remove_first_n_points)
        # 或者 (file_path, color, marker, display_name, remove_first_n_points)

        (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241029_2ppm铁离子污染和恢复测试\新版电解槽_firecloud\20241029_ion\循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv", (255, 230, 204), '+', '2ppm Fe3+ 0h_1029_firecloud',1),
        (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241029_2ppm铁离子污染和恢复测试\新版电解槽_firecloud\20241029_ion\循环1／1_工步组2(工步组)(1／80)_工步3(阻抗).csv", (255, 230, 204), '+', '2ppm Fe3+ 2h_1029_firecloud',1),
        (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241029_2ppm铁离子污染和恢复测试\新版电解槽_firecloud\20241029_ion\循环1／1_工步组2(工步组)(6／80)_工步3(阻抗).csv", (255, 230, 204), '+', '2ppm Fe3+ 12h_1029_firecloud',1),
        (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241029_2ppm铁离子污染和恢复测试\新版电解槽_firecloud\20241029_ion\循环1／1_工步组2(工步组)(12／80)_工步3(阻抗).csv", (255, 230, 204), '+', '2ppm Fe3+ 24h_1029_firecloud',1),
        #
        # # # #

    ]

    file_specifications_gamry = [
    ]

    plot_impedance_nyquist(output_folder, file_specifications_zhiyun, file_specifications_gamry)



