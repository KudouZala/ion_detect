import pandas as pd
import matplotlib.pyplot as plt
import os

from drt_eis_mega import plot_excel_data






# 示例输入
# excel_file = r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\autoeis\autoeis\AutoEIS\examples\校内测试\20241010_2ppm钠离子污染测试\DRT_eis_mega_firecloud\ion0-3.xlsx"
# sheet_info = {
#     '已恢复_Sheet1': {'color': (255, 220, 220), 'marker': 'o','label':'0 h'},  # 红色圆形标记
#     '已恢复_Sheet2': {'color': (255, 180, 180), 'marker': 'x','label':'2 h'},  # 蓝色叉形标记
#     '已恢复_Sheet3': {'color': (220, 100, 100), 'marker': 's','label':'4 h'},  # 绿色方形标记
#     '已恢复_Sheet4': {'color': (180, 50, 50), 'marker': '^','label':'6 h'},  # 绿色方形标记
# }

# excel_file = r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\autoeis\autoeis\AutoEIS\examples\校内测试\20241028_2ppm钠离子污染和恢复测试\DRT_eis_mega_gamry\ion0-3.xlsx"
# sheet_info = {
#     'cm2_20241029_ion_0_no_title_gre': {'color': (220, 255, 255), 'marker': 'o','label':'0 h'},  # 红色圆形标记
#     'cm2_20241029_ion_1_no_title_gre': {'color': (180, 255, 255), 'marker': 'x','label':'2 h'},  # 蓝色叉形标记
#     'cm2_20241029_ion_2_no_title_gre': {'color': (100, 255, 220), 'marker': 's','label':'4 h'},  # 绿色方形标记
#     'cm2_20241029_ion_3_no_title_gre': {'color': (50, 220, 200), 'marker': '^','label':'6 h'},  # 绿色方形标记
# }
#
# excel_file = r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\autoeis\autoeis\AutoEIS\examples\校内测试\20241028_2ppm钠离子污染和恢复测试\DRT_eis_mega_gamry\ion0-3.xlsx"
# sheet_info = {
#     'cm2_20241029_ion_0_no_title_gre': {'color': (220, 255, 255), 'marker': 'o','label':'0 h'},  # 红色圆形标记
#     'cm2_20241029_ion_1_no_title_gre': {'color': (180, 255, 255), 'marker': 'x','label':'2 h'},  # 蓝色叉形标记
#     'cm2_20241029_ion_2_no_title_gre': {'color': (100, 255, 220), 'marker': 's','label':'4 h'},  # 绿色方形标记
#     'cm2_20241029_ion_3_no_title_gre': {'color': (50, 220, 200), 'marker': '^','label':'6 h'},  # 绿色方形标记
# }
#


# excel_file = r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\autoeis\autoeis\AutoEIS\examples\校内测试\20241213_2ppm钠离子污染及恢复测试\DRT_eis_mega_gamry\ion0-3.xlsx"
# sheet_info = {
#     'cm2_20241214_ion_0_no_title_gre': {'color': (220, 255, 220), 'marker': 'o','label':'0 h'},  # 红色圆形标记
#     'cm2_20241214_ion_1_no_title_gre': {'color': (180, 255, 180), 'marker': 'x','label':'2 h'},  # 蓝色叉形标记
#     'cm2_20241214_ion_2_no_title_gre': {'color':(100, 255, 100), 'marker': 's','label':'4 h'},  # 绿色方形标记
#     'cm2_20241214_ion_3_no_title_gre': {'color': (50, 220, 50), 'marker': '^','label':'6 h'},  # 绿色方形标记
# }

# excel_file = r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\autoeis\autoeis\AutoEIS\examples\校内测试\20241229_10ppm钠离子污染及恢复测试\DRT_eis_mega_gamry\ion0-3.xlsx"
# sheet_info = {
#     'cm2_20241230_ion_0_no_title_gre': {'color': (220, 255, 220), 'marker': 'o','label':'0 h'},  # 红色圆形标记
#     'cm2_20241230_ion_1_no_title_gre': {'color': (180, 255, 180), 'marker': 'x','label':'2 h'},  # 蓝色叉形标记
#     'cm2_20241230_ion_2_no_title_gre': {'color':(100, 255, 100), 'marker': 's','label':'4 h'},  # 绿色方形标记
#     'cm2_20241230_ion_3_no_title_gre': {'color': (50, 220, 50), 'marker': '^','label':'6 h'},  # 绿色方形标记
# }



output_dir = r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\autoeis\autoeis\AutoEIS\examples\校内测试\图片输出文件夹'

# 调用绘图函数
plot_excel_data(excel_file, sheet_info, output_dir)