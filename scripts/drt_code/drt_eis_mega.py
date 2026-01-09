import pandas as pd
import matplotlib.pyplot as plt
import os

# 将τ/s转换为频率f/Hz的函数，假设转换公式为 f = 1 / τ
def tau_to_frequency(tau):
    return 1 / tau
# 将RGB（0-255范围）转化为matplotlib可以使用的格式（0-1范围）
def rgb_to_normalized(rgb):
    return tuple([x / 255.0 for x in rgb])
# 绘制图表的函数
def plot_excel_data(excel_file, sheet_info, output_dir):
    # 创建一个图形
    plt.figure(figsize=(10, 6))
    
    # 遍历字典中每个工作表
    for sheet_name, settings in sheet_info.items():
        # 读取指定工作表的数据
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        print("df.head:",df.head())
        # 提取τ/s和γ/Ω数据
        tau = df['τ/s']
        gamma = df['γ/Ω']
        
        # 转换τ/s为频率f/Hz
        frequency = tau_to_frequency(tau)
        
        # 绘制数据，使用指定的颜色和marker
        plt.plot(frequency, gamma, label=settings['label'],
                 color=rgb_to_normalized(settings['color']), marker=settings['marker'],
                 linestyle='-', markersize=3)
    
    # 设置图表标题和标签
    plt.title('Frequency vs Gamma')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gamma (Ω)')
    plt.xscale('log')  # 通常使用对数坐标绘制频率
    # plt.yscale('log')  # 如果需要，可以将γ/Ω也设置为对数坐标
    plt.legend()
    
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 输出图片的路径
    output_path = os.path.join(output_dir, 'combined_plot.png')
    
    # 保存图表
    plt.savefig(output_path)
    print(f"图表已保存至：{output_path}")
    plt.close()








# 示例输入
excel_file = r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\autoeis\autoeis\AutoEIS\examples\校内测试\20240918_2ppm钙离子污染测试\DRT_eis_mega_gamry\cm2_20240917_ion_0-3_no_title_greater_than_0.xlsx"
sheet_info = {
    'cm2_20240917_ion_0_no_title_gre': {'color': (255, 200, 200), 'marker': 'o','label':'0 h'},  # 红色圆形标记
    'cm2_20240917_ion_1_no_title_gre': {'color': (255, 140, 140), 'marker': 'x','label':'2 h'},  # 蓝色叉形标记
    'cm2_20240917_ion_2_no_title_gre': {'color': (255, 100, 100), 'marker': 's','label':'4 h'},  # 绿色方形标记
    'cm2_20240917_ion_3_no_title_gre': {'color': (255, 20, 20), 'marker': '^','label':'6 h'},  # 绿色方形标记
}

# excel_file = r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\autoeis\autoeis\AutoEIS\examples\校内测试\20241101_2ppm钙离子污染和恢复测试\DRT_eis_mega_firecloud\ion0-3.xlsx"
# sheet_info = {
#     '已恢复_Sheet1': {'color': (200, 255, 200), 'marker': 'o','label':'0 h'},  # 红色圆形标记
#     '已恢复_Sheet2': {'color': (140, 255, 140), 'marker': 'x','label':'2 h'},  # 蓝色叉形标记
#     '已恢复_Sheet3': {'color': (100, 255, 100), 'marker': 's','label':'4 h'},  # 绿色方形标记
#     '已恢复_Sheet4': {'color': (20, 255, 20), 'marker': '^','label':'6 h'},  # 绿色方形标记
# }



output_dir = r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\autoeis\autoeis\AutoEIS\examples\校内测试\图片输出文件夹'

# 调用绘图函数
plot_excel_data(excel_file, sheet_info, output_dir)
