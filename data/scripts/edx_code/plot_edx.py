import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import os
from matplotlib.font_manager import FontProperties


def rgb_to_plotly(rgb_tuple):
    """
    将RGB元组（0-255）转换为Plotly可识别的'rgb(r, g, b)'格式。

    参数：
    - rgb_tuple (tuple): RGB颜色元组，如 (255, 255, 0)

    返回：
    - str: 转换后的颜色字符串，如 'rgb(255, 255, 0)'
    """
    return f"rgb({rgb_tuple[0]}, {rgb_tuple[1]}, {rgb_tuple[2]})"


def rgb_to_matplotlib(rgb_tuple):
    """
    将RGB元组（0-255）转换为Matplotlib可识别的0-1之间的浮点数格式。

    参数：
    - rgb_tuple (tuple): RGB颜色元组，如 (255, 255, 0)

    返回：
    - tuple: 转换后的RGB颜色元组，如 (1.0, 1.0, 0.0)
    """
    return tuple([x / 255 for x in rgb_tuple])


# 定义 Matplotlib 到 Plotly 的 Marker 映射
matplotlib_to_plotly_marker = {
    'o': 'circle',
    's': 'square',
    'D': 'diamond',
    '^': 'triangle-up',
    'v': 'triangle-down',
    '<': 'triangle-left',
    '>': 'triangle-right',
    'p': 'pentagon',
    '*': 'star',
    'h': 'hexagon',
    'H': 'hexagon2',
    '+': 'cross',
    'x': 'x',
    'd': 'diamond-tall',
    '|': 'line-ew',
    '_': 'line-ns',
    'P': 'star-open',
    'X': 'x-open',
    # 可以根据需要扩展更多映射
}


def plot_csv_data_both(csv_files, output_dir,
                       html_output_file='combined_plot_edx.html',
                       png_output_file='combined_plot_edx.png',
                       sep=',',
                       x_label='距离 (µm)',
                       y_label='元素分布',
                       title='edx',
                       figsize=(12, 8)):
    """
    绘制多个CSV文件的数据到同一张图形上，并同时保存为Plotly的HTML和Matplotlib的PNG文件。

    参数：
    - csv_files (list of dict): 每个字典包含以下键：
        - 'path' (str): CSV文件的绝对路径。
        - 'y_columns' (list of str): 需要绘制的纵坐标列名列表。
        - 'colors' (list of tuple or str): 绘图颜色列表，可以是颜色名称（如 'blue'）或RGB元组（如 (255, 255, 0)）。
        - 'markers' (list of str): 点标记样式列表（如 'o', 's', 'D', '^' 等）。
        - 'legends' (list of str): 图例名称列表，与'y_columns'一一对应。
        - 'reverse' (bool, optional): 是否反转“距离 (µm)”列，默认为False。
        - 'x_label' (str, optional): 该文件的X轴列名，如果与全局不同。
    - output_dir (str): 图形保存的目标文件夹路径。
    - html_output_file (str, optional): 输出HTML文件的名称，默认为 'combined_plot_edx.html'。
    - png_output_file (str, optional): 输出PNG文件的名称，默认为 'combined_plot_edx.png'。
    - sep (str, optional): CSV文件的分隔符，默认为逗号','。
    - x_label (str, optional): 全局X轴标签，默认为 '距离 (µm)'。
    - y_label (str, optional): Y轴标签，默认为 '元素分布'。
    - title (str, optional): 图形标题，默认为 'edx'。
    - figsize (tuple, optional): Matplotlib图形的尺寸，默认为 (12, 8)。
    """

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出文件夹: {output_dir}")

    # 设置 Matplotlib 的全局字体
    plt.rcParams['font.family'] = 'SimHei'  # 确保系统中已安装 SimHei 字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 如果需要手动指定字体路径，请取消以下注释并设置正确的路径
    # font_path = r'C:\Windows\Fonts\simhei.ttf'  # Windows 示例
    # font_prop = FontProperties(fname=font_path)

    # 初始化 Plotly 的图形对象
    plotly_fig = go.Figure()

    # 初始化 Matplotlib 图形
    plt.figure(figsize=figsize)

    # 遍历每个CSV文件并添加数据到图形中
    for csv in csv_files:
        csv_path = csv.get('path')
        y_columns = csv.get('y_columns', [])
        colors = csv.get('colors', ['blue'] * len(y_columns))
        markers = csv.get('markers', ['o'] * len(y_columns))
        legends = csv.get('legends', [f"{col}" for col in y_columns])
        reverse = csv.get('reverse', False)
        file_x_label = csv.get('x_label', x_label)  # 每个文件的 X 轴列名

        # 检查必要参数是否齐全
        if not (len(y_columns) == len(colors) == len(markers) == len(legends)):
            print(f"文件 {csv_path} 中'y_columns'、'colors'、'markers'、'legends' 的长度不一致。")
            continue

        # 检查文件是否存在
        if not os.path.isfile(csv_path):
            print(f"文件未找到: {csv_path}")
            continue

        try:
            # 读取 CSV 文件
            df = pd.read_csv(csv_path, sep=sep)
        except Exception as e:
            print(f"读取文件 {csv_path} 时出错: {e}")
            continue

        # 检查横坐标列是否存在
        if file_x_label not in df.columns:
            print(f"横坐标列 '{file_x_label}' 未在文件 {csv_path} 中找到。")
            print(f"文件的列名: {df.columns.tolist()}\n")
            continue

        # 处理距离列
        if reverse:
            # 获取最后一行的距离值
            last_distance = df[file_x_label].iloc[-1]
            # 计算新的距离列
            df['new_distance'] = last_distance - df[file_x_label]
            x = df['new_distance']
            # 更新 x_label 显示
            current_x_label = f"{file_x_label} (反转)"
        else:
            x = df[file_x_label]
            current_x_label = file_x_label

        # 遍历需要绘制的纵坐标列及其对应的颜色、marker和legend
        for y_col, color_input, marker_style, legend_name in zip(y_columns, colors, markers, legends):
            if y_col not in df.columns:
                print(f"纵坐标列 '{y_col}' 未在文件 {csv_path} 中找到。")
                print(f"文件的列名: {df.columns.tolist()}\n")
                continue

            # 处理颜色输入
            if isinstance(color_input, tuple) and len(color_input) == 3:
                # 转换颜色为 Plotly 和 Matplotlib 格式
                plotly_color = rgb_to_plotly(color_input)
                matplotlib_color = rgb_to_matplotlib(color_input)
            elif isinstance(color_input, str):
                # 直接使用颜色名称
                plotly_color = color_input
                matplotlib_color = color_input
            else:
                print(f"无效的颜色格式 '{color_input}' 在文件 {csv_path} 中。")
                plotly_color = 'blue'  # 默认颜色
                matplotlib_color = 'blue'  # 默认颜色

            # 映射 Matplotlib marker 到 Plotly marker
            plotly_marker = matplotlib_to_plotly_marker.get(marker_style, 'circle')  # 默认使用 'circle'

            # 添加 Trace 到 Plotly 图形中
            plotly_fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df[y_col],
                    mode='markers+lines',
                    name=legend_name,
                    marker=dict(
                        symbol=plotly_marker,  # 使用映射后的 Plotly marker
                        size=8,
                        color=plotly_color
                    ),
                    line=dict(
                        color=plotly_color
                    )
                )
            )

            # 添加到 Matplotlib 图形中
            plt.plot(x, df[y_col], marker=marker_style, color=matplotlib_color,
                     label=legend_name)

    # 更新 Plotly 图形布局
    plotly_fig.update_layout(
        title=title,
        xaxis_title=current_x_label,
        yaxis_title=y_label,
        font=dict(
            family='SimHei',  # 确保支持中文显示
            size=12,
            color='black'
        ),
        legend=dict(
            title="图例",
            itemsizing='constant'
        ),
        hovermode='closest'
    )

    # 构造保存文件的路径
    html_save_path = os.path.join(output_dir, html_output_file)
    png_save_path = os.path.join(output_dir, png_output_file)

    # 保存 Plotly 图形为 HTML 文件
    try:
        plotly_fig.write_html(html_save_path, include_plotlyjs='cdn')
        print(f"已保存交互式图形 (HTML): {html_save_path}")
    except Exception as e:
        print(f"保存 HTML 图形到 {html_save_path} 时出错: {e}")

    # 保存 Matplotlib 图形为 PNG 文件
    try:
        plt.title(title, fontsize=14)  # 已设置全局字体，无需再指定 fontproperties
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(png_save_path, dpi=300)
        print(f"已保存静态图形 (PNG): {png_save_path}")
    except Exception as e:
        print(f"保存 PNG 图形到 {png_save_path} 时出错: {e}")

    plt.close()


# 可用的Matplotlib Marker样式
# 常用的marker样式包括：
# 'o'  - 圆圈
# 's'  - 方形
# 'D'  - 菱形
# '^'  - 上三角
# 'v'  - 下三角
# '<'  - 左三角
# '>'  - 右三角
# 'p'  - 五边形
# '*'  - 星形
# 'h'  - 六边形1
# 'H'  - 六边形2
# '+'  - 加号
# 'x'  - 叉号
# 'd'  - 小菱形
# '|'  - 垂直线
# '_'  - 水平线
# 'P'  - 星形开口
# 'X'  - 叉形开口
# 更多marker样式请参考Matplotlib官方文档：
# https://matplotlib.org/stable/api/markers_api.html

# 可用的Plotly Marker样式
# 常用的Plotly marker样式包括：
# 'circle'
# 'square'
# 'diamond'
# 'triangle-up'
# 'triangle-down'
# 'triangle-left'
# 'triangle-right'
# 'pentagon'
# 'star'
# 'hexagon'
# 'hexagon2'
# 'cross'
# 'x'
# 'diamond-tall'
# 'line-ew'
# 'line-ns'
# 'star-open'
# 'x-open'
# 'diamond-open'
# 'pentagon-open'
# 'diamond-dot'
# 'star-dot'
# # 可以根据需要扩展更多Plotly marker样式
# 参考Plotly官方文档获取更多信息：
# https://plotly.com/python/marker-style/#custom-marker-symbols

# 示例用法
if __name__ == "__main__":
    # 定义CSV文件及其绘图参数
    csv_files = [
        # {
        #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240813-0814_10ppm铜离子污染和恢复测试（拔去离子柱跑误差较大）\edx\1-0815-L.csv',
        #     'y_columns': ['Cu Lα1_2 (计数)','Ir Mα1 (计数)'],
        #     'colors': [(102, 204, 255),(102, 204, 255)],  # RGB颜色元组，例如红色
        #     'markers': ['d','d'],  # 点标记样式，详见下方注释
        #     'legends': ['10ppm Cu2+ 8h H2SO4 0813','10ppm Cu2+ 8h H2SO4 0813 Ir'],
        #     'reverse': True,
        #     'x_label': '距离 (µm)'
        # },
        # {
        #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240815-0817_10ppm铁离子污染和恢复测试（拔去离子柱跑误差较大）\edx\2-0817-L.csv',
        #     'y_columns': ['Fe Kα1 (计数)','Ir Mα1 (计数)'],
        #     'colors': [ (204, 153, 102), (204, 153, 102)],  # RGB颜色元组，例如绿色
        #     'markers': ['d','d'],  # 点标记样式，详见下方注释，恢复后点标记用diamond，恢复前用x
        #     'legends': ['10ppm Fe3+ 8h H2SO4 0817','10ppm Fe3+ 8h H2SO4 0817 Ir'],
        #     'reverse': True,
        #     'x_label': '距离 (µm)'  # 根据实际列名调整
        # },

        # {
        #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240820_100ppm铜离子污染和恢复测试\edx\3-0821-L.csv',
        #     'y_columns': ['Cu Lα1_2 (计数)','Ir Mα1 (计数)'],
        #     'colors': [(0, 128, 255),(0, 128, 255)],  # RGB颜色元组，例如蓝色
        #     'markers': ['x','x'],  # 点标记样式，详见下方注释
        #     'legends': ['100ppm Cu2+ 8h H2SO4 0820','100ppm Cu2+ 8h H2SO4 0820 Ir'],
        #     'reverse': True,
        #     'x_label': '距离 (µm)'
        # },
        # {
        #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240820_100ppm铜离子污染和恢复测试\edx\3-0821-L.csv',
        #     'y_columns': ['Cu Lα1_2 (计数)'],
        #     'colors': [(0, 128, 255)],  # RGB颜色元组，例如蓝色
        #     'markers': ['x'],  # 点标记样式，详见下方注释
        #     'legends': ['100ppm Cu2+ 8h H2SO4 0820'],
        #     'reverse': True,
        #     'x_label': '距离 (µm)'
        # },


        # {
        #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240822_10ppm铜离子污染测试\edx\4-0823-L.csv',
        #     'y_columns': ['Cu Lα1_2 (计数)','Ir Mα1 (计数)'],
        #     'colors': [(102, 204, 255),(102, 204, 255)],  # RGB颜色元组，例如橙色
        #     'markers': ['x','x'],  # 点标记样式，详见下方注释
        #     'legends': ['10ppm Cu2+ 24h 0822','10ppm Cu2+ 24h 0822 Ir'],
        #     'reverse': False,
        #     'x_label': '距离 (µm)'
        # },
        {
            'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240822_10ppm铜离子污染测试\edx\4-0823-L.csv',
            'y_columns': ['Cu Lα1_2 (计数)'],
            'colors': [(102, 204, 255)],  # RGB颜色元组，例如橙色
            'markers': ['x'],  # 点标记样式，详见下方注释
            'legends': ['10ppm Cu2+ 24h 0822'],
            'reverse': False,
            'x_label': '距离 (µm)'
        },

        #
        #
        # # {
        # #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240823_10ppm钙离子污染和恢复测试\edx\5-0827-L.csv',
        # #     'y_columns': ['Ca Kα1 (计数)','Ir Mα1 (计数)'],
        # #     'colors': [ (204, 153, 255), (204, 153, 255)],  # RGB颜色元组，例如紫色
        # #     'markers': ['x','x'],  # 点标记样式，详见下方注释
        # #     'legends': ['10ppm Ca2+ 24h H2SO4 0827','10ppm Ca2+ 24h H2SO4 0827 Ir'],
        # #     'reverse': True,
        # #     'x_label': '距离 (µm)'
        # # },
        # # {
        # #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240823_10ppm钙离子污染和恢复测试\edx\5-0827-L.csv',
        # #     'y_columns': ['Ca Kα1 (计数)'],
        # #     'colors': [ (204, 153, 255)],  # RGB颜色元组，例如紫色
        # #     'markers': ['x',],  # 点标记样式，详见下方注释
        # #     'legends': ['10ppm Ca2+ 24h H2SO4 0827'],
        # #     'reverse': True,
        # #     'x_label': '距离 (µm)'
        # # },


        # # {
        # #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240827_10ppm铬离子污染和恢复测试\edx\6-0830-L.csv',
        # #     'y_columns': ['Cr Kα1 (计数)','Ir Mα1 (计数)'],
        # #     'colors': [ (102, 204, 102), (102, 204, 102)],  # RGB颜色元组，例如青色
        # #     'markers': ['d','d'],  # 点标记样式，详见下方注释
        # #     'legends': ['10ppm Cr3+ 24h H2SO4 0827','10ppm Cr3+ 24h H2SO4 0827 Ir'],
        # #     'reverse': True,
        # #     'x_label': '距离 (µm)'
        # # },
        # {
        #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240827_10ppm铬离子污染和恢复测试\edx\6-0830-L.csv',
        #     'y_columns': ['Cr Kα1 (计数)'],
        #     'colors': [ (102, 204, 102)],  # RGB颜色元组，例如青色
        #     'markers': ['d'],  # 点标记样式，详见下方注释
        #     'legends': ['10ppm Cr3+ 24h H2SO4 0827'],
        #     'reverse': True,
        #     'x_label': '距离 (µm)'
        # },



        # #
        # # {
        # #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240831_10ppm镍离子污染测试\edx\完整版2409039625\完整版2409039625\完整版reports\完整版\1-Line-2.csv',
        # #     'y_columns': ['Ni Kα1 (计数)','Ir Mα1 (计数)'],
        # #     'colors': [ (102, 204, 153), (102, 204, 153)],  # RGB颜色元组，例如深绿色
        # #     'markers': ['x','x'],  # 点标记样式，详见下方注释
        # #     'legends': ['10ppm Ni2+ 48h 0831','10ppm Ni2+ 48h 0831 Ir'],
        # #     'reverse': False,
        # #     'x_label': '距离 (µm)'
        # # },
        {
            'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240831_10ppm镍离子污染测试\edx\完整版2409039625\完整版2409039625\完整版reports\完整版\1-Line-2.csv',
            'y_columns': ['Ni Kα1 (计数)'],
            'colors': [ (102, 204, 153)],  # RGB颜色元组，例如深绿色
            'markers': ['x'],  # 点标记样式，详见下方注释
            'legends': ['10ppm Ni2+ 48h 0831'],
            'reverse': False,
            'x_label': '距离 (µm)'
        },


        # # {
        # #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240907_10ppm铁离子污染测试\edx\完整版3667_1726003521418_2409073394\完整版2409073394\完整版reports\完整版\1-X-2.csv',
        # #     'y_columns': ['Fe Kα1 (计数)','Ir Mα1 (计数)'],
        # #     'colors': [(204, 153, 102),(204, 153, 102)],  # RGB颜色元组，例如深绿色
        # #     'markers': ['x','x'],  # 点标记样式，详见下方注释
        # #     'legends': ['10ppm Fe3+ 48h 0907','10ppm Fe3+ 48h 0907 Ir'],
        # #     'reverse': False,
        # #     'x_label': '距离 (µm)'
        # # },
        {
            'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240907_10ppm铁离子污染测试\edx\完整版3667_1726003521418_2409073394\完整版2409073394\完整版reports\完整版\1-X-2.csv',
            'y_columns': ['Fe Kα1 (计数)'],
            'colors': [(204, 153, 102)],  # RGB颜色元组，例如深绿色
            'markers': ['x',],  # 点标记样式，详见下方注释
            'legends': ['10ppm Fe3+ 48h 0907',],
            'reverse': False,
            'x_label': '距离 (µm)'
        },


        # # {
        # #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240910_10ppm钙离子污染测试\edx\完整版2409186812\完整版reports\完整版\60PPM-CA-X-2.csv',
        # #     'y_columns': ['Ca Kα1 (计数)','Ir Mα1 (计数)'],
        # #     'colors': [(204, 153, 255),(204, 153, 255)],  # RGB颜色元组，例如靛青
        # #     'markers': ['x','x'],  # 点标记样式，详见下方注释
        # #     'legends': ['10ppm Ca2+ 48h 0910','10ppm Ca2+ 48h 0910 Ir'],
        # #     'reverse': True,
        # #     'x_label': '距离 (µm)'
        # # },
        {
            'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240910_10ppm钙离子污染测试\edx\完整版2409186812\完整版reports\完整版\60PPM-CA-X-2.csv',
            'y_columns': ['Ca Kα1 (计数)'],
            'colors': [(204, 153, 255)],  # RGB颜色元组，例如靛青
            'markers': ['x',],  # 点标记样式，详见下方注释
            'legends': ['10ppm Ca2+ 48h 0910'],
            'reverse': True,
            'x_label': '距离 (µm)'
        },


        # {
        #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240915_2ppm铜离子污染测试\edx\旧版电解槽_firecloud\20PPM-CU-J-2.csv',
        #     'y_columns': ['Cu Lα1_2 (计数)','Ir Mα1 (计数)'],
        #     'colors': [  (204, 255, 255), (204, 255, 255)],  # RGB颜色元组，例如番茄色
        #     'markers': ['x','x'],  # 点标记样式，详见下方注释
        #     'legends': ['2ppm Cu2+ 24h 0915 firecloud','2ppm Cu2+ 24h 0915 firecloud Ir'],
        #     'reverse': False,
        #     'x_label': '距离 (µm)'
        # },
        # {
        #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240915_2ppm铜离子污染测试\edx\旧版电解槽_firecloud\20PPM-CU-J-2.csv',
        #     'y_columns': ['Cu Lα1_2 (计数)'],
        #     'colors': [(204, 255, 255)],  # RGB颜色元组，例如番茄色
        #     'markers': [ 'x'],  # 点标记样式，详见下方注释
        #     'legends': ['2ppm Cu2+ 24h 0915 firecloud'],
        #     'reverse': False,
        #     'x_label': '距离 (µm)'
        # },


        # {
        #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240915_2ppm铜离子污染测试\edx\新版电解槽_gamry\20PPM-CU-X-2.csv',
        #     'y_columns': ['Cu Lα1_2 (计数)','Ir Mα1 (计数)'],
        #     'colors': [ (204, 255, 255), (204, 255, 255)],  # RGB颜色元组，例如番茄色
        #     'markers': ['x','x'],  # 点标记样式，详见下方注释
        #     'legends': ['2ppm Cu2+ 24h 0915','2ppm Cu2+ 24h 0915 Ir'],
        #     'reverse': True,
        #     'x_label': '距离 (µm)'
        # },
        # {
        #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240915_2ppm铜离子污染测试\edx\新版电解槽_gamry\20PPM-CU-X-2.csv',
        #     'y_columns': ['Cu Lα1_2 (计数)'],
        #     'colors': [(204, 255, 255)],  # RGB颜色元组，例如番茄色
        #     'markers': ['x'],  # 点标记样式，详见下方注释
        #     'legends': ['2ppm Cu2+ 24h 0915 gamry'],
        #     'reverse': True,
        #     'x_label': '距离 (µm)'
        # },
        # #
        # #
        # # {
        # #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240918_2ppm钙离子污染测试\edx\7-0920-L.csv',
        # #     'y_columns': ['Ca Kα1 (计数)', 'Ir Mα1 (计数)'],
        # #     'colors': [(230, 200, 255), (230, 200, 255)],  # RGB颜色元组，例如靛青
        # #     'markers': ['x', 'x'],  # 点标记样式，详见下方注释
        # #     'legends': ['2ppm Ca2+ 48h 0918', '2ppm Ca2+ 48h 0918 Ir'],
        # #     'reverse': True,
        # #     'x_label': '距离 (µm)'
        # # },
        # {
        #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240918_2ppm钙离子污染测试\edx\7-0920-L.csv',
        #     'y_columns': ['Ca Kα1 (计数)'],
        #     'colors': [(230, 200, 255)],  # RGB颜色元组，例如靛青
        #     'markers': ['x'],  # 点标记样式，详见下方注释
        #     'legends': ['2ppm Ca2+ 48h 0918'],
        #     'reverse': True,
        #     'x_label': '距离 (µm)'
        # },
        #
        # # {
        # #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241001_2ppm铁离子污染测试\edx\旧版电解槽_firecloud\1002J-X.csv',
        # #     'y_columns': ['Fe Kα1 (计数)','Ir Mα1 (计数)'],
        # #     'colors': [ (255, 230, 204),  (255, 230, 204)],  # RGB颜色元组，例如靛青
        # #     'markers': ['x', 'x'],  # 点标记样式，详见下方注释
        # #     'legends': ['2ppm Fe3+ 24h 1001 firecloud', '2ppm Fe3+ 24h 1001 Ir firecloud'],
        # #     'reverse': False,
        # #     'x_label': '距离 (µm)'
        # # },
        # {
        #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241001_2ppm铁离子污染测试\edx\旧版电解槽_firecloud\1002J-X.csv',
        #     'y_columns': ['Fe Kα1 (计数)'],
        #     'colors': [(255, 230, 204)],  # RGB颜色元组，例如靛青
        #     'markers': ['x'],  # 点标记样式，详见下方注释
        #     'legends': ['2ppm Fe3+ 24h 1001 firecloud'],
        #     'reverse': False,
        #     'x_label': '距离 (µm)'
        # },
        #
        #
        # # {
        # #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241001_2ppm铁离子污染测试\edx\新版电解槽_gamry\1002X-X.csv',
        # #     'y_columns': ['Fe Kα1 (计数)', 'Ir Mα1 (计数)'],
        # #     'colors': [(255, 230, 204), (255, 230, 204)],  # RGB颜色元组，例如靛青
        # #     'markers': ['x', 'x'],  # 点标记样式，详见下方注释
        # #     'legends': ['2ppm Fe3+ 24h 1001 ', '2ppm Fe3+ 24h 1001 Ir'],
        # #     'reverse': False,
        # #     'x_label': '距离 (µm)'
        # # },
        # # {
        # #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241001_2ppm铁离子污染测试\edx\新版电解槽_gamry\1002X-X.csv',
        # #     'y_columns': ['Fe Kα1 (计数)'],
        # #     'colors': [(255, 230, 204)],  # RGB颜色元组，例如靛青
        # #     'markers': ['x'],  # 点标记样式，详见下方注释
        # #     'legends': ['2ppm Fe3+ 24h 1001 '],
        # #     'reverse': False,
        # #     'x_label': '距离 (µm)'
        # # },
        #
        #
        # #
        # # {
        # #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241003_2ppm镍离子污染测试\edx\旧版电解槽_firecloud\1005J-X.csv',
        # #     'y_columns': ['Ni Kα1 (计数)','Ir Mα1 (计数)'],
        # #     'colors': [ (173, 255, 204),  (173, 255, 204)],  # RGB颜色元组，例如靛青
        # #     'markers': ['x', 'x'],  # 点标记样式，详见下方注释
        # #     'legends': ['2ppm Ni2+ 48h 1003 firecloud', '2ppm Ni2+ 48h 1003 firecloud Ir'],
        # #     'reverse': False,
        # #     'x_label': '距离 (µm)'
        # # },
        # # {
        # #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241003_2ppm镍离子污染测试\edx\旧版电解槽_firecloud\1005J-X.csv',
        # #     'y_columns': ['Ni Kα1 (计数)'],
        # #     'colors': [(173, 255, 204)],  # RGB颜色元组，例如靛青
        # #     'markers': ['x'],  # 点标记样式，详见下方注释
        # #     'legends': ['2ppm Ni2+ 48h 1003 firecloud'],
        # #     'reverse': False,
        # #     'x_label': '距离 (µm)'
        # # },
        #
        #
        # # {
        # #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241003_2ppm镍离子污染测试\edx\新版电解槽-gamry\1005X-X.csv',
        # #     'y_columns': ['Ni Kα1 (计数)','Ir Mα1 (计数)'],
        # #     'colors': [ (173, 255, 204),  (173, 255, 204)],  # RGB颜色元组，例如靛青
        # #     'markers': ['x', 'x'],  # 点标记样式，详见下方注释
        # #     'legends': ['2ppm Ni2+ 48h 1003', '2ppm Ni2+ 48h 1003 Ir'],
        # #     'reverse': False,
        # #     'x_label': '距离 (µm)'
        # # },
        #
        # {
        #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241003_2ppm镍离子污染测试\edx\新版电解槽_gamry\1005X-X.csv',
        #     'y_columns': ['Ni Kα1 (计数)'],
        #     'colors': [ (173, 255, 204)],  # RGB颜色元组，例如靛青
        #     'markers': ['x'],  # 点标记样式，详见下方注释
        #     'legends': ['2ppm Ni2+ 48h 1003'],
        #     'reverse': False,
        #     'x_label': '距离 (µm)'
        # },
        #
        # #
        # # {
        # #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241005未使用膜电极送检edx\edx\1005NOUSE-X.csv',
        # #     'y_columns': ['Pt Mα1 (计数)','Ir Mα1 (计数)'],
        # #     'colors': [(192, 192, 192), (192, 192, 192)],  # RGB颜色元组，例如靛青
        # #     'markers': ['o', 'o'],  # 点标记样式，详见下方注释
        # #     'legends': ['no run Pt 1005', 'no run Ir 1005'],
        # #     'reverse': True,
        # #     'x_label': '距离 (µm)'
        # # },
        # # #
        # # {
        # #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241006_2ppm铬离子污染测试\edx\新版电解槽_gamry\1008X-X.csv',
        # #     'y_columns': ['Cr Kα1 (计数)', 'Ir Mα1 (计数)'],
        # #     'colors': [(200, 255, 200), (200, 255, 200)],  # RGB颜色元组，例如青色
        # #     'markers': ['x', 'x'],  # 点标记样式，详见下方注释
        # #     'legends': ['2ppm Cr3+ 48h 1006', '2ppm Cr3+ 48h 1006 Ir'],
        # #     'reverse': False,
        # #     'x_label': '距离 (µm)'
        # # },
        # {
        #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241006_2ppm铬离子污染测试\edx\新版电解槽_gamry\1008X-X.csv',
        #     'y_columns': ['Cr Kα1 (计数)'],
        #     'colors': [(200, 255, 200)],  # RGB颜色元组，例如青色
        #     'markers': ['x'],  # 点标记样式，详见下方注释
        #     'legends': ['2ppm Cr3+ 48h 1006'],
        #     'reverse': False,
        #     'x_label': '距离 (µm)'
        # },
        # {
        #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241006_2ppm铬离子污染测试\edx\旧版电解槽_firecloud\1008J-X.csv',
        #     'y_columns': ['Cr Kα1 (计数)', 'Ir Mα1 (计数)'],
        #     'colors': [(200, 255, 200), (200, 255, 200)],  # RGB颜色元组，例如青色
        #     'markers': ['x', 'x'],  # 点标记样式，详见下方注释
        #     'legends': ['2ppm Cr3+ 48h 1006 firecloud', '2ppm Cr3+ 48h 1006 firecloud Ir'],
        #     'reverse': True,
        #     'x_label': '距离 (µm)'
        # },
        # {
        #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241006_2ppm铬离子污染测试\edx\旧版电解槽_firecloud\1008J-X.csv',
        #     'y_columns': ['Cr Kα1 (计数)',],
        #     'colors': [(200, 255, 200), ],  # RGB颜色元组，例如青色
        #     'markers': ['x', ],  # 点标记样式，详见下方注释
        #     'legends': ['2ppm Cr3+ 48h 1006 firecloud',],
        #     'reverse': True,
        #     'x_label': '距离 (µm)'
        # },
        #
        # {
        #     'path': r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241008_无离子污染测试\edx\1010-X.csv',
        #     'y_columns': ['Pt Mα1 (计数)', 'Ir Mα1 (计数)'],
        #     'colors': [(192, 192, 192), (192, 192, 192)],  # RGB颜色元组，例如靛青
        #     'markers': ['o', 'o'],  # 点标记样式，详见下方注释
        #     'legends': ['no ion run Pt 1008 firecloud', 'no ion run Ir 1008 firecloud'],
        #     'reverse': False,
        #     'x_label': '距离 (µm)'
        # },

        # 可以继续添加更多的CSV文件及其参数
    ]

    # 指定保存图形的文件夹
    output_directory = r'C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\图片输出文件夹'  # 替换为您希望保存图形的文件夹路径

    # 调用函数绘图并保存
    plot_csv_data_both(
        csv_files=csv_files,
        output_dir=output_directory,
        html_output_file='combined_plot_edx.html',  # 指定输出HTML文件名
        png_output_file='combined_plot_edx.png',  # 指定输出PNG文件名
        sep=',',  # CSV分隔符
        x_label='距离 (µm)',
        y_label='元素分布',
        title='edx',
        figsize=(12, 8)  # Matplotlib图形尺寸
    )



