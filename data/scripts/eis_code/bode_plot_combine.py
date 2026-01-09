import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go  # 用于生成交互式绘图
import plotly.offline as pyo    # 用于保存HTML文件
import numpy as np

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
    # 添加其他需要的符号映射
}

def find_data_starting_indices(lines, keywords):
    indices = {key: None for key in keywords}
    for i, line in enumerate(lines):
        columns = line.strip().split('\t')
        for j, column in enumerate(columns):
            if column in indices:
                indices[column] = (i, j)
    return indices

def extract_eis_data_from_lines(lines, indices):
    data = {'Freq': [], 'Zreal': [], 'Zimag': []}
    start_row = max(index[0] for index in indices.values()) + 2
    for line in lines[start_row:]:
        columns = line.strip().split('\t')
        if len(columns) > max(index[1] for index in indices.values()):  # 确保行有足够的列
            try:
                freq = float(columns[indices['Freq'][1]])
                zreal = float(columns[indices['Zreal'][1]])
                zimag = float(columns[indices['Zimag'][1]])
                data['Freq'].append(freq)
                data['Zreal'].append(zreal)
                data['Zimag'].append(zimag)
            except ValueError:
                # 跳过无法转换为浮点数的行
                continue
    return data

def plot_magnitude(ax_mag, freq, magnitude, label, color, marker):
    ax_mag.plot(freq, magnitude, label=label, color=color, marker=marker, linestyle='-', linewidth=1)

def plot_phase(ax_phase, freq, phase, label, color, marker):
    ax_phase.plot(freq, phase, label=label, color=color, marker=marker, linestyle='-', linewidth=1)

def convert_and_plot_eis(file_specifications):
    output_folder_plot = os.path.join(os.getcwd(), "output_plot")
    os.makedirs(output_folder_plot, exist_ok=True)

    output_folder_xlsx = os.path.join(os.getcwd(), "output_xlsx")
    os.makedirs(output_folder_xlsx, exist_ok=True)

    output_folder_csv = os.path.join(os.getcwd(), "output_csv")
    os.makedirs(output_folder_csv, exist_ok=True)

    # 创建Matplotlib图形
    fig_mag, ax_mag = plt.subplots(figsize=(12, 6))
    fig_phase, ax_phase = plt.subplots(figsize=(12, 6))

    # 初始化Plotly轨迹列表
    plotly_traces_mag = []
    plotly_traces_phase = []

    for file_spec in file_specifications:
        if len(file_spec) == 4:
            file_path, color, marker, display_name = file_spec
        else:
            print(f"Invalid specification for {file_spec}")
            continue

        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                lines = file.readlines()

            indices = find_data_starting_indices(lines, ['Freq', 'Zreal', 'Zimag'])
            if None in indices.values():
                print(f"Required columns not found in {file_path}")
                continue

            data = extract_eis_data_from_lines(lines, indices)
            if not data['Freq']:
                print(f"No valid data found in {file_path}")
                continue

            df = pd.DataFrame(data)

            # 计算幅值和相位
            df['Magnitude'] = np.sqrt(df['Zreal']**2 + df['Zimag']**2)
            df['Phase'] = np.degrees(np.arctan2(-df['Zimag'], df['Zreal']))  # 取负以匹配奈奎斯特图的定义

            # 保存为Excel文件
            excel_file_path = os.path.join(output_folder_xlsx, f"{os.path.basename(file_path).replace('.DTA', '.xlsx')}")
            df.to_excel(excel_file_path, index=False)

            # 保存为CSV文件（不含标题行）
            csv_file_path = os.path.join(output_folder_csv, f"{os.path.basename(file_path).replace('.DTA', '.csv')}")
            df.to_csv(csv_file_path, index=False, header=False)

            # 绘制幅值图 (Matplotlib)
            plot_magnitude(ax_mag, df['Freq'], df['Magnitude'], label=display_name, color=color, marker=marker)

            # 绘制相位图 (Matplotlib)
            plot_phase(ax_phase, df['Freq'], df['Phase'], label=display_name, color=color, marker=marker)

            # 将 Matplotlib 标记符号映射到 Plotly 标记符号
            plotly_marker = marker_symbols.get(marker, 'circle')  # 默认使用 'circle'

            # 为 Plotly 生成幅值轨迹
            trace_mag = go.Scatter(
                x=df['Freq'],
                y=df['Magnitude'],
                mode='lines+markers',
                name=f"{display_name} Magnitude",
                line=dict(color=color),
                marker=dict(symbol=plotly_marker)
            )
            plotly_traces_mag.append(trace_mag)

            # 为 Plotly 生成相位轨迹
            trace_phase = go.Scatter(
                x=df['Freq'],
                y=df['Phase'],
                mode='lines+markers',
                name=f"{display_name} Phase",
                line=dict(color=color),
                marker=dict(symbol=plotly_marker)
            )
            plotly_traces_phase.append(trace_phase)
        else:
            print(f"File not found: {file_path}")

    # 配置Matplotlib的幅值子图
    ax_mag.set_ylabel('Magnitude (Ohm)')
    ax_mag.set_xscale('log')
    ax_mag.set_xlabel('Frequency (Hz)')
    ax_mag.set_title('Bode Plot - Magnitude')
    ax_mag.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_mag.legend(fontsize='small')

    # 保存幅值图
    magnitude_file_name = "bode_magnitude.png"
    fig_mag.tight_layout()
    fig_mag.savefig(os.path.join(output_folder_plot, magnitude_file_name))

    # 配置Matplotlib的相位子图
    ax_phase.set_ylabel('Phase (Degrees)')
    ax_phase.set_xscale('log')
    ax_phase.set_xlabel('Frequency (Hz)')
    ax_phase.set_title('Bode Plot - Phase')
    ax_phase.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_phase.legend(fontsize='small')

    # 保存相位图
    phase_file_name = "bode_phase.png"
    fig_phase.tight_layout()
    fig_phase.savefig(os.path.join(output_folder_plot, phase_file_name))

    plt.show()

    # 使用Plotly分别创建交互式幅值图和相位图
    # 幅值图
    layout_mag = go.Layout(
        xaxis=dict(title='Frequency (Hz)', type='log'),
        yaxis=dict(title='Magnitude (Ohm)'),
        title='Bode Plot - Magnitude',
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor='center')
    )
    fig_plotly_mag = go.Figure(data=plotly_traces_mag, layout=layout_mag)

    # 相位图
    layout_phase = go.Layout(
        xaxis=dict(title='Frequency (Hz)', type='log'),
        yaxis=dict(title='Phase (Degrees)'),
        title='Bode Plot - Phase',
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor='center')
    )
    fig_plotly_phase = go.Figure(data=plotly_traces_phase, layout=layout_phase)

    # 创建output_html文件夹
    output_folder_html = os.path.join(os.getcwd(), "output_html")
    os.makedirs(output_folder_html, exist_ok=True)

    # 保存幅值图的HTML文件
    html_file_path_mag = os.path.join(output_folder_html, "bode_magnitude_interactive.html")
    pyo.plot(fig_plotly_mag, filename=html_file_path_mag, auto_open=False)

    # 保存相位图的HTML文件
    html_file_path_phase = os.path.join(output_folder_html, "bode_phase_interactive.html")
    pyo.plot(fig_plotly_phase, filename=html_file_path_phase, auto_open=False)





def main():
    # 指定每个文件的绝对路径、颜色、符号和图例名
    file_specifications = [
        # 示例文件规范（取消注释并根据需要添加更多）
        # ("C:\\路径\\到\\文件1.DTA", 'black', 'o', '样本1'),
        # ("C:\\路径\\到\\文件2.DTA", 'red', 's', '样本2'),

        # ("C:\\Users\\asus\\Desktop\\ROG桌面其他文件\\电解水\\校内测试\\20240915_2ppm铜离子污染测试\\20240915\\EISGALV_60℃_150ml_1A\\cm2_20240914_ion_0.DTA",
        #  'black', 'o', '2ppm Cu2+ 0h 1_0.4'),
        # ("C:\\Users\\asus\\Desktop\\ROG桌面其他文件\\电解水\\校内测试\\20240915_2ppm铜离子污染测试\\20240915\\EISGALV_60℃_150ml_1A\\cm2_20240914_ion_1.DTA",
        #  'black', 'o', '2ppm Cu2+ 2h 1_0.4'),
        ("C:\\Users\\asus\\Desktop\\ROG桌面其他文件\\电解水\\校内测试\\20240823_10ppm铜离子污染测试\\20240823\\EISGALV_60℃_150ml_1A\\cm2_20240822_ion_0.DTA",
        'cyan', 'o', '10ppmCu2+ 0h 2_1'),
        ("C:\\Users\\asus\\Desktop\\ROG桌面其他文件\\电解水\\校内测试\\20240823_10ppm铜离子污染测试\\20240823\\EISGALV_60℃_150ml_1A\\cm2_20240822_ion_1.DTA",
        'cyan', 'o', '10ppmCu2+ 2h 2_1'),
        # ("C:\\Users\\asus\\Desktop\\ROG桌面其他文件\\电解水\\校内测试\\20240820_100ppm铜离子污染测试\\20240820Cuion\\EISGALV_60℃_150ml_1A\\cm2_20240819_ion_column_38.DTA",
        # 'gray', 'o', '100ppm Cu2+ 0h 2_1'),
        # ("C:\\Users\\asus\\Desktop\\ROG桌面其他文件\\电解水\\校内测试\\20240820_100ppm铜离子污染测试\\20240820Cuion\\EISGALV_60℃_150ml_1A\\cm2_20240820_ion_7.DTA",
        # 'gray', 'o', '100ppm Cu2+ 2h 2_1'),

        ("C:\\Users\\asus\\Desktop\\ROG桌面其他文件\\电解水\\校内测试\\20240910_10ppm钙离子污染测试\\20240910\\EISGALV_60℃_150ml_1A\\cm2_20240908_ion_0.DTA",
        'orange', 's', '10ppmCa2+ 0h 1_0.4'),
        ("C:\\Users\\asus\\Desktop\\ROG桌面其他文件\\电解水\\校内测试\\20240910_10ppm钙离子污染测试\\20240910\\EISGALV_60℃_150ml_1A\\cm2_20240908_ion_1.DTA",
        'orange', 's', '10ppmCa2+ 2h 1_0.4'),
        #
        ("C:\\Users\\asus\\Desktop\\ROG桌面其他文件\\电解水\\校内测试\\20240823_10ppm钙离子污染测试\\20240902\\EISGALV_60℃_150ml_1A\\cm2_20240824_ion_0.DTA",
        'purple', 's', '10ppmCa2+ 0h 1_0.5'),
        ("C:\\Users\\asus\\Desktop\\ROG桌面其他文件\\电解水\\校内测试\\20240823_10ppm钙离子污染测试\\20240902\\EISGALV_60℃_150ml_1A\\cm2_20240824_ion_1.DTA",
        'purple', 's', '10ppmCa2+ 2h 1_0.5'),

        ("C:\\Users\\asus\\Desktop\\ROG桌面其他文件\\电解水\\校内测试\\20240907_10ppm铁离子污染测试\\20240907\\EISGALV_60℃_150ml_1A\\cm2_20240905_ion_0.DTA",
        'brown', '+', '10ppmFe3+ 0h 1_0.4'),
        ("C:\\Users\\asus\\Desktop\\ROG桌面其他文件\\电解水\\校内测试\\20240907_10ppm铁离子污染测试\\20240907\\EISGALV_60℃_150ml_1A\\cm2_20240905_ion_1.DTA",
        'brown', '+', '10ppmFe3+ 2h 1_0.4'),

        # ("C:\\Users\\asus\\Desktop\\ROG桌面其他文件\\电解水\\校内测试\\20240815-0817_10ppm铁离子污染测试\\20240817_all\\EISGALV_60℃_150ml_1A\\cm2_20240815_no_ion_column_39.DTA",
        # 'pink', '+', '10ppm Fe3+ 0h 3_0.5'),
        # ("C:\\Users\\asus\\Desktop\\ROG桌面其他文件\\电解水\\校内测试\\20240815-0817_10ppm铁离子污染测试\\20240817_all\\EISGALV_60℃_150ml_1A\\cm2_20240815_ion_7.DTA",
        # 'pink', '+', '10ppm Fe3+ 2h 3_0.5'),

        ("C:\\Users\\asus\\Desktop\\ROG桌面其他文件\\电解水\\校内测试\\20240831_10ppm镍离子污染测试\\20240902\\EISGALV_60℃_150ml_1A\\cm2_20240831_ion_0.DTA",
        'green', 'x', '10ppmNi2+ 0h 1_0.4'),
        ("C:\\Users\\asus\\Desktop\\ROG桌面其他文件\\电解水\\校内测试\\20240831_10ppm镍离子污染测试\\20240902\\EISGALV_60℃_150ml_1A\\cm2_20240831_ion_1.DTA",
        'green', 'x', '10ppmNi2+ 2h 1_0.4'),

        ("C:\\Users\\asus\\Desktop\\ROG桌面其他文件\\电解水\\校内测试\\20240827_10ppm铬离子污染测试\\20240902\\EISGALV_60℃_150ml_1A\\cm2_20240828_ion_0.DTA",
        'blue', '<', '10ppmCr2+ 0h 1_0.5'),
        ("C:\\Users\\asus\\Desktop\\ROG桌面其他文件\\电解水\\校内测试\\20240827_10ppm铬离子污染测试\\20240902\\EISGALV_60℃_150ml_1A\\cm2_20240828_ion_1.DTA",
        'blue', '<', '10ppmCr2+ 2h 1_0.5'),

        # 添加更多文件及其相应的颜色、符号、图例名
    ]

    convert_and_plot_eis(file_specifications)

if __name__ == "__main__":
    main()





