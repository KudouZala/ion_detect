import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objs as go
import plotly.offline as pyo

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


# 炙云数据提取
def extract_eis_data_from_csv(file_path, remove_first_n_points=0):
    df = pd.read_csv(file_path)
    if 'Freq/Hz' not in df.columns or 'Re(Z)/Ohm' not in df.columns or 'Im(Z)/Ohm' not in df.columns:
        raise ValueError(f"CSV文件 {file_path} 中没有找到正确的列名")

    # 移除前n个数据点
    df = df.iloc[remove_first_n_points:]

    data = {
        'Freq': df['Freq/Hz'].values,
        'Zreal': df['Re(Z)/Ohm'].values,
        'Zimag': df['Im(Z)/Ohm'].values
    }
    return data


# Gamry数据提取
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
        if len(columns) > max(index[1] for index in indices.values()):
            data['Freq'].append(float(columns[indices['Freq'][1]]))
            data['Zreal'].append(float(columns[indices['Zreal'][1]]))
            data['Zimag'].append(float(columns[indices['Zimag'][1]]))
    return data


# RGB颜色转换函数，将RGB三元组转换为Matplotlib可用的格式
def rgb_to_hex(rgb_tuple):
    return mcolors.to_hex([x/255.0 for x in rgb_tuple])

# 绘制奈奎斯特图
def plot_nyquist(ax, zreal, zimag, label, color, marker):
    ax.plot(zreal, -zimag, label=label, color=color, marker=marker)

# 数据转换和绘制
def convert_and_plot_eis(file_specifications, source_type):
    plotly_traces = []

    for file_spec in file_specifications:
        if len(file_spec) == 4:
            file_path, rgb_color, marker, remove_first_n_points = file_spec
            display_name = os.path.basename(file_path)
        elif len(file_spec) == 5:
            file_path, rgb_color, marker, display_name, remove_first_n_points = file_spec
        else:
            continue

        # 将RGB三元组转换为Matplotlib和Plotly可用的颜色格式
        color = rgb_to_hex(rgb_color)

        if os.path.isfile(file_path):
            if source_type == 'zhiyun':
                data = extract_eis_data_from_csv(file_path, remove_first_n_points)
            elif source_type == 'gamry':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    lines = file.readlines()
                indices = find_data_starting_indices(lines, ['Freq', 'Zreal', 'Zimag'])
                if None in indices.values():
                    print(f"Could not find required columns in {file_path}")
                    continue
                data = extract_eis_data_from_lines(lines, indices)
            else:
                continue

            df = pd.DataFrame(data)

            # 使用指定颜色和标记符号绘制奈奎斯特图
            plot_nyquist(ax_nyquist, df['Zreal'], df['Zimag'], label=display_name, color=color, marker=marker)
            plotly_marker = marker_symbols.get(marker, 'circle')
            trace = go.Scatter(
                x=df['Zreal'],
                y=-df['Zimag'],
                mode='lines+markers',
                name=display_name,
                line=dict(color=color),  # 使用转换后的颜色
                marker=dict(symbol=plotly_marker)
            )
            plotly_traces.append(trace)
        else:
            print(f"File not found: {file_path}")

    return plotly_traces


def main(output_folder):
    file_specifications_zhiyun = [
        # 你可以继续添加其他文件，并为每个文件指定remove_first_n_points
        # (file_path, color, marker, remove_first_n_points)
        # 或者 (file_path, color, marker, display_name, remove_first_n_points)


        # # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241010_2ppm钠离子污染测试\新版电解槽-炙云设备\20241011_ion-20241011-131456-默认1728834048305\循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv", (128, 0, 128), '+', '2ppm Na+ 0h',1),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241010_2ppm钠离子污染测试\新版电解槽-炙云设备\20241011_ion-20241011-131456-默认1728834048305\循环1／1_工步组2(工步组)(1／80)_工步3(阻抗).csv",(128, 0, 128), 's', '2ppm Na+ 2h',1),        # 更多文件...
        # # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241010_2ppm钠离子污染测试\新版电解槽-炙云设备\20241011_ion-20241011-131456-默认1728834048305\循环1／1_工步组2(工步组)(6／80)_工步3(阻抗).csv",(128, 0, 128), 'o', '2ppm Na+ 12h',1),        # 更多文件...
        # # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241010_2ppm钠离子污染测试\新版电解槽-炙云设备\20241011_ion-20241011-131456-默认1728834048305\循环1／1_工步组2(工步组)(12／80)_工步3(阻抗).csv",(128, 0, 128), '*', '2ppm Na+ 24h',1),  # 更多文件...
        #
        # # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241013_2ppm铝离子污染测试\新版电解槽-炙云设备\20241015_ion-20241015-035648-默认1728987741540\循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv",(0, 255, 255), '+', '2ppm Al+ 0h',1),  # 更多文件...
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241013_2ppm铝离子污染测试\新版电解槽-炙云设备\20241015_ion-20241015-035648-默认1728987741540\循环1／1_工步组2(工步组)(1／80)_工步3(阻抗).csv",(0, 255, 255), 's', '2ppm Al+ 2h', 1),  # 更多文件...
        # # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241013_2ppm铝离子污染测试\新版电解槽-炙云设备\20241015_ion-20241015-035648-默认1728987741540\循环1／1_工步组2(工步组)(2／80)_工步3(阻抗).csv",(0, 255, 255), 'o', '2ppm Al+ 4h',1),  # 更多文件...


    ]

    file_specifications_gamry = [
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240915_2ppm铜离子污染测试\20240915\EISGALV_60℃_150ml_1A\cm2_20240914_ion_0.DTA", (100, 100, 100), '+', '2ppm Cu2+ 0h',0),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240915_2ppm铜离子污染测试\20240915\EISGALV_60℃_150ml_1A\cm2_20240914_ion_1.DTA", (100, 100, 100), 's', '2ppm Cu2+ 2h',0),
        # # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240915_2ppm铜离子污染测试\20240915\EISGALV_60℃_150ml_1A\cm2_20240914_ion_5.DTA", 'black', 'o', '2ppm Cu2+ 12h',0),
        # # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240915_2ppm铜离子污染测试\20240915\EISGALV_60℃_150ml_1A\cm2_20240914_ion_10.DTA", 'black', '*', '2ppm Cu2+ 24h',0),
        #
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240918_2ppm钙离子污染测试\20240920\EISGALV_60℃_150ml_1A\cm2_20240917_ion_0.DTA",(255, 0, 0), '+', '2ppm Ca2+ 0h',0),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240918_2ppm钙离子污染测试\20240920\EISGALV_60℃_150ml_1A\cm2_20240917_ion_1.DTA",(255, 0, 0), 's', '2ppm Ca2+ 2h',0),
        # # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240918_2ppm钙离子污染测试\20240920\EISGALV_60℃_150ml_1A\cm2_20240917_ion_5.DTA",'red', 'o', '2ppm Ca2+ 12h',0),
        # # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240918_2ppm钙离子污染测试\20240920\EISGALV_60℃_150ml_1A\cm2_20240917_ion_10.DTA",'red', '*', '2ppm Ca2+ 24h',0),
        #
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241001_2ppm铁离子污染测试\新版电解槽-gamry\EISGALV_60℃_150ml_1A\cm2_20241001_ion_0.DTA",(0, 0, 100), '+', '2ppm Fe3+ 0h',0),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241001_2ppm铁离子污染测试\新版电解槽-gamry\EISGALV_60℃_150ml_1A\cm2_20241001_ion_1.DTA",(0, 0, 100), 's', '2ppm Fe3+ 2h',0),
        # # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241001_2ppm铁离子污染测试\新版电解槽-gamry\EISGALV_60℃_150ml_1A\cm2_20241001_ion_5.DTA",'orange', 'o', '2ppm Fe3+ 12h',0),
        # # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241001_2ppm铁离子污染测试\新版电解槽-gamry\EISGALV_60℃_150ml_1A\cm2_20241001_ion_10.DTA",'orange', '*', '2ppm Fe3+ 24h',0),
        #
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241003_2ppm镍离子污染测试\新版电解槽-gamry\EISGALV_60℃_150ml_1A\cm2_20241003_ion_0.DTA",(200, 200, 0), '+', '2ppm Ni2+ 0h',0),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241003_2ppm镍离子污染测试\新版电解槽-gamry\EISGALV_60℃_150ml_1A\cm2_20241003_ion_1.DTA",(200, 200, 0), 's', '2ppm Ni2+ 2h',0),
        # # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241003_2ppm镍离子污染测试\新版电解槽-gamry\EISGALV_60℃_150ml_1A\cm2_20241003_ion_5.DTA",'yellow', 'o', '2ppm Ni2+ 12h',0),
        # # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241003_2ppm镍离子污染测试\新版电解槽-gamry\EISGALV_60℃_150ml_1A\cm2_20241003_ion_10.DTA",'yellow', '*', '2ppm Ni2+ 24h',0),
        #
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241006_2ppm铬离子污染测试\新版电解槽-gamry\EISGALV_60℃_150ml_1A\cm2_20241006_ion_0.DTA",(0, 100, 0), '+', '2ppm Cr3+ 0h',0),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241006_2ppm铬离子污染测试\新版电解槽-gamry\EISGALV_60℃_150ml_1A\cm2_20241006_ion_1.DTA",(0, 100, 0), 's', '2ppm Cr3+ 2h',0),
        # # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241006_2ppm铬离子污染测试\新版电解槽-gamry\EISGALV_60℃_150ml_1A\cm2_20241006_ion_5.DTA",'green', 'o', '2ppm Cr3+ 12h',0),
        # # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241006_2ppm铬离子污染测试\新版电解槽-gamry\EISGALV_60℃_150ml_1A\cm2_20241006_ion_10.DTA",'green', '*', '2ppm Cr3+ 24h',0),


        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240910_10ppm钙离子污染测试\20240910\EISGALV_60℃_150ml_1A\cm2_20240908_ion_0.DTA",(100, 0, 0), '+', '10ppm Ca2+ 0h',0),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240910_10ppm钙离子污染测试\20240910\EISGALV_60℃_150ml_1A\cm2_20240908_ion_2.DTA",(100, 0, 0), 's', '10ppm Ca2+ 2h', 0),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240910_10ppm钙离子污染测试\20240910\EISGALV_60℃_150ml_1A\cm2_20240908_ion_5.DTA",'red', 'o', '10ppm Ca2+ 12h',0),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240910_10ppm钙离子污染测试\20240910\EISGALV_60℃_150ml_1A\cm2_20240908_ion_10.DTA",'red', '*', '10ppm Ca2+ 24h',0),
        #
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240907_10ppm铁离子污染测试\20240907\EISGALV_60℃_150ml_1A\cm2_20240905_ion_0.DTA",(0, 0, 255), '+', '10ppm Fe3+ 0h', 0),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240907_10ppm铁离子污染测试\20240907\EISGALV_60℃_150ml_1A\cm2_20240905_ion_2.DTA",(0, 0, 255), 's', '10ppm Fe3+ 2h', 0),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240907_10ppm铁离子污染测试\20240907\EISGALV_60℃_150ml_1A\cm2_20240905_ion_5.DTA",'orange', 'o', '10ppm Fe3+ 12h', 0),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240907_10ppm铁离子污染测试\20240907\EISGALV_60℃_150ml_1A\cm2_20240905_ion_10.DTA",'orange', '*', '10ppm Fe3+ 24h', 0),
        #
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240831_10ppm镍离子污染测试\20240902\EISGALV_60℃_150ml_1A\cm2_20240831_ion_0.DTA",(255, 255, 0), '+', '10ppm Ni2+ 0h', 0),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240831_10ppm镍离子污染测试\20240902\EISGALV_60℃_150ml_1A\cm2_20240831_ion_2.DTA",(255, 255, 0), 's', '10ppm Ni2+ 2h', 0),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240831_10ppm镍离子污染测试\20240902\EISGALV_60℃_150ml_1A\cm2_20240831_ion_5.DTA",'yellow', 'o', '10ppm Ni2+ 12h', 0),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240831_10ppm镍离子污染测试\20240902\EISGALV_60℃_150ml_1A\cm2_20240831_ion_10.DTA",'yellow', '*', '10ppm Ni2+ 24h', 0),
        #
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240827_10ppm铬离子污染测试\20240902\EISGALV_60℃_150ml_1A\cm2_20240828_ion_0.DTA",(0, 255, 0), '+', '10ppm Cr3+ 0h', 0),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240827_10ppm铬离子污染测试\20240902\EISGALV_60℃_150ml_1A\cm2_20240828_ion_2.DTA",(0, 255, 0), 's', '10ppm Cr3+ 2h', 0),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240827_10ppm铬离子污染测试\20240902\EISGALV_60℃_150ml_1A\cm2_20240828_ion_5.DTA",'green', 'o', '10ppm Cr3+ 12h', 0),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240827_10ppm铬离子污染测试\20240902\EISGALV_60℃_150ml_1A\cm2_20240828_ion_10.DTA",'green', '*', '10ppm Cr3+ 24h', 0),
        #
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240822_10ppm铜离子污染测试\新版电解槽-gamry\EISGALV_60℃_150ml_1A\cm2_20240822_ion_0.DTA",(0, 0, 0), '+', '10ppm Cu2+ 0h', 0),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240822_10ppm铜离子污染测试\新版电解槽-gamry\EISGALV_60℃_150ml_1A\cm2_20240822_ion_2.DTA",(0, 0, 0), 's', '10ppm Cu2+ 2h', 0),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240822_10ppm铜离子污染测试\新版电解槽-gamry\EISGALV_60℃_150ml_1A\cm2_20240822_ion_5.DTA",'black', 'o', '10ppm Cu2+ 12h', 0),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240822_10ppm铜离子污染测试\新版电解槽-gamry\EISGALV_60℃_150ml_1A\cm2_20240822_ion_10.DTA",'black', '*', '10ppm Cu2+ 24h', 0),

        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240813-0814_10ppm铜离子污染和恢复测试\20240815-ion-renew\EISGALV_60℃_150ml_1A\cm2_20240812_ion_column_1.DTA",(0, 0, 0), 'o', '10ppm Cu2+_0812_ion_column_0h', 0),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240813-0814_10ppm铜离子污染和恢复测试\20240815-ion-renew\EISGALV_60℃_150ml_1A\cm2_20240814_ion_39.DTA",(100, 100, 100), 'x', '10ppm Cu2+_0812_ion_8h', 0),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240813-0814_10ppm铜离子污染和恢复测试\20240815-ion-renew\EISGALV_60℃_150ml_1A\cm2_20240814_ion_renew_1.DTA",(200,200, 200), 'd', '10ppm Cu2+_0812_ion_renew_0h', 0),

        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240815-0817_10ppm铁离子污染和恢复测试\20240817_all\EISGALV_60℃_150ml_1A\cm2_20240815_ion_column_1.DTA",(0, 0, 0), 'o', '10ppm Fe3+_0815_ion_column_0h', 0),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240815-0817_10ppm铁离子污染和恢复测试\20240817_all\EISGALV_60℃_150ml_1A\cm2_20240815_ion_39.DTA",(0, 0, 100), 'x', '10ppm Fe3+_0815_ion_8h', 0),
        # (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240815-0817_10ppm铁离子污染和恢复测试\20240817_all\EISGALV_60℃_150ml_1A\cm2_20240816_ion_renew_1.DTA",(0,0, 200), 'd', '10ppm Fe3+_0815_ion_renew_0h', 0),

        (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240827_10ppm铬离子污染和恢复测试\20240902\EISGALV_60℃_150ml_1A\cm2_20240827_ion_column_1.DTA",(0, 0, 0), 'o', '10ppm Cr3+_0827_ion_column_0h', 0),
        (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240827_10ppm铬离子污染和恢复测试\20240902\EISGALV_60℃_150ml_1A\cm2_20240828_ion_11.DTA",(0, 100, 0), 'x', '10ppm Cr3+_0827_ion_24h', 0),
        (r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240827_10ppm铬离子污染和恢复测试\20240902\EISGALV_60℃_150ml_1A\cm2_20240829_ion_renew_1.DTA",(0,200, 0), 'd', '10ppm Cr3+_0827_ion_renew_0h', 0),
    ]

    global ax_nyquist
    fig_nyquist, ax_nyquist = plt.subplots(figsize=(12, 6))

    # 绘制zhiyun数据的Nyquist图
    plotly_traces_zhiyun = convert_and_plot_eis(file_specifications_zhiyun, 'zhiyun')

    # 绘制gamry数据的Nyquist图
    plotly_traces_gamry = convert_and_plot_eis(file_specifications_gamry, 'gamry')

    # Matplotlib最终绘制设置
    ax_nyquist.set_xlabel('Zreal (Ohm)')
    ax_nyquist.set_ylabel('-Zimag (Ohm)')
    ax_nyquist.set_title('Nyquist Plot (Combined)')
    ax_nyquist.grid(True)
    ax_nyquist.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    fig_nyquist.tight_layout(rect=[0, 0, 1, 0.95])

    # 创建指定的输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 保存静态图像为PNG
    nyquist_file_name = os.path.join(output_folder, "nyquist_combined_zhiyun2gamry.png")
    fig_nyquist.savefig(nyquist_file_name)

    plt.show()

    # 使用Plotly绘制交互式图形
    layout = go.Layout(
        xaxis=dict(title='Zreal (Ohm)'),
        yaxis=dict(title='-Zimag (Ohm)'),
        title='Nyquist Plot (Combined)',
        legend=dict(orientation="h", y=-0.2)
    )
    fig = go.Figure(data=plotly_traces_zhiyun + plotly_traces_gamry, layout=layout)

    # 保存交互式图形为HTML
    html_file_path = os.path.join(output_folder, "nyquist_interactive_combined.html")
    pyo.plot(fig, filename=html_file_path, auto_open=False)


if __name__ == "__main__":
    # 这里指定输出文件夹
    output_folder = r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\图片输出文件夹"
    main(output_folder)




