import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as pyo

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
            data['Freq'].append(float(columns[indices['Freq'][1]]))
            data['Zreal'].append(float(columns[indices['Zreal'][1]]))
            data['Zimag'].append(float(columns[indices['Zimag'][1]]))
    return data

def plot_bode(ax, freq, value, label, color, marker):
    ax.plot(freq, value, label=label, color=color, marker=marker)
    ax.set_xscale('log')  # 频率通常在对数尺度上显示
    ax.grid(True)

def convert_and_plot_bode(folder_path, file_specifications):
    output_folder_plot = os.path.join(folder_path, "output_plot")
    os.makedirs(output_folder_plot, exist_ok=True)

    output_folder_xlsx = os.path.join(folder_path, "output_xlsx")
    os.makedirs(output_folder_xlsx, exist_ok=True)

    output_folder_csv = os.path.join(folder_path, "output_csv")
    os.makedirs(output_folder_csv, exist_ok=True)

    fig_bode_real, ax_bode_real = plt.subplots(figsize=(12, 6))
    fig_bode_imag, ax_bode_imag = plt.subplots(figsize=(12, 6))

    # For Plotly interactive graphs
    plotly_real_traces = []
    plotly_imag_traces = []

    for file_spec in file_specifications:
        # 根据指定的参数，使用文件名作为默认显示名字
        if len(file_spec) == 3:
            file_name, color, marker = file_spec
            display_name = file_name
        elif len(file_spec) == 4:
            file_name, color, marker, display_name = file_spec

        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                lines = file.readlines()

            indices = find_data_starting_indices(lines, ['Freq', 'Zreal', 'Zimag'])
            if None in indices.values():
                print(f"Required columns not found in {file_path}")
                continue

            data = extract_eis_data_from_lines(lines, indices)
            df = pd.DataFrame(data)

            # 保存为Excel文件
            excel_file_path = os.path.join(output_folder_xlsx, f"{os.path.basename(file_name).replace('.DTA', '.xlsx')}")
            df.to_excel(excel_file_path, index=False)

            # 保存为CSV文件（不含标题行）
            csv_file_path = os.path.join(output_folder_csv, f"{os.path.basename(file_name).replace('.DTA', '.csv')}")
            df.to_csv(csv_file_path, index=False, header=False)

            # 绘制实部-频率图 (Matplotlib)
            plot_bode(ax_bode_real, df['Freq'], df['Zreal'], label=display_name, color=color, marker=marker)

            # 绘制虚部-频率图 (Matplotlib)
            plot_bode(ax_bode_imag, df['Freq'], df['Zimag'], label=display_name, color=color, marker=marker)

            # 添加 Plotly 的 traces
            real_trace = go.Scatter(x=df['Freq'], y=df['Zreal'], mode='lines', name=f"Real - {display_name}",
                                    line=dict(color=color))
            imag_trace = go.Scatter(x=df['Freq'], y=df['Zimag'], mode='lines', name=f"Imag - {display_name}",
                                    line=dict(color=color))
            plotly_real_traces.append(real_trace)
            plotly_imag_traces.append(imag_trace)
        else:
            print(f"File not found: {file_path}")

    # Matplotlib 保存实部和虚部的Bode图
    ax_bode_real.set_xlabel('Frequency (Hz)')
    ax_bode_real.set_ylabel('Zreal (Ohm)')
    ax_bode_real.set_title('Bode Plot - Real Part')
    ax_bode_real.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    fig_bode_real.tight_layout(rect=[0, 0, 1, 0.95])
    bode_real_file_name = "bode_real_combined.png"
    fig_bode_real.savefig(os.path.join(output_folder_plot, bode_real_file_name))

    ax_bode_imag.set_xlabel('Frequency (Hz)')
    ax_bode_imag.set_ylabel('Zimag (Ohm)')
    ax_bode_imag.set_title('Bode Plot - Imaginary Part')
    ax_bode_imag.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    fig_bode_imag.tight_layout(rect=[0, 0, 1, 0.95])
    bode_imag_file_name = "bode_imag_combined.png"
    fig_bode_imag.savefig(os.path.join(output_folder_plot, bode_imag_file_name))

    plt.show()

    # 使用 Plotly 保存为交互式 HTML
    plotly_real_layout = go.Layout(title='Bode Plot - Real Part', xaxis=dict(title='Frequency (Hz)', type='log'),
                                   yaxis=dict(title='Zreal (Ohm)'), legend=dict(orientation="h", y=-0.2))
    plotly_imag_layout = go.Layout(title='Bode Plot - Imaginary Part', xaxis=dict(title='Frequency (Hz)', type='log'),
                                   yaxis=dict(title='Zimag (Ohm)'), legend=dict(orientation="h", y=-0.2))

    plotly_real_fig = go.Figure(data=plotly_real_traces, layout=plotly_real_layout)
    plotly_imag_fig = go.Figure(data=plotly_imag_traces, layout=plotly_imag_layout)

    output_folder_html = os.path.join(folder_path, "output_html")
    os.makedirs(output_folder_html, exist_ok=True)

    pyo.plot(plotly_real_fig, filename=os.path.join(output_folder_html, "bode_real_interactive.html"), auto_open=False)
    pyo.plot(plotly_imag_fig, filename=os.path.join(output_folder_html, "bode_imag_interactive.html"), auto_open=False)

def main():
    base_folder = r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240915_2ppm铜离子污染测试\20240915\EISGALV_60℃_150ml_1A"

    # 指定每个文件的文件名、颜色和标记符号
    file_specifications = [
        # ("cm2_20240914_ion_column_0.DTA", 'red', 's','0 h'),  # 红色圆形
        # ("cm2_20240914_ion_column_1.DTA", 'orange', 's','2 h'),  # 红色方形
        # ("cm2_20240914_ion_column_2.DTA", 'yellow', 's','4 h'),  # 红色方形
        # ("cm2_20240914_ion_column_3.DTA", 'green', 's','6 h'),  # 红色方形
        # ("cm2_20240914_ion_column_4.DTA", 'blue', 's','8 h'),  # 红色方形



        ("cm2_20240914_ion_0.DTA", 'black', '+', '0 h'),  # 红色方形
        ("cm2_20240914_ion_1.DTA", 'red', '+', '2 h'),  # 红色方形
        ("cm2_20240914_ion_2.DTA", 'orange', '+', '4 h'),  # 红色方形
        ("cm2_20240914_ion_3.DTA", 'yellow', '+', '6 h'),  # 红色方形
        ("cm2_20240914_ion_4.DTA", 'green', '+', '8 h'),  # 红色方形
        ("cm2_20240914_ion_5.DTA", 'blue', '+', '10 h'),  # 红色方形
        ("cm2_20240914_ion_6.DTA", 'cyan', '+', '12 h'),  # 红色方形
        ("cm2_20240914_ion_7.DTA", 'purple', '+', '14 h'),  # 红色方形
        ("cm2_20240914_ion_8.DTA", 'gray', '+', '16 h'),  # 红色方形
        ("cm2_20240914_ion_9.DTA", 'pink', '+', '18 h'),  # 红色方形
        ("cm2_20240914_ion_10.DTA", 'brown', '+', '20 h'),  # 红色方形
        ("cm2_20240914_ion_11.DTA", 'gold', '+', '22 h'),  # 红色方形

        # ("cm2_20240908_ion_13.DTA", 'red', '+', '26 h'),  # 淡紫色
        # ("cm2_20240908_ion_14.DTA", 'orange', '+', '28 h'),  # 栗色
        # ("cm2_20240908_ion_15.DTA", 'yellow', '+', '30 h'),  # 桃色
        # ("cm2_20240908_ion_16.DTA", 'green', '+', '32 h'),  # 卡其色
        # ("cm2_20240908_ion_17.DTA", 'blue', '+', '34 h'),  # 水鸭色
        # ("cm2_20240908_ion_18.DTA", 'cyan', '+', '36 h'),  # 靛蓝色
        # ("cm2_20240908_ion_19.DTA", 'purple', '+', '38 h'),  # 玫瑰色
        # ("cm2_20240908_ion_20.DTA", 'gray', '+', '40 h'),  # 青绿色
        # ("cm2_20240908_ion_21.DTA", 'pink', '+', '42 h'),  # 米色
        # ("cm2_20240908_ion_22.DTA", 'brown', '+', '44 h'),  # 米色
        # ("cm2_20240908_ion_23.DTA", 'gold', '+', '46 h'),  # 米色
        # ("cm2_20240908_ion_24.DTA", 'silver', '+', '48 h'),  # 米色
        # ("cm2_20240908_ion_25.DTA", 'black', '+', '50 h'),  # 米色

        # ("cm2_20240816_ion_renew_1.DTA", 'red', 'x'),  # 红色圆形
        # ("cm2_20240816_ion_renew_4.DTA", 'orange', 'x'),  # 红色方形
        # ("cm2_20240816_ion_renew_8.DTA", 'yellow', 'x'),  # 红色方形
        # ("cm2_20240816_ion_renew_12.DTA", 'green', 'x'),  # 红色方形
        # ("cm2_20240816_ion_renew_16.DTA", 'blue', 'x'),  # 红色方形
        # ("cm2_20240816_ion_renew_20.DTA", 'cyan', 'x'),  # 红色方形
        # ("cm2_20240816_ion_renew_24.DTA", 'purple', 'x'),  # 红色方形
        # ("cm2_20240816_ion_renew_28.DTA", 'brown', 'x'),  # 红色方形
        # ("cm2_20240816_ion_renew_32.DTA", 'black', 'x'),  # 红色方形


        # ("cm2_20240816_ion_renew_ocv_1.DTA", 'red', 'x'),  # 红色圆形
        # ("cm2_20240816_ion_renew_ocv_4.DTA", 'orange', 'x'),  # 红色方形
        # ("cm2_20240816_ion_renew_ocv_8.DTA", 'yellow', 'x'),  # 红色方形
        # ("cm2_20240816_ion_renew_ocv_12.DTA", 'green', 'x'),  # 红色方形
        # ("cm2_20240816_ion_renew_ocv_16.DTA", 'blue', 'x'),  # 红色方形
        # ("cm2_20240816_ion_renew_ocv_20.DTA", 'cyan', 'x'),  # 红色方形
        # ("cm2_20240816_ion_renew_ocv_24.DTA", 'purple', 'x'),  # 红色方形
        # ("cm2_20240816_ion_renew_ocv_28.DTA", 'brown', 'x'),  # 红色方形
]

    convert_and_plot_bode(base_folder, file_specifications)

if __name__ == "__main__":
    main()












