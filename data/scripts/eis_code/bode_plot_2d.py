import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


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


def plot_2d_diff_for_frequencies(folder_path, frequencies, file_specifications, all_freq, all_delta_zreal,
                                 all_delta_zimag):
    output_folder_plot = os.path.join(folder_path, "output_plot")
    os.makedirs(output_folder_plot, exist_ok=True)

    file_names = [os.path.basename(f[0]).replace('.DTA', '') for f in file_specifications]

    for freq in frequencies:
        freq_log = np.log10(freq)

        for data_type, all_delta_data, label in [('Zreal', all_delta_zreal, 'Real'),
                                                 ('Zimag', all_delta_zimag, 'Imaginary')]:
            fig, ax = plt.subplots(figsize=(10, 6))

            for i in range(len(all_freq)):
                closest_idx = np.argmin(np.abs(all_freq[i] - freq_log))
                ax.plot(file_names, [delta[closest_idx] for delta in all_delta_data], marker='o',
                        label=f"{freq}Hz {data_type}")

            ax.set_xlabel('File Index')
            ax.set_ylabel(f'Delta {label} Part (Ohm)')
            ax.set_title(f'{freq}Hz - Delta {label} Part vs File Index')
            ax.grid(True)
            plt.xticks(rotation=45, ha='right')

            plot_filename = f"bode_{data_type.lower()}_plot_diff_2d_{freq}Hz.png"
            plt.savefig(os.path.join(output_folder_plot, plot_filename))
            plt.close()


def convert_and_plot_3d_surface_and_2d_diff(folder_path, file_specifications, frequencies):
    output_folder_xlsx = os.path.join(folder_path, "output_xlsx")
    os.makedirs(output_folder_xlsx, exist_ok=True)

    output_folder_csv = os.path.join(folder_path, "output_csv")
    os.makedirs(output_folder_csv, exist_ok=True)

    output_folder_plot = os.path.join(folder_path, "output_plot")
    os.makedirs(output_folder_plot, exist_ok=True)

    fig_3d_real = plt.figure(figsize=(12, 8))
    ax_3d_real = fig_3d_real.add_subplot(111, projection='3d')

    fig_3d_imag = plt.figure(figsize=(12, 8))
    ax_3d_imag = fig_3d_imag.add_subplot(111, projection='3d')

    y_values = []

    all_freq_real = []
    all_freq_imag = []
    all_zreal = []
    all_zimag = []

    for idx, (file_name, color, marker) in enumerate(file_specifications):
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
            excel_file_path = os.path.join(output_folder_xlsx,
                                           f"{os.path.basename(file_name).replace('.DTA', '.xlsx')}")
            df.to_excel(excel_file_path, index=False)

            # 保存为CSV文件（不含标题行）
            csv_file_path = os.path.join(output_folder_csv, f"{os.path.basename(file_name).replace('.DTA', '.csv')}")
            df.to_csv(csv_file_path, index=False, header=False)

            y_value = [idx] * len(df['Freq'])
            y_values.append(y_value)

            all_freq_real.append(np.log10(df['Freq']))
            all_freq_imag.append(np.log10(df['Freq']))

            all_zreal.append(df['Zreal'])
            all_zimag.append(df['Zimag'])

            # 绘制实部的三维曲面
            ax_3d_real.plot(np.log10(df['Freq']), y_value, df['Zreal'], label=f"{file_name}", color=color,
                            marker=marker)

            # 绘制虚部的三维曲面
            ax_3d_imag.plot(np.log10(df['Freq']), y_value, df['Zimag'], label=f"{file_name}", color=color,
                            marker=marker)
        else:
            print(f"File not found: {file_path}")

    # 将数据转换为二维矩阵形式以绘制曲面
    y_values = pd.DataFrame(y_values).values
    all_freq_real = pd.DataFrame(all_freq_real).values
    all_zreal = pd.DataFrame(all_zreal).values
    all_freq_imag = pd.DataFrame(all_freq_imag).values
    all_zimag = pd.DataFrame(all_zimag).values

    # 绘制实部曲面
    ax_3d_real.plot_surface(all_freq_real, y_values, all_zreal, cmap='viridis', alpha=0.6)

    ax_3d_real.set_xlabel('Log(Frequency) (Hz)')
    ax_3d_real.set_ylabel('File Index')
    ax_3d_real.set_zlabel('Zreal (Ohm)')
    ax_3d_real.set_title('3D Bode Surface Plot - Real Part')

    # 绘制虚部曲面
    ax_3d_imag.plot_surface(all_freq_imag, y_values, all_zimag, cmap='plasma', alpha=0.6)

    ax_3d_imag.set_xlabel('Log(Frequency) (Hz)')
    ax_3d_imag.set_ylabel('File Index')
    ax_3d_imag.set_zlabel('Zimag (Ohm)')
    ax_3d_imag.set_title('3D Bode Surface Plot - Imaginary Part')

    # 保存图像
    real_plot_path = os.path.join(output_folder_plot, "bode_real_plot_3d.png")
    imag_plot_path = os.path.join(output_folder_plot, "bode_imag_plot_3d.png")

    fig_3d_real.savefig(real_plot_path)
    fig_3d_imag.savefig(imag_plot_path)

    # 绘制并保存指定频率下的2D图
    plot_2d_diff_for_frequencies(folder_path, frequencies, file_specifications, all_freq_real, all_zreal,
                                 all_zimag)

    plt.show()

def main():
    base_folder = r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\20240823_10ppm铜离子污染测试\20240823\EISGALV_60℃_150ml_1A"

    # 指定每个文件的文件名、颜色和标记符号
    file_specifications = [
        # ("cm2_20240821_ion_column_1.DTA", 'red', 's'),  # 红色圆形
        # ("cm2_20240821_ion_column_4.DTA", 'orange', 's'),  # 红色方形
        # ("cm2_20240821_ion_column_8.DTA", 'yellow', 's'),  # 红色方形
        # ("cm2_20240821_ion_column_12.DTA", 'green', 's'),  # 红色方形
        # ("cm2_20240821_ion_column_16.DTA", 'blue', 's'),  # 红色方形
        # ("cm2_20240821_ion_column_20.DTA", 'cyan', 's'),  # 红色方形
        # ("cm2_20240821_ion_column_24.DTA", 'purple', 's'),  # 红色方形
        # ("cm2_20240821_ion_column_28.DTA", 'gray', 's'),  # 红色方形
        # ("cm2_20240821_ion_column_31.DTA", 'pink', 's'),  # 红色方形

        # ("cm2_20240822_ion_column_1.DTA", 'red', 's'),  # 红色圆形
        # ("cm2_20240822_ion_column_2.DTA", 'orange', 's'),  # 红色方形
        # ("cm2_20240822_ion_column_3.DTA", 'yellow', 's'),  # 红色方形
        # ("cm2_20240822_ion_column_4.DTA", 'green', 's'),  # 红色方形
        # ("cm2_20240822_ion_column_5.DTA", 'blue', 's'),  # 红色方形
        # ("cm2_20240822_ion_column_6.DTA", 'cyan', 's'),  # 红色方形
        # ("cm2_20240822_ion_column_7.DTA", 'purple', 's'),  # 红色方形
        # ("cm2_20240822_ion_column_8.DTA", 'gray', 's'),  # 红色方形
        # ("cm2_20240822_ion_column_9.DTA", 'pink', 's'),  # 红色方形
        # ("cm2_20240822_ion_column_10.DTA", 'black', 's'),  # 红色方形

        ("cm2_20240822_ion_0.DTA", 'lightcoral', '+'),  # 红色圆形
        ("cm2_20240822_ion_1.DTA", 'red', '+'),  # 红色圆形
        ("cm2_20240822_ion_2.DTA", 'orange', '+'),  # 红色方形
        ("cm2_20240822_ion_3.DTA", 'yellow', '+'),  # 红色方形
        ("cm2_20240822_ion_4.DTA", 'green', '+'),  # 红色方形
        ("cm2_20240822_ion_5.DTA", 'blue', '+'),  # 红色方形
        ("cm2_20240822_ion_6.DTA", 'cyan', '+'),  # 红色方形
        ("cm2_20240822_ion_7.DTA", 'purple', '+'),  # 红色方形
        ("cm2_20240822_ion_8.DTA", 'gray', '+'),  # 红色方形
        ("cm2_20240822_ion_9.DTA", 'pink', '+'),  # 红色方形
        ("cm2_20240822_ion_10.DTA", 'black', '+'),  # 红色方形


    ]
    # 指定感兴趣的频率
    frequencies = [0.01, 0.1, 1, 10, 100, 1000, 10000]

    convert_and_plot_3d_surface_and_2d_diff(base_folder, file_specifications, frequencies)



if __name__ == "__main__":
    main()