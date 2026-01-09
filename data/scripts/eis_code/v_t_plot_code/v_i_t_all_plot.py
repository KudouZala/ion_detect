import pandas as pd
import plotly.graph_objects as go
import os

# 文件夹路径设置
base_path = r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\20240907_10ppm铁离子污染测试\20240907"
pwrgalv_folder = os.path.join(base_path, "PWRGALVANOSTATIC_60℃_150ml_1A")
ocp_folder = os.path.join(base_path, "OCP_60℃_150ml_1A")
eisgalv_folder = os.path.join(base_path, "EISGALV_60℃_150ml_1A")


# 转换文件格式为 Excel
def convert_to_excel(input_file, output_file):
    try:
        df = pd.read_csv(input_file, sep='\t', header=None, encoding='utf-8', error_bad_lines=False)
        df.to_excel(output_file, index=False, header=False)
        print(f"Converted to Excel: {output_file}")
    except Exception as e:
        print(f"Error converting file: {e}")


# 读取 Excel 文件
def read_excel_file(filepath):
    try:
        df = pd.read_excel(filepath, header=None)
        print(f"Successfully read Excel file: {filepath}")
        return df
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return pd.DataFrame()


# 找到 T 或 Time 所在的列
def find_time_column(df):
    for col in df.columns:
        if df.iloc[0, col] in ['T', 'Time'] and df.iloc[1, col] == 's':
            return col
    return None


# 处理文件
def process_files():
    fig = go.Figure()
    time_offset = 0

    # 处理 PWRGALVANOSTATIC 文件
    for i in range(21):
        input_file = os.path.join(pwrgalv_folder, f"cm2_20240905_ion_{i}.DTA")
        excel_file = input_file.replace('.DTA', '.xlsx')
        convert_to_excel(input_file, excel_file)

        df = read_excel_file(excel_file)
        time_col = find_time_column(df)

        if time_col is not None:
            df.columns = ['T', 'Vf', 'Im']
            df['T'] = pd.to_numeric(df[time_col], errors='coerce') + time_offset
            df['Vf'] = pd.to_numeric(df['Vf'], errors='coerce')
            df['Im'] = pd.to_numeric(df['Im'], errors='coerce')
            time_offset = df['T'].max()

            fig.add_trace(go.Scatter(x=df['T'], y=df['Vf'], mode='lines', name=f'PWRGALVANOSTATIC Vf {i}'))
            fig.add_trace(go.Scatter(x=df['T'], y=df['Im'], mode='lines', name=f'PWRGALVANOSTATIC Im {i}'))

    # 处理 OCP 文件
    for i in range(22):
        input_file = os.path.join(ocp_folder, f"cm2_20240905_ion_{i}.DTA")
        excel_file = input_file.replace('.DTA', '.xlsx')
        convert_to_excel(input_file, excel_file)

        df = read_excel_file(excel_file)
        time_col = find_time_column(df)

        if time_col is not None:
            df.columns = ['T', 'Vf']
            df['T'] = pd.to_numeric(df[time_col], errors='coerce') + time_offset
            df['Vf'] = pd.to_numeric(df['Vf'], errors='coerce')
            time_offset = df['T'].max()

            fig.add_trace(go.Scatter(x=df['T'], y=df['Vf'], mode='lines', name=f'OCP Vf {i}'))

    # 处理 EISGALV 文件
    for i in range(22):
        input_file = os.path.join(eisgalv_folder, f"cm2_20240905_ion_{i}.DTA")
        excel_file = input_file.replace('.DTA', '.xlsx')
        convert_to_excel(input_file, excel_file)

        df = read_excel_file(excel_file)
        df.columns = ['Idc', 'Vdc']
        df['T'] = pd.to_numeric(df.index, errors='coerce') + time_offset
        df['Idc'] = pd.to_numeric(df['Idc'], errors='coerce')
        df['Vdc'] = pd.to_numeric(df['Vdc'], errors='coerce')

        fig.add_trace(go.Scatter(x=df['T'], y=df['Vdc'], mode='lines', name=f'EISGALV Vdc {i}'))
        fig.add_trace(go.Scatter(x=df['T'], y=df['Idc'], mode='lines', name=f'EISGALV Idc {i}'))

    fig.update_layout(
        title='Voltage and Current vs Time',
        xaxis_title='Time',
        yaxis_title='Voltage / Current',
        legend_title='Legend',
        template='plotly_dark'
    )

    # 保存为 HTML 文件
    fig.write_html(
        r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\20240907_10ppm铁离子污染测试\20240907\output_plot.html")
    print("Plot saved as output_plot.html")


process_files()


