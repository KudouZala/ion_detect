import os
import pandas as pd
import platform
import unicodedata, re, numpy as np

#本代码文件用于将.DTA文件转化为供机器学习训练的样本数据excel文件
#输入,如：20240918_2ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A,这个地址加上代码所在地址的上两级地址作为前缀即是绝对地址，自动对该文件夹下的所有如下每组文件DTA文件进行提取：
# *_ion_数字.DTA（*代表不论是数字还是文字都可以）
# *_ion_column_数字.DTA,（*代表不论是数字还是文字都可以）
# *_ion_column_renew_H2SO4_数字.DTA（*代表不论是数字还是文字都可以）
#每个DTA文件下包含阻抗数据，阻抗数据包含频率、实部、虚部。频率数据在D列，上一行单元格内容为Freq且该行内容为Hz的行的下一行开始,到最后一行;
# 实部数据在E列，上一行单元格内容为Zreal且该行内容为ohm的行的下一行开始,到最后一行；
# 虚部数据在F列，上一行单元格内容为Zimag且该行内容为ohm的行的下一行开始,到最后一行；


#要求最后输出文件名为：20240918_2ppm钙离子污染测试_ion.xlsx、20240918_2ppm钙离子污染测试_ion_column.xlsx、20240918_2ppm钙离子污染测试_ion_column_renew_H2SO4.xlsx即绝对地址的上三级文件夹名
#每个输出的excel中需要包含的title：第一列Time(h),第二列Freq，第三列：Zreal，第四列，Zimag，第五列：ppm，第六列Label，
#你要按照DTA文件中的数字的顺序从0开始，依次提取数据，填充到输出的excel表格中，
# *_ion_数字.DTA填充到20240918_2ppm钙离子污染测试_ion.xlsx中；
#*_ion_column_数字.DTA填充到20240918_2ppm钙离子污染测试_ion_column.xlsx
#*_ion_column_renew_H2SO4_数字.DTA填充到20240918_2ppm钙离子污染测试_ion_column_renew_H2SO4.xlsx中

#第一列：Time(h)，如果DTA文件名中的数字为x，那么Time为2x；
#第二列Freq，取自DTA文件中频率数据；
#第三列Zreal，取自DTA文件中实部数据；
#第四列Zimag，取自DTA文件中虚部数据；
#第五列ppm，取自绝对地址的上三级文件夹名中ppm前面的数字，比方说：“20240918_2ppm钙离子污染测试”中的2
#第六列Label，
# 如果是*_ion_数字.DTA，那么取自绝对地址的上三级文件夹名，比方说：“20240918_2ppm钙离子污染测试”，文件名会用中文标注出钙离子、铝离子、钠离子、铁离子、镍离子、铬离子，那么便成为Ca2+_ion、Al3+_ion、Na+_ion、Fe3+_ion、Ni2+_ion、Cr3+_ion
#如果是*_ion_column_数字.DTA，那么便是no_ion
#如果是*_ion_column_renew_H2SO4_数字.DTA，那么便是no_ion

#输出文件夹在运行的python代码的上三级目录下的名字叫“数据整理”文件夹下

#
#
import os
import pandas as pd
import glob
import re
def extract_data_from_dta(dta_file, ppm_value, time_value, my_range, mean_voltage, current=1, temperature=60, flow=150):
    try:
        with open(dta_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        print(f"成功读取文件 {dta_file} (UTF-8编码)")
    except UnicodeDecodeError:
        with open(dta_file, 'r', encoding='ISO-8859-1', errors='ignore') as f:
            lines = f.readlines()
        print(f"成功读取文件 {dta_file} (ISO-8859-1编码)")

    print(f"文件内容预览（前5行）：\n{''.join(lines[:5])}")

    freq_start, zreal_start, zimag_start = None, None, None
    for i in range(len(lines) - 1):
        if 'Freq' in lines[i] and 'Hz' in lines[i + 1]:
            freq_start = i + 2
        if 'Zreal' in lines[i] and 'ohm' in lines[i + 1]:
            zreal_start = i + 2
        if 'Zimag' in lines[i] and 'ohm' in lines[i + 1]:
            zimag_start = i + 2

    print(f"频率数据起始行: {freq_start}")
    print(f"实部数据起始行: {zreal_start}")
    print(f"虚部数据起始行: {zimag_start}")

    freq_data, zreal_data, zimag_data = [], [], []
    if freq_start and zreal_start and zimag_start:
        for line in lines[freq_start:]:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    freq_data.append(float(parts[2]))
                except ValueError:
                    break
        for line in lines[zreal_start:]:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    zreal_data.append(float(parts[3]))
                except ValueError:
                    break
        for line in lines[zimag_start:]:
            parts = line.split()
            if len(parts) >= 6:
                try:
                    zimag_data.append(float(parts[4]))
                except ValueError:
                    break

    print(f"提取的频率数据（前5个）：{freq_data[:5]}")
    print(f"提取的实部数据（前5个）：{zreal_data[:5]}")
    print(f"提取的虚部数据（前5个）：{zimag_data[:5]}")

    file_name = str(dta_file)
    if "ion_column" in file_name:
        label = 'no_ion'
    else:
        if '钙离子' in file_name:
            label = 'Ca2+_ion'
        elif '铝离子' in file_name:
            label = 'Al3+_ion'
        elif '钠离子' in file_name:
            label = 'Na+_ion'
        elif '铁离子' in file_name:
            label = 'Fe3+_ion'
        elif '镍离子' in file_name:
            label = 'Ni2+_ion'
        elif '铬离子' in file_name:
            label = 'Cr3+_ion'
        elif '铜离子' in file_name:
            label = 'Cu2+_ion'
        else:
            label = 'no_ion'

    data = {
        'Time(h)': [time_value] * len(freq_data),
        'Freq': freq_data,
        'Zreal': zreal_data,
        'Zimag': zimag_data,
        'ppm': [ppm_value] * len(freq_data),
        'mean_voltage': [mean_voltage] * len(freq_data),
        'current': [current] * len(freq_data),
        'temperature': [temperature] * len(freq_data),
        'flow': [flow] * len(freq_data),
        'Label': [label] * len(freq_data)
    }
    df = pd.DataFrame(data)
    print(f"生成的DataFrame（前5行）：\n{df.head()}")
    return df

def get_voltage_average(dta_file):
    try:
        with open(dta_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        print(f"成功读取文件 {dta_file} (UTF - 8编码)")
    except UnicodeDecodeError:
        with open(dta_file, 'r', encoding='ISO-8859-1', errors='ignore') as f:
            lines = f.readlines()
        print(f"成功读取文件 {dta_file} (ISO - 8859 - 1编码)")

    print(f"文件内容预览（前5行）：\n{''.join(lines[:5])}")

    voltage_col_index, voltage_start = None, None
    for i in range(len(lines) - 1):
        line_parts = lines[i].split()
        next_line_parts = lines[i + 1].split()
        for j, part in enumerate(line_parts):
            if part == 'Vf' and j < len(next_line_parts) and next_line_parts[j] == 'V':
                voltage_col_index = j
                voltage_start = i + 2
                break
        if voltage_col_index is not None:
            break

    print(f"电压数据起始行: {voltage_start}")
    print(f"电压数据所在列索引: {voltage_col_index}")

    voltage_data = []
    if voltage_col_index is not None and voltage_start is not None:
        for line in lines[voltage_start:]:
            parts = line.split()
            if len(parts) > voltage_col_index:
                try:
                    voltage = float(parts[voltage_col_index])
                    voltage_data.append(voltage)
                except ValueError:
                    break

    if voltage_data:
        return sum(voltage_data) / len(voltage_data)
    else:
        print("未找到有效的电压数据。")
        return None

# ✅ 新增：电解槽版本识别
def detect_cell_version(path_str: str) -> str:
    if '新版电解槽' in path_str:
        return '新版电解槽'
    if '旧版电解槽' in path_str:
        return '旧版电解槽'
    return '未知电解槽'

def save_data_to_excel(input_dir, dta_files, output_dir, my_range, voltage_files): 
    import re
    temp_value = 60
    flow_value = 150
    if '摄氏度' in input_dir:
        temp_value = re.search(r'(\d*\.?\d+)摄氏度', input_dir).group(1)
    if 'mlmin' in input_dir:
        flow_value = re.search(r'(\d*\.?\d+)mlmin', input_dir).group(1)

    # 先解析 ppm（可能会被下面覆盖）
    import unicodedata, re, numpy as np

    # 统一全角/半角，防止奇怪字符
    s = unicodedata.normalize("NFKC", input_dir)

    # 方案A：去掉 \b（最简单稳妥）
    m = re.search(r'_(\d+(?:\.\d+)?)(?=\s*ppm)', s, flags=re.IGNORECASE)

    ppm_value = float(m.group(1)) if m else np.nan   # 建议用 float，后续数值更安全
    print(f"[ppm] parsed from path: {ppm_value}  | path={input_dir}")

    # ✅ 电解槽版本（加入到文件名中）
    cell_version = detect_cell_version(input_dir)

    # ✅ 统一前缀（包含日期文件夹 + 电解槽版本）
    prefix_date = input_dir.split(os.sep)[-3]  # 如：20240823_10ppm钙离子污染和恢复测试
    prefix = f"{prefix_date}_{cell_version}"

    # ✅ 在循环前先判定文件类型，并在此处就锁定 ppm_value
    if '_ion_column_renew_H2SO4_' in dta_files[0]:
        output_file = os.path.join(output_dir, f"{prefix}_ion_column_renew_H2SO4_gamry_{my_range}.xlsx")
        ppm_value = 0
        # 续酸洗/恢复场景：沿用从路径解析到的 ppm_value
    elif '_ion_column_' in dta_files[0]:
        output_file = os.path.join(output_dir, f"{prefix}_ion_column_gamry_{my_range}.xlsx")
        ppm_value = 0  # ✅ 关键：ion_column 实验固定 ppm=0（在调用前锁定）
    elif '_ion_' in dta_files[0]:
        output_file = os.path.join(output_dir, f"{prefix}_ion_gamry_{my_range}.xlsx")
        # 常规离子污染：保持解析到的 ppm_value
    else:
        # 兜底：未知类型
        output_file = os.path.join(output_dir, f"{prefix}_unknown_gamry_{my_range}.xlsx")

    output_data = []
    for i, dta_file in enumerate(dta_files):
        print("dta_files:\n", dta_files)
        print("i:", i)
        print("voltage_files:\n", voltage_files)

        mean_voltage = get_voltage_average(voltage_files[i])

        file_name = os.path.basename(dta_file)
        x_value = re.search(r'_(\d+)\.DTA$', file_name)
        x_value = int(x_value.group(1)) if x_value else 0
        time_value = 2 * x_value

        df = extract_data_from_dta(
            dta_file, ppm_value, time_value, my_range, mean_voltage,
            temperature=temp_value, flow=flow_value
        )
        output_data.append(df)

    final_df = pd.concat(output_data, ignore_index=True)

    filtered_df = final_df[
        final_df.iloc[:, 0].isin(my_range) & (final_df.iloc[:, 1] < 22000)
    ]
    if all(num in filtered_df.iloc[:, 0].values for num in my_range):
        filtered2_df = filtered_df.copy()
        filtered2_df.iloc[:, 0] = filtered2_df.iloc[:, 0] - my_range[0]
        filtered2_df.to_excel(output_file, index=False)
        print(f"已生成: {output_file}")
    else:
        print(f"由于range中的数字不在，因此无法生成: {output_file}")


def sort_files_by_number(file_list):
    def extract_number(file_path):
        file_name = os.path.basename(file_path)
        match = re.search(r'_(\d+)\.DTA$', file_name)
        return int(match.group(1)) if match else 0
    return sorted(file_list, key=extract_number)

def process_dta_files(input_dirs, my_range):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../校内测试/'))
    output_dir = os.path.join(base_dir, '数据整理_range')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for input_dir in input_dirs:
        input_folder = os.path.join(base_dir, input_dir)
        if not os.path.exists(input_folder):
            print(f"警告: 输入文件夹 {input_folder} 不存在，跳过该文件夹")
            continue
        parent_folder = os.path.dirname(input_folder)
        voltage_folder = None
        for item in os.listdir(parent_folder):
            item_path = os.path.join(parent_folder, item)
            if os.path.isdir(item_path) and item.startswith('PWRGALVANOSTATIC_'):
                voltage_folder = item_path
                break

        ion_files = glob.glob(os.path.join(input_folder, '*_ion_*.DTA'))
        voltage_ion_files = glob.glob(os.path.join(voltage_folder, '*_ion_*.DTA')) if voltage_folder else []
        print("voltage_ion_files:", voltage_ion_files)

        ion_column_files = glob.glob(os.path.join(input_folder, '*_ion_column_*.DTA'))
        voltage_ion_column_files = glob.glob(os.path.join(voltage_folder, '*_ion_column_*.DTA')) if voltage_folder else []

        ion_column_renew_H2SO4_files = glob.glob(os.path.join(input_folder, '*_ion_column_renew_H2SO4_*.DTA'))
        voltage_ion_column_renew_H2SO4_files = glob.glob(os.path.join(voltage_folder, '*_ion_column_renew_H2SO4_*.DTA')) if voltage_folder else []
        print("voltage_ion_column_renew_H2SO4_files:", voltage_ion_column_renew_H2SO4_files)

        ion_files = [f for f in ion_files if re.match(r'.*_ion_\d+\.DTA$', os.path.basename(f))]
        voltage_ion_files = [f for f in voltage_ion_files if re.match(r'.*_ion_\d+\.DTA$', os.path.basename(f))]
        ion_column_files = [f for f in ion_column_files if re.match(r'.*_ion_column_\d+\.DTA$', os.path.basename(f))]
        voltage_ion_column_files = [f for f in voltage_ion_column_files if re.match(r'.*_ion_column_\d+\.DTA$', os.path.basename(f))]
        ion_column_renew_H2SO4_files = [f for f in ion_column_renew_H2SO4_files if re.match(r'.*_ion_column_renew_H2SO4_\d+\.DTA$', os.path.basename(f))]
        voltage_ion_column_renew_H2SO4_files = [f for f in voltage_ion_column_renew_H2SO4_files if re.match(r'.*_ion_column_renew_H2SO4_\d+\.DTA$', os.path.basename(f))]

        ion_files = sort_files_by_number(ion_files)
        voltage_ion_files = sort_files_by_number(voltage_ion_files)
        ion_column_files = sort_files_by_number(ion_column_files)
        voltage_ion_column_files = sort_files_by_number(voltage_ion_column_files)
        ion_column_renew_H2SO4_files = sort_files_by_number(ion_column_renew_H2SO4_files)
        voltage_ion_column_renew_H2SO4_files = sort_files_by_number(voltage_ion_column_renew_H2SO4_files)

        if ion_files:
            save_data_to_excel(input_dir, ion_files, output_dir, my_range, voltage_ion_files)
        if ion_column_files:
            save_data_to_excel(input_dir, ion_column_files, output_dir, my_range, voltage_ion_column_files)
        if ion_column_renew_H2SO4_files:
            save_data_to_excel(input_dir, ion_column_renew_H2SO4_files, output_dir, my_range, voltage_ion_column_renew_H2SO4_files)

# 使用示例
input_dirs = [
    r"20240822_10ppm铜离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20240823_10ppm钙离子污染和恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20240827_10ppm铬离子污染和恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20240831_10ppm镍离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20240907_10ppm铁离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20240910_10ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20240915_2ppm铜离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20240918_2ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20241001_2ppm铁离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20241003_2ppm镍离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20241006_2ppm铬离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20241017_2ppm铬离子污染和恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20241028_2ppm钠离子污染和恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20241101_2ppm铜离子污染和恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20241107_0.1ppm铬离子污染及恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20241112_2ppm钠离子污染和恢复测试80摄氏度\新版电解槽_gamry\EISGALV_80℃_150ml_1A",
    r"20241112_2ppm镍离子污染及恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20241117_2ppm钠离子污染及恢复测试10mlmin\旧版电解槽_gamry\EISGALV_60℃_10ml_1A",
    r"20241117_2ppm钠离子污染及恢复测试40摄氏度\新版电解槽_gamry\EISGALV_40℃_150ml_1A",
    r"20241122_2ppm钠离子污染及恢复测试300mlmin\旧版电解槽_gamry\EISGALV_60℃_300ml_1A",
    r"20241122_2ppm铜离子污染及恢复测试40摄氏度\新版电解槽_gamry\EISGALV_40℃_150ml_1A",
    r"20241209_无离子污染80摄氏度\新版电解槽_gamry\EISGALV_80℃_150ml_1A",
    r"20241211_2ppm铜离子污染及恢复测试80摄氏度\旧版电解槽_gamry\EISGALV_80℃_150ml_1A",
    r"20241213_2ppm钠离子污染及恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20241214_2ppm铜离子污染及恢复测试300mlmin\旧版电解槽_gamry\EISGALV_60℃_300ml_1A",
    r"20241227_10ppm铜离子污染及恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20241229_10ppm钠离子污染及恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20250101_10ppm铬离子污染及恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20250103_无离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20250317_2ppm钠离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20250321_2ppm铝离子污染测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A",
]
# # 判断系统类型
if platform.system().lower() != "windows":
    # Ubuntu / Linux / macOS，替换为斜杠
    input_dirs = [p.replace("\\", "/") for p in input_dirs]

# # ✅ Debug打印确认
# for p in input_dirs:
#     print(p)
for start in range(0, 91, 2):
    my_range = [start, start + 2, start + 4, start + 6]
    process_dta_files(input_dirs, my_range)






