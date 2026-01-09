import os
import pandas as pd



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

#我希望输出文件夹在运行的python代码的上三级目录下的名字叫“数据整理”文件夹下
#请帮我写出python代码，并加入调试bug的代码，供我报错后直接看出哪里有问题。
#
#
import os
import pandas as pd
import glob
import re

def extract_data_from_dta(dta_file, ppm_value, time_value):
    """
    从 DTA 文件中提取数据，包括频率、实部、虚部等。

    参数:
    dta_file : str
        DTA 文件的路径。
    ppm_value : str
        ppm值，从文件夹路径中提取。
    time_value : int
        Time值，从 DTA 文件名中的数字提取。

    返回:
    pd.DataFrame
        包含提取数据的 DataFrame。
    """
    try:
        # 使用utf-8编码或ISO-8859-1编码打开文件
        with open(dta_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        print(f"成功读取文件 {dta_file} (UTF-8编码)")
    except UnicodeDecodeError:
        # 如果 utf-8 编码失败，尝试 ISO-8859-1 编码
        with open(dta_file, 'r', encoding='ISO-8859-1', errors='ignore') as f:
            lines = f.readlines()
        print(f"成功读取文件 {dta_file} (ISO-8859-1编码)")

    # 打印前几行数据进行调试
    print(f"文件内容预览（前5行）：\n{''.join(lines[:5])}")

    # 查找频率、实部和虚部数据的起始行
    freq_start, zreal_start, zimag_start = None, None, None
    for i in range(len(lines) - 1):  # 遍历文件中的每一行，并且避免超出索引
        # 查找 'Freq' 行且下一行包含 'Hz'
        if 'Freq' in lines[i] and 'Hz' in lines[i + 1]:
            freq_start = i + 2  # 频率数据从下一行开始
        # 查找 'Zreal' 行且下一行包含 'ohm'
        if 'Zreal' in lines[i] and 'ohm' in lines[i + 1]:
            zreal_start = i + 2  # 实部数据从下一行开始
        # 查找 'Zimag' 行且下一行包含 'ohm'
        if 'Zimag' in lines[i] and 'ohm' in lines[i + 1]:
            zimag_start = i + 2  # 虚部数据从下一行开始

    # 打印找到的起始行位置
    print(f"频率数据起始行: {freq_start}")
    print(f"实部数据起始行: {zreal_start}")
    print(f"虚部数据起始行: {zimag_start}")

    # 提取频率、实部、虚部数据
    freq_data = []
    zreal_data = []
    zimag_data = []

    if freq_start and zreal_start and zimag_start:
        # 提取频率数据（在D列）
        for line in lines[freq_start:]:
            parts = line.split()
            if len(parts) >= 4:  # 第4列是频率数据（D列）
                try:
                    freq_data.append(float(parts[2]))  # D列的数据
                except ValueError:
                    break

        # 提取实部数据（在E列）
        for line in lines[zreal_start:]:
            parts = line.split()
            if len(parts) >= 5:  # 第5列是实部数据（E列）
                try:
                    zreal_data.append(float(parts[3]))  # E列的数据
                except ValueError:
                    break

        # 提取虚部数据（在F列）
        for line in lines[zimag_start:]:
            parts = line.split()
            if len(parts) >= 6:  # 第6列是虚部数据（F列）
                try:
                    zimag_data.append(float(parts[4]))  # F列的数据
                except ValueError:
                    break

    # 打印提取的数据预览
    print(f"提取的频率数据（前5个）：{freq_data[:5]}")
    print(f"提取的实部数据（前5个）：{zreal_data[:5]}")
    print(f"提取的虚部数据（前5个）：{zimag_data[:5]}")
    
    # 提取文件名中的离子信息
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

    #创建DataFrame
    data = {
        'Time(h)': [time_value] * len(freq_data),
        'Freq': freq_data,
        'Zreal': zreal_data,
        'Zimag': zimag_data,
        'ppm': [ppm_value] * len(freq_data),
        'Label': [label] * len(freq_data)
    }
    #下面的用于检查文件的排序是否正确
    # data = {
    #     'Time(h)': [time_value] * len(freq_data),
    #     'Freq': freq_data,
    #     'Zreal': zreal_data,
    #     'Zimag': zimag_data,
    #     'ppm': [ppm_value] * len(freq_data),
    #     'Label': [file_name] * len(freq_data)
    # }
    
    # 打印生成的 DataFrame 预览
    df = pd.DataFrame(data)
    print(f"生成的DataFrame（前5行）：\n{df.head()}")

    return df




def save_data_to_excel(input_dir, dta_files, output_dir):
    """
    从多个 DTA 文件中提取数据并保存为 Excel 文件。

    参数:
    input_dir : str
        输入的文件夹路径，用于提取ppm值和文件夹名称。
    dta_files : list
        需要处理的 DTA 文件路径列表。
    output_dir : str
        输出文件夹路径，用于保存生成的 Excel 文件。
    """
    # 获取文件夹路径中的ppm值
    ppm_value = re.search(r'(\d*\.?\d+)ppm', input_dir)
    if ppm_value:
        ppm_value = ppm_value.group(1)
    else:
        ppm_value = "未知"

    # 提取 DTA 文件的基本信息
    output_data = []
    for dta_file in dta_files:
        # 提取 DTA 文件名中的最后一个下划线后的数字值，用于计算 Time
        file_name = os.path.basename(dta_file)
        x_value = re.search(r'_(\d+)\.DTA$', file_name)  # 匹配最后一个下划线后的数字
        if x_value:
            x_value = int(x_value.group(1))
        else:
            x_value = 0  # 如果没有匹配到数字，则默认值为0
        
        time_value = 2 * x_value

        # 提取数据
        df = extract_data_from_dta(dta_file, ppm_value, time_value)
        output_data.append(df)
    
    # 合并所有数据
    final_df = pd.concat(output_data, ignore_index=True)
    
    # 根据DTA文件类型决定输出文件名
    if '_ion_column_renew_H2SO4_' in dta_file:
        output_file = os.path.join(output_dir, f"{input_dir.split(os.sep)[-3]}_ion_column_renew_H2SO4_gamry_10h.xlsx")
    elif '_ion_column_' in dta_file:
        output_file = os.path.join(output_dir, f"{input_dir.split(os.sep)[-3]}_ion_column_gamry_10h.xlsx")
    elif '_ion_' in dta_file:
        output_file = os.path.join(output_dir, f"{input_dir.split(os.sep)[-3]}_ion_gamry_10h.xlsx")
    

    # 假设 final_df 是你的 DataFrame
    # 筛选第一列数据为 0、2、4、6、8、10 的行
    filtered_df = final_df[
    final_df.iloc[:, 0].isin([0, 2, 4, 6, 8, 10]) & (final_df.iloc[:, 1] < 22000)
    ]

    print(f"生成的filtered_df（前5行）：\n{filtered_df.head()}")

    # 将筛选后的 DataFrame 保存为 Excel 文件
    filtered_df.to_excel(output_file, index=False)
    print(f"已生成: {output_file}")
    # # 将合并后的 DataFrame 保存为 Excel 文件
    # final_df.to_excel(output_file, index=False)
    # print(f"已生成: {output_file}")

import os
import glob
import re

def sort_files_by_number(file_list):
    """
    按照文件名中的数字部分对文件列表进行排序。

    参数:
    file_list : list
        需要排序的文件路径列表。

    返回:
    list
        排序后的文件路径列表。
    """
    def extract_number(file_path):
        # 提取文件名中的数字部分
        file_name = os.path.basename(file_path)
        match = re.search(r'_(\d+)\.DTA$', file_name)  # 匹配最后一个下划线后的数字
        if match:
            return int(match.group(1))
        return 0  # 如果没有匹配到数字，则返回0作为默认值

    # 按照数字部分排序文件列表
    return sorted(file_list, key=extract_number)

def process_dta_files(input_dirs):
    """
    处理多个文件夹中的 DTA 文件，提取数据并保存为 Excel 文件。

    参数:
    input_dirs : list
        输入文件夹路径的列表，每个文件夹中包含多个 DTA 文件。
    """
    # 获取上三级目录路径，并构建“数据整理”文件夹路径
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    output_dir = os.path.join(base_dir, '数据整理_10h')

    # 如果"数据整理"文件夹不存在，则创建该文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历每个文件夹
    for input_dir in input_dirs:
        input_folder = os.path.join(base_dir, input_dir)
        if not os.path.exists(input_folder):
            print(f"警告: 输入文件夹 {input_folder} 不存在，跳过该文件夹")
            continue
        
        # 获取文件夹中的所有 DTA 文件，并进行精确过滤
        ion_files = glob.glob(os.path.join(input_folder, '*_ion_*.DTA'))
        ion_column_files = glob.glob(os.path.join(input_folder, '*_ion_column_*.DTA'))
        ion_column_renew_H2SO4_files = glob.glob(os.path.join(input_folder, '*_ion_column_renew_H2SO4_*.DTA'))

        # 过滤符合规则的文件（仅保留符合数字模式的文件）
        ion_files = [f for f in ion_files if re.match(r'.*_ion_\d+\.DTA$', os.path.basename(f))]
        ion_column_files = [f for f in ion_column_files if re.match(r'.*_ion_column_\d+\.DTA$', os.path.basename(f))]
        ion_column_renew_H2SO4_files = [f for f in ion_column_renew_H2SO4_files if re.match(r'.*_ion_column_renew_H2SO4_\d+\.DTA$', os.path.basename(f))]

        # 对每个文件列表进行排序
        ion_files = sort_files_by_number(ion_files)
        ion_column_files = sort_files_by_number(ion_column_files)
        ion_column_renew_H2SO4_files = sort_files_by_number(ion_column_renew_H2SO4_files)

        # 分别处理每种类型的文件
        if ion_files:
            save_data_to_excel(input_dir, ion_files, output_dir)
        if ion_column_files:
            save_data_to_excel(input_dir, ion_column_files, output_dir)
        if ion_column_renew_H2SO4_files:
            save_data_to_excel(input_dir, ion_column_renew_H2SO4_files, output_dir)

# 使用示例
input_dirs = [
    # r"20240822_10ppm铜离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    # r"20240823_10ppm钙离子污染和恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    # r"20240827_10ppm铬离子污染和恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    # r"20240831_10ppm镍离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    # r"20240907_10ppm铁离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    # r"20240910_10ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20240915_2ppm铜离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20240918_2ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20241001_2ppm铁离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20241003_2ppm镍离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20241006_2ppm铬离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20241017_2ppm铬离子污染和恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20241028_2ppm钠离子污染和恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20241101_2ppm铜离子污染和恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A",
    # r"20241107_0.1ppm铬离子污染及恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20241112_2ppm钠离子污染和恢复测试80摄氏度\新版电解槽_gamry\EISGALV_80℃_150ml_1A",
    r"20241112_2ppm镍离子污染及恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20241117_2ppm钠离子污染及恢复测试10mlmin\旧版电解槽_gamry\EISGALV_60℃_10ml_1A",
    r"20241117_2ppm钠离子污染及恢复测试40摄氏度\新版电解槽_gamry\EISGALV_40℃_150ml_1A",
    r"20241122_2ppm钠离子污染及恢复测试300mlmin\旧版电解槽_gamry\EISGALV_60℃_300ml_1A",
    r"20241122_2ppm铜离子污染及恢复测试40摄氏度\新版电解槽_gamry\EISGALV_40℃_150ml_1A",
    # r"20241209_无离子污染80摄氏度\新版电解槽_gamry\EISGALV_80℃_150ml_1A",
    r"20241211_2ppm铜离子污染及恢复测试80摄氏度\旧版电解槽_gamry\EISGALV_80℃_150ml_1A",
    r"20241213_2ppm钠离子污染及恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20241214_2ppm铜离子污染及恢复测试300mlmin\旧版电解槽_gamry\EISGALV_60℃_300ml_1A",

    # r"20241227_10ppm铜离子污染及恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A",
    # r"20241229_10ppm钠离子污染及恢复测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",
    # r"20250101_10ppm铬离子污染及恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A",
    r"20250103_无离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A",

   
]

process_dta_files(input_dirs)






