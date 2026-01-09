import os
import pandas as pd

#本代码文件用于将firecloud设备的csv数据文件转化为供机器学习训练的样本数据excel文件
#输入,如：20241101_2ppm铜离子污染和恢复测试\旧版电解槽_firecloud,这个地址加上代码所在地址的上两级地址作为前缀即是绝对地址，
# 该文件夹下可能有三种文件夹，*_ion_column，*_ion，*_ion_column_renew
# 自动对每个文件夹下的所有如下每组文件csv文件进行数据提取：
#名字为：循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv作为第0组数据
#循环1／1_工步组2(工步组)(1／80)_工步3(阻抗).csv作为第1组数据
#循环1／1_工步组2(工步组)(2／80)_工步3(阻抗).csv作为第2组数据
#后面的以此类推，工步组2(工步组)(与/之间的数字，作为第几组数据

#每个csv文件下包含阻抗数据，阻抗数据包含频率、实部、虚部。频率数据在title为Freq/Hz列；
# 实部数据在Re(Z)/Ohm列
# 虚部数据在Im(Z)/Ohm列


#要求最后输出文件名为：20241101_2ppm铜离子污染和恢复测试_firecloud_ion.xlsx、20241101_2ppm铜离子污染和恢复测试_firecloud_ion_column.xlsx、20240918_2ppm钙离子污染测试_firecloud_ion_column_renew.xlsx即输入的地址的上一级文件夹名加上后缀
#每个输出的excel中需要包含的title：第一列Time(h),第二列Freq，第三列：Zreal，第四列，Zimag，第五列：ppm，第六列Label，

#你要按照csv文件中的的顺序从0开始，依次提取数据，填充到输出的excel表格中，
# *_ion_column下的csv文件提取的数据填充到20241101_2ppm铜离子污染和恢复测试_ion_column.xlsx中；
# *_ion下的csv文件提取的数据填充到20241101_2ppm铜离子污染和恢复测试_ion_column.xlsx中；
# *_ion_column_renew下的csv文件提取的数据填充到20241101_2ppm铜离子污染和恢复测试_ion_column_renew.xlsx中；

#第一列：Time(h)，如果csv文件是第x组，那么Time为2x；
#第二列Freq，取自csv文件中频率数据；
#第三列Zreal，取自csv文件中实部数据；
#第四列Zimag，取自csv文件中虚部数据；
#第五列ppm，取自绝对地址的上三级文件夹名中ppm前面的数字，比方说：“20241101_2ppm铜离子污染和恢复测试”中的2
#第六列Label，
# 如果是*_ion_column和*_ion_column_renew文件夹，那么label=no_ion，
# 如果是*_ion，那么取自绝对地址的上三级文件夹名，比方说：“20240918_2ppm钙离子污染测试”，文件名会用中文标注出钙离子、铝离子、钠离子、铁离子、镍离子、铬离子，那么便成为Ca2+_ion、Al3+_ion、Na+_ion、Fe3+_ion、Ni2+_ion、Cr3+_ion

#我希望输出文件夹在运行的python代码的上三级目录下的名字叫“数据整理”文件夹下
#请帮我写出python代码，并加入调试bug的代码，供我报错后直接看出哪里有问题。
import os
import glob
import re
import pandas as pd
import os
import glob
import re
import pandas as pd

def extract_data_from_csv(csv_file, time_value, ppm_value, label_value):
    """
    从 CSV 文件中提取阻抗数据，并返回 DataFrame。

    参数:
    csv_file : str
        CSV 文件路径。
    time_value : int
        当前组的时间值（2 * 当前组编号）。
    ppm_value : str
        ppm值，从文件夹路径中提取。
    label_value : str
        离子标签，根据文件夹名称和CSV文件的类型确定。

    返回:
    pd.DataFrame
        包含提取数据的 DataFrame。
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        print(f"成功读取文件 {csv_file}")
    except Exception as e:
        print(f"读取文件 {csv_file} 时发生错误: {e}")
        return None

    # 检查必要列是否存在
    if 'Freq/Hz' not in df.columns or 'Re(Z)/Ohm' not in df.columns or 'Im(Z)/Ohm' not in df.columns:
        print(f"文件 {csv_file} 中缺少必要的列（Freq/Hz, Re(Z)/Ohm, Im(Z)/Ohm）。")
        return None

    # 提取数据
    freq_data = df['Freq/Hz'].tolist()
    zreal_data = df['Re(Z)/Ohm'].tolist()
    zimag_data = df['Im(Z)/Ohm'].tolist()

    # 构建 DataFrame
    data = {
        'Time(h)': [time_value] * len(freq_data),
        'Freq': freq_data,
        'Zreal': zreal_data,
        'Zimag': zimag_data,
        'ppm': [ppm_value] * len(freq_data),
        'Label': [label_value] * len(freq_data)
    }
    
    result_df = pd.DataFrame(data)
    print(f"生成的DataFrame（前5行）：\n{result_df.head()}")
    
    return result_df
def custom_sort(file_path):
    """
    自定义排序函数，根据文件名中的两个数字进行排序：
    1. 根据 '_工步组' 和 第一个 '(' 之间的数字进行排序（第一层排序）。
    2. 根据 '工步组)' 和 第一个 '/' 之间的数字进行排序（第二层排序）。
    """
    filename = os.path.basename(file_path)

    # 提取第一层排序：'_工步组' 和 '(' 之间的数字
    group_number = re.search(r'_工步组(\d+)\(', filename)  # 确保括号是英文括号
    if group_number:
        group_number = int(group_number.group(1))  # 转为整数
    else:
        group_number = float('inf')  # 如果没有找到，返回最大值，排在最后

    # 提取第二层排序：'工步组)' 和 '/' 之间的数字
    # 正则表达式修改
    step_number = re.search(r'工步组\)\((\d+)／', filename)  # 匹配"工步组)"和中文"/"之间的数字

    if step_number:
        step_number = int(step_number.group(1))  # 转为整数
    else:
        step_number = float('inf')  # 如果没有找到，返回最大值，排在最后

    # 返回的元组将用于排序：首先按 group_number 排序，其次按 step_number 排序
    return (group_number, step_number)

def process_csv_files(input_dir, output_dir, range):
    """
    处理每个文件夹下的 CSV 文件，提取数据并保存为 Excel 文件。

    参数:
    input_dir : str
        输入的文件夹路径，用于提取ppm值和文件夹名称。
    output_dir : str
        输出文件夹路径，用于保存生成的 Excel 文件。
    """
    # 获取文件夹路径中的 ppm 值
    
    
    ppm_value = re.search(r'(\d*\.?\d+)ppm', input_dir)
    if ppm_value:
        ppm_value = ppm_value.group(1)
    else:
        ppm_value = "未知"

    # 获取文件夹名称中的离子信息
    label_value = "no_ion"
    if '离子污染' in input_dir:
        if '钙' in input_dir:
            label_value = "Ca2+_ion"
        elif '铝' in input_dir:
            label_value = "Al3+_ion"
        elif '钠' in input_dir:
            label_value = "Na+_ion"
        elif '铁' in input_dir:
            label_value = "Fe3+_ion"
        elif '镍' in input_dir:
            label_value = "Ni2+_ion"
        elif '铬' in input_dir:
            label_value = "Cr3+_ion"
        elif '铜' in input_dir:
            label_value = "Cu2+_ion"
        else:
            label_value = "no_ion"

    # 获取文件夹中的所有子文件夹（只获取包含“（阻抗）”字样的 CSV 文件）
    ion_column_files = glob.glob(os.path.join(input_dir, '*_ion_column', '*(阻抗)*.csv'))
    ion_files = glob.glob(os.path.join(input_dir, '*_ion', '*(阻抗)*.csv'))
    ion_column_renew_files = glob.glob(os.path.join(input_dir, '*_ion_column_renew', '*(阻抗)*.csv'))

    # 排序 CSV 文件，确保文件按名称顺序（如循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv）
    ion_column_files.sort(key=custom_sort)
    ion_files.sort(key=custom_sort)
    ion_column_renew_files.sort(key=custom_sort)

    # 获取 input_dir 的上一级目录的文件夹名字
    parent_dir_name = os.path.basename(os.path.dirname(input_dir))


    range_str=f"{range[0]}-{range[-1]}"
    # 处理 *_ion_column 目录下的 CSV 文件
    if ion_column_files:
        output_data = []
        for i, csv_file in enumerate(ion_column_files):
            time_value = 2 * i
            df = extract_data_from_csv(csv_file, time_value, ppm_value, "no_ion")
            if df is not None:
                output_data.append(df)
        
        # 将数据保存到 Excel 文件
        if output_data:
            final_df = pd.concat(output_data, ignore_index=True)
            output_file = os.path.join(output_dir, f"{os.path.basename(parent_dir_name)}_ion_column_firecloud_{range_str}.xlsx")
            # 假设 final_df 是你的 DataFrame
            # 筛选第一列数据为 0、2、4、6、8、10 的行
            filtered_df = final_df[final_df.iloc[:, 0].isin(range)]
            print(f"生成的filtered_df（前5行）：\n{filtered_df.head()}")

            # 将筛选后的 DataFrame 保存为 Excel 文件
            filtered_df.to_excel(output_file, index=False)
            print(f"已生成: {output_file}")
            # final_df.to_excel(output_file, index=False)
            # print(f"已生成: {output_file}")

    # 处理 *_ion 目录下的 CSV 文件
    if ion_files:
        output_data = []
        for i, csv_file in enumerate(ion_files):
            time_value = 2 * i
            df = extract_data_from_csv(csv_file, time_value, ppm_value, label_value)
            if df is not None:
                output_data.append(df)
        
        # 将数据保存到 Excel 文件
        if output_data:
            final_df = pd.concat(output_data, ignore_index=True)
            output_file = os.path.join(output_dir, f"{os.path.basename(parent_dir_name)}_ion_firecloud_{range_str}.xlsx")
            # 假设 final_df 是你的 DataFrame
            # 筛选第一列数据为 0、2、4、6、8、10 的行
            filtered_df = final_df[final_df.iloc[:, 0].isin(range)]
            print(f"生成的filtered_df（前5行）：\n{filtered_df.head()}")

            # 将筛选后的 DataFrame 保存为 Excel 文件
            filtered_df.to_excel(output_file, index=False)
            print(f"已生成: {output_file}")
            # final_df.to_excel(output_file, index=False)
            # print(f"已生成: {output_file}")

    # 处理 *_ion_column_renew 目录下的 CSV 文件
    if ion_column_renew_files:
        output_data = []
        for i, csv_file in enumerate(ion_column_renew_files):
            time_value = 2 * i
            df = extract_data_from_csv(csv_file, time_value, ppm_value, "no_ion")
            if df is not None:
                output_data.append(df)
        
        # 将数据保存到 Excel 文件
        if output_data:
            final_df = pd.concat(output_data, ignore_index=True)
            output_file = os.path.join(output_dir, f"{os.path.basename(parent_dir_name)}_ion_column_renew_firecloud_{range_str}.xlsx")
            # 假设 final_df 是你的 DataFrame
            # 筛选第一列数据为 0、2、4、6、8、10 的行
            filtered_df = final_df[final_df.iloc[:, 0].isin(range)]
            print(f"生成的filtered_df（前5行）：\n{filtered_df.head()}")

            # 将筛选后的 DataFrame 保存为 Excel 文件
            filtered_df.to_excel(output_file, index=False)
            print(f"已生成: {output_file}")
            # final_df.to_excel(output_file, index=False)
            # print(f"已生成: {output_file}")

def process_dta_files(input_dirs,range):
    """
    处理多个文件夹中的 DTA 文件，提取数据并保存为 Excel 文件。

    参数:
    input_dirs : list
        输入文件夹路径的列表，每个文件夹中包含多个 DTA 文件。
    """
    # 获取上三级目录路径，并构建“数据整理”文件夹路径
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    output_dir = os.path.join(base_dir, '数据整理_6h')

    # 如果"数据整理"文件夹不存在，则创建该文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历每个文件夹
    for input_dir in input_dirs:
        input_folder = os.path.join(base_dir, input_dir)
        if not os.path.exists(input_folder):
            print(f"警告: 输入文件夹 {input_folder} 不存在，跳过该文件夹")
            continue

        # 处理该文件夹下的 CSV 文件
        process_csv_files(input_folder, output_dir,range)

# 调用示例
input_dirs = [
            # r"20240915_2ppm铜离子污染测试\旧版电解槽_firecloud",
            # r"20241001_2ppm铁离子污染测试\旧版电解槽_firecloud",
            # r"20241003_2ppm镍离子污染测试\旧版电解槽_firecloud",
            # r"20241006_2ppm铬离子污染测试\旧版电解槽_firecloud",
            # r"20241008_无离子污染测试\新版电解槽_firecloud",
            # r"20241010_2ppm钠离子污染测试\新版电解槽_firecloud",
            # r"20241013_2ppm铝离子污染测试\新版电解槽_firecloud",
            # r"20241024_2ppm铁离子污染和恢复测试\新版电解槽_firecloud",
            # r"20241029_2ppm铁离子污染和恢复测试\新版电解槽_firecloud",
            # r"20241101_2ppm钙离子污染和恢复测试\新版电解槽_firecloud",
            # r"20241101_2ppm铜离子污染和恢复测试\旧版电解槽_firecloud",
            r"20241107_0.1ppm钙离子污染及恢复测试\新版电解槽_firecloud",
            r"20241107_0.1ppm铬离子污染及恢复测试\旧版电解槽_firecloud",
            # r"20241201_2ppm铜离子污染及恢复测试\新版电解槽_firecloud",

              ]
range=[2,4,6,8]
process_dta_files(input_dirs,range)








