import os
import pandas as pd
from pathlib import Path

def extract_eis_data_from_csv(file_path, remove_first_n_points=0):
    """
    从炙云的CSV文件中提取EIS数据。

    参数：
    - file_path: CSV文件的路径。
    - remove_first_n_points: 要移除的前n个数据点。

    返回：
    - 包含 'Freq', 'Zreal', 'Zimag' 列的数据字典。
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"无法读取文件 {file_path}: {e}")
        return None

    required_columns = ['Freq/Hz', 'Re(Z)/Ohm', 'Im(Z)/Ohm']
    if not all(col in df.columns for col in required_columns):
        print(f"CSV文件 {file_path} 中没有找到所有必需的列: {required_columns}")
        return None

    # 移除前n个数据点
    if remove_first_n_points > 0:
        df = df.iloc[remove_first_n_points:]

    try:
        data = {
            'Freq': df['Freq/Hz'].astype(float).values,
            'Zreal': df['Re(Z)/Ohm'].astype(float).values,
            'Zimag': df['Im(Z)/Ohm'].astype(float).values
        }
    except ValueError as ve:
        print(f"数据转换错误在文件 {file_path}: {ve}")
        return None

    return data

def extract_positive_imaginary_data_from_csv(file_path, remove_first_n_points=0):
    """
    从炙云的CSV文件中提取虚部大于0的数据。

    参数：
    - file_path: CSV文件的路径。
    - remove_first_n_points: 要移除的前n个数据点。

    返回：
    - 包含 'Freq', 'Zreal', 'Zimag' 列的过滤后数据字典。
    """
    data = extract_eis_data_from_csv(file_path, remove_first_n_points)
    
    if data is None:
        return None

    # 筛选虚部大于0的数据
    positive_imaginary_mask = data['Zimag'] < 0
    filtered_data = {
        'Freq': data['Freq'][positive_imaginary_mask],
        'Zreal': data['Zreal'][positive_imaginary_mask],
        'Zimag': data['Zimag'][positive_imaginary_mask]
    }

    return filtered_data

def convert_zhiyun_to_txt(base_folder, zhiyun_files, output_folder, remove_first_n_points=0):
    """
    将指定的炙云CSV文件转换为TXT文件，并保存在output_folder的output_txt子文件夹中。

    参数：
    - base_folder: 包含炙云CSV文件的基准文件夹路径。
    - zhiyun_files: 包含炙云CSV文件相对路径的列表。
    - output_folder: TXT文件的输出文件夹路径。
    - remove_first_n_points: 要移除的前n个数据点。
    """
    # 创建output_txt子文件夹路径
    txt_output_folder = os.path.join(output_folder, "output_txt")
    os.makedirs(txt_output_folder, exist_ok=True)
    print(f"TXT文件将保存在: {txt_output_folder}")

    for file_path_simp in zhiyun_files:
        # 构建完整的文件路径
        file_path = os.path.join(base_folder, file_path_simp)
        if not os.path.isfile(file_path):
            print(f"文件未找到: {file_path}")
            continue

        print(f"处理文件 (TXT): {file_path}")
        data = extract_eis_data_from_csv(file_path, remove_first_n_points)
        folder_name = Path(base_folder).name

        if data is None:
            print(f"跳过文件 {file_path} 因为数据提取失败。")
            continue

        df = pd.DataFrame(data)

        # 生成输出文件名
        base_name = os.path.basename(file_path)
        txt_name = folder_name + os.path.splitext(base_name)[0] + '.txt'
        txt_path = os.path.join(txt_output_folder, txt_name)

        try:
            # 将数据保存为TXT文件，列之间用空格分隔，且没有标题行
            df.to_csv(txt_path, sep=' ', index=False, header=False)
            print(f"已保存 TXT 文件: {txt_path}")
        except Exception as e:
            print(f"无法保存 TXT 文件 {txt_path}: {e}")

def convert_zhiyun_to_txt_greater_than_0(base_folder, zhiyun_files, output_folder, remove_first_n_points=0):
    """
    将指定的炙云CSV文件转换为TXT文件，并保存在output_folder的output_txt子文件夹中。

    参数：
    - base_folder: 包含炙云CSV文件的基准文件夹路径。
    - zhiyun_files: 包含炙云CSV文件相对路径的列表。
    - output_folder: TXT文件的输出文件夹路径。
    - remove_first_n_points: 要移除的前n个数据点。
    """
    # 创建output_txt子文件夹路径
    txt_output_folder = os.path.join(output_folder, "output_txt")
    os.makedirs(txt_output_folder, exist_ok=True)
    print(f"TXT文件将保存在: {txt_output_folder}")

    for file_path_simp in zhiyun_files:
        # 构建完整的文件路径
        file_path = os.path.join(base_folder, file_path_simp)
        folder_name = Path(base_folder).name
        if not os.path.isfile(file_path):
            print(f"文件未找到: {file_path}")
            continue

        print(f"处理文件 (TXT): {file_path}")
        data = extract_positive_imaginary_data_from_csv(file_path, remove_first_n_points)

        if data is None:
            print(f"跳过文件 {file_path} 因为数据提取失败。")
            continue

        df = pd.DataFrame(data)

        # 生成输出文件名
        base_name = os.path.basename(file_path)
        txt_name = folder_name + os.path.splitext(base_name)[0] + '_greater_than_0.txt'
        txt_path = os.path.join(txt_output_folder, txt_name)

        try:
            # 将数据保存为TXT文件，列之间用空格分隔，且没有标题行
            df.to_csv(txt_path, sep=' ', index=False, header=False)
            print(f"已保存 TXT 文件: {txt_path}")
        except Exception as e:
            print(f"无法保存 TXT 文件 {txt_path}: {e}")


# def main():
#     """
#     主函数，指定炙云文件并执行转换。
#     """
#     base_folder = r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240915_2ppm铜离子污染测试\旧版电解槽_firecloud\20240914_ion-20240914_旧版电解槽_炙云设备"

#     # 指定炙云CSV文件列表（相对于base_folder）
#     zhiyun_files = [
#         r"循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv",
#         r"循环1／1_工步组2(工步组)(1／24)_工步3(阻抗).csv",
#         r"循环1／1_工步组2(工步组)(2／24)_工步3(阻抗).csv",
#         r"循环1／1_工步组2(工步组)(3／24)_工步3(阻抗).csv",
#         r"循环1／1_工步组2(工步组)(4／24)_工步3(阻抗).csv",
#         r"循环1／1_工步组2(工步组)(5／24)_工步3(阻抗).csv",
#         r"循环1／1_工步组2(工步组)(6／24)_工步3(阻抗).csv",
#         r"循环1／1_工步组2(工步组)(7／24)_工步3(阻抗).csv",
#         r"循环1／1_工步组2(工步组)(8／24)_工步3(阻抗).csv",
#         r"循环1／1_工步组2(工步组)(9／24)_工步3(阻抗).csv",
#         r"循环1／1_工步组2(工步组)(10／24)_工步3(阻抗).csv",
#         r"循环1／1_工步组2(工步组)(11／24)_工步3(阻抗).csv",
#         r"循环1／1_工步组2(工步组)(12／24)_工步3(阻抗).csv",
#         # 在此添加更多炙云文件路径...
#     ]

#     # 指定输出文件夹
#     output_folder = base_folder

#     # 要移除的前n个数据点
#     remove_first_n_points = 0

#     # 执行XLSX转换
#     convert_zhiyun_to_xlsx(base_folder, zhiyun_files, output_folder, remove_first_n_points)
#     convert_zhiyun_to_txt(base_folder, zhiyun_files, output_folder, remove_first_n_points)
# ################分割线#######################
#     base_folder_2 = r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240915_2ppm铜离子污染测试\旧版电解槽_firecloud\20240914当天_ion_column_新版电解槽"

#     # 指定炙云CSV文件列表（相对于base_folder）
#     zhiyun_files_2 = [
#         r"循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv",
#         r"循环1／1_工步组1(工步组)(1／24)_工步3(阻抗).csv",
#         r"循环1／1_工步组1(工步组)(2／24)_工步3(阻抗).csv",
#         r"循环1／1_工步组1(工步组)(3／24)_工步3(阻抗).csv",
#         r"循环1／1_工步组1(工步组)(4／24)_工步3(阻抗).csv",
#         # 在此添加更多炙云文件路径...
#     ]

#     # 指定输出文件夹
#     output_folder_2 = base_folder_2

#     # 要移除的前n个数据点
#     remove_first_n_points_2 = 0

#     convert_zhiyun_to_xlsx(base_folder_2, zhiyun_files_2, output_folder_2, remove_first_n_points_2)
#     convert_zhiyun_to_txt(base_folder_2, zhiyun_files_2, output_folder_2, remove_first_n_points_2)

if __name__ == "__main__":
    main()
