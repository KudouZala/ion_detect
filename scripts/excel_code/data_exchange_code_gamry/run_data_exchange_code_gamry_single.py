import sys
import os
import glob

# 将库的绝对路径添加到系统路径中
import glob
# 将库的绝对路径添加到系统路径中
lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '最新_绘图和输出文件代码_更新位置', 'excel_code', 'data_exchange_code_gamry')

sys.path.append(lib_path)


from data_to_excel import convert_to_csv_and_xlsx
from data_to_excel_greater_than_0 import convert_to_csv_and_xlsx_greater_than_0
from data_to_txt import convert_to_txt
from data_to_txt_greater_than_0 import convert_to_txt_greater_than_0
from data_to_excel_notitle import convert_to_csv_and_xlsx_no_title
from data_to_excel_notitle_greater_than_0 import convert_to_csv_and_xlsx_no_title_greater_than_0

def main():

    # 获取当前文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取上一级目录
    parent_dir = os.path.dirname(current_dir)
    # 查找包含'_gamry'的文件夹路径
    gamry_folder = next((d for d in os.listdir(parent_dir) if '电解槽_gamry' in d), None)

    if gamry_folder:
        # 获取所有与 "EISGALV_*" 匹配的文件夹路径
        base_folder_pattern = os.path.join(parent_dir, gamry_folder, "EISGALV_*")
        matching_folders = glob.glob(base_folder_pattern)

        # 如果找到匹配的文件夹，选择第一个匹配的文件夹
        if matching_folders:
            base_folder = matching_folders[0]  # 选择第一个匹配的文件夹
            print(f"使用的 base_folder: {base_folder}")

            # 自动搜索该 base_folder 中的 .DTA 文件
            file_specifications = glob.glob(os.path.join(base_folder, "*.DTA"))
            print(f"找到的 .DTA 文件: {file_specifications}")
        else:
            print("没有找到匹配的 EISGALV_* 文件夹。")


    else:
        print("未找到包含 '_gamry' 的文件夹。")
        return
    convert_to_csv_and_xlsx(base_folder, file_specifications)
    convert_to_txt(base_folder, file_specifications)
    convert_to_csv_and_xlsx_greater_than_0(base_folder, file_specifications)
    convert_to_txt_greater_than_0(base_folder, file_specifications)
    convert_to_csv_and_xlsx_greater_than_0(base_folder, file_specifications)
    convert_to_csv_and_xlsx_no_title_greater_than_0(base_folder, file_specifications)


if __name__ == "__main__":
    main()