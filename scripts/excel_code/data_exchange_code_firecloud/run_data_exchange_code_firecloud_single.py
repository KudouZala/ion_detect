import sys
import os
import glob
# 将库的绝对路径添加到系统路径中
lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '最新_绘图和输出文件代码_更新位置', 'excel_code', 'data_exchange_code_firecloud')

sys.path.append(lib_path)

from data_to_excel_firecloud import convert_zhiyun_to_xlsx
from data_to_txt_firecloud import convert_zhiyun_to_txt
from data_to_csv_firecloud import convert_zhiyun_to_csv
from data_to_csv_firecloud import convert_zhiyun_to_csv_notitle
from data_to_excel_firecloud import convert_zhiyun_to_xlsx_notitle
from data_to_txt_firecloud import convert_zhiyun_to_txt_greater_than_0

def main():
    # 获取当前文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取上一级目录
    parent_dir = os.path.dirname(current_dir)

    # 查找以'_firecloud'结尾的文件夹
    firecloud_folder = next((d for d in os.listdir(parent_dir) if d.endswith('电解槽_firecloud')), None)

    if firecloud_folder:
        firecloud_path = os.path.join(parent_dir, firecloud_folder)
        # 查找以'_ion_column'结尾的文件夹
        base_folder = next((d for d in os.listdir(firecloud_path) if d.endswith('_ion_column')), None)

        if base_folder:
            base_folder = os.path.join(firecloud_path, base_folder)
            print(f"使用的 base_folder: {base_folder}")

            # 自动获取该文件夹中的所有 .csv 文件
            zhiyun_files = glob.glob(os.path.join(base_folder, "*.csv"))
            print(f"找到的 .csv 文件: {zhiyun_files}")

            # 指定输出文件夹
            output_folder = base_folder

            # 要移除的前n个数据点
            remove_first_n_points = 1

            # 执行XLSX转换
            convert_zhiyun_to_xlsx(base_folder, zhiyun_files, output_folder, remove_first_n_points)
            convert_zhiyun_to_txt(base_folder, zhiyun_files, output_folder, remove_first_n_points)
            convert_zhiyun_to_txt_greater_than_0(base_folder, zhiyun_files, output_folder, remove_first_n_points)
            convert_zhiyun_to_csv(base_folder, zhiyun_files, output_folder, remove_first_n_points)
            convert_zhiyun_to_csv_notitle(base_folder, zhiyun_files, output_folder, remove_first_n_points)
            convert_zhiyun_to_xlsx_notitle(base_folder, zhiyun_files, output_folder, remove_first_n_points)

        else:
            print("未找到以 '_ion_column' 结尾的文件夹。")
            
    else:
        print("未找到以 '_firecloud' 结尾的文件夹。")
        


    
    """
        主函数，指定炙云文件并执行转换。
        """


    if firecloud_folder:
        firecloud_path = os.path.join(parent_dir, firecloud_folder)
        # 查找以'_ion_column'结尾的文件夹
        base_folder_ion = next((d for d in os.listdir(firecloud_path) if d.endswith('_ion')), None)

        if  base_folder_ion:
            base_folder_ion = os.path.join(firecloud_path,  base_folder_ion)
            print(f"使用的 base_folder_ion: { base_folder_ion}")

            # 自动获取该文件夹中的所有 .csv 文件
            zhiyun_files_ion = glob.glob(os.path.join( base_folder_ion, "*.csv"))
            print(f"找到的 .csv 文件: {zhiyun_files_ion}")

             # 指定输出文件夹
            output_folder_ion = base_folder_ion

            # 要移除的前n个数据点
            remove_first_n_points_ion = 1

            # 执行XLSX转换
            convert_zhiyun_to_xlsx(base_folder_ion, zhiyun_files_ion, output_folder_ion, remove_first_n_points_ion)
            convert_zhiyun_to_txt(base_folder_ion, zhiyun_files_ion, output_folder_ion, remove_first_n_points_ion)
            convert_zhiyun_to_txt_greater_than_0(base_folder_ion, zhiyun_files_ion, output_folder_ion, remove_first_n_points_ion)
            convert_zhiyun_to_csv(base_folder_ion, zhiyun_files_ion, output_folder_ion, remove_first_n_points_ion)
            convert_zhiyun_to_csv_notitle(base_folder_ion, zhiyun_files_ion, output_folder_ion, remove_first_n_points_ion)
            convert_zhiyun_to_xlsx_notitle(base_folder_ion, zhiyun_files_ion, output_folder_ion, remove_first_n_points_ion)
        else:
            print("未找到以 '_ion' 结尾的文件夹。")

    else:
        print("未找到以 '_firecloud' 结尾的文件夹。")


   



    if firecloud_folder:
        firecloud_path = os.path.join(parent_dir, firecloud_folder)
        # 查找以'_ion_column'结尾的文件夹
        base_folder_ion_renew = next((d for d in os.listdir(firecloud_path) if d.endswith('_ion_column_renew')), None)

        if base_folder_ion_renew:
            base_folder_ion_renew = os.path.join(firecloud_path, base_folder_ion_renew)
            print(f"使用的 base_folder_ion_renew: {base_folder_ion_renew}")

            # 自动获取该文件夹中的所有 .csv 文件
            zhiyun_files_ion_renew = glob.glob(os.path.join(base_folder_ion_renew, "*.csv"))
            print(f"找到的 .csv 文件: {zhiyun_files_ion_renew}")

            # 指定输出文件夹
            output_folder_ion_renew = base_folder_ion_renew

            # 要移除的前n个数据点
            remove_first_n_points_ion_renew = 1

            # 执行XLSX转换
            convert_zhiyun_to_xlsx(base_folder_ion_renew, zhiyun_files_ion_renew, output_folder_ion_renew, remove_first_n_points_ion_renew)
            convert_zhiyun_to_txt(base_folder_ion_renew, zhiyun_files_ion_renew, output_folder_ion_renew, remove_first_n_points_ion_renew)
            convert_zhiyun_to_txt_greater_than_0(base_folder_ion_renew, zhiyun_files_ion_renew, output_folder_ion_renew, remove_first_n_points_ion_renew)
            convert_zhiyun_to_csv(base_folder_ion_renew, zhiyun_files_ion_renew, output_folder_ion_renew, remove_first_n_points_ion_renew)
            convert_zhiyun_to_csv_notitle(base_folder_ion_renew, zhiyun_files_ion_renew, output_folder_ion_renew, remove_first_n_points_ion_renew)
            convert_zhiyun_to_xlsx_notitle(base_folder_ion_renew, zhiyun_files_ion_renew, output_folder_ion_renew, remove_first_n_points_ion_renew)
        else:
            print("未找到以 '_ion_column_renew' 结尾的文件夹。")

    else:
        print("未找到以 '_firecloud' 结尾的文件夹。")


    


if __name__ == "__main__":
    main()