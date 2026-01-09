import os

def rename_files_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.startswith("cm2_20241115_ion_column_") and filename.endswith(".DTA"):
            new_filename = filename.replace("cm2_20241115_ion_column_", "cm2_20241115_ion_column_renew_H2SO4_", 1)
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")

# 使用示例
directory = r"/home/cagalii/Application/autoeis/AutoEIS/examples/校内测试/20241112_2ppm镍离子污染及恢复测试/旧版电解槽_gamry/EISGALV_60℃_150ml_1A"  # 替换为你文件夹的路径
rename_files_in_directory(directory)
