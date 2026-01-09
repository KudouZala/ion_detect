import os

def rename_files_in_directory(directory,replace_name,new_name):
    # 检查指定目录是否存在
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return
    
    print(f"Scanning directory: {directory}")
    
    # 遍历指定文件夹中的所有文件
    for filename in os.listdir(directory):
        print(f"Found file: {filename}")  # 输出找到的文件名
        
        # 检查文件名是否包含 'ion_column_renew'
        if replace_name in filename:
            print(f"Renaming file: {filename}")  # 输出正在重命名的文件名
            
            # 创建新的文件名
            new_filename = filename.replace(replace_name, new_name)
            # 获取完整的文件路径
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)
            
            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {old_file_path} -> {new_file_path}')
        else:
            print(f"No match for: {filename}")  # 输出没有匹配的文件名

# gamry常用改名模板
# directory_path = r"/home/cagalii/Application/autoeis/AutoEIS/examples/校内测试/20241122_2ppm铜离子污染及恢复测试40摄氏度/新版电解槽_gamry/EISGALV_40℃_150ml_1A"
directory_path = r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\autoeis\autoeis\AutoEIS\examples\校内测试\20241107_0.1ppm铬离子污染及恢复测试\旧版电解槽_gamry\EISGALV_60℃_150ml_1A"
replace_name = "cm2_20241109_ion_column_"#要替换的名字
new_name = 'cm2_20241109_ion_column_renew_H2SO4_'#替换后的名字
rename_files_in_directory(directory_path,replace_name,new_name)
