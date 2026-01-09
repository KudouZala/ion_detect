
import os
import re

# 指定文件夹路径
folder_path = '/home/cagalii/Application/autoeis/AutoEIS/examples/校内测试/20241107_0.1ppm铬离子污染及恢复测试/旧版电解槽_firecloud/20241106_ion/output_txt'

file_startswith = 'cm2_20241107_ion_'
# file_startswith = 'cm2_20241107_ion_'
files = [f for f in os.listdir(folder_path) if f.startswith(file_startswith) and f.endswith('_大于0.txt')]


files_sorted = sorted(files, key=lambda x: int(re.search(f'{file_startswith}(\d+)_大于0.txt', x).group(1)))


# 遍历排序后的文件，进行重命名
for file in files_sorted:
    # 提取文件名中的数字部分（x）
    match = re.search(f'{file_startswith}(\d+)_大于0.txt', file)
    if match:
        x = match.group(1)  # 获取文件名中的数字部分

        # 构建新的文件名，替换/为_
        new_filename = f"{file_startswith}循环1／1_工步组1(工步组)({x}／80)_工步3(阻抗)_greater_than_0.txt"
        
        # 获取完整路径
        old_file_path = os.path.join(folder_path, file)
        new_file_path = os.path.join(folder_path, new_filename)

        # 打印调试信息
        # print(f"Attempting to rename: {old_file_path}")
        # print(f"New file path: {new_file_path}")

        # 确认文件是否存在
        if not os.path.exists(old_file_path):
            print(f"Error: {old_file_path} does not exist.")
            continue

        # 重命名文件
        try:
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {file} -> {new_filename}")
        except Exception as e:
            print(f"Error renaming file {file}: {e}")
