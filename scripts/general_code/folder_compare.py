import os

# 文件夹路径
folder1 = "/home/cagalii/Application/train_machine_learning/数据整理_range_ion"
folder2 = "/home/cagalii/Application/train_machine_learning/数据整理_range_ion（复件）"

# 获取文件名集合
files1 = set(os.listdir(folder1))
files2 = set(os.listdir(folder2))

# 计算重复文件与独有文件
common_files = files1 & files2  # 两个文件夹中都有的文件
only_in_folder1 = files1 - files2  # 只在 folder1 中的文件
only_in_folder2 = files2 - files1  # 只在 folder2 中的文件

# 打印结果
print("✅ 重复的文件（两个文件夹都有）:")
for f in sorted(common_files):
    print("  ", f)

print("\n📁 仅在原始文件夹中的文件:")
for f in sorted(only_in_folder1):
    print("  ", f)

print("\n📁 仅在复件文件夹中的文件:")
for f in sorted(only_in_folder2):
    print("  ", f)
