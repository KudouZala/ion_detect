#我要写一个python脚本，只保留[/home/cagalii/Application/train_machine_learning/数据整理_range_ion_02462468该绝对路径下所有以_[0, 2, 4, 6].xlsx、_[2, 4, 6， 8].xlsx、_[4, 6, 8, 10].xlsx,结尾的文件
import os

# 目标目录
folder = "/home/wangruyi3/Application/ion_detect/datasets/datasets_for_range_ion_0_10_test"

# 明确指定允许保留的结尾（包括方括号）
allowed_suffixes = [
    "_[0, 2, 4, 6].xlsx",
    "_[2, 4, 6, 8].xlsx",
    "_[4, 6, 8, 10].xlsx"
]

# 遍历文件夹中的所有文件
for filename in os.listdir(folder):
    if filename.endswith(".xlsx"):
        full_path = os.path.join(folder, filename)
        # 检查是否为允许的结尾
        if not any(filename.endswith(suffix) for suffix in allowed_suffixes):
            print(f"删除：{filename}")
            os.remove(full_path)
        else:
            print(f"保留：{filename}")
