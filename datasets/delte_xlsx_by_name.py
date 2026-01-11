import os

# 目标文件夹
target_dir = "/home/cagalii/Application/ion_detect/datasets/datasets_for_all_2ppm"

# # 检查路径是否存在
if not os.path.exists(target_dir):
    raise FileNotFoundError(f"路径不存在: {target_dir}")

# 遍历目标目录
files_to_delete = []
for fname in os.listdir(target_dir):
    fpath = os.path.join(target_dir, fname)
    if os.path.isfile(fpath):
        # 如果包含 则加入删除列表
        if "2ppm" not in fname :#在这里编辑
            files_to_delete.append(fpath)
        # elif "恢复" not in fname:
        #     files_to_delete.append(fpath)
        # elif "_ion_gamry_[0, 2, 4, 6]" not in fname and "_ion_firecloud_[0, 2, 4, 6]" not in fname \
        # and "_ion_gamry_[6, 8, 10, 12]" not in fname and "_ion_firecloud_[6, 8, 10, 12]"not in fname \
        # and "_ion_column_renew_H2SO4_gamry_[0, 2, 4, 6]" not in fname and "_ion_column_renew_H2SO4_firecloud_[0, 2, 4, 6]" not in fname: #在这里编辑
        #     files_to_delete.append(fpath)
        
        # elif ("[0, 2, 4, 6]" not  in fname) and ('[2, 4, 6, 8]' not  in fname) and ('[4, 6, 8, 10]' not  in fname) and ('[6, 8, 10, 12]' not  in fname):#在这里编辑
        #     files_to_delete.append(fpath)
        # if "2ppm" not in fname :#在这里编辑20240822_10ppm铜离子污染测试_新版电解槽_ion_column_gamry_[0, 2, 4, 6]
        #     files_to_delete.append(fpath)

# 打印要删除的文件
print("以下文件将被删除:")
for f in files_to_delete:
    print(f)

# 确认删除
confirm = input("是否确认删除这些文件？(y/n): ").strip().lower()
if confirm == 'y':
    for f in files_to_delete:
        os.remove(f)
    print(f"已删除 {len(files_to_delete)} 个文件。")
else:
    print("取消删除。")
