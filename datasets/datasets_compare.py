import os
import pandas as pd
import re

# ===== 配置 =====
folder1 = "/home/wangruyi3/Application/ion_detect/datasets/datasets_for_ion_train"
folder2 = "/home/wangruyi3/Downloads/datasets_for_ion_train"

# ===== 主程序 =====
def normalize_filename(name):
    """
    去掉文件名中的'_新版电解槽'或'_旧版电解槽'，方便匹配
    """
    # 正则替换
    return re.sub(r"_(新|旧)版电解槽", "", name)

def compare_excel(file1, file2):
    """比较两个 Excel 文件内容是否一致"""
    try:
        df1 = pd.read_excel(file1)
        df2 = pd.read_excel(file2)
    except Exception as e:
        return False, f"读取失败: {e}"

    if df1.equals(df2):
        return True, None
    else:
        return False, "内容不同"

if __name__ == "__main__":
    files1 = {normalize_filename(f): f for f in os.listdir(folder1) if f.endswith(".xlsx")}
    files2 = {normalize_filename(f): f for f in os.listdir(folder2) if f.endswith(".xlsx")}

    all_keys = set(files1.keys()) & set(files2.keys())
    missing_in_f2 = set(files1.keys()) - set(files2.keys())
    missing_in_f1 = set(files2.keys()) - set(files1.keys())

    if missing_in_f2:
        print("❌ 下列文件在 folder2 中缺失：")
        for k in missing_in_f2:
            print("   ", files1[k])
    if missing_in_f1:
        print("❌ 下列文件在 folder1 中缺失：")
        for k in missing_in_f1:
            print("   ", files2[k])

    print("\n=== 开始比较内容 ===")
    for key in sorted(all_keys):
        f1_path = os.path.join(folder1, files1[key])
        f2_path = os.path.join(folder2, files2[key])
        same, err = compare_excel(f1_path, f2_path)
        if same:
            print(f"✅ 一致: {files1[key]}")
        else:
            print(f"❌ 不一致: {files1[key]}  -> {err}")
