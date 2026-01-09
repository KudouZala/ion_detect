import os
import pandas as pd
import matplotlib.pyplot as plt
from excel_PnPw_sequence import process_multiple_files


import os
import re

def find_target_files_and_titles(root_dir):
    matched_files = []
    custom_titles = []

    # 中文离子名称到符号映射
    ion_map = {
        "钠离子": "Na+",
        "钙离子": "Ca2+",
        "铬离子": "Cr3+",
        "镍离子": "Ni2+",
        "铜离子": "Cu2+",
        "铁离子": "Fe3+",
        "铝离子": "Al3+"
    }

    # 文件名匹配规则
    target_pattern = re.compile(
        r"output_values_R1-\[P2,R3\]-\[P4,R5\](?:-\[P6,R7\])?\.xlsx"
    )

    for sub_dir_name in os.listdir(root_dir):
        sub_dir_path = os.path.join(root_dir, sub_dir_name)
        if os.path.isdir(sub_dir_path) and "铜离子" in sub_dir_name:
            # 提取信息：日期、ppm、离子种类
            match = re.search(r"(\d{8})_(.*?)(" + "|".join(ion_map.keys()) + ")", sub_dir_name)
            if match:
                date = match.group(1)
                ppm_raw = match.group(2).strip()
                ion_cn = match.group(3)
                ion_symbol = ion_map.get(ion_cn, ion_cn)
                custom_title = f"{ppm_raw}_{ion_symbol}_{date}"
            else:
                custom_title = sub_dir_name  # fallback

            # 查找目标文件
            for root, dirs, files in os.walk(sub_dir_path):
                for file in files:
                    if target_pattern.fullmatch(file):
                        full_path = os.path.join(root, file)
                        matched_files.append(full_path)
                        custom_titles.append(custom_title)

    return matched_files, custom_titles


# ✅ 自动获取脚本路径的上三级目录
current_file = os.path.abspath(__file__)
parent_3 = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

# ✅ 拼接目标根目录
root_folder = os.path.join(parent_3, "eis_fit_results", "20250724")

# ✅ 调用主函数
file_paths, custom_title = find_target_files_and_titles(root_folder)

# ✅ 示例 legend_dict
legend_dict = {
    'ion_column': {'color': 'green', 'marker': 'o', 'label': 'ion_column'},
    'ion': {'color': 'red', 'marker': 'x', 'label': 'ion'},
    'ion_column_renew': {'color': 'gray', 'marker': 's', 'label': 'ion_column_renew_H2SO4'}
}
print("file_paths:",file_paths)
# ✅ 最终调用
process_multiple_files(file_paths, legend_dict=legend_dict, custom_title=custom_title)
