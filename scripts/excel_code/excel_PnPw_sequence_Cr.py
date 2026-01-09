import os
import pandas as pd
import matplotlib.pyplot as plt
from excel_PnPw_sequence import process_multiple_files

if __name__ == '__main__':
    # 输入多个文件路径
    file_paths = [
        # 'eis_results_20250110_nopreprocess\20250101_10ppm铬离子污染及恢复测试_R1-[P2,R3]-[P4,R5]\output_ecm_gamry\output_values_R1-[P2,R3]-[P4,R5].xlsx',
        # r"eis_results_20241107\20240827_10ppm铬离子污染和恢复测试_R1-[P2,R3]-[P4,R5]\output_ecm_gamry\output_values_R1-[P2,R3]-[P4,R5].xlsx"
        r"eis_results_20241113\20241006_2ppm铬离子污染测试_R1-[P2,R3]-[P4,R5]\output_ecm_firecloud\output_values_R1-[P2,R3]-[P4,R5].xlsx",
        r"eis_results_20241113\20241006_2ppm铬离子污染测试_R1-[P2,R3]-[P4,R5]\output_ecm_gamry\output_values_R1-[P2,R3]-[P4,R5].xlsx",
        r"eis_results_20241113\20241017_2ppm铬离子污染和恢复测试_R1-[P2,R3]-[P4,R5]\output_ecm_gamry\output_values_R1-[P2,R3]-[P4,R5].xlsx",
        # r"eis_results_20241113/20241107_0.1ppm铬离子污染及恢复测试_R1-[P2,R3]-[P4,R5]/output_ecm_firecloud/output_values_R1-[P2,R3]-[P4,R5].xlsx",
        # r"eis_results_20250110_nopreprocess\20250101_10ppm铬离子污染及恢复测试_R1-[P2,R3]-[P4,R5]\output_ecm_gamry\output_values_R1-[P2,R3]-[P4,R5].xlsx",
    ]
    # 示例：自定义图例字典
    legend_dict = {
        'ion_column': {'color': 'green', 'marker': 'o', 'label': 'ion_column'},
        'ion': {'color': 'red', 'marker': 'x', 'label': 'ion'},
        'ion_column_renew': {'color': 'gray', 'marker': 's', 'label': 'ion_column_renew_H2SO4'}
    }
    custom_title = ["2ppm_Cr3+_20241006",
                    "2ppm_Cr3+_20241006",
                    "2ppm_Cr3+_20241017",
    ]
    # 调用处理多个文件的函数
    process_multiple_files(file_paths,legend_dict=legend_dict,custom_title = custom_title)
