import os
import pandas as pd
import matplotlib.pyplot as plt
from excel_PnPw_sequence import process_multiple_files

if __name__ == '__main__':
    # 输入多个文件路径
    file_paths = [
        # 'eis_results_20241107/20240910_10ppm钙离子污染测试_R1-[P2,R3]-[P4,R5]/output_ecm_gamry/output_values_R1-[P2,R3]-[P4,R5].xlsx',
        'eis_results_20241113/20240918_2ppm钙离子污染测试_R1-[P2,R3]-[P4,R5]/output_ecm_gamry/output_values_R1-[P2,R3]-[P4,R5].xlsx',
        # 'eis_results_20241107/20240918_2ppm钙离子污染测试_R1-[P2,R3]-[P4,R5]-[P6,R7]/output_ecm_gamry/output_values_R1-[P2,R3]-[P4,R5]-[P6,R7].xlsx',
        'eis_results_20241113/20241101_2ppm钙离子污染和恢复测试_R1-[P2,R3]-[P4,R5]/output_ecm_firecloud/output_values_R1-[P2,R3]-[P4,R5].xlsx',
        # r"eis_results_20241113\20241107_0.1ppm钙离子污染及恢复测试_R1-[P2,R3]-[P4,R5]\output_ecm_firecloud\output_values_R1-[P2,R3]-[P4,R5].xlsx",
    ]
    # 示例：自定义图例字典
    legend_dict = {
        'ion_column': {'color': 'green', 'marker': 'o', 'label': 'ion_column'},
        'ion': {'color': 'red', 'marker': 'x', 'label': 'ion'},
        'ion_column_renew': {'color': 'gray', 'marker': 's', 'label': 'ion_column_renew_H2SO4'}
    }
    custom_title=["2ppm_Ca2+_20240918",
                    # "2ppm_Ca2+_20240918",
                    "2ppm_Ca2+_20241101",
    ]
    # 调用处理多个文件的函数
    process_multiple_files(file_paths,legend_dict=legend_dict,custom_title = custom_title)
