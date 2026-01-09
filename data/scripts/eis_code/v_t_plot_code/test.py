import os
gamry_data_list = [
     {
            'folder': [
                r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241010_2ppm钠离子污染测试\新版电解槽-炙云设备\20241010_ion_column-20241010-150335-默认1728834036317",
                r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241010_2ppm钠离子污染测试\新版电解槽-炙云设备\20241011_ion-20241011-131456-默认1728834048305",
            ],
            'files': [
                [
                    "循环1／1_工步组1(工步组)(1／1)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(1／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(2／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(3／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(4／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(5／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(6／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(7／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(8／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(9／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(10／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(11／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(12／80)_工步1(CC).csv",
                    # 你可以继续添加其他文件名
                ],
                [
                    "循环1／1_工步组1(工步组)(1／1)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(1／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(2／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(3／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(4／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(5／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(6／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(7／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(8／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(9／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(10／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(11／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(12／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(13／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(14／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(15／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(16／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(17／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(18／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(19／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(20／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(21／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(22／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(23／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(24／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(25／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(26／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(27／80)_工步1(CC).csv",
                    "循环1／1_工步组2(工步组)(28／80)_工步1(CC).csv",
                    # 你可以继续添加其他文件名
                ],
            ],
            'color': [
                (255, 255, 200),  # 2ppm Na+_1011_firecloud
                (255, 255, 200),
            ],
            'legend': [
                '2ppm Na+_ion_column_1011_firecloud',
                '2ppm Na+_1011_firecloud',
            ],
            'marker': [
                'o',
                'x',  # 对应 Na+
            ]
        },
        {
            'folder': [
                r"20240813-0814_10ppm铜离子污染和恢复测试（拔去离子柱跑误差较大）\新版电解槽_gamry\PWRGALVANOSTATIC_60℃_150ml_1A",
                r"20240813-0814_10ppm铜离子污染和恢复测试（拔去离子柱跑误差较大）\新版电解槽_gamry\PWRGALVANOSTATIC_60℃_150ml_1A",
                r"20240813-0814_10ppm铜离子污染和恢复测试（拔去离子柱跑误差较大）\新版电解槽_gamry\PWRGALVANOSTATIC_60℃_150ml_1A",
                r"20240813-0814_10ppm铜离子污染和恢复测试（拔去离子柱跑误差较大）\新版电解槽_gamry\PWRGALVANOSTATIC_60℃_150ml_1A",
            ],
            'patterns': [
                {"pattern": "cm2_20240812_ion_column_", "range": list(range(0, 50))},
                {"pattern": "cm2_20240813_no_ion_column_", "range": list(range(0, 50))},
                {"pattern": "cm2_20240814_ion_", "range": list(range(0, 50))},
                {"pattern": "cm2_20240814_ion_renew_", "range": list(range(0, 50))},
            ],
            'color': [],
            'legend': [
                '10ppm Cu2+_0813_ion_column_gamry',
                '10ppm Cu2+_0813_no_ion_column_gamry',
                '10ppm Cu2+_0813_ion_gamry',
                '10ppm Cu2+_0813_ion_H2SO4_renew_gamry',
            ],
            'marker': [
                'o',  # 对应 Ni2+
                'o',  # 对应 Cr3+
                'x',  # 对应 Fe3+
                'd',  # 对应 Ca2+
            ]
        },
        {
            'folder': [
                r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240815-0817_10ppm铁离子污染和恢复测试（拔去离子柱跑误差较大）\新版电解槽_gamry\PWRGALVANOSTATIC_60℃_150ml_1A",
                r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240815-0817_10ppm铁离子污染和恢复测试（拔去离子柱跑误差较大）\新版电解槽_gamry\PWRGALVANOSTATIC_60℃_150ml_1A",
                r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240815-0817_10ppm铁离子污染和恢复测试（拔去离子柱跑误差较大）\新版电解槽_gamry\PWRGALVANOSTATIC_60℃_150ml_1A",
                r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240815-0817_10ppm铁离子污染和恢复测试（拔去离子柱跑误差较大）\新版电解槽_gamry\PWRGALVANOSTATIC_60℃_150ml_1A",
            ],
            'patterns': [
                {"pattern": "cm2_20240815_ion_column_", "range": list(range(0, 50))},
                {"pattern": "cm2_20240815_no_ion_column_", "range": list(range(0, 50))},
                {"pattern": "cm2_20240815_ion_", "range": list(range(0, 50))},
                {"pattern": "cm2_20240816_ion_renew_", "range": list(range(0, 50))},
            ],
            'color': [
                (204, 153, 102),  # 2ppm Ni2+_1003_gamry
                (204, 153, 102),  # 2ppm Cu2+_0915_gamry
                (204, 153, 102),
                (204, 153, 102),
            ],
            'legend': [
                '10ppm Fe3+_0815_ion_column_gamry',
                '10ppm Fe3+_0815_no_ion_column_gamry',
                '10ppm Fe3+_0815_ion_gamry',
                '10ppm Fe3+_0815_ion_H2SO4_renew_gamry',
            ],
            'marker': [
                'o',  # 对应 Ni2+
                'o',  # 对应 Cr3+
                'x',  # 对应 Fe3+
                'd',  # 对应 Ca2+
            ]
        },


]
import os
# 获取当前文件的路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上三级目录并将其中的 '/' 替换为 '\\'
parent_dir_filename = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir, os.pardir)).replace('/', '\\')

x=1
# 遍历 gamry_data_list，修改 'folder' 中的路径
for filename_data in gamry_data_list:
    if x ==1:
        filename_data['folder'] = [
            os.path.join(parent_dir_filename, folder_path.lstrip(os.sep)).replace('/', '\\')  # 在这里拼接路径时，不修改 folder_path
            for folder_path in filename_data['folder']
        ]
        x=2

for i in range (0,2):
    print("gamry_data_list:",gamry_data_list[i],"\n")

