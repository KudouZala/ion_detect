"""AutoEIS/
待改进：我希望修改下面的代码：import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import seaborn as sns
from IPython.display import display

import autoeis as ae

ae.visualization.set_plot_style()

# Set this to True if you're running the notebook locally
interactive = False

fpath = r"/home/cagalii/Application/autoeis/AutoEIS/examples/校内测试/20240915_2ppm铜离子污染测试/新版电解槽_gamry/EISGALV_60℃_150ml_1A/output_txt/cm2_20240914_ion_10_大于0.txt"
freq, Zreal, Zimag = np.loadtxt(fpath, skiprows=0, unpack=True, usecols=(0, 1, 2))
# Convert to complex impedance (the file contains -Im(Z) hence the minus sign)
Z = Zreal + 1j * Zimag
preprocess=True


if preprocess:
    freq, Z = ae.utils.preprocess_impedance_data(freq, Z)
ax = ae.visualization.plot_impedance_combo(freq, Z)

freq, Z, aux = ae.utils.preprocess_impedance_data(freq, Z, tol_linKK=5e-2, return_aux=True)



use_custom_circuit = True

if not use_custom_circuit:
    circuit = circuits.iloc[0]["circuitstring"]
    p = circuits.iloc[0]["Parameters"]
    # Refine the circuit parameters
    p = ae.utils.fit_circuit_parameters(circuit, freq, Z, p0=p)
else:
    circuit = "R1-[P2,R3]-[P4,R5]-[P6,R7]"
    p = ae.utils.fit_circuit_parameters(circuit, freq, Z)

# Simulate Z using the circuit and the fitted parameters
circuit_fn = ae.utils.generate_circuit_fn(circuit)
Z_sim = circuit_fn(freq, list(p.values()))


# Plot against ground truth
fig, ax = plt.subplots(figsize=(5.5, 4))
ae.visualization.plot_nyquist(Z_sim, fmt="-", ax=ax, label="simulated")
ae.visualization.plot_nyquist(Z, fmt=".", ax=ax, label="data");
ax.set_title(circuit)


print(ae.parser.get_component_labels(circuit))
print(ae.parser.get_parameter_labels(circuit))
print(p.values())实现功能：我希望能够自己指定多个txt文件，从而得到所有的输出（因此你要用函数的方式进行调用），输出分别是：1ax = ae.visualization.plot_impedance_combo(freq, Z)这个将生成两个图片，分别命名为txt文件名_nyquist和_bode；2是ae.visualization.plot_linKK_residuals(aux.freq, aux.res.real, aux.res.imag)将生成一个图片命名为txt文件名_linKK_residuals；3是ae.visualization.plot_nyquist(Z_sim, fmt="-", ax=ax, label="simulated")和
ae.visualization.plot_nyquist(Z, fmt=".", ax=ax, label="data")将生成1张图片,命名为_nyquist_simu；4是print(ae.parser.get_component_labels(circuit))
print(ae.parser.get_parameter_labels(circuit))
print(p.values())将生成三组数值，前两组数值放到excel文件的前两行，第三个value的放到后面，value的第一列是txt的文件名，我指定的txt所生成的value都放到这一个excel中第一行和第二行就是print(ae.parser.get_component_labels(circuit))
print(ae.parser.get_parameter_labels(circuit))，后面依次是各个txt的结果value，前两行的第一列空出来

"""
import traceback
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import autoeis as ae
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
from pathlib import Path
import sys
import os
import glob
import os
from datetime import datetime
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import logging
from multiprocessing import Pool, cpu_count
# 设置字体路径
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'  # 替换为你需要的字体路径
font_prop = font_manager.FontProperties(fname=font_path)

# 设置全局字体
rcParams['font.family'] = font_prop.get_name()
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 将库的相对路径添加到系统路径中
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print(current_path)
from data_analysis import process_xlsx_data_analysis

# 设置绘图样式
ae.visualization.set_plot_style()

import os
import sys

# 获取当前文件所在文件夹的路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 拼接出目标文件 'ion_color.py' 的路径
ion_color_path = os.path.join(current_dir, 'ion_color.py')

# 将该目录添加到 sys.path 中，使得可以导入 ion_color.py
if os.path.exists(ion_color_path):
    sys.path.append(os.path.dirname(ion_color_path))

# 导入 get_ion_color 函数
from ion_color import get_ion_color


def process_files(file_paths, output_folder, custom_circuit = None, output_excel="output_values.xlsx",KK_pre=True):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    data = {"File Name": ["", ""]}
    component_labels_row = [""]  
    parameter_labels_row = [""]  
    values_rows = []  
    image_paths = []  

    for fpath in file_paths:
        freq, Zreal, Zimag = np.loadtxt(fpath, skiprows=0, unpack=True, usecols=(0, 1, 2))
        Z = Zreal + 1j * Zimag  
        
        preprocess = True
        if preprocess:#是否进行预处理
            if KK_pre:
                freq, Z = ae.utils.preprocess_impedance_data(freq, Z,KK_pre=True)
            else:
                freq, Z = ae.utils.preprocess_impedance_data(freq, Z,KK_pre=False)
        
        base_filename = os.path.splitext(os.path.basename(fpath))[0]

        # Nyquist and Bode plots
        nyquist_path = os.path.join(output_folder, f"{base_filename}_nyquist.png")
        bode_path = os.path.join(output_folder, f"{base_filename}_bode.png")
        fig, ax = plt.subplots()
        ae.visualization.plot_impedance_combo(freq, Z)
        # ax.set_title(f"{base_filename} Nyquist and Bode")
        plt.savefig(nyquist_path)
        plt.savefig(bode_path)
        plt.close(fig)
        plt.close('all')  # 关闭所有打开的图形
        
        if KK_pre:
            # lin-KK residuals
            freq, Z, aux = ae.utils.preprocess_impedance_data(freq, Z, tol_linKK=5e-2, return_aux=True)
            linKK_residuals_path = os.path.join(output_folder, f"{base_filename}_linKK_residuals.png")
            fig, ax = plt.subplots()
            ae.visualization.plot_linKK_residuals(aux.freq, aux.res.real, aux.res.imag)
            # ax.set_title(f"{base_filename} lin-KK Residuals")
            plt.savefig(linKK_residuals_path)
            plt.close(fig)
            plt.close('all')  # 关闭所有打开的图形
        
        
        if not custom_circuit:#由于我们是使用自定义等效电路，因此这里为False
            circuit = circuits.iloc[0]["circuitstring"]
            p = circuits.iloc[0]["Parameters"]
            p = ae.utils.fit_circuit_parameters(circuit, freq, Z, p0=p)
        else:
            circuit = custom_circuit
            # 1. 设定合理的初值（p0）和边界
            param_names =  ae.parser.get_parameter_labels(circuit)
            default_p0 = [1.0 for _ in param_names]  # 粗略设为1.0，可自定义每类元件更合理值
            lower_bounds = [1e-8 for _ in param_names]
            upper_bounds = [1e6 for _ in param_names]

            # 2. 调用拟合函数（高精度配置）
            p = ae.utils.fit_circuit_parameters(
                circuit=circuit,
                freq=freq,
                Z=Z,
                p0=default_p0,
                bounds=(lower_bounds, upper_bounds),
                max_iters=100,
                min_iters=50,
                tol_chi_squared=1e-8,
                max_nfev=10000,  # 允许更多计算
                ftol=1e-20,     # 稍微放松但仍高精度
                xtol=1e-20,
                method='chi-squared',  # 更平衡稳定的目标函数
                verbose=True
            )#核心拟合函数



        
        nyquist_simu_path = os.path.join(output_folder, f"{base_filename}_nyquist_simu.png")
        circuit_fn = ae.utils.generate_circuit_fn(circuit)
        Z_sim = circuit_fn(freq, list(p.values()))
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ae.visualization.plot_nyquist(Z_sim, fmt="-", ax=ax, label="simulated")
        ae.visualization.plot_nyquist(Z, fmt=".", ax=ax, label="data")
        ax.set_title(f"{base_filename} Nyquist Simulation")
        plt.savefig(nyquist_simu_path)
        plt.close(fig)
        plt.close('all')  # 关闭所有打开的图形
        
        component_labels = ae.parser.get_component_labels(circuit)
        parameter_labels = ae.parser.get_parameter_labels(circuit)
        parameter_values = list(p.values())
        
        if len(component_labels_row) == 1:
            component_labels_row.extend(component_labels)
            parameter_labels_row.extend(parameter_labels)
        
        values_row = [base_filename] + parameter_values
        values_rows.append(values_row)
        if KK_pre:
            image_paths.append([nyquist_path, bode_path, linKK_residuals_path, nyquist_simu_path])
        else:
            image_paths.append([nyquist_path, bode_path, nyquist_simu_path])

    data["File Name"] = component_labels_row
    data["Parameter Labels"] = parameter_labels_row
    df = pd.DataFrame([data["File Name"], data["Parameter Labels"]] + values_rows)

    excel_path = os.path.join(output_folder, output_excel)
    df.to_excel(excel_path, header=False, index=False)
    print(f"数据保存到 {excel_path}")

    wb = load_workbook(excel_path)
    ws = wb.active

    start_row = 3  
    for i, image_set in enumerate(image_paths):
        for j, image_path in enumerate(image_set):
            img = Image(image_path)
            col_letter = chr(79 + j)  
            cell = f"{col_letter}{start_row + i}"
            ws.add_image(img, cell)
    
    wb.save(excel_path)
    print(f"图片插入完成，保存到 {excel_path}")





def custom_sort_key_firecloud(filename):
    # 提取文件名中的工步组信息
    if '工步组1' in filename:
        group = 1
    elif '工步组2' in filename:
        group = 2
    else:
        group = 3  # 用于其他组或未找到的情况

    # 提取文件名中的第一层（文件夹类型）
    if 'ion_column' in filename and 'ion_column_renew' not in filename:
        folder_type = 1
    elif 'ion' in filename and 'ion_column_renew' not in filename and 'ion_column' not in filename:
        folder_type = 2
    elif 'ion_column_renew' in filename:
        folder_type = 3
    else:
        folder_type = float('inf')  # 如果无法识别，设置为无穷大

    # 提取"(工步组)("后面的数字
    try:
        step_number = int(filename.split('(工步组)(')[-1].split('／')[0].strip())
    except (IndexError, ValueError):
        step_number = float('inf')  # 如果无法提取数字，设置为无穷大

    return (folder_type, group, step_number)


def custom_sort_key_gamry(filename):
    # 确定文件的组别
    if "ion_column" in filename and "ion_column_renew_H2SO4" not in filename and "ion_column_renew" not in filename and "ion_renew" not in filename:
        group = 1
        # 提取“ion_column_”后面的数字
        ion_number = int(filename.split('ion_column_')[-1].split('_')[0])  
    elif "ion" in filename and "ion_column" not in filename and "ion_column_renew_H2SO4" not in filename and "ion_renew" not in filename and "ion_column_renew" not in filename:
        group = 2
        # 提取“ion_”后面的数字
        ion_number = int(filename.split('ion_')[-1].split('_')[0])  
    elif "ion_column_renew_H2SO4" in filename:
        group = 3
        # 提取“ion_column_renew_H2SO4_”后面的数字
        ion_number = int(filename.split('ion_column_renew_H2SO4_')[-1].split('_')[0])  
    else:
        group = 4  # 用于不符合任何组的情况
        ion_number = float('inf')  # 设置为无穷大，以便排到最后

    return (group, ion_number)




def ecm_plot_all(folder_paths,custom_circuit,KK_pre_whether=True):
    
    custom_circuit = custom_circuit
    for folder_path in folder_paths:
        #获取指定文件夹下所有符合条件的文件路径
        print("现在正在处理folder_path和custom_circuit:",folder_path,custom_circuit)
        try:
            if "firecloud" in folder_path:
                    # 提取文件路径
                file_paths = []

                # 1. 处理以 "ion_column" 结尾的文件夹
                print("处理以 ion_column结尾的文件夹")
                ion_column_paths = list(Path(folder_path).glob('**/*ion_column/output_txt/*_工步3(阻抗)_greater_than_0.txt'))
                print(f"找到 {len(ion_column_paths)} 个 ion_column 文件")
                for path in Path(folder_path).glob('**/*ion_column/output_txt/*_工步3(阻抗)_greater_than_0.txt'):
                    print(f"找到文件: {path}")
                    file_paths.append(str(path))

                # 2. 处理以 "ion" 结尾的文件夹
                print("处理以 ion结尾的文件夹")
                for path in Path(folder_path).glob('**/*ion/output_txt/*_工步3(阻抗)_greater_than_0.txt'):
                    file_paths.append(str(path))

                # 3. 处理以 "ion_column_renew" 结尾的文件夹
                print("处理以 ion_column_renew结尾的文件夹")
                for path in Path(folder_path).glob('**/*ion_column_renew/output_txt/*_工步3(阻抗)_greater_than_0.txt'):
                    file_paths.append(str(path))
                print("file_paths:",file_paths)
                # 按照自定义排序函数排序
                file_paths.sort(key=custom_sort_key_firecloud)


            elif "gamry" in folder_path:
                file_paths = []

                # 1. 处理以 "ion_column" 结尾的文件夹
                for path in Path(folder_path).glob('**/EISGALV_*/output_txt/*_大于0.txt'):
                    file_paths.append(str(path))
                file_paths.sort(key=custom_sort_key_gamry)

            # 输出排序后的文件路径
            for path in file_paths:
                print(path)

            # 确保路径兼容性
            file_paths = [Path(path) for path in file_paths]
            # 检查每个文件路径是否存在
            for path in file_paths:
                print(f"处理路径: {path}")
                if path.exists():
                    print("文件存在:", path)
                else:
                    print("文件不存在:", path)

            # 获取父文件夹路径
            parent_folder = Path(folder_path).parent

            # 提取文件夹名称
            target_folder_name = parent_folder.name


            # 获取当前脚本所在路径的上一层目录
            current_file_path = os.path.abspath(__file__)
            parent_dir = os.path.dirname(os.path.dirname(current_file_path))

            # 获取当前时间字符串，格式如：20250723（你也可以只用日期）
            timestamp = datetime.now().strftime("%Y%m%d")
            # 获取该目录的父级目录名称
            if "gamry" in folder_path:
            # 拼接目标路径
                output_folder = os.path.join(
                    parent_dir,
                    "eis_fit_results",
                    timestamp,
                    f"{target_folder_name}_{custom_circuit}",
                    "output_ecm_gamry"
                )            
            elif "firecloud" in folder_path:
                output_folder = os.path.join(
                    parent_dir,
                    "eis_fit_results",
                    timestamp,
                    f"{target_folder_name}_{custom_circuit}",
                    "output_ecm_fircloud"
                )                 
            print("output_folder:",output_folder)
            print("file_paths:",file_paths)
            
            # # 处理文件并生成输出
            process_files(file_paths, output_folder, custom_circuit,f"output_values_{custom_circuit}.xlsx",KK_pre_whether)
            
            ####
            #####
            #############################下方是数据分析环节#################################
            ####
            #####
            time.sleep(2)
            print(f"开始绘制{target_folder_name}等效电路 {custom_circuit} 的参数变化图及数据分析")

            color_ion = tuple(c / 255 for c in get_ion_color(target_folder_name))
            print("color_ion:",tuple(c * 255 for c in color_ion))

            process_xlsx_data_analysis(output_folder, f"output_values_{custom_circuit}.xlsx", output_folder, f"data_analysis_{custom_circuit}.png", f"data_analysis_{custom_circuit}.png", color_ion,custom_circuit)

            print(f"处理文件夹 {folder_path} 成功")
        except Exception as e:
            print(f"处理文件夹 {folder_path} 时出错: {e}")
             # 捕获所有异常，并打印堆栈信息
            print("An error occurred:")
            traceback.print_exc()  # 这将输出详细的堆栈追踪信息
            # 或者如果您想捕捉异常信息并进行处理，可以使用:
            error_message = traceback.format_exc()
            print(f"Detailed error message:\n{error_message}")
            continue


def process_single_folder(folder_path, custom_circuit, KK_pre_whether):
    try:
        # ✅ 获取当前 py 文件的上一级 logs 路径
        current_file_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(current_file_path)               # 当前 py 所在目录
        logs_dir = os.path.join(os.path.dirname(script_dir), "logs")  # 上一级 logs 目录
        os.makedirs(logs_dir, exist_ok=True)

        # ✅ 构造日志文件路径
        folder_name = folder_path.replace("/", "_").replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"{folder_name}_{timestamp}.log")

        # ✅ 初始化 logger
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            filemode="w"
        )
        logger = logging.getLogger()

        logger.info(f"🚀 开始处理文件夹: {folder_path}")
        ecm_plot_all([folder_path], custom_circuit, KK_pre_whether)
        logger.info(f"✅ 成功处理完成: {folder_path}")

    except Exception as e:
        # logger 可能未成功初始化，故使用 print
        print(f"❌ 处理 {folder_path} 出错: {e}")
        print(traceback.format_exc())
if __name__ == "__main__":
    import os

    # 获取当前脚本文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 获取当前脚本所在文件夹的父目录（即目标“校内测试”所在目录）
    parent_dir = os.path.dirname(os.path.dirname(current_file_path))
    # 拼接“校内测试”路径
    xiao_nei_ce_shi_dir = os.path.join(parent_dir, "校内测试")

    folder_paths_1 = [

        r"20240822_10ppm铜离子污染测试/新版电解槽_gamry",
        r"20240823_10ppm钙离子污染和恢复测试/新版电解槽_gamry",
        r"20240823_10ppm钙离子污染和恢复测试/新版电解槽_gamry",
        r"20240827_10ppm铬离子污染和恢复测试/新版电解槽_gamry",
        r"20240831_10ppm镍离子污染测试/新版电解槽_gamry",
        r"20240907_10ppm铁离子污染测试/新版电解槽_gamry",
        r"20240910_10ppm钙离子污染测试/新版电解槽_gamry",
        r"20240915_2ppm铜离子污染测试/新版电解槽_gamry",
        r"20240915_2ppm铜离子污染测试/旧版电解槽_firecloud",
        r"20240918_2ppm钙离子污染测试/新版电解槽_gamry",
        r"20241001_2ppm铁离子污染测试/旧版电解槽_firecloud",
        r"20241001_2ppm铁离子污染测试/新版电解槽_gamry",
        r"20241003_2ppm镍离子污染测试/旧版电解槽_firecloud",
        r"20241006_2ppm铬离子污染测试/旧版电解槽_firecloud",
        r"20241006_2ppm铬离子污染测试/新版电解槽_gamry",
        r"20241008_无离子污染测试/新版电解槽_firecloud",
        r"20241010_2ppm钠离子污染测试/新版电解槽_firecloud",
        r"20241013_2ppm铝离子污染测试/新版电解槽_firecloud",
        r"20241017_2ppm铬离子污染和恢复测试/新版电解槽_gamry",
        r"20241020_2ppm镍离子污染和恢复测试/新版电解槽_firecloud",
        r"20241024_2ppm铁离子污染和恢复测试/新版电解槽_firecloud",
        r"20241028_2ppm钠离子污染和恢复测试/旧版电解槽_gamry",
        r"20241029_2ppm铁离子污染和恢复测试/新版电解槽_firecloud",
        r"20241101_2ppm钙离子污染和恢复测试/新版电解槽_firecloud",
        r"20241101_2ppm铜离子污染和恢复测试/旧版电解槽_firecloud",
        r"20241107_0.1ppm钙离子污染及恢复测试/新版电解槽_firecloud",
        r"20241107_0.1ppm铬离子污染及恢复测试/旧版电解槽_firecloud",
        r"20241112_2ppm钠离子污染和恢复测试80摄氏度/新版电解槽_gamry",
        r"20241112_2ppm镍离子污染及恢复测试/旧版电解槽_gamry",
        r"20241117_2ppm钠离子污染及恢复测试10mlmin/旧版电解槽_gamry",
        r"20241117_2ppm钠离子污染及恢复测试40摄氏度/新版电解槽_gamry",
        r"20241122_2ppm钠离子污染及恢复测试300mlmin/旧版电解槽_gamry",
        r"20241122_2ppm铜离子污染及恢复测试40摄氏度/新版电解槽_gamry",
        r"20241201_2ppm铜离子污染及恢复测试/新版电解槽_firecloud",
        r"20241209_无离子污染80摄氏度/新版电解槽_gamry",
        r"20241211_2ppm铜离子污染及恢复测试80摄氏度/旧版电解槽_gamry",
        r"20241213_2ppm钠离子污染及恢复测试/新版电解槽_gamry",
        r"20241214_2ppm铜离子污染测试300mlmin/旧版电解槽_gamry",
        r"20241227_10ppm铜离子污染及恢复测试/旧版电解槽_gamry",
        r"20241229_10ppm钠离子污染及恢复测试/新版电解槽_gamry",
        r"20250101_10ppm铬离子污染及恢复测试/旧版电解槽_gamry",
        r"20250103_无离子污染测试/新版电解槽_gamry"

        # 可以继续添加其他文件夹路径
    ]
    
    # 加上前缀后的完整路径
    folder_paths_1 = [os.path.join(xiao_nei_ce_shi_dir, path) for path in folder_paths_1]
    
    # custom_circuit_2 = "R1-[P2,R3]-[P4,R5]"
    KK_pre_whether = False

    # ecm_plot_all(folder_paths_1,custom_circuit_2,KK_pre_whether=False)

    custom_circuit_2 = "R1-[P2,R3]-[P4,R5]-[P6,R7]"
    # ecm_plot_all(folder_paths_1,custom_circuit_1,KK_pre_whether=False)

    
    
    # ⚡ 多进程运行
    num_workers = min(cpu_count(), len(folder_paths_1))  # 不要超过核心数
    print(f"🧠 使用 {num_workers} 个进程进行并行计算")

    args_list = [(folder_path, custom_circuit_2, KK_pre_whether) for folder_path in folder_paths_1]

    with Pool(processes=num_workers) as pool:
        pool.starmap(process_single_folder, args_list)




