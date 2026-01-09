import pandas as pd
import matplotlib.pyplot as plt
import os


def vitplot_zhiyun(folder_path, file_names):
    """
    读取多个CSV文件，按时间叠加并绘制电流和电压随时间变化的图，并将图像保存到指定目录。

    参数:
    - folder_path: 包含CSV文件的文件夹路径。
    - file_names: CSV文件名的列表。

    返回:
    - 将电流和电压随时间变化的图像保存到output_plot目录中。
    """

    # 初始化时间、电流、电压
    total_time = []
    total_current = []
    total_voltage = []
    cumulative_time = 0

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 确保列名无误
        time_col = 'StepTime/s'
        current_col = 'Current/A'
        voltage_col = 'Voltage/V'

        # 获取时间、电流、电压数据
        time = df[time_col].to_list()
        current = df[current_col].to_list()
        voltage = df[voltage_col].to_list()

        # 将时间叠加
        adjusted_time = [t + cumulative_time for t in time]
        cumulative_time = adjusted_time[-1]  # 更新累计时间

        # 将当前文件的数据添加到总数据中
        total_time.extend(adjusted_time)
        total_current.extend(current)
        total_voltage.extend(voltage)

    # 创建输出文件夹
    output_folder = os.path.join(folder_path, 'output_plot')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 绘制电流和电压在同一图中
    plt.figure(figsize=(10, 6))
    plt.plot(total_time, total_current, label='Current (A)', color='b')
    plt.plot(total_time, total_voltage, label='Voltage (V)', color='r')

    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Current and Voltage vs Time')
    plt.legend()
    plt.grid(True)

    # 保存图像到指定路径
    output_path = os.path.join(output_folder, 'vitplot_zhiyun.png')
    plt.savefig(output_path)
    plt.close()

    print(f"图像已保存到: {output_path}")


def main():
    # folder_path = r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241003_2ppm镍离子污染测试\旧版电解槽-炙云设备\20241002_ion_column-20241002-170207-默认1728123271707"
    # file_names = [
    #     "循环1／1_工步组1(工步组)(1／1)_工步1(CC).csv",
    #     "循环1／1_工步组2(工步组)(1／80)_工步1(CC).csv",
    #     "循环1／1_工步组2(工步组)(2／80)_工步1(CC).csv",
    #     "循环1／1_工步组2(工步组)(3／80)_工步1(CC).csv",
    #     "循环1／1_工步组2(工步组)(4／80)_工步1(CC).csv"
    #     # 你可以继续添加其他文件名
    # ]
    folder_path = r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20241003_2ppm镍离子污染测试\旧版电解槽-炙云设备\20241003_ion-20241003-025748-默认1728123214071"
    file_names = [
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
        "循环1／1_工步组2(工步组)(29／80)_工步1(CC).csv",
        "循环1／1_工步组2(工步组)(30／80)_工步1(CC).csv",
        # 你可以继续添加其他文件名
    ]
    vitplot_zhiyun(folder_path, file_names)


main()

