import os
import pandas as pd
import matplotlib.pyplot as plt
import math

def calculate_fc(Q, R, alpha):
    fc = (1 / (Q * R))**(1 / alpha) / (2 * math.pi)
    return fc

def reorder_data(file_path):
    # 读取Excel文件，跳过前两行（假设第一行是列名，第二行是数据）
    df = pd.read_excel(file_path, header=1)
    
    # 读取Excel文件的前两行，以便保留这些信息
    header = pd.read_excel(file_path, header=None, nrows=2)
    
    # 创建一个新列来存储排序后的数据
    sorted_data = []
    
    for index, row in df.iterrows():
        # 获取C、F、I列的数据（列索引从0开始）
        c_val, f_val = row.iloc[2], row.iloc[5]  # C列是索引2，F列是索引5
        d_val, e_val = row.iloc[3], row.iloc[4]  # C列是索引2，F列是索引5
        g_val, h_val = row.iloc[6], row.iloc[7]  # C列是索引2，F列是索引5
        fc1 = calculate_fc(c_val, e_val, d_val)
        fc2 = calculate_fc(f_val, h_val, g_val)

        if len(row) > 8:  # 判断I列是否存在
            i_val = row.iloc[8]  # I列是索引8
            j_val = row.iloc[9]  # I列是索引8
            k_val = row.iloc[10]  # I列是索引8
            fc3 = calculate_fc(i_val, k_val, j_val)
        else:
            i_val = None
        
        # 如果没有I列，只考虑C和F的排序
        if pd.isna(i_val):  
            no_I=True
            # 根据C和F进行从小到大排序
            # sorted_values = sorted([(c_val, row.iloc[3], row.iloc[4]), (f_val, row.iloc[6], row.iloc[7])], key=lambda x: x[0])
            if fc1 < fc2:  #先低频后高频
                sorted_values = [(c_val, d_val, e_val), (f_val, g_val, h_val)]  
            else:  
                sorted_values = [(f_val, g_val, h_val), (c_val, d_val, e_val)] 
            sorted_values = sorted([(c_val, d_val, e_val), (f_val,g_val, h_val)], key=lambda x: fc1 if x == (c_val, d_val, e_val) else fc2 )

        else:
            # 根据C, F, I进行从小到大排序
            no_I=False
            # 假设有三个特征频率 fc1, fc2, fc3  
            if fc1 < fc2 < fc3:  # 低频在前，高频在后  
                sorted_values = [(c_val, d_val, e_val), (f_val, g_val, h_val), (i_val, j_val, k_val)]  
            elif fc1 < fc3 < fc2:  # fc3 介于 fc1 和 fc2 之间  
                sorted_values = [(c_val, d_val, e_val), (i_val, j_val, k_val), (f_val, g_val, h_val)]  
            elif fc2 < fc1 < fc3:  # fc1 介于 fc2 和 fc3 之间  
                sorted_values = [(f_val, g_val, h_val), (c_val, d_val, e_val), (i_val, j_val, k_val)]  
            elif fc2 < fc3 < fc1:  # fc3 介于 fc2 和 fc1 之间  
                sorted_values = [(f_val, g_val, h_val), (i_val, j_val, k_val), (c_val, d_val, e_val)]  
            elif fc3 < fc1 < fc2:  # fc1 介于 fc3 和 fc2 之间  
                sorted_values = [(i_val, j_val, k_val), (c_val, d_val, e_val), (f_val, g_val, h_val)]  
            elif fc3 < fc2 < fc1:  
                sorted_values = [(i_val, j_val, k_val), (f_val, g_val, h_val), (c_val, d_val, e_val)]  
            else:  # 如果出现了特征频率相等的情况或其他未处理的情况 
                print("特殊情况，注意：特征频率相等") 
                # sorted_values = [(c_val, d_val, e_val), (f_val, g_val, h_val), (i_val, j_val, k_val)]  # 默认顺序  

            # sorted_values = sorted([(c_val, row.iloc[3], row.iloc[4]), (f_val, row.iloc[6], row.iloc[7]), (i_val, row.iloc[9], row.iloc[10])], key=lambda x: x[0])

        # 提取排序后的数据，并将其添加到新的列
        reordered = []
        for val in sorted_values:
            reordered.extend(val[0:])  # 提取排序后的组数据

        # 将排序后的数据添加到新的列表
        sorted_data.append(reordered)

    # 将排序后的数据转换为DataFrame
    if no_I==True:
        sorted_df = pd.DataFrame(sorted_data, columns=[2, 3, 4, 5, 6, 7])  # 使用列索引命名
        # 确保sorted_df的列数与目标区域匹配
        df.iloc[0:, 2:8] = sorted_df.values  # 注意从第三行（即索引2）开始修改数据
    else:
        sorted_df = pd.DataFrame(sorted_data, columns=[2, 3, 4, 5, 6, 7, 8, 9, 10])  # 使用列索引命名
        # 确保sorted_df的列数与目标区域匹配
        df.iloc[0:, 2:11] = sorted_df.values  # 注意从第三行（即索引2）开始修改数据
    
    # 添加标题行
    if no_I:
        title = [''] + ['R0', 'P1w', 'P1n', 'R1', 'P2w', 'P2n', 'R2']
    else:
        title = [''] + ['R0', 'P1w', 'P1n', 'R1', 'P2w', 'P2n', 'R2', 'P3w', 'P3n', 'R3']

    # 将标题添加到数据的第一行
    df.columns = title + df.columns[len(title):].tolist()

    # 保存新的Excel文件
    new_file_path = file_path.replace('.xlsx', '_sorted.xlsx')
    df.to_excel(new_file_path, index=False, header=True)
    print(f"排序后的数据已经保存到 {new_file_path}")
    return new_file_path


def plot_excel_data(file_path, legend_dict,custom_title):
    """
    读取Excel文件并绘制数据图，每个图例根据legend_dict进行自定义。
    
    :param file_path: 输入的Excel文件路径
    :param legend_dict: 包含颜色、标记、图例文本的字典
    """
    # 读取Excel文件，标题在第一行
    print(f"正在读取Excel文件: {file_path}")
    df = pd.read_excel(file_path, header=0)
    print(f"Excel文件读取完成, 数据的前几行:\n{df.head()}")

    # 获取标题行（假设从第二列开始是数据列）
    titles = df.columns[1:]  # 排除第一列
    print(f"列标题（去除第一列）: {titles.tolist()}")

    # 生成时间列，根据行数 * 2 来生成
    time_data = [(i * 2) for i in range(len(df))]
    print(f"生成的时间数据: {time_data[:5]}... (总计 {len(time_data)} 个时间点)")

    # 创建图表
    num_titles = len(titles)
    print(f"共有 {num_titles} 个标题需要绘制")

    # 计算行数和列数，最大每行3个图
    num_rows = (num_titles // 3) + (1 if num_titles % 3 != 0 else 0)
    num_cols = min(3, num_titles)
    print(f"图表的排布: {num_rows} 行, {num_cols} 列")

    # 创建图形对象
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    print(f"创建了 {num_rows} x {num_cols} 的子图格局")

    # 如果是单行单列，axes不是一个二维数组，因此需要扁平化处理
    if num_rows == 1:
        axes = axes.reshape(1, num_cols)
        print("图表是单行布局，已将axes调整为1行")
    if num_cols == 1:
        axes = axes.reshape(num_rows, 1)
        print("图表是单列布局，已将axes调整为1列")

    # 绘制每个标题的数据
    for i, title in enumerate(titles):
        print(f"正在绘制: {title}")

        # 获取当前列的数据，并确保其为数值型
        data = pd.to_numeric(df[title], errors='coerce')  # 将非数值项转为 NaN
        data = data.dropna()  # 去掉 NaN 值
        print(f"{title} 的数据：\n{data.head()}... (总计 {len(data)} 个有效数据点)")

        # 绘制每个数据点，使用不同的颜色和标记
        row = i // num_cols
        col = i % num_cols

        # 初始化图例标签，直接从legend_dict中获取
        for j in range(len(data) - 1):  # 遍历数据点，但不包括最后一个点
            time1, value1 = time_data[j], data.iloc[j]
            time2, value2 = time_data[j + 1], data.iloc[j + 1]
            
            # 获取每一行的第一列内容，并根据其内容确定颜色和标记
            row_first_column_value = str(df.iloc[j, 0])  # 获取当前行的第一列值

            # # 判断并选择自定义图例配置
            # color, marker, label = None, None, None
            # for key, value_dict in legend_dict.items():
            #     if key.lower() in row_first_column_value.lower():  # 判断是否匹配
            #         color = value_dict['color']
            #         marker = value_dict['marker']
            #         label = value_dict['label']
            #         break

            # # 如果没有匹配到，则使用默认的红色和X标记
            # if not color or not marker or not label:
            #     color = 'red'
            #     marker = 'x'
            #     label = 'Default'
            if "ion_column_renew" in row_first_column_value.lower():  # 判断是否匹配
                color = "gray"
                marker = "s"
            elif "ion_column" in row_first_column_value.lower():  # 判断是否匹配
                color = "green"
                marker = "o"
            else:
                color = "red"
                marker = "x"

            # 如果是离群点（nan 或其他），可以跳过
            if pd.notna(value1) and pd.notna(value2):
                # 绘制数据点
                axes[row, col].plot(time1, value1, color=color, marker=marker, markersize=5)
                axes[row, col].plot(time2, value2, color=color, marker=marker, markersize=5)

                # 绘制每对相邻点之间的线段，保证线的颜色与点一致
                axes[row, col].plot([time1, time2], [value1, value2], color=color, lw=1)

        # 设置标题和标签
        axes[row, col].set_title(custom_title)
        axes[row, col].set_xlabel('Time (h)')
        axes[row, col].set_ylabel(title)

        # 直接根据legend_dict生成图例
        handles = []
        for key, value_dict in legend_dict.items():
            handle = plt.Line2D([0], [0], marker=value_dict['marker'], color=value_dict['color'], markerfacecolor=value_dict['color'], markersize=10, label=value_dict['label'])
            handles.append(handle)

        # 将自定义图例加入图表，放置在图片下方
        plt.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)


    # 调整图表布局，防止重叠
    plt.tight_layout()

    # 保存图表为图片
    output_image_path = file_path.replace('.xlsx', '_plot.png')
    try:
        plt.savefig(output_image_path)
        plt.close()
        print(f"图表已保存为 {output_image_path}")
    except Exception as e:
        print(f"保存图表时出错: {e}")





def process_multiple_files(file_paths,legend_dict,custom_title):
    # 获取当前py文件的所在路径
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    num=0
    for file_path in file_paths:
        # 为每个文件添加前缀
        full_path = os.path.join(base_dir, file_path)
        print("full_path:",full_path)
        sorted_file_path = reorder_data(full_path)
        plot_excel_data(sorted_file_path,legend_dict,custom_title=custom_title[num])  # 生成图表并保存为图片
        num+=1

if __name__ == '__main__':
    # 输入多个文件路径
    file_paths = [
        # 'eis_results_20241113/20240918_2ppm钙离子污染测试_R1-[P2,R3]-[P4,R5]/output_ecm_gamry/output_values_R1-[P2,R3]-[P4,R5].xlsx',
        # 'eis_results_20241107/20240918_2ppm钙离子污染测试_R1-[P2,R3]-[P4,R5]-[P6,R7]/output_ecm_gamry/output_values_R1-[P2,R3]-[P4,R5]-[P6,R7].xlsx',
        'eis_results_20241107/20240919_10ppm钙离子污染测试_R1-[P2,R3]-[P4,R5]/output_ecm_gamry/output_values_R1-[P2,R3]-[P4,R5].xlsx',
        # 'eis_results_20241113/20241101_2ppm钙离子污染和恢复测试_R1-[P2,R3]-[P4,R5]/output_ecm_firecloud/output_values_R1-[P2,R3]-[P4,R5].xlsx'
    ]
    # 示例：自定义图例字典
    legend_dict = {
        'ion_column': {'color': 'green', 'marker': 'o', 'label': 'ion_column'},
        'ion': {'color': 'red', 'marker': 'x', 'label': 'ion'},
        # 'ion_column_renew': {'color': 'gray', 'marker': 's', 'label': 'ion_column_renew_H2SO4'}
    }
    custom_title = ["2ppm_Ca2+_20241107","20240918_2ppm钙离子"]
    # 调用处理多个文件的函数
    process_multiple_files(file_paths,legend_dict=legend_dict,custom_title=custom_title)
