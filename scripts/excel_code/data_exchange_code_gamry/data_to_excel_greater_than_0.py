import os
import pandas as pd


def find_data_starting_indices(lines, keywords):
    indices = {key: None for key in keywords}
    for i, line in enumerate(lines):
        columns = line.strip().split('\t')
        for j, column in enumerate(columns):
            if column in indices:
                indices[column] = (i, j)
    return indices


def extract_eis_data_from_lines(lines, indices):
    data = {'Freq': [], 'Zreal': [], 'Zimag': []}
    start_row = max(index[0] for index in indices.values()) + 2
    for line in lines[start_row:]:
        columns = line.strip().split('\t')
        if len(columns) > max(index[1] for index in indices.values()):  # 确保行有足够的列
            try:
                freq = float(columns[indices['Freq'][1]])
                zreal = float(columns[indices['Zreal'][1]])
                zimag = float(columns[indices['Zimag'][1]])
                if zimag < 0:  # 只保留虚部小于0的点
                    data['Freq'].append(freq)
                    data['Zreal'].append(zreal)
                    data['Zimag'].append(zimag)
            except ValueError:
                # 如果转换失败，跳过该行
                continue
    return data


def convert_to_csv_and_xlsx_greater_than_0(folder_path, file_specifications):
    output_folder_csv = os.path.join(folder_path, "output_csv")
    output_folder_xlsx = os.path.join(folder_path, "output_xlsx")
    os.makedirs(output_folder_csv, exist_ok=True)
    os.makedirs(output_folder_xlsx, exist_ok=True)

    for file_name in file_specifications:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                lines = file.readlines()

            indices = find_data_starting_indices(lines, ['Freq', 'Zreal', 'Zimag'])
            if None in indices.values():
                print(f"在文件 {file_path} 中未找到所有必需的列。")
                continue

            data = extract_eis_data_from_lines(lines, indices)

            if not data['Freq']:
                print(f"文件 {file_path} 中没有符合条件的数据。")
                continue

            df = pd.DataFrame(data)

            base_name = os.path.basename(file_name).replace('.DTA', '')

            # 保存为 CSV 文件
            csv_file_path = os.path.join(output_folder_csv, f"{base_name}_大于0_no_title.csv")
            df.to_csv(csv_file_path, index=False,header=False)
            print(f"已保存 CSV 文件: {csv_file_path}")

            # 保存为 Excel 文件
            xlsx_file_path = os.path.join(output_folder_xlsx, f"{base_name}_大于0.xlsx")
            df.to_excel(xlsx_file_path, index=False)
            print(f"已保存 Excel 文件: {xlsx_file_path}")

        else:
            print(f"未找到文件: {file_path}")


def main():
    base_folder = r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\校内测试\20240823_10ppm钙离子污染测试\新版电解槽_gamry\EISGALV_60℃_150ml_1A"

    # 只保留文件名列表
    file_specifications = [
        "cm2_20240823_ion_column_0.DTA",
        "cm2_20240823_ion_column_10.DTA",

        "cm2_20240824_ion_0.DTA",
        "cm2_20240824_ion_1.DTA",
        "cm2_20240824_ion_2.DTA",
        "cm2_20240824_ion_3.DTA",
        "cm2_20240824_ion_4.DTA",
        "cm2_20240824_ion_5.DTA",
        "cm2_20240824_ion_6.DTA",
        "cm2_20240824_ion_7.DTA",
        "cm2_20240824_ion_8.DTA",
        "cm2_20240824_ion_9.DTA",
        "cm2_20240824_ion_10.DTA",
        "cm2_20240824_ion_11.DTA",
        "cm2_20240824_ion_12.DTA",

        "cm2_20240826_ion_renew_0.DTA",
        "cm2_20240826_ion_renew_10.DTA",

    ]

    convert_to_csv_and_xlsx_greater_than_0(base_folder, file_specifications)


if __name__ == "__main__":
    main()

