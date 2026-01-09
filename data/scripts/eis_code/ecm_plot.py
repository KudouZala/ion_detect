import os
import pandas as pd
import matplotlib.pyplot as plt
from impedance.models.circuits import CustomCircuit

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
            data['Freq'].append(float(columns[indices['Freq'][1]]))
            data['Zreal'].append(float(columns[indices['Zreal'][1]]))
            data['Zimag'].append(float(columns[indices['Zimag'][1]]))
    return data

def load_eis_data(file_path, freq_min=None, freq_max=None):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()

    indices = find_data_starting_indices(lines, ['Freq', 'Zreal', 'Zimag'])
    if None in indices.values():
        raise ValueError(f"Required columns not found in {file_path}")

    data = extract_eis_data_from_lines(lines, indices)

    # 转换为 DataFrame 以便进行筛选
    df = pd.DataFrame(data)

    # 如果指定了频率范围，则筛选出该范围内的数据
    if freq_min is not None:
        df = df[df['Freq'] >= freq_min]
    if freq_max is not None:
        df = df[df['Freq'] <= freq_max]

    frequencies = df['Freq'].values
    Z = df['Zreal'].values + 1j * df['Zimag'].values
    return frequencies, Z

def plot_nyquist(ax, zreal, zimag, label, color, marker):
    ax.plot(zreal, -zimag, label=label, color=color, marker=marker)

def plot_nyquist_comparison(frequencies, Z, Z_fit, file_name, output_folder):
    plt.figure(figsize=(6, 6))
    plt.plot(Z.real, -Z.imag, 'o', label='Data')
    plt.plot(Z_fit.real, -Z_fit.imag, '-', label='Fit')
    plt.xlabel('Zreal (Ohm)')
    plt.ylabel('-Zimag (Ohm)')
    plt.legend()
    plt.grid(True)
    plt.title(f'Nyquist Plot - {file_name}')

    # Save the plot
    plot_file_path = os.path.join(output_folder, f"{file_name.replace('.DTA', '')}_nyquist.png")
    plt.savefig(plot_file_path)
    plt.show()

def fit_circuit_and_plot(file_path, circuit_string, initial_guess, output_folder, freq_min=None, freq_max=None, max_iter=1000, ftol=1e-8, xtol=1e-8):
    # Load the data with frequency filtering
    frequencies, Z = load_eis_data(file_path, freq_min=freq_min, freq_max=freq_max)

    # Create and fit the circuit model with specified maximum iterations and tolerances
    circuit = CustomCircuit(circuit=circuit_string, initial_guess=initial_guess)
    circuit.fit(frequencies, Z, maxfev=max_iter, ftol=ftol, xtol=xtol)

    # Predict the impedance using the fitted model
    Z_fit = circuit.predict(frequencies)

    # Print the fitted parameters
    print("Fitted Parameters:")
    print(circuit.parameters_)

    # Plot the Nyquist plot comparing the data and the fitted model
    plot_nyquist_comparison(frequencies, Z, Z_fit, os.path.basename(file_path), output_folder)

def convert_and_plot_eis(folder_path, file_specifications, circuit_string, initial_guess, freq_min=None, freq_max=None, max_iter=1000, ftol=1e-8, xtol=1e-8):
    output_folder_plot = os.path.join(folder_path, "output_plot")
    os.makedirs(output_folder_plot, exist_ok=True)

    fig_nyquist, ax_nyquist = plt.subplots(figsize=(12, 6))

    for file_name, color, marker in file_specifications:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            fit_circuit_and_plot(file_path, circuit_string, initial_guess, output_folder_plot, freq_min, freq_max, max_iter, ftol, xtol)
            frequencies, Z = load_eis_data(file_path, freq_min=freq_min, freq_max=freq_max)
            plot_nyquist(ax_nyquist, Z.real, Z.imag, label=f"{file_name}", color=color, marker=marker)
        else:
            print(f"File not found: {file_path}")

    ax_nyquist.set_xlabel('Zreal (Ohm)')
    ax_nyquist.set_ylabel('-Zimag (Ohm)')
    ax_nyquist.set_title('Nyquist Plot')
    ax_nyquist.grid(True)
    ax_nyquist.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    fig_nyquist.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局，给图例留出更多空间
    nyquist_file_name = "nyquist_combined.png"
    fig_nyquist.savefig(os.path.join(output_folder_plot, nyquist_file_name))

    plt.show()

def main():
    base_folder = r"C:\Users\asus\Desktop\ROG桌面其他文件\电解水\20240813-0814铜离子污染测试\20240815-ion-renew\EISGALV_60℃_150ml_1A"

    # 指定每个文件的文件名、颜色和标记符号
    # file_specifications = [
    #     ("cm2_20240814_ion_1.DTA", 'red', '+'),
    #     ("cm2_20240814_ion_4.DTA", 'orange', '+'),
    #     ("cm2_20240814_ion_8.DTA", 'yellow', '+'),
    #     ("cm2_20240814_ion_12.DTA", 'green', '+'),
    #     ("cm2_20240814_ion_16.DTA", 'blue', '+'),
    #     ("cm2_20240814_ion_20.DTA", 'cyan', '+'),
    #     ("cm2_20240814_ion_24.DTA", 'purple', '+'),
    #     ("cm2_20240814_ion_28.DTA", 'brown', '+'),
    #     ("cm2_20240814_ion_32.DTA", 'black', '+'),
    # ]
    file_specifications = [
        ("cm2_20240814_ion_1.DTA", 'red', '+')
    ]

    # 指定等效电路
    circuit_string = 'R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)'

    # 设置初始参数
    initial_guess = [0.3821, 0.514, 0.096,0.757, 0.03, 0.05, 0.02, 0.03, 0.05, 0.5]

    # 设置频率范围（可选）
    freq_min = 0.1
    freq_max = 10000

    # 设置最大迭代次数和误差容忍度
    max_iter = 5000
    ftol = 1e-10  # 设置误差容忍度
    xtol = 1e-10  # 设置参数变化容忍度

    convert_and_plot_eis(base_folder, file_specifications, circuit_string, initial_guess, freq_min, freq_max, max_iter, ftol, xtol)

if __name__ == "__main__":
    main()

