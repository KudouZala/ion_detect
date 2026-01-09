import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from numpy import log, pi
import cvxopt
from basics import compute_epsilon, assemble_A_re, assemble_A_im, assemble_M_1, quad_format_combined, cvxopt_solve_qpr, \
    optimal_lambda, is_PD, nearest_PD, x_to_gamma, pretty_plot

from scipy.integrate import simps

from scipy.signal import savgol_filter

def plot_nyquist(freq, Z_real, Z_imag, Z_real_smooth=None, Z_imag_smooth=None, title="Nyquist Plot"):
    plt.figure(figsize=(8, 6))
    plt.plot(Z_real, -Z_imag, 'o', label='Original Data')
    if Z_real_smooth is not None and Z_imag_smooth is not None:
        plt.plot(Z_real_smooth, -Z_imag_smooth, '-', label='Smoothed Data')
    plt.xlabel(r'$Z^{\prime} (\Omega)$')
    plt.ylabel(r'$-Z^{\prime\prime} (\Omega)$')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Example of adjusting the smoothing parameters
def smooth_data(frequencies, Z_real, Z_imag, window_length=9, polyorder=3):
    Z_real_smooth = savgol_filter(Z_real, window_length=window_length, polyorder=polyorder)
    Z_imag_smooth = savgol_filter(Z_imag, window_length=window_length, polyorder=polyorder)
    return Z_real_smooth, Z_imag_smooth

# In the main function, you can call smooth_data with different parameters
#Z_prime_smooth, Z_double_prime_smooth = smooth_data(eis_entry.freq, eis_entry.Z_prime, eis_entry.Z_double_prime, window_length=11, polyorder=2)


def kramers_kronig_check(frequencies, Z_real, Z_imag):
    """ Perform KK validation to check if the real and imaginary parts are consistent. """

    # Convert frequency to angular frequency
    omega = 2 * np.pi * frequencies

    # Calculate the integral of Z_real and Z_imag
    integral_real = np.trapz(Z_imag / omega, frequencies)
    integral_imag = np.trapz(-Z_real / omega, frequencies)

    # Calculate the error between the two integrals
    error = np.abs(integral_real - integral_imag)

    # Print the integrals and the error
    print(f"Integral of Z_imag/omega: {integral_real}")
    print(f"Integral of -Z_real/omega: {integral_imag}")
    print(f"KK validation error: {error}")

    # The KK condition is that these integrals should be equal
    kk_valid = np.isclose(integral_real, integral_imag, atol=1e-2)  # Adjust the tolerance as needed
    return kk_valid, error


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


class EISObject:

    def __init__(self, freq, Z_prime, Z_double_prime):
        self.freq = freq
        self.Z_prime = Z_prime
        self.Z_double_prime = Z_double_prime
        self.Z_exp = Z_prime + 1j * Z_double_prime

        self.freq_0 = freq
        self.Z_prime_0 = Z_prime
        self.Z_double_prime_0 = Z_double_prime
        self.Z_exp_0 = Z_prime + 1j * Z_double_prime

        self.tau = 1 / freq
        self.tau_fine = np.logspace(np.log10(self.tau.min()) - 0.5, np.log10(self.tau.max()) + 0.5, 10 * freq.shape[0])

        self.method = 'none'

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()
        indices = find_data_starting_indices(lines, ['Freq', 'Zreal', 'Zimag'])
        if None in indices.values():
            raise ValueError(f"Required columns not found in {filename}")
        data = extract_eis_data_from_lines(lines, indices)
        freq = np.array(data['Freq'])
        Z_prime = np.array(data['Zreal'])
        Z_double_prime = np.array(data['Zimag'])
        return cls(freq, Z_prime, Z_double_prime)


def simple_run(entry, rbf_type='Gaussian', data_used='Combined Re-Im Data', induct_used=1, der_used='1st order',
               cv_type='custom', reg_param=1E-3, shape_control='FWHM Coefficient', coeff=0.5):
    N_freqs = entry.freq.shape[0]
    N_taus = entry.tau.shape[0]
    entry.b_re = entry.Z_exp.real
    entry.b_im = entry.Z_exp.imag
    entry.epsilon = compute_epsilon(entry.freq, coeff, rbf_type, shape_control)

    entry.A_re_temp = assemble_A_re(entry.freq, entry.tau, entry.epsilon, rbf_type, flag1='simple', flag2='impedance')
    entry.A_im_temp = assemble_A_im(entry.freq, entry.tau, entry.epsilon, rbf_type, flag1='simple', flag2='impedance')

    if der_used == '1st order':
        entry.M_temp = assemble_M_1(entry.tau, entry.epsilon, rbf_type, flag='simple')

    if data_used == 'Combined Re-Im Data':
        if induct_used == 0 or induct_used == 2:
            N_RL = 1
            entry.A_re = np.zeros((N_freqs, N_taus + N_RL))
            entry.A_re[:, N_RL:] = entry.A_re_temp
            entry.A_re[:, 0] = 1

            entry.A_im = np.zeros((N_freqs, N_taus + N_RL))
            entry.A_im[:, N_RL:] = entry.A_im_temp

            entry.M = np.zeros((N_taus + N_RL, N_taus + N_RL))
            entry.M[N_RL:, N_RL:] = entry.M_temp
        elif induct_used == 1:
            N_RL = 2
            entry.A_re = np.zeros((N_freqs, N_taus + N_RL))
            entry.A_re[:, N_RL:] = entry.A_re_temp
            entry.A_re[:, 1] = 1

            entry.A_im = np.zeros((N_freqs, N_taus + N_RL))
            entry.A_im[:, N_RL:] = entry.A_im_temp
            entry.A_im[:, 0] = 2 * pi * entry.freq

            entry.M = np.zeros((N_taus + N_RL, N_taus + N_RL))
            entry.M[N_RL:, N_RL:] = entry.M_temp

        log_lambda_0 = log(reg_param)
        entry.lambda_value = optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, log_lambda_0,
                                            cv_type)
        H_combined, c_combined = quad_format_combined(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M,
                                                      entry.lambda_value)
        lb = np.zeros([entry.b_re.shape[0] + N_RL])
        bound_mat = np.eye(lb.shape[0])
        x = cvxopt_solve_qpr(H_combined, c_combined, -bound_mat, lb)

        entry.mu_Z_re = entry.A_re @ x
        entry.mu_Z_im = entry.A_im @ x
        entry.res_re = entry.mu_Z_re - entry.b_re
        entry.res_im = entry.mu_Z_im - entry.b_im
        sigma_re_im = np.std(np.concatenate([entry.res_re, entry.res_im]))
        inv_V = 1 / sigma_re_im ** 2 * np.eye(N_freqs)

        Sigma_inv = (entry.A_re.T @ inv_V @ entry.A_re) + (entry.A_im.T @ inv_V @ entry.A_im) + (
                    entry.lambda_value / sigma_re_im ** 2) * entry.M
        mu_numerator = entry.A_re.T @ inv_V @ entry.b_re + entry.A_im.T @ inv_V @ entry.b_im

    entry.Sigma_inv = (Sigma_inv + Sigma_inv.T) / 2

    if not is_PD(entry.Sigma_inv):
        entry.Sigma_inv = nearest_PD(entry.Sigma_inv)

    L_Sigma_inv = np.linalg.cholesky(entry.Sigma_inv)
    entry.mu = np.linalg.solve(L_Sigma_inv, mu_numerator)
    entry.mu = np.linalg.solve(L_Sigma_inv.T, entry.mu)

    if N_RL == 0:
        entry.L, entry.R = 0, 0
    elif N_RL == 1 and data_used == 'Im Data':
        entry.L, entry.R = x[0], 0
    elif N_RL == 1 and data_used != 'Im Data':
        entry.L, entry.R = 0, x[0]
    elif N_RL == 2:
        entry.L, entry.R = x[0:2]

    entry.x = x[N_RL:]
    entry.out_tau_vec, entry.gamma = x_to_gamma(x[N_RL:], entry.tau_fine, entry.tau, entry.epsilon, rbf_type)
    entry.N_RL = N_RL
    entry.method = 'simple'

    return entry


def save_drt_to_excel(entry, output_folder, filename):
    # 打印目标文件夹路径
    print(f"Output folder: {output_folder}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # 确保目标文件夹存在

    df = pd.DataFrame({
        'tau': entry.out_tau_vec,
        'gamma': entry.gamma
    })

    # 在保存文件时包含子文件夹路径
    subfolder_name = os.path.basename(os.path.dirname(filename))
    base_filename = os.path.basename(filename)
    output_filename = f"{subfolder_name}_{base_filename.replace('.DTA', '.xlsx')}"
    output_path = os.path.join(output_folder, output_filename)

    # 打印输出路径
    print(f"Saving Excel file to: {output_path}")

    df.to_excel(output_path, index=False)


def load_drt_from_excel(file_path):
    df = pd.read_excel(file_path)
    return df['tau'].values, df['gamma'].values

def plot_all_drt(entries, folder_path):
    pretty_plot(4, 4)
    plt.rc('font', family='serif', size=15)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    # plt.rc('text', usetex=True)  # 禁用 LaTeX 渲染

    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(entries)))

    for entry, color in zip(entries, colors):
        ax.plot(entry['tau'], entry['gamma'], label=entry['label'], color=color, markersize=4)

        # Print KK validation result and error
        print(f"KK Validation result for {entry['label']}: {'Passed' if entry['kk_valid'] else 'Failed'}, Error: {entry['error']}")

    ax.set_xscale('log')
    ax.set_xlabel(r'$\tau$/s', fontsize=20)
    ax.set_ylabel(r'$\gamma(\log \tau)/\Omega$', fontsize=20)
    ax.legend(fontsize=4, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)  # 将图例放置在下方并设置为多列
    ax.grid(True)

    fig.tight_layout()
    drt_file_name = "drt_combined.png"
    fig.savefig(os.path.join(folder_path, drt_file_name))
    plt.show()


# 使用平滑后的数据进行KK验证
def main():
    base_folder = "C:\\Users\\asus\\Desktop\\ROG桌面其他文件\\电解水\\20240720测试0720-1"
    subfolders = [
        "EISGALV_70℃_150ml_1A",
        "EISGALV_80℃_150ml_1A",
        "EISGALV_90℃_150ml_1A"
    ]

    file_patterns = []
    for subfolder in subfolders:
        for i in range(1, 6):
            file_patterns.append(os.path.join(subfolder, f"cm2_20240720_no_inject_{i}.DTA"))
    output_folder = os.path.join(base_folder, "output_drt_smooth")
    entries = []

    for file_name in file_patterns:
        file_path = os.path.join(base_folder, file_name)
        subfolder_name = os.path.basename(os.path.dirname(file_name))
        output_filename = f"{subfolder_name}_{os.path.basename(file_name).replace('.DTA', '.xlsx')}"
        output_path = os.path.join(output_folder, output_filename)
        if os.path.isfile(file_path):
            if os.path.isfile(output_path):
                tau, gamma = load_drt_from_excel(output_path)
                kk_valid = False  # Load KK validation result from the file if available
                entry = {'tau': tau, 'gamma': gamma, 'label': output_filename, 'kk_valid': kk_valid, 'error': None}
            else:
                eis_entry = EISObject.from_file(file_path)
                eis_entry = simple_run(eis_entry)
                save_drt_to_excel(eis_entry, output_folder, file_name)

                # Apply smoothing
                Z_prime_smooth, Z_double_prime_smooth = smooth_data(eis_entry.freq, eis_entry.Z_prime, eis_entry.Z_double_prime)

                # Plot Nyquist plot before and after smoothing
                plot_nyquist(eis_entry.freq, eis_entry.Z_prime, eis_entry.Z_double_prime,
                             Z_prime_smooth, Z_double_prime_smooth,
                             title=f"Nyquist Plot: {output_filename}")

                # Perform KK validation
                kk_valid, error = kramers_kronig_check(eis_entry.freq, Z_prime_smooth, Z_double_prime_smooth)
                entry = {'tau': eis_entry.out_tau_vec, 'gamma': eis_entry.gamma, 'label': output_filename,
                         'kk_valid': kk_valid, 'error': error}
            entries.append(entry)
        else:
            print(f"File not found: {file_path}")

    plot_all_drt(entries, output_folder)

if __name__ == "__main__":
    main()




if __name__ == "__main__":
    main()





