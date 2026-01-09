import numpy as np
import torch
from scipy.optimize import minimize

# 优化函数：拟合 true_voltage_first_time
def optimize_electrochemical_parameters(temperature, flow, current, true_voltage_first_time,
                                        theta_ca, phi_ca, theta_an, phi_an, psi,
                                        initial_guess=[0.1, 0.4, 0.5, 1e-2, 1e-7]):
    def loss_fn(params):
        sigma_mem, alpha_ca, alpha_an, i_0ca, i_0an = params
        T = temperature + 273.15
        R = 8.314
        F = 96485
        i = current * 4  # 假设膜面积 4 cm²
        E_rev = 1.229 - 0.9e-3 * (T - 298)
        try:
            V_ocv = E_rev + R * T / (2 * F) * np.log(1.0 * np.sqrt(1.0) / 1.0)
            V_act = R * T / (theta_ca * alpha_ca * F) * np.arcsinh(current / (2 * i_0ca * (1 - phi_ca))) + \
                    R * T / (theta_an * alpha_an * F) * np.arcsinh(current / (2 * i_0an * (1 - phi_an)))

            t_an, t_ca1, t_ca2, t_mem = 0.012, 0.012, 0.002, 135e-6
            sigma_an = 2.38e6
            sigma_ca1 = 2.38e6
            sigma_ca2 = 5.96e7

            V_ohm = i * (t_an / sigma_an + t_ca1 / sigma_ca1 + t_ca2 / sigma_ca2 + t_mem / sigma_mem*100 / (1 - psi))

            predicted = V_ocv + V_act + V_ohm
        except Exception:
            return np.inf
        return abs(predicted - true_voltage_first_time)

    # 改进后的物理范围 bounds
    bounds = [
        (0.01, 1),      # sigma_mem
        (0.25, 0.7),        # alpha_ca
        (0.25, 2),        # alpha_an
        (1e-3, 1e-1),      # i_0ca
        (1e-9, 1e-6),      # i_0an
    ]

    result = minimize(loss_fn, initial_guess, bounds=bounds, method='L-BFGS-B')
    return result.x  # 返回最优解

# 预测电压函数（Torch 实现）
def calculate_predicted_voltage(theta_ca, phi_ca, theta_an, phi_an, psi, temperature, flow, current,
                                sigma_mem, alpha_ca, alpha_an, i_0ca, i_0an):
    R = 8.314
    F = 96485
    T = temperature + 273.15

    # 转换为 Tensor 类型
    i = torch.tensor(current * 4, dtype=torch.float32)
    i_0ca = torch.tensor(i_0ca, dtype=torch.float32)
    i_0an = torch.tensor(i_0an, dtype=torch.float32)
    alpha_ca = torch.tensor(alpha_ca, dtype=torch.float32)
    alpha_an = torch.tensor(alpha_an, dtype=torch.float32)
    sigma_mem = torch.tensor(sigma_mem, dtype=torch.float32)

    V_ocv = (1.229 - 0.9 * 10 ** -3 * (T - 298)) + R * T / (2 * F) * torch.log(torch.tensor(1.0) * torch.sqrt(torch.tensor(1.0)) / torch.tensor(1.0))

    V_act = R * T / (theta_ca * alpha_ca * F) * torch.arcsinh(current / (2 * i_0ca * (1 - phi_ca))) + \
            R * T / (theta_an * alpha_an * F) * torch.arcsinh(current / (2 * i_0an * (1 - phi_an)))

    t_an = 0.012
    t_ca1 = 0.012
    t_ca2 = 0.002
    t_mem = 135e-6
    sigma_an = 2.38e6
    sigma_ca1 = 2.38e6
    sigma_ca2 = 5.96e7

    V_ohm = i  * (t_an / sigma_an + t_ca1 / sigma_ca1 + t_ca2 / sigma_ca2 + t_mem / sigma_mem*100 / (1 - psi))

    V_diff = torch.tensor(0.0)

    predicted_voltage = V_ocv + V_act + V_diff + V_ohm
    return predicted_voltage.item(), V_ocv.item(), V_act.item(), V_diff.item(), V_ohm.item()


# ===== 示例测试 =====

if __name__ == "__main__":
    # 输入数据示例
    temperature = 60       # 摄氏度
    flow = 0.1             # 未使用
    current = 1          # A
    true_voltage_first_time = 1.65  # 实测第一时刻电压

    theta_ca = 1.0
    phi_ca = 0
    theta_an = 1.0
    phi_an = 0
    psi = 0

    # 执行优化
    optimal_params = optimize_electrochemical_parameters(
        temperature, flow, current, true_voltage_first_time,
        theta_ca, phi_ca, theta_an, phi_an, psi
    )

    sigma_mem, alpha_ca, alpha_an, i_0ca, i_0an = optimal_params
    print("\n=== 最优参数 ===")
    print(f"sigma_mem = {sigma_mem:.4f} S/cm")
    print(f"alpha_ca  = {alpha_ca:.4f}")
    print(f"alpha_an  = {alpha_an:.4f}")
    print(f"i_0ca     = {i_0ca:.4e} A/cm²")
    print(f"i_0an     = {i_0an:.4e} A/cm²")

    # 验证预测电压分量
    pred, vocv, vact, vdiff, vohm = calculate_predicted_voltage(
        theta_ca, phi_ca, theta_an, phi_an, psi, temperature, flow, current,
        sigma_mem, alpha_ca, alpha_an, i_0ca, i_0an
    )
    print("\n=== 预测电压分解 ===")
    print(f"Predicted Voltage = {pred:.4f} V")
    print(f"V_ocv = {vocv:.4f} V, V_act = {vact:.4f} V, V_ohm = {vohm:.4f} V")
