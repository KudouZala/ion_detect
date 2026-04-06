import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch


LABEL_MAPPING = {
    "Ca2+_ion": 0,
    "Na+_ion": 1,
    "Ni2+_ion": 2,
    "Cr3+_ion": 3,
    "Cu2+_ion": 4,
    "Fe3+_ion": 5,
    "no_ion": 6,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAPPING.items()}
ION_CHINESE_MAP = {
    "Ca2+_ion": "钙离子",
    "Na+_ion": "钠离子",
    "Ni2+_ion": "镍离子",
    "Cr3+_ion": "铬离子",
    "Cu2+_ion": "铜离子",
    "Fe3+_ion": "铁离子",
    "no_ion": "无污染",
}

_WINDOW_RE = re.compile(
    r"^(?P<prefix>.*?_)\[(?P<times>\d+(?:\s*,\s*\d+)*)\]\s*(?:\.\w+)?$"
)


def evaluate_prediction(xlsx_path, predicted_label):
    if predicted_label is not None:
        predicted_ion_chinese = ION_CHINESE_MAP.get(predicted_label, "未知")
    else:
        predicted_ion_chinese = "无预测"

    true_ion_chinese = "未知"
    if "ion_column" in xlsx_path:
        true_ion_chinese = "无污染"
    else:
        for name in ION_CHINESE_MAP.values():
            if name in str(xlsx_path):
                true_ion_chinese = name
                break

    correct = predicted_ion_chinese == true_ion_chinese
    return correct, predicted_ion_chinese, true_ion_chinese


def _parse_name_times(xlsx_path: str):
    fname = os.path.basename(xlsx_path)
    match = _WINDOW_RE.match(fname.strip())
    if not match:
        return None
    return [int(x.strip()) for x in match.group("times").split(",")]


def _remap_df_time_by_filename(df: pd.DataFrame, xlsx_path: str):
    name_times = _parse_name_times(xlsx_path)
    if name_times is None or "Time(h)" not in df.columns:
        return df, None

    slot_values = sorted(df["Time(h)"].dropna().unique().tolist())
    if len(slot_values) != len(name_times):
        return df, None

    pieces = []
    for idx, slot_t in enumerate(slot_values):
        abs_t = int(name_times[idx])
        t_df = df[df["Time(h)"] == slot_t].copy()
        t_df["Time(h)"] = abs_t
        pieces.append(t_df)
    mapped = pd.concat(pieces, axis=0, ignore_index=True)
    return mapped, name_times


def _get_electrolyzer_parameters(file_name: str, device: torch.device):
    if "新版电解槽" in file_name:
        values = [0.012, 0.012, 0.002, 135e-6, 2.38e6, 2.38e6, 5.96e7, 4]
    elif "旧版电解槽" in file_name:
        values = [0.012, 0.012, 0.002, 135e-6, 2.38e6, 2.38e6, 5.96e7, 4]
    else:
        values = [0.012, 0.012, 0.002, 135e-6, 2.38e6, 2.38e6, 5.96e7, 4]
    return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0)


def _load_single_plus_sample(
    xlsx_path: str,
    device: torch.device,
    num_freq_points: int,
    stats_file: str,
):
    df = pd.read_excel(xlsx_path)

    if "ppm" in df.columns:
        try:
            concentration_value = float(df["ppm"].iloc[0])
        except Exception:
            concentration_value = -1.0
    else:
        concentration_value = -1.0
    concentration = torch.tensor([[concentration_value]], dtype=torch.float32, device=device)

    df, parsed_times = _remap_df_time_by_filename(df, xlsx_path)
    if parsed_times is None:
        if "Time(h)" not in df.columns:
            raise ValueError("Missing Time(h) column")
        parsed_times = sorted(df["Time(h)"].dropna().unique().tolist())
    if len(parsed_times) < 1:
        raise ValueError("No valid time points found")

    volt_list, impe_list = [], []
    for t in parsed_times:
        t_df = df[df["Time(h)"] == t].copy()
        if t_df.empty:
            raise ValueError(f"Missing mapped time point {t}")
        if "Freq" not in t_df.columns:
            raise ValueError("Missing column 'Freq'")
        t_df = t_df.sort_values("Freq", ascending=False).reset_index(drop=True)
        if len(t_df) < num_freq_points:
            raise ValueError(f"Impedance points {len(t_df)} < {num_freq_points}")

        volt = torch.tensor([t_df["mean_voltage"].iloc[0]], dtype=torch.float32)
        impedance_np = t_df.loc[: num_freq_points - 1, ["Zreal", "Zimag"]].to_numpy()
        volt_list.append(volt)
        impe_list.append(torch.tensor(impedance_np, dtype=torch.float32))

    volt_tensor = torch.stack(volt_list).unsqueeze(0).to(device)
    true_volt = volt_tensor[0, -1, 0]
    impe_tensor = torch.stack(impe_list).unsqueeze(0).to(device)

    env_param = torch.tensor(
        [[df["temperature"].mean(), df["flow"].mean(), df["current"].mean()]],
        dtype=torch.float32,
        device=device,
    )

    with open(stats_file, "r", encoding="utf-8") as f:
        stats = json.load(f)

    volt_min = stats["volt_min"]
    volt_max = stats["volt_max"]
    impe_mag_mean = torch.tensor(stats["impe_mag_mean"], dtype=torch.float32, device=device)
    impe_mag_std = torch.tensor(stats["impe_mag_std"], dtype=torch.float32, device=device)
    impe_phase_mean = torch.tensor(stats["impe_phase_mean"], dtype=torch.float32, device=device)
    impe_phase_std = torch.tensor(stats["impe_phase_std"], dtype=torch.float32, device=device)

    volt_tensor = (volt_tensor - volt_min) / max(volt_max - volt_min, 1e-8)
    real = impe_tensor[..., 0]
    imag = impe_tensor[..., 1]
    z_complex = real + 1j * imag
    z_mag = torch.log1p(torch.abs(z_complex))
    z_phase = (torch.angle(z_complex) + np.pi) / (2 * np.pi)
    z_mag = (z_mag - impe_mag_mean) / (impe_mag_std + 1e-8)
    z_phase = (z_phase - impe_phase_mean) / (impe_phase_std + 1e-8)
    impe_tensor = torch.stack((z_mag, z_phase), dim=-1)

    file_name = os.path.basename(xlsx_path)
    electrolyzer_parameters = _get_electrolyzer_parameters(file_name, device)
    return {
        "df": df,
        "file_name": file_name,
        "parsed_times": parsed_times,
        "volt_tensor": volt_tensor,
        "impe_tensor": impe_tensor,
        "env_param": env_param,
        "electrolyzer_parameters": electrolyzer_parameters,
        "concentration": concentration,
        "true_volt": true_volt,
    }


def _save_gradient_csv(voltage_grad, impedance_grad, env_grad, para_grad, out_path: Path):
    def ensure_2d(x):
        if isinstance(x, np.ndarray) and x.ndim == 1:
            return x[:, None]
        return x

    voltage_grad = ensure_2d(voltage_grad)
    env_grad = ensure_2d(env_grad)
    para_grad = ensure_2d(para_grad)
    rows = []

    for t in range(voltage_grad.shape[0]):
        rows.append([t, "volt", "", "", float(voltage_grad[t, 0])])

    t_size, f_size, d_size = impedance_grad.shape
    for t in range(t_size):
        for f in range(f_size):
            for d in range(d_size):
                rows.append([t, "impe", f, d, float(impedance_grad[t, f, d])])

    for idx in range(env_grad.shape[0]):
        rows.append([0, "env", idx, "", float(env_grad[idx, 0])])

    for idx in range(para_grad.shape[0]):
        rows.append([0, "electrolyzer_param", idx, "", float(para_grad[idx, 0])])

    pd.DataFrame(rows, columns=["time_idx", "type", "freq_idx", "dim", "value"]).to_csv(
        out_path, index=False
    )
    print(f"✅ 保存到 {out_path}")


def _save_time_aggregates_csv(impedance_grad, voltage_grad, out_path: Path):
    volt_time_abs = np.abs(voltage_grad.squeeze(-1))
    impe_time_sum = np.abs(impedance_grad).sum(axis=(1, 2))
    pd.DataFrame(
        {
            "t": np.arange(volt_time_abs.shape[0]),
            "volt_abs": volt_time_abs,
            "impe_abs_sum": impe_time_sum,
        }
    ).to_csv(out_path, index=False)
    print(f"✅ 时间聚合归因保存: {out_path}")


def test_single_xlsx_plus(
    xlsx_path,
    model,
    device,
    num_freq_points,
    stats_file,
    generate_explanations: bool = False,
    output_dir: Optional[str] = None,
):
    model.eval()
    sample = _load_single_plus_sample(
        xlsx_path=xlsx_path,
        device=device,
        num_freq_points=num_freq_points,
        stats_file=stats_file,
    )

    volt_tensor = sample["volt_tensor"]
    impe_tensor = sample["impe_tensor"]
    env_param = sample["env_param"]
    electrolyzer_parameters = sample["electrolyzer_parameters"]
    concentration = sample["concentration"]
    true_volt = sample["true_volt"]
    file_name = sample["file_name"]

    if generate_explanations:
        volt_tensor = volt_tensor.clone().detach().requires_grad_(True)
        impe_tensor = impe_tensor.clone().detach().requires_grad_(True)

    outputs = model(volt_tensor, impe_tensor, env_param, electrolyzer_parameters, concentration)
    prob_output = outputs["prob"] if isinstance(outputs, dict) else outputs[0]
    predicted_class = torch.argmax(prob_output.detach(), dim=1).item()
    predicted_label = IDX_TO_LABEL.get(int(predicted_class), f"Unknown({predicted_class})")

    if generate_explanations:
        if output_dir is None:
            raise ValueError("generate_explanations=True 时必须提供 output_dir")
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base_filename = os.path.splitext(file_name)[0]

        wuxing_names = ["sigma_mem", "alpha_ca", "alpha_an", "i_0ca", "i_0an"]
        influence_names = ["psi", "theta_ca", "theta_an", "phi_ca", "phi_an"]
        wuxing = outputs.get("wuxing", [torch.zeros((1, 1), device=device) for _ in range(5)])
        influence = outputs.get("influence", [torch.zeros((1, 1), device=device) for _ in range(5)])
        wuxing_values = [float(t.flatten()[0].item()) for t in wuxing]
        influence_values = [float(t.flatten()[0].item()) for t in influence]

        rows = []
        for idx in range(5):
            rows.append([wuxing_names[idx], wuxing_values[idx], influence_names[idx], influence_values[idx]])
        rows.append(["predicted_class", predicted_label, "", ""])

        prob_values = prob_output.detach().cpu().numpy().flatten()
        for ion, idx in LABEL_MAPPING.items():
            rows.append([f"prob_{ion}", prob_values[idx], "", ""])

        pred_voltage = outputs.get("pred_voltage", torch.zeros((1, 1), device=device))
        pred_conc = outputs.get("pred_conc", torch.zeros((1,), device=device))
        rows.append(["predicted_volt", float(pred_voltage.squeeze().item()), "", ""])
        rows.append(["true_volt", float(true_volt.item()), "", ""])
        rows.append(["predicted_conc", float(pred_conc.squeeze().item()), "", ""])
        rows.append(["true_conc", float(concentration.item()), "", ""])

        if "clip_logit_scale" in outputs:
            rows.append(["clip_logit_scale", float(outputs["clip_logit_scale"].item()), "", ""])

        structured_path = out_dir / f"{base_filename}_phys_params_structured.csv"
        pd.DataFrame(
            rows,
            columns=["wuxing_name", "wuxing_value", "yingxiang_name", "yingxiang_value"],
        ).to_csv(structured_path, index=False)
        print(f"✅ Structured parameter table saved: {structured_path}")

        cls_attn_mean = outputs.get("cls_attn_mean")
        if cls_attn_mean is not None:
            attn_path = out_dir / f"{base_filename}_attn_pred{predicted_class}.csv"
            pd.DataFrame(cls_attn_mean[0].detach().cpu().numpy().reshape(1, -1)).to_csv(
                attn_path, index=False, header=False
            )
            print(f"✅ Attention heatmap (CLS) saved: {attn_path}")

        param_attn_mean = outputs.get("param_attn_mean")
        if param_attn_mean is not None:
            param_attn_path = out_dir / f"{base_filename}_param_attn_pred{predicted_class}.csv"
            pd.DataFrame(param_attn_mean[0].detach().cpu().numpy().reshape(1, -1)).to_csv(
                param_attn_path, index=False, header=False
            )
            print(f"✅ Attention heatmap (PARAM) saved: {param_attn_path}")

        cross_output = outputs.get("cross_output")
        if cross_output is not None:
            latent_rows = []
            latent_np = cross_output[0].detach().cpu().numpy()
            for token_idx in range(latent_np.shape[0]):
                for dim_idx in range(latent_np.shape[1]):
                    latent_rows.append([token_idx, dim_idx, float(latent_np[token_idx, dim_idx])])
            latent_path = out_dir / f"{base_filename}_latent_tokens.csv"
            pd.DataFrame(latent_rows, columns=["token_idx", "dim_idx", "value"]).to_csv(
                latent_path, index=False
            )
            print(f"✅ Latent tokens saved: {latent_path}")

        from captum.attr import IntegratedGradients, Saliency

        def forward_func(v, i, e, p, c):
            out = model(v, i, e, p, c)
            return out.get("logits", out["prob"])

        volt_in = volt_tensor.clone().detach().requires_grad_(True)
        impe_in = impe_tensor.clone().detach().requires_grad_(True)
        env_in = env_param.clone().detach().requires_grad_(True)
        para_in = electrolyzer_parameters.clone().detach().requires_grad_(True)
        conc_arg = concentration

        saliency = Saliency(forward_func)
        sal_attr = saliency.attribute(
            inputs=(volt_in, impe_in, env_in, para_in),
            additional_forward_args=(conc_arg,),
            target=predicted_class,
        )
        sal_v = sal_attr[0].detach().cpu().numpy()[0]
        sal_i = sal_attr[1].detach().cpu().numpy()[0]
        sal_env = sal_attr[2].detach().cpu().numpy()[0]
        sal_para = sal_attr[3].detach().cpu().numpy()[0]

        ig = IntegratedGradients(forward_func)
        ig_attr = ig.attribute(
            inputs=(volt_in, impe_in, env_in, para_in),
            baselines=(
                torch.zeros_like(volt_in),
                torch.zeros_like(impe_in),
                torch.zeros_like(env_in),
                torch.zeros_like(para_in),
            ),
            additional_forward_args=(conc_arg,),
            target=predicted_class,
            internal_batch_size=6,
        )
        ig_v = ig_attr[0].detach().cpu().numpy()[0]
        ig_i = ig_attr[1].detach().cpu().numpy()[0]
        ig_env = ig_attr[2].detach().cpu().numpy()[0]
        ig_para = ig_attr[3].detach().cpu().numpy()[0]

        sal_path = out_dir / f"{base_filename}_saliency_pred{predicted_class}.csv"
        ig_path = out_dir / f"{base_filename}_ig_pred{predicted_class}.csv"
        _save_gradient_csv(sal_v, sal_i, sal_env, sal_para, sal_path)
        _save_gradient_csv(ig_v, ig_i, ig_env, ig_para, ig_path)

        sal_time_path = out_dir / f"{base_filename}_saliency_time_aggregates.csv"
        ig_time_path = out_dir / f"{base_filename}_ig_time_aggregates.csv"
        _save_time_aggregates_csv(sal_i, sal_v, sal_time_path)
        _save_time_aggregates_csv(ig_i, ig_v, ig_time_path)

    correct, predict, truth = evaluate_prediction(
        xlsx_path=xlsx_path,
        predicted_label=predicted_label,
    )
    return correct, predict, truth
