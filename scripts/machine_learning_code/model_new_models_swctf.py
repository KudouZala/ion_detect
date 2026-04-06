"""
model_new_models_swctf.py

新增 SWC-TF 架构（不替换原有 IonDetectModel）:
- 时序+频率感知生成 5 个初始状态 token (S tokens)
- 时序+频率感知生成 5 个影响因子 token (U tokens)
- S 作为 Q, U 作为 KV 做 cross-attn 融合
- 融合后接分类（linear / clip / clip_v2）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_models import (
    AdjustableMLP,
    MyTransformerWithAttn,
    MyCrossAttnTransformer,
    ion_attr_list_plus,
    calculate_predicted_voltage_plus,
)


def _get_freq_hz(num_freq_points: int) -> torch.Tensor:
    if num_freq_points == 63:
        return torch.tensor([
            19950, 15850, 12590, 10000, 7943, 6310, 5010, 3980, 3160, 2510,
            1990, 1590, 1260, 1000,
            794.3, 631.0, 501.2, 398.1, 316.2, 251.2, 199.5, 158.5, 125.9,
            100.0,
            79.43, 63.10, 50.12, 39.81, 31.62, 25.12, 19.95, 15.85, 12.59,
            10.0,
            7.94, 6.31, 5.01, 3.98, 3.16, 2.51, 1.99, 1.59, 1.26, 1.0,
            0.7943, 0.6310, 0.5012, 0.3981, 0.3162, 0.2512, 0.1995, 0.1585,
            0.1259, 0.1,
            0.07943, 0.06310, 0.05012, 0.03981, 0.03162, 0.02512, 0.01995,
            0.01585, 0.01259,
        ], dtype=torch.float32)
    if num_freq_points == 64:
        return torch.tensor([
            19950, 15850, 12590, 10000, 7943, 6310, 5010, 3980, 3160, 2510,
            1990, 1590, 1260, 1000,
            794.3, 631.0, 501.2, 398.1, 316.2, 251.2, 199.5, 158.5, 125.9,
            100.0,
            79.43, 63.10, 50.12, 39.81, 31.62, 25.12, 19.95, 15.85, 12.59,
            10.0,
            7.94, 6.31, 5.01, 3.98, 3.16, 2.51, 1.99, 1.59, 1.26, 1.0,
            0.7943, 0.6310, 0.5012, 0.3981, 0.3162, 0.2512, 0.1995, 0.1585,
            0.1259, 0.1,
            0.07943, 0.06310, 0.05012, 0.03981, 0.03162, 0.02512, 0.01995,
            0.01585, 0.01259, 0.01,
        ], dtype=torch.float32)
    raise ValueError(f"Unsupported num_freq_points: {num_freq_points}")


def _zscore_columns(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
    return (x - mean) / std


def _build_clip_v2_attribute_tensor(
    use_chem_only: bool = True,
    preprocess_attrs: bool = True,
) -> torch.Tensor:
    attr = torch.tensor(ion_attr_list_plus, dtype=torch.float32)
    chem_dim = 7

    if use_chem_only:
        clip_attr = attr[:, :chem_dim].clone()
        if preprocess_attrs:
            clip_attr = _zscore_columns(clip_attr)
        return clip_attr

    clip_attr = attr.clone()
    if preprocess_attrs and clip_attr.size(1) > chem_dim:
        ratio_part = clip_attr[:, chem_dim:-2]
        freq_part = clip_attr[:, -2:]
        ratio_part = torch.sign(ratio_part) * torch.log1p(torch.abs(ratio_part))
        freq_part = torch.log10(freq_part.clamp_min(1e-6))
        clip_attr = torch.cat([clip_attr[:, :chem_dim], ratio_part, freq_part], dim=1)
        clip_attr = _zscore_columns(clip_attr)
    return clip_attr


class IonDetectModelSWCTF(nn.Module):
    """
    SWC-TF 新架构：
    - 使用时频记忆库生成 S/U 两组 5-token
    - S 组偏向首时刻（anchor boost）
    - Cross-Attn 融合后用于分类
    """

    def __init__(self, cfg: dict):
        super().__init__()
        mcfg = cfg["new_model"]
        dcfg = cfg["data"]

        d = int(mcfg["d_model"])
        nhead = int(mcfg["nhead"])
        num_layers = int(mcfg["num_layers"])
        dropout = float(mcfg.get("dropout", 0.1))
        num_classes = int(mcfg.get("num_classes", 7))
        num_freq_points = int(dcfg["num_freq_points"])

        self.d_model = d
        self.num_classes = num_classes
        self.classify_mode = mcfg.get("classify_mode", "clip_v2")
        self.use_env = bool(mcfg.get("use_env", True))
        self.use_physics = bool(mcfg.get("use_physics", True))
        self.use_cross_attn = bool(mcfg.get("use_cross_attn", True))
        self.use_hierarchical = bool(mcfg.get("use_hierarchical", False))
        self.use_rule = bool(mcfg.get("use_rule", False))
        self.use_band_feat = bool(mcfg.get("use_band_feat", False))
        self.use_conc = bool(mcfg.get("use_conc", False))

        # SWC-TF 新参数
        self.swc_use_window_calibration = bool(
            mcfg.get("swc_use_window_calibration", True)
        )
        self.swc_anchor_boost = float(mcfg.get("swc_anchor_boost", 0.5))
        self.swc_cross_gate_init = float(mcfg.get("swc_cross_gate_init", 0.2))

        # Core encoders
        self.volt_encoder = AdjustableMLP(1, [d], d, 1)
        self.impe_encoder = AdjustableMLP(2, [d], d, 1)
        self.time_encoder = AdjustableMLP(1, [d], d, 1)
        self.freq_encoder = AdjustableMLP(1, [d], d, 1)

        if self.swc_use_window_calibration:
            self.volt_delta_encoder = AdjustableMLP(1, [d], d, 1)
            self.impe_delta_encoder = AdjustableMLP(2, [d], d, 1)

        freq_hz = _get_freq_hz(num_freq_points)
        freq_log = torch.log10(freq_hz)
        freq_norm = (freq_log - freq_log.min()) / (freq_log.max() - freq_log.min())
        self.register_buffer("freq_values_tensor", freq_norm)

        if self.use_env:
            self.env_encoder = AdjustableMLP(3, [d], d, 1)
            self.ep_encoder = AdjustableMLP(8, [d], d, 1)
            self.z_encoder = MyTransformerWithAttn(d, nhead, 1, dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.transformer = MyTransformerWithAttn(d, nhead, num_layers, dropout)
        self.cls_norm = nn.LayerNorm(d)

        if self.use_band_feat:
            band_edges = [20000.0, 1000.0, 100.0, 10.0, 1.0, 0.1, 0.0]
            self.num_bands = len(band_edges) - 1
            freq_band_ids = torch.zeros(len(freq_hz), dtype=torch.long)
            for k in range(self.num_bands):
                high, low = band_edges[k], band_edges[k + 1]
                if low > 0:
                    mask = (freq_hz <= high) & (freq_hz > low)
                else:
                    mask = (freq_hz <= high) & (freq_hz >= low)
                freq_band_ids[mask] = k
            self.register_buffer("freq_band_ids", freq_band_ids)
            self.band_proj = nn.Linear(self.num_bands, d)

        if self.use_physics:
            self.s_query_tokens = nn.Parameter(torch.zeros(1, 5, d))
            self.u_query_tokens = nn.Parameter(torch.zeros(1, 5, d))
            nn.init.trunc_normal_(self.s_query_tokens, std=0.02)
            nn.init.trunc_normal_(self.u_query_tokens, std=0.02)

            self.s_token_generator = MyCrossAttnTransformer(d, nhead, 1, dropout)
            self.u_token_generator = MyCrossAttnTransformer(d, nhead, 1, dropout)

            self.param_compress = AdjustableMLP(d, [], 1, 0)
            self.influence_compress = AdjustableMLP(d, [], 1, 0)

            if self.use_cross_attn:
                self.cross_attn = MyCrossAttnTransformer(d, nhead, 1, dropout)
                self.cross_gate = nn.Parameter(
                    torch.tensor(self.swc_cross_gate_init, dtype=torch.float32)
                )

        # Classification heads
        if self.classify_mode == "linear":
            self.classifier = nn.Sequential(
                nn.LayerNorm(d),
                nn.Dropout(dropout),
                nn.Linear(d, num_classes),
            )
        elif self.classify_mode == "clip":
            ion_attr_tensor = torch.tensor(ion_attr_list_plus, dtype=torch.float32)
            self.register_buffer("ion_attr_tensor", ion_attr_tensor)
            self.ion_attr_dim = 7
            self.rule_dim = ion_attr_tensor.size(1) - self.ion_attr_dim
            self.register_buffer("ion_rule_proto", ion_attr_tensor[:, self.ion_attr_dim:])
            self.ion_attr_embed = AdjustableMLP(ion_attr_tensor.size(1), [d], d, 1)
            self.ion_encoder = MyTransformerWithAttn(d, nhead, 1, dropout)
            self.ion_post_mlp = AdjustableMLP(d, [d], d, 1)
            self.ion_norm = nn.LayerNorm(d)
            self.clip_proj = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, d),
            )
        elif self.classify_mode == "clip_v2":
            self.clip_use_chem_only = bool(mcfg.get("clip_use_chem_only", True))
            self.clip_preprocess_attrs = bool(mcfg.get("clip_preprocess_attrs", True))
            self.clip_learnable_no_ion = bool(mcfg.get("clip_learnable_no_ion", True))
            self.clip_attr_dropout = float(mcfg.get("clip_attr_dropout", 0.0))
            clip_attr_tensor = _build_clip_v2_attribute_tensor(
                use_chem_only=self.clip_use_chem_only,
                preprocess_attrs=self.clip_preprocess_attrs,
            )
            self.register_buffer("clip_attr_tensor_v2", clip_attr_tensor)
            attr_dim = int(clip_attr_tensor.size(1))
            self.clip_attr_embed_v2 = AdjustableMLP(attr_dim, [d], d, 1)
            self.clip_attr_dropout_layer = nn.Dropout(self.clip_attr_dropout)
            self.clip_ion_encoder_v2 = MyTransformerWithAttn(d, nhead, 1, dropout)
            self.clip_ion_post_mlp_v2 = AdjustableMLP(d, [d], d, 1)
            self.clip_ion_norm_v2 = nn.LayerNorm(d)
            self.clip_proj_v2 = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, d),
            )
            init_temp = float(mcfg.get("clip_logit_scale_init", 1.0 / 0.07))
            self.logit_scale_v2 = nn.Parameter(
                torch.log(torch.tensor(init_temp, dtype=torch.float32))
            )
            if self.clip_learnable_no_ion:
                self.no_ion_prototype = nn.Parameter(torch.zeros(d))
                nn.init.trunc_normal_(self.no_ion_prototype, std=0.02)
        else:
            raise ValueError(f"Unknown classify_mode: {self.classify_mode}")

        if self.use_conc:
            conc_dim = 5 * d if (self.use_physics and self.use_cross_attn) else d
            self.conc_head = nn.Linear(conc_dim, 1)

        if self.use_hierarchical:
            self.head_group = nn.Linear(d, 2)
            self.head_high3 = nn.Linear(d, 3)
            self.head_low3 = nn.Linear(d, 3)

        if self.use_rule:
            if not hasattr(self, "ion_rule_proto"):
                ion_attr_tensor = torch.tensor(ion_attr_list_plus, dtype=torch.float32)
                self.register_buffer("ion_attr_tensor_rule", ion_attr_tensor)
                self.ion_attr_dim = 7
                self.rule_dim = ion_attr_tensor.size(1) - self.ion_attr_dim
                self.register_buffer("ion_rule_proto", ion_attr_tensor[:, self.ion_attr_dim:])
            self.rule_head = nn.Linear(d, self.rule_dim)

    def _encode_ions(self):
        ion_proj = self.ion_attr_embed(self.ion_attr_tensor)
        ion_encoded, _ = self.ion_encoder(ion_proj.unsqueeze(0))
        ion_raw = self.ion_post_mlp(ion_encoded.squeeze(0))
        return self.ion_norm(ion_encoded.squeeze(0) + ion_raw)

    def _encode_ions_v2(self):
        ion_proj = self.clip_attr_embed_v2(self.clip_attr_tensor_v2)
        ion_proj = self.clip_attr_dropout_layer(ion_proj)
        ion_encoded, _ = self.clip_ion_encoder_v2(ion_proj.unsqueeze(0))
        ion_raw = self.clip_ion_post_mlp_v2(ion_encoded.squeeze(0))
        ion_embeddings = self.clip_ion_norm_v2(ion_encoded.squeeze(0) + ion_raw)
        if self.clip_learnable_no_ion:
            ion_embeddings = ion_embeddings.clone()
            ion_embeddings[-1] = self.no_ion_prototype
        return F.normalize(ion_embeddings, dim=-1)

    def _apply_anchor_boost(self, memory: torch.Tensor, T: int, Freq: int) -> torch.Tensor:
        if self.swc_anchor_boost <= 0:
            return memory
        B, N, D = memory.shape
        weights = torch.ones(N, device=memory.device, dtype=memory.dtype)
        # token 布局: [volt(T), impe(T*F), cls(1)]
        weights[0] = 1.0 + self.swc_anchor_boost
        impe_start = T
        weights[impe_start:impe_start + Freq] = 1.0 + self.swc_anchor_boost
        return memory * weights.view(1, N, 1)

    def forward(self, volt_data, impe_data, env_params, electrolyzer_parameters, concentration):
        B, T, Freq, C = impe_data.shape
        device = volt_data.device
        out = {}

        # 1) Encode voltage / impedance
        volt_feat = self.volt_encoder(volt_data)
        if self.swc_use_window_calibration:
            volt_delta = volt_data - volt_data[:, 0:1, :]
            volt_feat = volt_feat + self.volt_delta_encoder(volt_delta)

        time_input = torch.arange(T, device=device).float().view(T, 1)
        time_enc = self.time_encoder(time_input).unsqueeze(0).expand(B, T, -1)
        volt_feat = volt_feat + time_enc

        impe_flat = impe_data.view(B, T * Freq, C)
        impe_feat = self.impe_encoder(impe_flat).view(B, T, Freq, -1)
        if self.swc_use_window_calibration:
            impe_delta = impe_data - impe_data[:, 0:1, :, :]
            impe_delta_feat = self.impe_delta_encoder(
                impe_delta.view(B, T * Freq, C)
            ).view(B, T, Freq, -1)
            impe_feat = impe_feat + impe_delta_feat

        freq_input = self.freq_values_tensor.view(Freq, 1)
        freq_enc = self.freq_encoder(freq_input).unsqueeze(0).unsqueeze(0).expand(B, T, Freq, -1)
        time_enc_impe = time_enc.unsqueeze(2).expand(B, T, Freq, -1)
        impe_feat = impe_feat + time_enc_impe + freq_enc
        impe_feat = impe_feat.view(B, T * Freq, -1)

        # 2) CLS pre-encoding with env/ep
        cls_token = self.cls_token.expand(B, -1, -1)
        if self.use_env:
            env_feat = self.env_encoder(env_params).unsqueeze(1)
            ep_feat = self.ep_encoder(electrolyzer_parameters).unsqueeze(1)
            z_inputs = torch.cat([env_feat, ep_feat, cls_token], dim=1)
            z_encoded, _ = self.z_encoder(z_inputs)
            cls_token_feat = z_encoded[:, -1:, :]
        else:
            cls_token_feat = cls_token

        tokens = torch.cat([volt_feat, impe_feat, cls_token_feat], dim=1)
        encoded, attn_all = self.transformer(tokens)
        cls_output = self.cls_norm(encoded[:, -1, :])

        if attn_all:
            attn_map = attn_all[-1]
            cls_attn = attn_map[:, :, -1, :]
            cls_attn_mean = cls_attn.mean(dim=1)
            out["cls_attn_mean"] = cls_attn_mean
            out["freq_attn"] = cls_attn_mean[:, T:-1].view(B, T, Freq)

        if self.use_band_feat:
            mag = impe_data[..., 0]
            band_ids = self.freq_band_ids
            band_feats = []
            for k in range(self.num_bands):
                mask_k = (band_ids == k).view(1, 1, -1).float().to(device)
                band_mag = (mag * mask_k).sum(dim=2) / (mask_k.sum(dim=2) + 1e-8)
                band_feats.append(band_mag.mean(dim=1))
            band_feats = torch.stack(band_feats, dim=1)
            cls_output = cls_output + self.band_proj(band_feats)

        cross_pooled = None
        s_tokens = None
        u_tokens = None
        if self.use_physics:
            memory = encoded
            memory_s = self._apply_anchor_boost(memory, T, Freq)
            q_s = self.s_query_tokens.expand(B, -1, -1)
            q_u = self.u_query_tokens.expand(B, -1, -1)
            s_tokens, _ = self.s_token_generator(query=q_s, key_value=memory_s)
            u_tokens, _ = self.u_token_generator(query=q_u, key_value=memory)

            param_values = self.param_compress(s_tokens).squeeze(-1)
            influence_values = self.influence_compress(u_tokens).squeeze(-1)

            sig = torch.sigmoid
            sigma_mem = 0.01 + 1.99 * sig(param_values[:, 0:1])
            alpha_ca = 0.2 + 0.7 * sig(param_values[:, 1:2])
            alpha_an = 0.2 + 0.7 * sig(param_values[:, 2:3])
            log_i0ca = -9.0 + 9.0 * sig(param_values[:, 3:4])
            log_i0an = -9.0 + 9.0 * sig(param_values[:, 4:5])
            i_0ca = torch.pow(10.0, log_i0ca)
            i_0an = torch.pow(10.0, log_i0an)

            theta_ca = sig(influence_values[:, 0:1])
            phi_ca = sig(influence_values[:, 1:2])
            theta_an = sig(influence_values[:, 2:3])
            phi_an = sig(influence_values[:, 3:4])
            psi = sig(influence_values[:, 4:5])

            out["wuxing"] = [sigma_mem, alpha_ca, alpha_an, i_0ca, i_0an]
            out["influence"] = [psi, theta_ca, theta_an, phi_ca, phi_an]

            temperature = env_params[:, 0].unsqueeze(1)
            flow = env_params[:, 1].unsqueeze(1)
            current = env_params[:, 2].unsqueeze(1)
            pred_voltages = []
            for i in range(B):
                pv, *_ = calculate_predicted_voltage_plus(
                    theta_ca[i], phi_ca[i], theta_an[i], phi_an[i], psi[i],
                    temperature[i], flow[i], current[i],
                    sigma_mem[i], alpha_ca[i], alpha_an[i], i_0ca[i], i_0an[i],
                    electrolyzer_parameters[i],
                )
                pred_voltages.append(pv)
            out["pred_voltage"] = torch.stack(pred_voltages).unsqueeze(1)

            if self.use_cross_attn:
                cross_raw, _ = self.cross_attn(query=s_tokens, key_value=u_tokens)
                gate = torch.sigmoid(self.cross_gate)
                cross_output = s_tokens + gate * cross_raw
                out["cross_output"] = cross_output
                cross_pooled = cross_output.mean(dim=1)
            else:
                cross_pooled = 0.5 * (s_tokens.mean(dim=1) + u_tokens.mean(dim=1))

        feat_for_classify = cross_pooled if cross_pooled is not None else cls_output

        if self.classify_mode == "linear":
            logits = self.classifier(feat_for_classify)
        elif self.classify_mode == "clip":
            ion_embeddings = self._encode_ions()
            clip_feat = self.clip_proj(feat_for_classify)
            logits = torch.matmul(clip_feat, ion_embeddings.T)
        else:
            clip_feat = self.clip_proj_v2(feat_for_classify)
            clip_feat = F.normalize(clip_feat, dim=-1)
            ion_embeddings = self._encode_ions_v2()
            logit_scale = self.logit_scale_v2.exp().clamp(max=100.0)
            logits = logit_scale * torch.matmul(clip_feat, ion_embeddings.T)
            out["clip_logit_scale"] = logit_scale.detach()

        out["logits"] = logits
        out["prob"] = torch.softmax(logits, dim=1)

        if self.use_conc:
            if self.use_physics and self.use_cross_attn and "cross_output" in out:
                conc_input = out["cross_output"].flatten(start_dim=1)
            else:
                conc_input = feat_for_classify
            out["pred_conc"] = self.conc_head(conc_input).squeeze(-1)

        if self.use_hierarchical:
            out["group_logits"] = self.head_group(feat_for_classify)
            out["high3_logits"] = self.head_high3(feat_for_classify)
            out["low3_logits"] = self.head_low3(feat_for_classify)

        if self.use_rule:
            out["rule_pred"] = self.rule_head(feat_for_classify)

        return out
