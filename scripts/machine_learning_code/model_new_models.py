"""
model_new_models.py — 模块化离子检测模型（消融实验用）

架构总览：
  ┌────────────────────────────────────────────────────────────┐
  │ 始终存在:                                                   │
  │   Z_encoder: [env, ep, CLS] → CLS_feat（环境预编码）         │
  │   主 Transformer: [volt, impe, CLS_feat] → cls_output       │
  │   （全时间点融合表示，所有实验共用）                              │
  └────────────────────────────────────────────────────────────┘
        │
        ├─ use_physics=False → cls_output 直接分类
        │
        ├─ use_physics=True, use_cross_attn=False
        │     param_transformer([第1时间点]) → param_raw（初始状态）
        │     influence_mlp(cls_output)     → physic_raw（影响因子）
        │     → 电压重建等物理约束
        │     → 分类仍然走 cls_output（全时间点）
        │
        └─ use_physics=True, use_cross_attn=True
              param_raw (query, 初始状态标定)
              physic_raw (key/value, 全时间点影响因子)
              → cross_attn → cross_pooled
              → 分类走 cross_pooled（融合了初始标定 + 时序影响因子）

YAML 开关：
  classify_mode: "linear" | "clip" | "clip_v2"
  use_env / use_physics / use_cross_attn
  use_hierarchical / use_rule / use_band_feat / use_conc
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
    elif num_freq_points == 64:
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
    else:
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


class IonDetectModel(nn.Module):
    """模块化离子检测模型，所有组件通过 cfg 控制开关。"""

    def __init__(self, cfg: dict):
        super().__init__()
        mcfg = cfg["new_model"]
        dcfg = cfg["data"]

        d = mcfg["d_model"]
        nhead = mcfg["nhead"]
        num_layers = mcfg["num_layers"]
        dropout = mcfg.get("dropout", 0.1)
        num_classes = mcfg.get("num_classes", 7)
        num_freq_points = dcfg["num_freq_points"]

        self.d_model = d
        self.classify_mode = mcfg.get("classify_mode", "linear")
        self.use_env = mcfg.get("use_env", True)
        self.use_physics = mcfg.get("use_physics", False)
        self.use_cross_attn = mcfg.get("use_cross_attn", False)
        self.use_hierarchical = mcfg.get("use_hierarchical", False)
        self.use_rule = mcfg.get("use_rule", False)
        self.use_band_feat = mcfg.get("use_band_feat", False)
        self.use_conc = mcfg.get("use_conc", False)
        self.num_classes = num_classes

        if self.use_cross_attn and not self.use_physics:
            raise ValueError("use_cross_attn=True 需要 use_physics=True"
                             "（cross_attn 的 query 来自物理分支的初始状态标定）")

        # ==================== Core Input Encoders ====================
        self.volt_encoder = AdjustableMLP(1, [d], d, 1)
        self.impe_encoder = AdjustableMLP(2, [d], d, 1)
        self.time_encoder = AdjustableMLP(1, [d], d, 1)
        self.freq_encoder = AdjustableMLP(1, [d], d, 1)

        # ==================== Frequency position buffer ====================
        freq_hz = _get_freq_hz(num_freq_points)
        freq_log = torch.log10(freq_hz)
        freq_norm = (freq_log - freq_log.min()) / (freq_log.max() - freq_log.min())
        self.register_buffer("freq_values_tensor", freq_norm)

        # ==================== Environment Encoder (optional) ====================
        if self.use_env:
            self.env_encoder = AdjustableMLP(3, [d], d, 1)
            self.ep_encoder = AdjustableMLP(8, [d], d, 1)
            self.z_encoder = MyTransformerWithAttn(d, nhead, 1, dropout)

        # ==================== CLS Token ====================
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ==================== Shared Backbone Transformer ====================
        self.transformer = MyTransformerWithAttn(d, nhead, num_layers, dropout)
        self.cls_norm = nn.LayerNorm(d)

        # ==================== Band Features (optional) ====================
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

        # ==================== Physics Branch (optional) ====================
        if self.use_physics:
            self.param_transformer = MyTransformerWithAttn(d, nhead, 1, dropout)
            self.param_mlp = AdjustableMLP(d, [d], 5 * d, 1)
            self.param_compress = AdjustableMLP(d, [], 1, 0)
            self.influence_mlp = AdjustableMLP(d, [d], 5 * d, 1)
            self.influence_compress = AdjustableMLP(d, [], 1, 0)
            if self.use_cross_attn:
                self.cross_attn = MyCrossAttnTransformer(d, nhead, 1, dropout)

        # ==================== Classification Head ====================
        if self.classify_mode == "linear":
            self.classifier = nn.Sequential(
                nn.LayerNorm(d),
                nn.Dropout(dropout),
                nn.Linear(d, num_classes),
            )
        elif self.classify_mode == "clip":
            ion_attr_tensor = torch.tensor(
                ion_attr_list_plus, dtype=torch.float32
            )
            self.register_buffer("ion_attr_tensor", ion_attr_tensor)
            self.ion_attr_dim = 7
            self.rule_dim = ion_attr_tensor.size(1) - self.ion_attr_dim
            self.register_buffer(
                "ion_rule_proto", ion_attr_tensor[:, self.ion_attr_dim:]
            )
            self.ion_attr_embed = AdjustableMLP(
                ion_attr_tensor.size(1), [d], d, 1
            )
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

        # ==================== Concentration Head (optional) ====================
        if self.use_conc:
            conc_dim = 5 * d if self.use_cross_attn else d
            self.conc_head = nn.Linear(conc_dim, 1)

        # ==================== Hierarchical Heads (optional) ====================
        if self.use_hierarchical:
            self.head_group = nn.Linear(d, 2)
            self.head_high3 = nn.Linear(d, 3)
            self.head_low3 = nn.Linear(d, 3)

        # ==================== Rule Head (optional) ====================
        if self.use_rule:
            if not hasattr(self, "ion_rule_proto"):
                ion_attr_tensor = torch.tensor(
                    ion_attr_list_plus, dtype=torch.float32
                )
                self.register_buffer("ion_attr_tensor_rule", ion_attr_tensor)
                self.ion_attr_dim = 7
                self.rule_dim = ion_attr_tensor.size(1) - self.ion_attr_dim
                self.register_buffer(
                    "ion_rule_proto", ion_attr_tensor[:, self.ion_attr_dim:]
                )
            self.rule_head = nn.Linear(d, self.rule_dim)

    # ------------------------------------------------------------------
    def _encode_ions(self):
        """CLIP 模式下预计算离子嵌入。"""
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

    # ------------------------------------------------------------------
    def forward(self, volt_data, impe_data, env_params,
                electrolyzer_parameters, concentration):
        B, T, num_freqs, C = impe_data.shape
        device = volt_data.device
        out = {}

        # ================================================================
        # 1. Input Encoding
        # ================================================================
        volt_feat = self.volt_encoder(volt_data)  # (B, T, d)
        time_input = torch.arange(T, device=device).float().view(T, 1)
        time_enc = self.time_encoder(time_input).unsqueeze(0).expand(B, T, -1)
        volt_feat = volt_feat + time_enc

        impe_feat = self.impe_encoder(impe_data.view(B, T * num_freqs, C))
        impe_feat = impe_feat.view(B, T, num_freqs, -1)
        freq_input = self.freq_values_tensor.view(num_freqs, 1)
        freq_enc = self.freq_encoder(freq_input) \
                       .unsqueeze(0).unsqueeze(0).expand(B, T, num_freqs, -1)
        time_enc_impe = time_enc.unsqueeze(2).expand(B, T, num_freqs, -1)
        impe_feat = impe_feat + time_enc_impe + freq_enc
        impe_feat = impe_feat.view(B, T * num_freqs, -1)  # (B, T*F, d)

        # ================================================================
        # 2. CLS Token 预编码
        #    与旧模型 Z_encoder 保持一致：先用 env+ep 给 CLS 注入环境信息，
        #    避免 CLS "冷启动" 被 ~190 个 token 淹没
        # ================================================================
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, d)

        if self.use_env:
            env_feat = self.env_encoder(env_params).unsqueeze(1)   # (B, 1, d)
            ep_feat = self.ep_encoder(electrolyzer_parameters).unsqueeze(1)
            z_inputs = torch.cat([env_feat, ep_feat, cls_token], dim=1)
            z_encoded, _ = self.z_encoder(z_inputs)  # (B, 3, d)
            cls_token_feat = z_encoded[:, -1:, :]    # (B, 1, d)
        else:
            cls_token_feat = cls_token

        # ================================================================
        # 3. Shared Backbone Transformer
        # ================================================================
        tokens = torch.cat([volt_feat, impe_feat, cls_token_feat], dim=1)
        encoded, attn_all = self.transformer(tokens)
        cls_output = self.cls_norm(encoded[:, -1, :])  # (B, d)
        if attn_all:
            # 最后一层注意力: (B, nhead, tgt_len, src_len)
            attn_map = attn_all[-1]
            cls_attn = attn_map[:, :, -1, :]            # (B, nhead, seq_len)
            cls_attn_mean = cls_attn.mean(dim=1)        # (B, seq_len)
            out["cls_attn_mean"] = cls_attn_mean

            # 与旧模型保持一致：去掉前 T 个电压 token 和最后 1 个 cls token
            freq_attn = cls_attn_mean[:, T:-1].view(B, T, num_freqs)
            out["freq_attn"] = freq_attn

        # ================================================================
        # 4. Band Features (optional)
        # ================================================================
        if self.use_band_feat:
            mag = impe_data[..., 0]
            band_ids = self.freq_band_ids
            band_feats = []
            for k in range(self.num_bands):
                mask_k = (band_ids == k).view(1, 1, -1).float().to(device)
                band_mag = (mag * mask_k).sum(dim=2) / \
                           (mask_k.sum(dim=2) + 1e-8)
                band_feats.append(band_mag.mean(dim=1))
            band_feats = torch.stack(band_feats, dim=1)
            cls_output = cls_output + self.band_proj(band_feats)

        # ================================================================
        # 5. Physics Branch (optional)
        # ================================================================
        cross_pooled = None

        if self.use_physics:
            # 初始状态分支（第一个时间点）→ param_raw
            volt_first = volt_feat[:, 0:1, :]
            impe_first = impe_feat.view(B, T, num_freqs, -1)[:, 0, :, :]
            param_tokens = torch.cat(
                [cls_token_feat, volt_first, impe_first], dim=1
            )
            param_encoded, param_attn_all = self.param_transformer(param_tokens)
            first_cls = param_encoded[:, 0, :]
            param_raw = self.param_mlp(first_cls).view(B, 5, self.d_model)
            param_values = self.param_compress(param_raw).squeeze(-1)
            if param_attn_all:
                param_attn_map = param_attn_all[-1]        # (B, nhead, N, N)
                param_attn_weights = param_attn_map[:, :, 0, :]
                param_attn_mean = param_attn_weights.mean(dim=1)  # (B, N)
                out["param_attn_mean"] = param_attn_mean
                # 对应旧模型: 跳过 cls+volt 两个 token，只保留频率 token
                out["freq_attn_param"] = param_attn_mean[:, 2:]

            # 影响因子分支（cls_output, 全时间点）→ physic_raw
            physic_raw = self.influence_mlp(cls_output).view(B, 5, self.d_model)
            influence_values = self.influence_compress(physic_raw).squeeze(-1)

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
                    sigma_mem[i], alpha_ca[i], alpha_an[i],
                    i_0ca[i], i_0an[i],
                    electrolyzer_parameters[i],
                )
                pred_voltages.append(pv)
            out["pred_voltage"] = torch.stack(pred_voltages).unsqueeze(1)

            if self.use_cross_attn:
                cross_output, _ = self.cross_attn(
                    query=param_raw, key_value=physic_raw
                )
                cross_pooled = cross_output.mean(dim=1)
                out["cross_output"] = cross_output

        # ================================================================
        # 6. Classification
        # ================================================================
        feat_for_classify = cross_pooled if cross_pooled is not None else cls_output

        if self.classify_mode == "linear":
            logits = self.classifier(feat_for_classify)
        elif self.classify_mode == "clip":
            ion_embeddings = self._encode_ions()
            clip_feat = self.clip_proj(feat_for_classify)
            logits = torch.matmul(clip_feat, ion_embeddings.T)
        elif self.classify_mode == "clip_v2":
            clip_feat = self.clip_proj_v2(feat_for_classify)
            clip_feat = F.normalize(clip_feat, dim=-1)
            ion_embeddings = self._encode_ions_v2()
            logit_scale = self.logit_scale_v2.exp().clamp(max=100.0)
            logits = logit_scale * torch.matmul(clip_feat, ion_embeddings.T)
            out["clip_logit_scale"] = logit_scale.detach()

        out["logits"] = logits
        out["prob"] = torch.softmax(logits, dim=1)

        # ================================================================
        # 7. Concentration (optional)
        # ================================================================
        if self.use_conc:
            if self.use_cross_attn and "cross_output" in out:
                conc_input = out["cross_output"].flatten(start_dim=1)
            else:
                conc_input = cls_output
            out["pred_conc"] = self.conc_head(conc_input).squeeze(-1)

        # ================================================================
        # 8. Auxiliary Heads (optional)
        # ================================================================
        head_feat = feat_for_classify

        if self.use_hierarchical:
            out["group_logits"] = self.head_group(head_feat)
            out["high3_logits"] = self.head_high3(head_feat)
            out["low3_logits"] = self.head_low3(head_feat)

        if self.use_rule:
            out["rule_pred"] = self.rule_head(head_feat)

        return out
