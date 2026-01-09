import os
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import json5


# 获取当前脚本所在目录
current_dir = Path(__file__).resolve().parent
# 拼接 label_mapping.json 的完整路径
json_path = current_dir / "ion_attributes_dict.jsonc"
# 读取 JSON 文件
with open(json_path, "r", encoding="utf-8") as f:
    ion_attr_dict = json5.load(f)
# 例如按以下顺序固定排列
ion_order_plus = ["Ca2+", "Na+",  "Ni2+", "Cr3+", "Cu2+","Fe3+","no_ion"]
ion_attr_list_plus = [ion_attr_dict[ion] for ion in ion_order_plus]



json_path_1117 = current_dir / "ion_attributes_dict_1117.jsonc"
# 读取 JSON 文件
with open(json_path_1117, "r", encoding="utf-8") as f:
    ion_attr_dict_1117 = json5.load(f)
ion_attr_list_plus_1117 = [ion_attr_dict_1117[ion] for ion in ion_order_plus]




# 例如按以下顺序固定排列
ion_order = ["Ca2+", "Na+",  "Ni2+", "Cr3+", "Cu2+","Fe3+"]
ion_attr_list = [ion_attr_dict[ion] for ion in ion_order]
ion_attr_list_1117 = [ion_attr_dict_1117[ion] for ion in ion_order]


def load_ion_rule_prototypes(cfg_path: str):
    """
    从 ion_rule_prototypes.yaml 读取各离子的 rule prototype 向量 r_k。
    返回:
        rule_proto: (6, D_r) 的 tensor
        feat_mean:  (1, D_r) 均值（用于归一化）
        feat_std:   (1, D_r) 标准差（用于归一化）
        ion_order:  list[str]，离子顺序
    """
    cfg_path = str(cfg_path)
    if not os.path.exists(cfg_path):
        print(f"[WARN] ion_rule_prototypes.yaml not found at {cfg_path}, "
              f"rule-based prior will be disabled.")
        # 占位：6 个类，每类 1 维 0
        rule_proto = torch.zeros(6, 1, dtype=torch.float32)
        feat_mean = rule_proto.mean(dim=0, keepdim=True)
        feat_std = rule_proto.std(dim=0, keepdim=True) + 1e-6
        ion_order = ["Ca2+", "Na+", "Ni2+", "Cr3+", "Cu2+", "Fe3+"]
        return rule_proto, feat_mean, feat_std, ion_order

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ion_order = cfg.get("ion_order", ["Ca2+", "Na+", "Ni2+", "Cr3+", "Cu2+", "Fe3+"])
    features_dict = cfg["features"]

    rule_mat = []
    for name in ion_order:
        if name not in features_dict:
            raise ValueError(f"ion_rule_prototypes: ion '{name}' not found in features section")
        vec = features_dict[name]
        rule_mat.append(vec)

    rule_proto = torch.tensor(rule_mat, dtype=torch.float32)  # (6, D_r)
    feat_mean = rule_proto.mean(dim=0, keepdim=True)
    feat_std = rule_proto.std(dim=0, keepdim=True) + 1e-6

    return rule_proto, feat_mean, feat_std, ion_order

class TransformerEncoderLayerWithReturnAttn(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, src):
        attn_output, attn_weights = self.self_attn(src, src, src, need_weights=True, average_attn_weights=False)
        src = self.norm1(src + self.dropout1(attn_output))
        return src, attn_weights

class MyTransformerWithAttn(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithReturnAttn(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src):
        attn_all_layers = []
        for layer in self.layers:
            src, attn_weights = layer(src)
            attn_all_layers.append(attn_weights)
        return src, attn_all_layers
class TransformerCrossAttnLayerWithReturnAttn(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):  # query: (B, T1, D), key_value: (B, T2, D)
        attn_output, attn_weights = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=True,
            average_attn_weights=False
        )
        query = self.norm(query + self.dropout(attn_output))  # residual + norm
        return query, attn_weights
class MyCrossAttnTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerCrossAttnLayerWithReturnAttn(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, query, key_value):
        attn_all_layers = []
        for layer in self.layers:
            query, attn_weights = layer(query, key_value)
            attn_all_layers.append(attn_weights)
        return query, attn_all_layers
    


class AdjustableMLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 output_dim,
                 num_layers=None,
                 dropout=None,
                 use_bn=False):
        """
        可调节的多层感知机模块（向后兼容版）

        兼容两种写法：
        1) 新：AdjustableMLP(input_dim, hidden_dims, output_dim, num_layers, dropout=0.1)
        2) 旧：AdjustableMLP(input_dim, hidden_dims, output_dim, use_bn=True, dropout=0.1)

        参数:
            input_dim (int): 输入维度
            hidden_dims (list[int]): 隐藏层维度列表，例如 [1024, 512, 256]
            output_dim (int): 输出维度
            num_layers (int|None): 使用的隐藏层层数；None 时默认使用 len(hidden_dims)
            dropout (float|None): 每层后的 dropout 概率
            use_bn (bool): 是否在每个隐藏层后加 BatchNorm1d（默认 False）
        """
        super().__init__()

        # ---- 兼容旧签名：如果第4个位置传的是 bool（以前当 use_bn 用）----
        if num_layers is not None and not isinstance(num_layers, int):
            # 旧版把第4个参数当作 use_bn 传进来了
            use_bn = bool(num_layers)
            num_layers = None

        hidden_dims = list(hidden_dims) if hidden_dims is not None else []
        # 推断层数
        if num_layers is None:
            num_layers = len(hidden_dims)

        # 实际使用的层数：取 num_layers 与 len(hidden_dims) 的最小值，避免越界
        use_layers = max(0, min(num_layers, len(hidden_dims)))

        layers = []
        in_dim = input_dim

        for li in range(use_layers):
            out_dim = hidden_dims[li]
            layers.append(nn.Linear(in_dim, out_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout is not None and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        # 最后一层到输出
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# 定义计算预测电压的函数
def calculate_predicted_voltage(theta_ca, phi_ca, theta_an, phi_an, psi, temperature, flow, current,
                                sigma_mem, alpha_ca, alpha_an, i_0ca, i_0an,
                                
                                
                                ):

    R = 8.314
    F = 96485
    T = temperature + 273.15
    i = current * 4 # 4是电解槽膜面积

    #------------------下面计算V_ocv-----------------
    E_rev = 1.229 - 0.9 * 10 ** -3 * (T - 298)
    P_H2 = torch.tensor(1.0, dtype=torch.float32)
    P_O2 = torch.tensor(1.0, dtype=torch.float32)
    P_H2O = torch.tensor(1.0, dtype=torch.float32)
    # V_ocv=1.229-0.9*10**-3*(T-298)+R*T/2/F*torch.log(alpha_H2*sqrt(alpha_O2)/alpha_H2O)
    V_ocv = E_rev + R * T / 2 / F * torch.log(P_H2*torch.sqrt(P_O2)/P_H2O)  # 保持 tensor 运算

    epsilon = 1e-9  # 设置最小值，可调小值,

    # 修改 safe_arcsinh：
    def safe_arcsinh(x):
        return torch.log(x + torch.sqrt(x ** 2 + 1) + epsilon)
    #--------------------- 下面计算V_act
    V_act = R * T / (theta_ca * alpha_ca * F) * safe_arcsinh(current / (2 * i_0ca * (1 - phi_ca) + epsilon)) + \
            R * T / (theta_an * alpha_an * F) * safe_arcsinh(current / (2 * i_0an * (1 - phi_an)+ epsilon))

    # ---------------------下面计算V_diff
    V_diff = 0
    # ---------------------下面计算V_ohm
    t_an = 0.012 # 阳极钛板厚度m
    t_ca1 = 0.012 # 阴极钛板厚度m
    t_ca2 = 0.002 # 阴极铜板厚度m
    t_mem = 135 * 10 ** -6 # 质子交换膜厚度m
    
    sigma_an = 2.38 * 10 ** 6
    sigma_ca1 = 2.38 * 10 ** 6
    sigma_ca2 = 5.96 * 10 ** 7



    V_ohm = i  * (t_an / sigma_an + t_ca1 / sigma_ca1 + t_ca2 / sigma_ca2 + t_mem / sigma_mem*100 / (1 - psi+epsilon))


    predicted_voltage = (V_ocv + V_act + V_diff + V_ohm).squeeze()

    return predicted_voltage,V_ocv,V_act,V_diff,V_ohm  
# 定义计算预测电压的函数
def calculate_predicted_voltage_plus(theta_ca, phi_ca, theta_an, phi_an, psi, temperature, flow, current,
                                sigma_mem, alpha_ca, alpha_an, i_0ca, i_0an,
                                electrolyzer_parameters                                 
                                
                                ):

    R = 8.314
    F = 96485
    T = temperature + 273.15

    #------------------下面计算V_ocv-----------------
    E_rev = 1.229 - 0.9 * 10 ** -3 * (T - 298)
    P_H2 = torch.tensor(1.0, dtype=torch.float32)
    P_O2 = torch.tensor(1.0, dtype=torch.float32)
    P_H2O = torch.tensor(1.0, dtype=torch.float32)
    # V_ocv=1.229-0.9*10**-3*(T-298)+R*T/2/F*torch.log(alpha_H2*sqrt(alpha_O2)/alpha_H2O)
    V_ocv = E_rev + R * T / 2 / F * torch.log(P_H2*torch.sqrt(P_O2)/P_H2O)  # 保持 tensor 运算

    epsilon = 1e-9  # 设置最小值，可调小值,

    # 修改 safe_arcsinh：
    def safe_arcsinh(x):
        return torch.log(x + torch.sqrt(x ** 2 + 1) + epsilon)
    #--------------------- 下面计算V_act
    V_act = R * T / (theta_ca * alpha_ca * F) * safe_arcsinh(current / (2 * i_0ca * (1 - phi_ca) + epsilon)) + \
            R * T / (theta_an * alpha_an * F) * safe_arcsinh(current / (2 * i_0an * (1 - phi_an)+ epsilon))#注意：该公式里的current是电流密度A/cm^2

    # ---------------------下面计算V_diff
    V_diff = 0
    # ---------------------下面计算V_ohm
    # t_an = 0.012 # 阳极钛板厚度m
    # t_ca1 = 0.012 # 阴极钛板厚度m
    # t_ca2 = 0.002 # 阴极铜板厚度m
    # t_mem = 135 * 10 ** -6 # 质子交换膜厚度m
    
    # sigma_an = 2.38 * 10 ** 6#阳极钛板导电率S/m
    # sigma_ca1 = 2.38 * 10 ** 6#阴极钛板导电率S/m
    # sigma_ca2 = 5.96 * 10 ** 7#阴极铜板导电率S/m
    # electrolyzer_parameters[7]是电解槽膜有效面积 cm^2

    # 分别赋值
    t_an       = electrolyzer_parameters[0].item()
    t_ca1      = electrolyzer_parameters[1].item()
    t_ca2      = electrolyzer_parameters[2].item()
    t_mem      = electrolyzer_parameters[3].item()
    sigma_an   = electrolyzer_parameters[4].item()
    sigma_ca1  = electrolyzer_parameters[5].item()
    sigma_ca2  = electrolyzer_parameters[6].item()
    i = current * electrolyzer_parameters[7].item() # 4是电解槽膜面积

    V_ohm = i  * (t_an / sigma_an + t_ca1 / sigma_ca1 + t_ca2 / sigma_ca2 + t_mem / sigma_mem*100 / (1 - psi+epsilon))


    predicted_voltage = (V_ocv + V_act + V_diff + V_ohm).squeeze()

    return predicted_voltage,V_ocv,V_act,V_diff,V_ohm           




# 定义完整模型
class Model_three_system(nn.Module):#三系统架构
    def __init__(self, volt_input_dim, volt_mlp_hidden_dims,mlp_output_dims, volt_mlp_num_layers,
                 impe_input_dim,impe_mlp_hidden_dims,  impe_mlp_num_layers,
                 transformer_d_model, nhead, transformer_num_layers,param_transformer_num_layers,
                 physic_mlp_hidden_dims, physic_mlp_num_layers,
                 ion_attr_embed_hidden_dims,ion_attr_embed_num_layers,
                 ion_encoder_num_layers,
                 ion_post_hidden_dims,ion_post_num_layers,
                 probMLP_input_dims,probMLP_hidden_dims,probMLP_num_layers,
                 param_mlp_hidden_dims,param_mlp_num_layers,
                 freq_encoder_hidden_dims,freq_encoder_num_layers,
                 time_encoder_hidden_dims,time_encoder_num_layers,
                 cross_transformer_num_layers,
                 param_embed_hidden_dims,param_embed_num_layers,
                 physic_embed_hidden_dims,physic_embed_num_layers,
                 cls_token_mlp_hidden_dims, cls_token_mlp_num_layers,
                 num_freq_points,num_time_points):
        
        
        super(Model_three_system, self).__init__()
        self.voltMLP = AdjustableMLP(volt_input_dim, volt_mlp_hidden_dims, mlp_output_dims, volt_mlp_num_layers)
        self.impeMLP = AdjustableMLP(impe_input_dim, impe_mlp_hidden_dims, mlp_output_dims, impe_mlp_num_layers)
        self.transformer = MyTransformerWithAttn(d_model=transformer_d_model, nhead=nhead, num_layers=transformer_num_layers)
        self.physicMLP = AdjustableMLP(transformer_d_model, physic_mlp_hidden_dims, 5*transformer_d_model, physic_mlp_num_layers)
        


        self.probMLP = AdjustableMLP(input_dim=probMLP_input_dims, hidden_dims=probMLP_hidden_dims, output_dim=mlp_output_dims, num_layers=probMLP_num_layers)  # 将 physic_output 投影到嵌入空间

        
        # 定义物理模型部分
        self.param_transformer =  MyTransformerWithAttn(d_model=transformer_d_model, nhead=nhead, num_layers=param_transformer_num_layers )
        self.paramMLP = AdjustableMLP(transformer_d_model,param_mlp_hidden_dims, 5*transformer_d_model, param_mlp_num_layers)
        self.norm = nn.LayerNorm(transformer_d_model)
        self.ion_attr_embed = AdjustableMLP(input_dim=len(ion_attr_list[0]), hidden_dims=ion_attr_embed_hidden_dims, output_dim=transformer_d_model, num_layers=ion_attr_embed_num_layers)  # 可学习离子嵌入器
        self.ion_encoder = MyTransformerWithAttn(d_model=transformer_d_model, nhead=nhead, num_layers=ion_encoder_num_layers)
        self.ion_postMLP = AdjustableMLP(input_dim=transformer_d_model, hidden_dims=ion_post_hidden_dims, output_dim=transformer_d_model, num_layers=ion_post_num_layers)  # 可学习离子嵌入器

        self.mlp_output_dims = mlp_output_dims
        self.my_cross_attn_transformer = MyCrossAttnTransformer(d_model=transformer_d_model, nhead=nhead, num_layers=cross_transformer_num_layers)


        self.param_compress = AdjustableMLP(transformer_d_model, param_embed_hidden_dims, 1, param_embed_num_layers)
        self.influence_compress = AdjustableMLP(transformer_d_model, physic_embed_hidden_dims, 1, physic_embed_num_layers)

        self.transformer_d_model = transformer_d_model
        # 对 physic_embed 添加残差和 norm
        self.prob_norm = nn.LayerNorm(transformer_d_model)

        # 对 ion_embeddings 添加残差和 norm
        self.ion_norm = nn.LayerNorm(transformer_d_model)

        # 真实频率列表（64个点）
        if num_freq_points==64:
            freq_values_hz = torch.tensor([
                19950, 15850, 12590, 10000, 7943, 6310, 5010, 3980, 3160, 2510, 1990, 1590, 1260, 1000,
                794.3, 631.0, 501.2, 398.1, 316.2, 251.2, 199.5, 158.5, 125.9, 100.0,
                79.43, 63.10, 50.12, 39.81, 31.62, 25.12, 19.95, 15.85, 12.59, 10.0,
                7.94, 6.31, 5.01, 3.98, 3.16, 2.51, 1.99, 1.59, 1.26, 1.0,
                0.7943, 0.6310, 0.5012, 0.3981, 0.3162, 0.2512, 0.1995, 0.1585, 0.1259, 0.1,
                0.07943, 0.06310, 0.05012, 0.03981, 0.03162, 0.02512, 0.01995, 0.01585, 0.01259, 0.01
            ], dtype=torch.float32)
        elif num_freq_points==63:
            freq_values_hz = torch.tensor([
            19950, 15850, 12590, 10000, 7943, 6310, 5010, 3980, 3160, 2510, 1990, 1590, 1260, 1000,
            794.3, 631.0, 501.2, 398.1, 316.2, 251.2, 199.5, 158.5, 125.9, 100.0,
            79.43, 63.10, 50.12, 39.81, 31.62, 25.12, 19.95, 15.85, 12.59, 10.0,
            7.94, 6.31, 5.01, 3.98, 3.16, 2.51, 1.99, 1.59, 1.26, 1.0,
            0.7943, 0.6310, 0.5012, 0.3981, 0.3162, 0.2512, 0.1995, 0.1585, 0.1259, 0.1,
            0.07943, 0.06310, 0.05012, 0.03981, 0.03162, 0.02512, 0.01995, 0.01585, 0.01259
            ], dtype=torch.float32)
        else:
            print("num_freq_points:",num_freq_points)
        # 归一化到 对数坐标系[0, 1]
        freq_values_log = torch.log10(freq_values_hz)  # log10变换
        freq_values_norm = (freq_values_log - freq_values_log.min()) / (freq_values_log.max() - freq_values_log.min())
        self.register_buffer("freq_values_tensor", freq_values_norm)     

        # 定义 frequency encoder MLP,将频率数据变为和MLP同样的大小，这样可以对阻抗数据进行频率编码
        self.freq_encoder = AdjustableMLP(1, freq_encoder_hidden_dims, mlp_output_dims, freq_encoder_num_layers)#(64，1)->(64,32)
        # 时间编码器
        self.time_embedding = AdjustableMLP(1, time_encoder_hidden_dims, mlp_output_dims, time_encoder_num_layers)  # nn.Embedding 默认只能处理离散索引（如 0、1、2、3），它并不会显式捕捉到这些时间点在物理上是“连续且间隔为2小时”的。那为什么还会用 nn.Embedding 表示时间？这是因为在很多场景下（特别是在 Transformer 等结构中），时间点只作为一个“位置标识符”存在，例如第几个时间步，它本身的绝对含义并不那么重要。

        # 环境参数和浓度的 MLP，用于生成cls_token
        self.cls_token_mlp = AdjustableMLP(
            input_dim=4,  # [温度, 流量, 电流密度, 浓度]
            hidden_dims=cls_token_mlp_hidden_dims,
            output_dim=self.mlp_output_dims,  # 保持与volt_output、impe_output维度一致（32）
            num_layers=cls_token_mlp_num_layers,
        )



        
        self.cross_attn = torch.nn.MultiheadAttention(embed_dim=mlp_output_dims, num_heads=nhead, batch_first=False)
        

        self.register_buffer("freq_mask", build_freq_mask(freq_values_hz))
        ion_attr_tensor = torch.tensor(ion_attr_list, dtype=torch.float32)
        self.register_buffer("ion_attr_tensor", ion_attr_tensor)


    def forward(self, volt_data, impe_data, env_params,concentrations):
        B, T, F, C = impe_data.shape  # (B,4,64,2)
        B2,T2,C2 = volt_data.shape
         # ==== 添加断言检查 ====
        assert B == B2, f"Batch size mismatch: impe_data B={B}, volt_data B={B2}"
        assert T == T2, f"Time steps mismatch: impe_data T={T}, volt_data T={T2}"

        batch_size = B
        num_time_points = T
        num_freq_points = F
        device = volt_data.device

        # ========== 电压特征 ==========
        volt_output = self.voltMLP(volt_data)  # (B,4,1)->(B,4,32)

        # ======== 时间编码(一次性计算) ========
        time_base_input = torch.arange(T, device=device).float().view(T, 1)   # (T,1)
        time_encoded_base = self.time_embedding(time_base_input)              # (T,32)

        # 扩展到 batch 维度 (B,T,32)
        time_encoded = time_encoded_base.unsqueeze(0).expand(B, T, -1)

        # 电压特征加时间编码
        volt_output = volt_output + time_encoded

        # ========== 阻抗特征 ==========
        impe_output = self.impeMLP(impe_data.view(B, T * F, -1))  # (B,256,32)
        impe_output = impe_output.view(B, T, F, -1)               # (B,4,64,32)

        # ======== 频率编码(一次性计算) ========
        freq_base_input = self.freq_values_tensor.view(F, 1)        # (64,1)
        freq_encoded_base = self.freq_encoder(freq_base_input)      # (64,32)
        freq_encoded = freq_encoded_base.unsqueeze(0).unsqueeze(0).expand(B, T, F, -1)

        # ======== 时间编码扩展到阻抗特征 ========
        time_encoded_impe = time_encoded.unsqueeze(2).expand(B, T, F, -1)    # (B,4,64,32)

        # 加入时间编码+频率编码
        impe_output = impe_output + time_encoded_impe + freq_encoded

        # reshape回原始形状
        impe_output = impe_output.view(B, T * F, -1)   # (B,256,32)


        # 提取环境参数
        temperature = env_params[:, 0].unsqueeze(1)      # (B,1)
        flow = env_params[:, 1].unsqueeze(1)             # (B,1)
        current = env_params[:, 2].unsqueeze(1)          # (B,1)

        # 提取浓度（假设作为forward参数传入，形状 (B,1)）
        # 训练阶段有值，推理阶段可以传入均值或0填充
        conc = concentrations.unsqueeze(1) if concentrations.dim()==1 else concentrations  # (B,1)

        # 拼接为 (B,4)
        cls_inputs = torch.cat([temperature, flow, current, conc], dim=1)

        # 通过MLP生成cls_token特征
        cls_token_feat = self.cls_token_mlp(cls_inputs)  # (B,32)
        cls_token_feat = cls_token_feat.unsqueeze(1)     # (B,1,32)

        # 拼接到volt_output和impe_output后面
        tokens = torch.cat([ volt_output, impe_output,cls_token_feat], dim=1)  # (B,1+4+256,32)
        transformer_output, attn_weights = self.transformer(tokens)  # (B, 261, 32)
 
        # 假设只用最后一层的transformer的注意力
        attn_map = attn_weights[-1]  # shape: (B, num_heads, tgt_len, src_len)
        cls_attn = attn_map[:, :, -1, :]  # (B, num_heads, 261)
        cls_attn_mean = cls_attn.mean(dim=1)    # (B, 261)  ← 平均所有头
        # 取最后一个输出 (blank input 的输出)
        last_output = transformer_output[:, -1, :] # Shape: (batch_size, feature_dim)

        
        # 提取阻抗频率部分注意力 (忽略电压4个token)
        freq_attn = cls_attn_mean[:, num_time_points:-1]  # (B, T*F)
        freq_attn = freq_attn.view(B, T, F)  # (B, T, F)


        # =============== 用于预测五个初始状态的物理参数 ===============
        # =============== 5️⃣ 使用主分支embedding预测初始物理参数 ===============

        # 1️⃣ 取电压第一个时间点的embedding (B,32)
        volt_first_feat = volt_output[:, 0, :].unsqueeze(1)  # (B,1,32)

        # 2️⃣ 取阻抗第一个时间点的embedding (B,64,32)
        impe_first_feat = impe_output[:, 0:num_freq_points, :]  # (B,64,32)

        # 3️⃣ 拼接 cls_token_feat、电压和阻抗
        param_tokens = torch.cat([ cls_token_feat,volt_first_feat, impe_first_feat], dim=1)  # (B,66,32)

        # 4️⃣ 送入参数 Transformer
        param_encoded,param_attn_weights = self.param_transformer(param_tokens)  # (B,66,32)

        # Transformer
        param_attn_map = param_attn_weights[-1]  # (B, nhead, N_tokens, N_tokens)
        param_attn_weights = param_attn_map[:, :, 0, :]  # (B, nhead, N_tokens)
        param_attn_mean = param_attn_weights.mean(dim=1)  # (B, N_tokens)
        
        # 提取阻抗频率部分注意力 (忽略电压1个token)
        freq_attn_param = param_attn_mean[:, 1:-1]  # (B, 1*F)
        freq_attn_param = freq_attn_param.view(B, F)  # (B, 1, F)
        
        first_cls_output = param_encoded[:, 0,:]  # (B, D)
        # print("first_cls_output.shape:",first_cls_output.shape)

        # 8. 通过 paramMLP 输出5个初始状态参数原始物理量（不做 clamp，不做 sigmoid）
        param_output = self.paramMLP(first_cls_output) # (B, 5*32)，允许其自由表达大幅环境变化
        # print("param_output.shape:",param_output.shape)
        param_output = param_output.view(B, 5, self.transformer_d_model)       # (B, 5, 32)
        # 每个参数独立压缩为标量
        param_values = self.param_compress(param_output) # (B, 5, 1)
        param_values = param_values.squeeze(-1)          # (B, 5)


        # ========== 2️⃣ 物理影响因子部分 ==========
        raw_physic_output = self.physicMLP(last_output)  # (B, 160)
        raw_physic_output = raw_physic_output.view(B, 5, self.transformer_d_model)

        # 每个影响因子独立压缩
        influence_values = self.influence_compress(raw_physic_output)  # (B, 5, 1)
        influence_values = influence_values.squeeze(-1)   
        # print("influence_values.shape:",influence_values.shape) 



        sigmoid = torch.nn.Sigmoid()
        
        # # ------------------- 行为影响因子（physic_output） -------------------
        theta_min, theta_max = 0.001, 0.999
        phi_min, phi_max     = 0.001, 0.999

        theta_ca = theta_min + (theta_max - theta_min) * sigmoid(influence_values[:, 0:1])
        phi_ca   = phi_min   + (phi_max   - phi_min)   * sigmoid(influence_values[:, 1:2])
        theta_an = theta_min + (theta_max - theta_min) * sigmoid(influence_values[:, 2:3])
        phi_an   = phi_min   + (phi_max   - phi_min)   * sigmoid(influence_values[:, 3:4])
        psi      = phi_min   + (phi_max   - phi_min)   * sigmoid(influence_values[:, 4:5])  # 同样限制


        # ------------------- 固有物性参数（param_output） -------------------
        sigma_min, sigma_max = 0.01, 2  # S/cm
        sigma_mem = sigma_min + (sigma_max - sigma_min) * sigmoid(param_values[:, 0:1])

        alpha_min, alpha_max = 0.01, 2# S/cm
        alpha_ca = alpha_min + (alpha_max - alpha_min) * sigmoid(param_values[:, 1:2])
        alpha_an = alpha_min + (alpha_max - alpha_min) * sigmoid(param_values[:, 2:3])

        log_i0ca_min, log_i0ca_max = -9, 0# S/cm
        log_i0an_min, log_i0an_max = -9, 0# S/cm
        log_i0ca = log_i0ca_min + (log_i0ca_max - log_i0ca_min) * sigmoid(param_values[:, 3:4])
        log_i0an = log_i0an_min + (log_i0an_max - log_i0an_min) * sigmoid(param_values[:, 4:5])
        i_0ca = torch.pow(10, log_i0ca)# S/cm
        i_0an = torch.pow(10, log_i0an)# S/cm


        
        
        # ========== Cross Attention 融合影响因子和物性参数 ==========
        # ========== 3️⃣ Cross Attention 交互 ==========
        # Query: param_output, Key: raw_physic_output
        cross_output, cross_attn_weights = self.my_cross_attn_transformer(
            query=param_output,
            key_value=raw_physic_output
        )

        # 1️⃣ 投影
        # physic_embed = self.probMLP(param_output)  # (B, 5, 32)
        physic_embed_raw = self.probMLP(cross_output)  # (B, 5, 32)
        physic_embed = self.prob_norm(param_output + physic_embed_raw)  # ✅ 残差 + norm

        # 1️⃣ 输入 ion_attr_tensor（静态属性表），形状为 (6, 7)
        ion_attr_proj = self.ion_attr_embed(self.ion_attr_tensor)  # (6, 32)

        # 2️⃣ 添加 batch 维度，输入 Transformer
        ion_attr_proj = ion_attr_proj.unsqueeze(0)  # (1, 6, 32)

        # 3️⃣ 编码，返回编码结果和注意力权重
        ion_encoded, ion_attn_weights = self.ion_encoder(ion_attr_proj)  # (1, 6, 32), list of attn

        # 4️⃣ 去掉 batch 维度，作为每个离子的嵌入向量
        ion_embeddings_raw = self.ion_postMLP(ion_encoded.squeeze(0))   # (6, 32)
        ion_embeddings = self.ion_norm(ion_encoded.squeeze(0) + ion_embeddings_raw)  # ✅ 残差 + norm

        # 5️⃣ 点积：param_output -> physic_embed (B, 5, 32)
        #         ion_embeddings -> (6, 32)
        raw_scores = torch.matmul(physic_embed, ion_embeddings.T)  # (B, 5, 6)

        # 6️⃣ 聚合
        raw_prob_output = raw_scores.mean(dim=1)  # (B, 6)

        # 4️⃣ Softmax
        prob_output = torch.softmax(raw_prob_output, dim=1)
        

    


        # ---------- 电压计算：基于防NaN、稳定后的参数 ----------
        predicted_voltages = []
        for i in range(batch_size):
            pred_v, _, _, _, _ = calculate_predicted_voltage(
                theta_ca[i], phi_ca[i], theta_an[i], phi_an[i], psi[i],
                temperature[i], flow[i], current[i],
                sigma_mem[i], alpha_ca[i], alpha_an[i], i_0ca[i], i_0an[i],
                         )
            predicted_voltages.append(pred_v)

        # 组合 batch 的电压结果
        predicted_voltage = torch.stack(predicted_voltages).unsqueeze(1)  # shape: (B, 1)

        wuxing = [sigma_mem,alpha_ca,alpha_an,i_0ca,i_0an]
        influence=[psi,theta_ca,theta_an,phi_ca,phi_an]

        return prob_output, predicted_voltage, wuxing,cls_attn_mean ,raw_prob_output,influence,param_attn_mean,freq_attn,freq_attn_param

def build_freq_mask(freq_values_hz: torch.Tensor, debug: bool = False) -> torch.Tensor:
    """
    根据预定义频带构造一个 0/1 掩码，用于后续做频带统计或可视化。

    Args:
        freq_values_hz: 频率列表，shape = (F,)
        debug: 是否打印匹配到的频率索引（默认 False）

    Returns:
        mask: shape = (F,) 的 0/1 tensor
    """
    freq_values_hz = freq_values_hz.to(torch.float32)
    mask = torch.zeros_like(freq_values_hz, dtype=torch.float32)

    # 判断是否要去掉 0.01 Hz
    last_band = [0.07943, 0.06310, 0.05012, 0.03981, 0.03162, 0.02512, 0.01995, 0.01585, 0.01259, 0.01]
    if len(freq_values_hz) == 63:
        last_band = last_band[:-1]  # 删除最后一个 0.01 Hz

    target_bands = [
        [19950, 15850, 12590, 10000, 7943, 6310, 5010, 3980, 3160, 2510, 1990, 1590, 1260, 1000],
        [794.3, 631.0, 501.2, 398.1, 316.2, 251.2, 199.5, 158.5, 125.9, 100.0],
        [79.43, 63.10, 50.12, 39.81, 31.62, 25.12, 19.95, 15.85, 12.59, 10.0],
        [7.94, 6.31, 5.01, 3.98, 3.16, 2.51, 1.99, 1.59, 1.26, 1.0],
        [0.7943, 0.6310, 0.5012, 0.3981, 0.3162, 0.2512, 0.1995, 0.1585, 0.1259, 0.1],
        last_band,
    ]

    # 用一个小的相对误差做匹配
    tol = 1e-4
    for band in target_bands:
        for f_target in band:
            diff = torch.abs(freq_values_hz - f_target)
            rel_err = diff / (torch.abs(freq_values_hz) + 1e-12)
            idx = torch.argmin(rel_err)
            if rel_err[idx] < tol:
                mask[idx] = 1.0

    if debug:
        print("🔍 freq_mask non-zero count:", int(mask.sum().item()))
        print("🔍 freq_mask indices:", torch.nonzero(mask).squeeze(-1).tolist())

    return mask





# 定义完整模型
class Model_three_system_one_encoder(nn.Module):#三系统架构
    def __init__(self, volt_input_dim, volt_mlp_hidden_dims,mlp_output_dims, volt_mlp_num_layers,
                 impe_input_dim,impe_mlp_hidden_dims,  impe_mlp_num_layers,
                 transformer_d_model, nhead, transformer_num_layers,param_transformer_num_layers,
                 physic_mlp_hidden_dims, physic_mlp_num_layers,
                 ion_attr_embed_hidden_dims,ion_attr_embed_num_layers,
                 ion_encoder_num_layers,
                 ion_post_hidden_dims,ion_post_num_layers,
                 probMLP_input_dims,probMLP_hidden_dims,probMLP_num_layers,
                 param_mlp_hidden_dims,param_mlp_num_layers,
                 freq_encoder_hidden_dims,freq_encoder_num_layers,
                 time_encoder_hidden_dims,time_encoder_num_layers,
                 cross_transformer_num_layers,
                 param_embed_hidden_dims,param_embed_num_layers,
                 physic_embed_hidden_dims,physic_embed_num_layers,
                 cls_token_mlp_hidden_dims, cls_token_mlp_num_layers,
                 num_freq_points,num_time_points):
        
        
        super(Model_three_system_one_encoder, self).__init__()
        self.voltMLP = AdjustableMLP(volt_input_dim, volt_mlp_hidden_dims, mlp_output_dims, volt_mlp_num_layers)
        self.impeMLP = AdjustableMLP(impe_input_dim, impe_mlp_hidden_dims, mlp_output_dims, impe_mlp_num_layers)
        self.transformer = MyTransformerWithAttn(d_model=transformer_d_model, nhead=nhead, num_layers=transformer_num_layers)
        self.physicMLP = AdjustableMLP(transformer_d_model, physic_mlp_hidden_dims, 5*transformer_d_model, physic_mlp_num_layers)
        


        self.probMLP = AdjustableMLP(input_dim=probMLP_input_dims, hidden_dims=probMLP_hidden_dims, output_dim=mlp_output_dims, num_layers=probMLP_num_layers)  # 将 physic_output 投影到嵌入空间

        
        # 定义物理模型部分
        self.paramMLP = AdjustableMLP(transformer_d_model,param_mlp_hidden_dims, 5*transformer_d_model, param_mlp_num_layers)
        self.norm = nn.LayerNorm(transformer_d_model)
        self.ion_attr_embed = AdjustableMLP(input_dim=len(ion_attr_list[0]), hidden_dims=ion_attr_embed_hidden_dims, output_dim=transformer_d_model, num_layers=ion_attr_embed_num_layers)  # 可学习离子嵌入器
        self.ion_encoder = MyTransformerWithAttn(d_model=transformer_d_model, nhead=nhead, num_layers=ion_encoder_num_layers)
        self.ion_postMLP = AdjustableMLP(input_dim=transformer_d_model, hidden_dims=ion_post_hidden_dims, output_dim=transformer_d_model, num_layers=ion_post_num_layers)  # 可学习离子嵌入器

        self.mlp_output_dims = mlp_output_dims
        self.my_cross_attn_transformer = MyCrossAttnTransformer(d_model=transformer_d_model, nhead=nhead, num_layers=cross_transformer_num_layers)


        self.param_compress = AdjustableMLP(transformer_d_model, param_embed_hidden_dims, 1, param_embed_num_layers)
        self.influence_compress = AdjustableMLP(transformer_d_model, physic_embed_hidden_dims, 1, physic_embed_num_layers)

        self.transformer_d_model = transformer_d_model
        # 对 physic_embed 添加残差和 norm
        self.prob_norm = nn.LayerNorm(transformer_d_model)

        # 对 ion_embeddings 添加残差和 norm
        self.ion_norm = nn.LayerNorm(transformer_d_model)

        # 真实频率列表（64个点）
        if num_freq_points==64:
            freq_values_hz = torch.tensor([
                19950, 15850, 12590, 10000, 7943, 6310, 5010, 3980, 3160, 2510, 1990, 1590, 1260, 1000,
                794.3, 631.0, 501.2, 398.1, 316.2, 251.2, 199.5, 158.5, 125.9, 100.0,
                79.43, 63.10, 50.12, 39.81, 31.62, 25.12, 19.95, 15.85, 12.59, 10.0,
                7.94, 6.31, 5.01, 3.98, 3.16, 2.51, 1.99, 1.59, 1.26, 1.0,
                0.7943, 0.6310, 0.5012, 0.3981, 0.3162, 0.2512, 0.1995, 0.1585, 0.1259, 0.1,
                0.07943, 0.06310, 0.05012, 0.03981, 0.03162, 0.02512, 0.01995, 0.01585, 0.01259, 0.01
            ], dtype=torch.float32)
        elif num_freq_points==63:
            freq_values_hz = torch.tensor([
            19950, 15850, 12590, 10000, 7943, 6310, 5010, 3980, 3160, 2510, 1990, 1590, 1260, 1000,
            794.3, 631.0, 501.2, 398.1, 316.2, 251.2, 199.5, 158.5, 125.9, 100.0,
            79.43, 63.10, 50.12, 39.81, 31.62, 25.12, 19.95, 15.85, 12.59, 10.0,
            7.94, 6.31, 5.01, 3.98, 3.16, 2.51, 1.99, 1.59, 1.26, 1.0,
            0.7943, 0.6310, 0.5012, 0.3981, 0.3162, 0.2512, 0.1995, 0.1585, 0.1259, 0.1,
            0.07943, 0.06310, 0.05012, 0.03981, 0.03162, 0.02512, 0.01995, 0.01585, 0.01259
            ], dtype=torch.float32)
        else:
            print("num_freq_points:",num_freq_points)
        # 归一化到 对数坐标系[0, 1]
        freq_values_log = torch.log10(freq_values_hz)  # log10变换
        freq_values_norm = (freq_values_log - freq_values_log.min()) / (freq_values_log.max() - freq_values_log.min())
        self.register_buffer("freq_values_tensor", freq_values_norm)     

        # 定义 frequency encoder MLP,将频率数据变为和MLP同样的大小，这样可以对阻抗数据进行频率编码
        self.freq_encoder = AdjustableMLP(1, freq_encoder_hidden_dims, mlp_output_dims, freq_encoder_num_layers)#(64，1)->(64,32)
        # 时间编码器
        self.time_embedding = AdjustableMLP(1, time_encoder_hidden_dims, mlp_output_dims, time_encoder_num_layers)  # nn.Embedding 默认只能处理离散索引（如 0、1、2、3），它并不会显式捕捉到这些时间点在物理上是“连续且间隔为2小时”的。那为什么还会用 nn.Embedding 表示时间？这是因为在很多场景下（特别是在 Transformer 等结构中），时间点只作为一个“位置标识符”存在，例如第几个时间步，它本身的绝对含义并不那么重要。

        # 环境参数和浓度的 MLP，用于生成cls_token
        self.cls_token_mlp = AdjustableMLP(
            input_dim=4,  # [温度, 流量, 电流密度, 浓度]
            hidden_dims=cls_token_mlp_hidden_dims,
            output_dim=self.mlp_output_dims,  # 保持与volt_output、impe_output维度一致（32）
            num_layers=cls_token_mlp_num_layers,
        )



        
        self.cross_attn = torch.nn.MultiheadAttention(embed_dim=mlp_output_dims, num_heads=nhead, batch_first=False)
        

        self.register_buffer("freq_mask", build_freq_mask(freq_values_hz))
        ion_attr_tensor = torch.tensor(ion_attr_list, dtype=torch.float32)
        self.register_buffer("ion_attr_tensor", ion_attr_tensor)


    def forward(self, volt_data, impe_data, env_params,concentrations):
        B, T, F, C = impe_data.shape  # (B,4,64,2)
        B2,T2,C2 = volt_data.shape
         # ==== 添加断言检查 ====
        assert B == B2, f"Batch size mismatch: impe_data B={B}, volt_data B={B2}"
        assert T == T2, f"Time steps mismatch: impe_data T={T}, volt_data T={T2}"

        batch_size = B
        num_time_points = T
        num_freq_points = F
        device = volt_data.device

        # ========== 电压特征 ==========
        volt_output = self.voltMLP(volt_data)  # (B,4,1)->(B,4,32)

        # ======== 时间编码(一次性计算) ========
        time_base_input = torch.arange(T, device=device).float().view(T, 1)   # (T,1)
        time_encoded_base = self.time_embedding(time_base_input)              # (T,32)

        # 扩展到 batch 维度 (B,T,32)
        time_encoded = time_encoded_base.unsqueeze(0).expand(B, T, -1)

        # 电压特征加时间编码
        volt_output = volt_output + time_encoded

        # ========== 阻抗特征 ==========
        impe_output = self.impeMLP(impe_data.view(B, T * F, -1))  # (B,256,32)
        impe_output = impe_output.view(B, T, F, -1)               # (B,4,64,32)

        # ======== 频率编码(一次性计算) ========
        freq_base_input = self.freq_values_tensor.view(F, 1)        # (64,1)
        freq_encoded_base = self.freq_encoder(freq_base_input)      # (64,32)
        freq_encoded = freq_encoded_base.unsqueeze(0).unsqueeze(0).expand(B, T, F, -1)

        # ======== 时间编码扩展到阻抗特征 ========
        time_encoded_impe = time_encoded.unsqueeze(2).expand(B, T, F, -1)    # (B,4,64,32)

        # 加入时间编码+频率编码
        impe_output = impe_output + time_encoded_impe + freq_encoded

        # reshape回原始形状
        impe_output = impe_output.view(B, T * F, -1)   # (B,256,32)


        # 提取环境参数
        temperature = env_params[:, 0].unsqueeze(1)      # (B,1)
        flow = env_params[:, 1].unsqueeze(1)             # (B,1)
        current = env_params[:, 2].unsqueeze(1)          # (B,1)

        # 提取浓度（假设作为forward参数传入，形状 (B,1)）
        # 训练阶段有值，推理阶段可以传入均值或0填充
        conc = concentrations.unsqueeze(1) if concentrations.dim()==1 else concentrations  # (B,1)

        # 拼接为 (B,4)
        cls_inputs = torch.cat([temperature, flow, current, conc], dim=1)

        # 通过MLP生成cls_token特征
        cls_token_feat = self.cls_token_mlp(cls_inputs)  # (B,32)
        cls_token_feat = cls_token_feat.unsqueeze(1)     # (B,1,32)

        # 拼接到volt_output和impe_output后面
        tokens = torch.cat([cls_token_feat,volt_output, impe_output,cls_token_feat], dim=1)  # (B,1+4+256,32)
        transformer_output, attn_weights = self.transformer(tokens)  # (B, 261, 32)
 
        # 假设只用最后一层的transformer的注意力
        attn_map = attn_weights[-1]  # shape: (B, num_heads, tgt_len, src_len)
        cls_attn = attn_map[:, :, -1, :]  # (B, num_heads, 261)
        cls_attn_mean = cls_attn.mean(dim=1)    # (B, 261)  ← 平均所有头
        # 取最后一个输出 (blank input 的输出)
        last_output = transformer_output[:, -1, :] # Shape: (batch_size, feature_dim)

        
        # 提取阻抗频率部分注意力 (忽略电压4个token)
        freq_attn = cls_attn_mean[:, 1+num_time_points:-1]  # (B, T*F)
        freq_attn = freq_attn.view(B, T, F)  # (B, T, F)


        # Transformer
        param_attn_map = attn_weights[-1]  # (B, nhead, N_tokens, N_tokens)
        param_attn_weights = attn_map[:, :, 0, :]  # (B, nhead, N_tokens)
        param_attn_mean = param_attn_weights.mean(dim=1)  # (B, N_tokens)
        
        # 提取阻抗频率部分注意力 (忽略电压1个token)
        freq_attn_param = param_attn_mean[:, 1:F+1]  # (B, 1*F)
        freq_attn_param = freq_attn_param.view(B, F)  # (B, 1, F)
        
        first_cls_output = transformer_output[:, 0,:]  # (B, D)
        # print("first_cls_output.shape:",first_cls_output.shape)

        # 8. 通过 paramMLP 输出5个初始状态参数原始物理量（不做 clamp，不做 sigmoid）
        param_output = self.paramMLP(first_cls_output) # (B, 5*32)，允许其自由表达大幅环境变化
        # print("param_output.shape:",param_output.shape)
        param_output = param_output.view(B, 5, self.transformer_d_model)       # (B, 5, 32)
        # 每个参数独立压缩为标量
        param_values = self.param_compress(param_output) # (B, 5, 1)
        param_values = param_values.squeeze(-1)          # (B, 5)


        # ========== 2️⃣ 物理影响因子部分 ==========
        raw_physic_output = self.physicMLP(last_output)  # (B, 160)
        raw_physic_output = raw_physic_output.view(B, 5, self.transformer_d_model)

        # 每个影响因子独立压缩
        influence_values = self.influence_compress(raw_physic_output)  # (B, 5, 1)
        influence_values = influence_values.squeeze(-1)   
        # print("influence_values.shape:",influence_values.shape) 



        sigmoid = torch.nn.Sigmoid()
        
        # # ------------------- 行为影响因子（physic_output） -------------------
        theta_min, theta_max = 0.001, 0.999
        phi_min, phi_max     = 0.001, 0.999

        theta_ca = theta_min + (theta_max - theta_min) * sigmoid(influence_values[:, 0:1])
        phi_ca   = phi_min   + (phi_max   - phi_min)   * sigmoid(influence_values[:, 1:2])
        theta_an = theta_min + (theta_max - theta_min) * sigmoid(influence_values[:, 2:3])
        phi_an   = phi_min   + (phi_max   - phi_min)   * sigmoid(influence_values[:, 3:4])
        psi      = phi_min   + (phi_max   - phi_min)   * sigmoid(influence_values[:, 4:5])  # 同样限制


        # ------------------- 固有物性参数（param_output） -------------------
        sigma_min, sigma_max = 0.01, 2  # S/cm
        sigma_mem = sigma_min + (sigma_max - sigma_min) * sigmoid(param_values[:, 0:1])

        alpha_min, alpha_max = 0.01, 2
        alpha_ca = alpha_min + (alpha_max - alpha_min) * sigmoid(param_values[:, 1:2])
        alpha_an = alpha_min + (alpha_max - alpha_min) * sigmoid(param_values[:, 2:3])

        log_i0ca_min, log_i0ca_max = -9, 0
        log_i0an_min, log_i0an_max = -9, 0
        log_i0ca = log_i0ca_min + (log_i0ca_max - log_i0ca_min) * sigmoid(param_values[:, 3:4])
        log_i0an = log_i0an_min + (log_i0an_max - log_i0an_min) * sigmoid(param_values[:, 4:5])
        i_0ca = torch.pow(10, log_i0ca)
        i_0an = torch.pow(10, log_i0an)


        
        
        # ========== Cross Attention 融合影响因子和物性参数 ==========
        # ========== 3️⃣ Cross Attention 交互 ==========
        # Query: param_output, Key: raw_physic_output
        cross_output, cross_attn_weights = self.my_cross_attn_transformer(
            query=param_output,
            key_value=raw_physic_output
        )

        # 1️⃣ 投影
        # physic_embed = self.probMLP(param_output)  # (B, 5, 32)
        physic_embed_raw = self.probMLP(cross_output)  # (B, 5, 32)
        physic_embed = self.prob_norm(param_output + physic_embed_raw)  # ✅ 残差 + norm

        # 1️⃣ 输入 ion_attr_tensor（静态属性表），形状为 (6, 7)
        ion_attr_proj = self.ion_attr_embed(self.ion_attr_tensor)  # (6, 32)

        # 2️⃣ 添加 batch 维度，输入 Transformer
        ion_attr_proj = ion_attr_proj.unsqueeze(0)  # (1, 6, 32)

        # 3️⃣ 编码，返回编码结果和注意力权重
        ion_encoded, ion_attn_weights = self.ion_encoder(ion_attr_proj)  # (1, 6, 32), list of attn

        # 4️⃣ 去掉 batch 维度，作为每个离子的嵌入向量
        ion_embeddings_raw = self.ion_postMLP(ion_encoded.squeeze(0))   # (6, 32)
        ion_embeddings = self.ion_norm(ion_encoded.squeeze(0) + ion_embeddings_raw)  # ✅ 残差 + norm

        # 5️⃣ 点积：param_output -> physic_embed (B, 5, 32)
        #         ion_embeddings -> (6, 32)
        raw_scores = torch.matmul(physic_embed, ion_embeddings.T)  # (B, 5, 6)

        # 6️⃣ 聚合
        raw_prob_output = raw_scores.mean(dim=1)  # (B, 6)

        # 4️⃣ Softmax
        prob_output = torch.softmax(raw_prob_output, dim=1)
        

    


        # ---------- 电压计算：基于防NaN、稳定后的参数 ----------
        predicted_voltages = []
        for i in range(batch_size):
            pred_v, _, _, _, _ = calculate_predicted_voltage(
                theta_ca[i], phi_ca[i], theta_an[i], phi_an[i], psi[i],
                temperature[i], flow[i], current[i],
                sigma_mem[i], alpha_ca[i], alpha_an[i], i_0ca[i], i_0an[i]
            )
            predicted_voltages.append(pred_v)

        # 组合 batch 的电压结果
        predicted_voltage = torch.stack(predicted_voltages).unsqueeze(1)  # shape: (B, 1)

        wuxing = [sigma_mem,alpha_ca,alpha_an,i_0ca,i_0an]
        influence=[psi,theta_ca,theta_an,phi_ca,phi_an]

        return prob_output, predicted_voltage, wuxing,cls_attn_mean ,raw_prob_output,influence,param_attn_mean,freq_attn,freq_attn_param


# 定义完整模型
class Model_three_system_plus(nn.Module):#三系统架构
    def __init__(self, volt_input_dim, volt_mlp_hidden_dims,mlp_output_dims, volt_mlp_num_layers,
                 impe_input_dim,impe_mlp_hidden_dims,  impe_mlp_num_layers,
                 transformer_d_model, nhead, transformer_num_layers,param_transformer_num_layers,
                 physic_mlp_hidden_dims, physic_mlp_num_layers,
                 ion_attr_embed_hidden_dims,ion_attr_embed_num_layers,
                 ion_encoder_num_layers,
                 ion_post_hidden_dims,ion_post_num_layers,
                 probMLP_input_dims,probMLP_hidden_dims,probMLP_num_layers,
                 param_mlp_hidden_dims,param_mlp_num_layers,
                 freq_encoder_hidden_dims,freq_encoder_num_layers,
                 time_encoder_hidden_dims,time_encoder_num_layers,
                 cross_transformer_num_layers,
                 param_embed_hidden_dims,param_embed_num_layers,
                 physic_embed_hidden_dims,physic_embed_num_layers,
                 envMLP_input_dim, env_mlp_hidden_dims, env_mlp_num_layers,
                 ep_input_dim, ep_mlp_hidden_dims,  ep_mlp_num_layers,
                 conc_input_dim, conc_mlp_hidden_dims, conc_mlp_num_layers,
                 Z_encoder_num_layers ,
                 num_freq_points,num_time_points):
        
        
        super(Model_three_system_plus, self).__init__()
        self.voltMLP = AdjustableMLP(volt_input_dim, volt_mlp_hidden_dims, mlp_output_dims, volt_mlp_num_layers)
        self.impeMLP = AdjustableMLP(impe_input_dim, impe_mlp_hidden_dims, mlp_output_dims, impe_mlp_num_layers)
        self.transformer = MyTransformerWithAttn(d_model=transformer_d_model, nhead=nhead, num_layers=transformer_num_layers)
        self.physicMLP = AdjustableMLP(transformer_d_model, physic_mlp_hidden_dims, 5*transformer_d_model, physic_mlp_num_layers)
        


        self.probMLP = AdjustableMLP(input_dim=probMLP_input_dims, hidden_dims=probMLP_hidden_dims, output_dim=mlp_output_dims, num_layers=probMLP_num_layers)  # 将 physic_output 投影到嵌入空间

        
        # 定义物理模型部分
        self.param_transformer =  MyTransformerWithAttn(d_model=transformer_d_model, nhead=nhead, num_layers=param_transformer_num_layers )
        self.paramMLP = AdjustableMLP(transformer_d_model,param_mlp_hidden_dims, 5*transformer_d_model, param_mlp_num_layers)
        self.norm = nn.LayerNorm(transformer_d_model)
        self.ion_attr_embed = AdjustableMLP(input_dim=len(ion_attr_list[0]), hidden_dims=ion_attr_embed_hidden_dims, output_dim=transformer_d_model, num_layers=ion_attr_embed_num_layers)  # 可学习离子嵌入器
        self.ion_encoder = MyTransformerWithAttn(d_model=transformer_d_model, nhead=nhead, num_layers=ion_encoder_num_layers)
        self.ion_postMLP = AdjustableMLP(input_dim=transformer_d_model, hidden_dims=ion_post_hidden_dims, output_dim=transformer_d_model, num_layers=ion_post_num_layers)  # 可学习离子嵌入器

        self.mlp_output_dims = mlp_output_dims
        self.my_cross_attn_transformer = MyCrossAttnTransformer(d_model=transformer_d_model, nhead=nhead, num_layers=cross_transformer_num_layers)


        self.param_compress = AdjustableMLP(transformer_d_model, param_embed_hidden_dims, 1, param_embed_num_layers)
        self.influence_compress = AdjustableMLP(transformer_d_model, physic_embed_hidden_dims, 1, physic_embed_num_layers)

        self.transformer_d_model = transformer_d_model
        # 对 physic_embed 添加残差和 norm
        self.prob_norm = nn.LayerNorm(transformer_d_model)

        # 对 ion_embeddings 添加残差和 norm
        self.ion_norm = nn.LayerNorm(transformer_d_model)


        self.envMLP = AdjustableMLP(envMLP_input_dim, env_mlp_hidden_dims, mlp_output_dims, env_mlp_num_layers)
        self.epMLP = AdjustableMLP(ep_input_dim, ep_mlp_hidden_dims, mlp_output_dims, ep_mlp_num_layers)
        self.concMLP = AdjustableMLP(conc_input_dim, conc_mlp_hidden_dims, mlp_output_dims, conc_mlp_num_layers)
        self.Z_encoder =  MyTransformerWithAttn(d_model=transformer_d_model, nhead=nhead, num_layers=Z_encoder_num_layers )

        # 1️⃣ 定义一个可学习的 cls_token (初始值可用正态分布)
        self.cls_token = nn.Parameter(torch.zeros(1, transformer_d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)




        # 真实频率列表（64个点）
        if num_freq_points==64:
            freq_values_hz = torch.tensor([
                19950, 15850, 12590, 10000, 7943, 6310, 5010, 3980, 3160, 2510, 1990, 1590, 1260, 1000,
                794.3, 631.0, 501.2, 398.1, 316.2, 251.2, 199.5, 158.5, 125.9, 100.0,
                79.43, 63.10, 50.12, 39.81, 31.62, 25.12, 19.95, 15.85, 12.59, 10.0,
                7.94, 6.31, 5.01, 3.98, 3.16, 2.51, 1.99, 1.59, 1.26, 1.0,
                0.7943, 0.6310, 0.5012, 0.3981, 0.3162, 0.2512, 0.1995, 0.1585, 0.1259, 0.1,
                0.07943, 0.06310, 0.05012, 0.03981, 0.03162, 0.02512, 0.01995, 0.01585, 0.01259, 0.01
            ], dtype=torch.float32)
        elif num_freq_points==63:
            freq_values_hz = torch.tensor([
            19950, 15850, 12590, 10000, 7943, 6310, 5010, 3980, 3160, 2510, 1990, 1590, 1260, 1000,
            794.3, 631.0, 501.2, 398.1, 316.2, 251.2, 199.5, 158.5, 125.9, 100.0,
            79.43, 63.10, 50.12, 39.81, 31.62, 25.12, 19.95, 15.85, 12.59, 10.0,
            7.94, 6.31, 5.01, 3.98, 3.16, 2.51, 1.99, 1.59, 1.26, 1.0,
            0.7943, 0.6310, 0.5012, 0.3981, 0.3162, 0.2512, 0.1995, 0.1585, 0.1259, 0.1,
            0.07943, 0.06310, 0.05012, 0.03981, 0.03162, 0.02512, 0.01995, 0.01585, 0.01259
            ], dtype=torch.float32)
        else:
            print("num_freq_points:",num_freq_points)
        # 归一化到 对数坐标系[0, 1]
        freq_values_log = torch.log10(freq_values_hz)  # log10变换
        freq_values_norm = (freq_values_log - freq_values_log.min()) / (freq_values_log.max() - freq_values_log.min())
        self.register_buffer("freq_values_tensor", freq_values_norm)     

        # 定义 frequency encoder MLP,将频率数据变为和MLP同样的大小，这样可以对阻抗数据进行频率编码
        self.freq_encoder = AdjustableMLP(1, freq_encoder_hidden_dims, mlp_output_dims, freq_encoder_num_layers)#(64，1)->(64,32)
        # 时间编码器
        self.time_embedding = AdjustableMLP(1, time_encoder_hidden_dims, mlp_output_dims, time_encoder_num_layers)  # nn.Embedding 默认只能处理离散索引（如 0、1、2、3），它并不会显式捕捉到这些时间点在物理上是“连续且间隔为2小时”的。那为什么还会用 nn.Embedding 表示时间？这是因为在很多场景下（特别是在 Transformer 等结构中），时间点只作为一个“位置标识符”存在，例如第几个时间步，它本身的绝对含义并不那么重要。





        
        self.cross_attn = torch.nn.MultiheadAttention(embed_dim=mlp_output_dims, num_heads=nhead, batch_first=False)
        
        ion_attr_tensor = torch.tensor(ion_attr_list_plus, dtype=torch.float32)
        self.register_buffer("ion_attr_tensor", ion_attr_tensor)


    def forward(self, volt_data, impe_data, env_params,electrolyzer_parameters,concentration):
        B, T, F, C = impe_data.shape  # (B,4,64,2)
        B2,T2,C2 = volt_data.shape
         # ==== 添加断言检查 ====
        assert B == B2, f"Batch size mismatch: impe_data B={B}, volt_data B={B2}"
        assert T == T2, f"Time steps mismatch: impe_data T={T}, volt_data T={T2}"

        batch_size = B
        num_time_points = T
        num_freq_points = F
        device = volt_data.device

        # ========== 电压特征 ==========
        volt_output = self.voltMLP(volt_data)  # (B,4,1)->(B,4,32)

        # ======== 时间编码(一次性计算) ========
        time_base_input = torch.arange(T, device=device).float().view(T, 1)   # (T,1)
        time_encoded_base = self.time_embedding(time_base_input)              # (T,32)

        # 扩展到 batch 维度 (B,T,32)
        time_encoded = time_encoded_base.unsqueeze(0).expand(B, T, -1)

        # 电压特征加时间编码
        volt_output = volt_output + time_encoded

        # ========== 阻抗特征 ==========
        impe_output = self.impeMLP(impe_data.view(B, T * F, -1))  # (B,256,32)
        impe_output = impe_output.view(B, T, F, -1)               # (B,4,64,32)

        # ======== 频率编码(一次性计算) ========
        freq_base_input = self.freq_values_tensor.view(F, 1)        # (64,1)
        freq_encoded_base = self.freq_encoder(freq_base_input)      # (64,32)
        freq_encoded = freq_encoded_base.unsqueeze(0).unsqueeze(0).expand(B, T, F, -1)

        # ======== 时间编码扩展到阻抗特征 ========
        time_encoded_impe = time_encoded.unsqueeze(2).expand(B, T, F, -1)    # (B,4,64,32)

        # 加入时间编码+频率编码
        impe_output = impe_output + time_encoded_impe + freq_encoded

        # reshape回原始形状
        impe_output = impe_output.view(B, T * F, -1)   # (B,256,32)


        # 提取环境参数
        temperature = env_params[:, 0].unsqueeze(1)      # (B,1)
        flow = env_params[:, 1].unsqueeze(1)             # (B,1)
        current = env_params[:, 2].unsqueeze(1)          # (B,1)


        # (1) 编码得到两个 32 维 token
        env_coding = self.envMLP(env_params)                 # (B, 32)
        ep_coding  = self.epMLP(electrolyzer_parameters)     # (B, 32)
        conc_coding  = self.concMLP(concentration)

        # (2) 扩展可学习 cls_token 到 batch
        B = env_coding.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)         # (B, 1, 32)

        # (3) 在“序列维”拼接成 3 个 token
        cls_inputs = torch.cat([
            env_coding.unsqueeze(1),                         # (B, 1, 32)
            ep_coding.unsqueeze(1),
            conc_coding.unsqueeze(1),                      # (B, 1, 32)
            cls_token                                        # (B, 1, 32)
        ], dim=1)                                            # => (B, 3, 32)

        # (4) 通过 Transformer 编码器
        enc_out,Z_attn = self.Z_encoder(cls_inputs)                 # 可能返回 Tensor 或 (Tensor, attn) / (Tensor, list_of_attn)

        # (5) 取 CLS 位置（最后一个 token）
        cls_token_feat = enc_out[:, -1:, :]               # (B, 1,32)
        # print("cls_token_feat.shape",cls_token_feat.shape)

        # 拼接到volt_output和impe_output后面
        tokens = torch.cat([ volt_output, impe_output,cls_token_feat], dim=1)  # (B,1+4+256,32)
        transformer_output, attn_weights = self.transformer(tokens)  # (B, 261, 32)
 
        # 假设只用最后一层的transformer的注意力
        attn_map = attn_weights[-1]  # shape: (B, num_heads, tgt_len, src_len)
        cls_attn = attn_map[:, :, -1, :]  # (B, num_heads, 261)
        cls_attn_mean = cls_attn.mean(dim=1)    # (B, 261)  ← 平均所有头
        # 取最后一个输出 (blank input 的输出)
        last_output = transformer_output[:, -1, :] # Shape: (batch_size, feature_dim)

        
        # 提取阻抗频率部分注意力 (忽略电压4个token)
        freq_attn = cls_attn_mean[:, num_time_points:-1]  # (B, T*F)
        freq_attn = freq_attn.view(B, T, F)  # (B, T, F)


        # =============== 用于预测五个初始状态的物理参数 ===============
        # =============== 5️⃣ 使用主分支embedding预测初始物理参数 ===============

        # 1️⃣ 取电压第一个时间点的embedding (B,32)
        volt_first_feat = volt_output[:, 0, :].unsqueeze(1)  # (B,1,32)

        # 2️⃣ 取阻抗第一个时间点的embedding (B,64,32)
        impe_first_feat = impe_output[:, 0:num_freq_points, :]  # (B,64,32)

        # 3️⃣ 拼接 cls_token_feat、电压和阻抗
        param_tokens = torch.cat([ cls_token_feat,volt_first_feat, impe_first_feat], dim=1)  # (B,66,32)

        # 4️⃣ 送入参数 Transformer
        param_encoded,param_attn_weights = self.param_transformer(param_tokens)  # (B,66,32)

        # Transformer
        param_attn_map = param_attn_weights[-1]  # (B, nhead, N_tokens, N_tokens)
        param_attn_weights = param_attn_map[:, :, 0, :]  # (B, nhead, N_tokens)
        param_attn_mean = param_attn_weights.mean(dim=1)  # (B, N_tokens)
        
        # 提取阻抗频率部分注意力 (忽略电压1个token)
        freq_attn_param = param_attn_mean[:, 1:-1]  # (B, 1*F)
        freq_attn_param = freq_attn_param.view(B, F)  # (B, 1, F)
        
        first_cls_output = param_encoded[:, 0,:]  # (B, D)
        # print("first_cls_output.shape:",first_cls_output.shape)

        # 8. 通过 paramMLP 输出5个初始状态参数原始物理量（不做 clamp，不做 sigmoid）
        param_output = self.paramMLP(first_cls_output) # (B, 5*32)，允许其自由表达大幅环境变化
        # print("param_output.shape:",param_output.shape)
        param_output = param_output.view(B, 5, self.transformer_d_model)       # (B, 5, 32)
        # 每个参数独立压缩为标量
        param_values = self.param_compress(param_output) # (B, 5, 1)
        param_values = param_values.squeeze(-1)          # (B, 5)


        # ========== 2️⃣ 物理影响因子部分 ==========
        raw_physic_output = self.physicMLP(last_output)  # (B, 160)
        raw_physic_output = raw_physic_output.view(B, 5, self.transformer_d_model)

        # 每个影响因子独立压缩
        influence_values = self.influence_compress(raw_physic_output)  # (B, 5, 1)
        influence_values = influence_values.squeeze(-1)   
        # print("influence_values.shape:",influence_values.shape) 



        sigmoid = torch.nn.Sigmoid()
        
        # # ------------------- 行为影响因子（physic_output） -------------------
        theta_min, theta_max = 0.001, 0.999
        phi_min, phi_max     = 0.001, 0.999

        theta_ca = theta_min + (theta_max - theta_min) * sigmoid(influence_values[:, 0:1])
        phi_ca   = phi_min   + (phi_max   - phi_min)   * sigmoid(influence_values[:, 1:2])
        theta_an = theta_min + (theta_max - theta_min) * sigmoid(influence_values[:, 2:3])
        phi_an   = phi_min   + (phi_max   - phi_min)   * sigmoid(influence_values[:, 3:4])
        psi      = phi_min   + (phi_max   - phi_min)   * sigmoid(influence_values[:, 4:5])  # 同样限制


        # ------------------- 固有物性参数（param_output） -------------------
        sigma_min, sigma_max = 0.01, 2  # S/cm
        sigma_mem = sigma_min + (sigma_max - sigma_min) * sigmoid(param_values[:, 0:1])

        alpha_min, alpha_max = 0.01, 2
        alpha_ca = alpha_min + (alpha_max - alpha_min) * sigmoid(param_values[:, 1:2])
        alpha_an = alpha_min + (alpha_max - alpha_min) * sigmoid(param_values[:, 2:3])

        log_i0ca_min, log_i0ca_max = -9, 0
        log_i0an_min, log_i0an_max = -9, 0
        log_i0ca = log_i0ca_min + (log_i0ca_max - log_i0ca_min) * sigmoid(param_values[:, 3:4])
        log_i0an = log_i0an_min + (log_i0an_max - log_i0an_min) * sigmoid(param_values[:, 4:5])
        i_0ca = torch.pow(10, log_i0ca)
        i_0an = torch.pow(10, log_i0an)


        
        
        # ========== Cross Attention 融合影响因子和物性参数 ==========
        # ========== 3️⃣ Cross Attention 交互 ==========
        # Query: param_output, Key: raw_physic_output
        cross_output, cross_attn_weights = self.my_cross_attn_transformer(
            query=param_output,
            key_value=raw_physic_output
        )

        # 1️⃣ 投影
        # physic_embed = self.probMLP(param_output)  # (B, 5, 32)
        physic_embed_raw = self.probMLP(cross_output)  # (B, 5, 32)
        physic_embed = self.prob_norm(param_output + physic_embed_raw)  # ✅ 残差 + norm

        # 1️⃣ 输入 ion_attr_tensor（静态属性表），形状为 (6, 7)
        ion_attr_proj = self.ion_attr_embed(self.ion_attr_tensor)  # (6, 32)

        # 2️⃣ 添加 batch 维度，输入 Transformer
        ion_attr_proj = ion_attr_proj.unsqueeze(0)  # (1, 6, 32)

        # 3️⃣ 编码，返回编码结果和注意力权重
        ion_encoded, ion_attn_weights = self.ion_encoder(ion_attr_proj)  # (1, 6, 32), list of attn

        # 4️⃣ 去掉 batch 维度，作为每个离子的嵌入向量
        ion_embeddings_raw = self.ion_postMLP(ion_encoded.squeeze(0))   # (6, 32)
        ion_embeddings = self.ion_norm(ion_encoded.squeeze(0) + ion_embeddings_raw)  # ✅ 残差 + norm

        # 5️⃣ 点积：param_output -> physic_embed (B, 5, 32)
        #         ion_embeddings -> (6, 32)
        raw_scores = torch.matmul(physic_embed, ion_embeddings.T)  # (B, 5, 6)

        # 6️⃣ 聚合
        raw_prob_output = raw_scores.mean(dim=1)  # (B, 6)

        # 4️⃣ Softmax
        prob_output = torch.softmax(raw_prob_output, dim=1)
        

    


        # ---------- 电压计算：基于防NaN、稳定后的参数 ----------
        predicted_voltages = []
        for i in range(batch_size):
            pred_v, _, _, _, _ = calculate_predicted_voltage_plus(
                theta_ca[i], phi_ca[i], theta_an[i], phi_an[i], psi[i],
                temperature[i], flow[i], current[i],
                sigma_mem[i], alpha_ca[i], alpha_an[i], i_0ca[i], i_0an[i],
                electrolyzer_parameters= electrolyzer_parameters[i]  
            )
            predicted_voltages.append(pred_v)

        # 组合 batch 的电压结果
        predicted_voltage = torch.stack(predicted_voltages).unsqueeze(1)  # shape: (B, 1)

        wuxing = [sigma_mem,alpha_ca,alpha_an,i_0ca,i_0an]
        influence=[psi,theta_ca,theta_an,phi_ca,phi_an]

        return prob_output, predicted_voltage, wuxing,cls_attn_mean ,raw_prob_output,influence,param_attn_mean,freq_attn,freq_attn_param




# 定义完整模型
class Model_three_system_1004(nn.Module):#三系统架构
    def __init__(self, volt_input_dim, volt_mlp_hidden_dims,mlp_output_dims, volt_mlp_num_layers,
                 impe_input_dim,impe_mlp_hidden_dims,  impe_mlp_num_layers,
                 transformer_d_model, nhead, transformer_num_layers,param_transformer_num_layers,
                 physic_mlp_hidden_dims, physic_mlp_num_layers,
                 ion_attr_embed_hidden_dims,ion_attr_embed_num_layers,
                 ion_encoder_num_layers,
                 ion_post_hidden_dims,ion_post_num_layers,
                 probMLP_input_dims,probMLP_hidden_dims,probMLP_num_layers,
                 param_mlp_hidden_dims,param_mlp_num_layers,
                 freq_encoder_hidden_dims,freq_encoder_num_layers,
                 time_encoder_hidden_dims,time_encoder_num_layers,
                 cross_transformer_num_layers,
                 param_embed_hidden_dims,param_embed_num_layers,
                 physic_embed_hidden_dims,physic_embed_num_layers,
                 envMLP_input_dim, env_mlp_hidden_dims, env_mlp_num_layers,
                 ep_input_dim, ep_mlp_hidden_dims,  ep_mlp_num_layers,
                 Z_encoder_num_layers ,
                 conc_input_dim, conc_mlp_hidden_dims, conc_mlp_num_layers,
                 num_freq_points,num_time_points):
        
        
        super(Model_three_system_1004, self).__init__()
        self.voltMLP = AdjustableMLP(volt_input_dim, volt_mlp_hidden_dims, mlp_output_dims, volt_mlp_num_layers)
        self.impeMLP = AdjustableMLP(impe_input_dim, impe_mlp_hidden_dims, mlp_output_dims, impe_mlp_num_layers)
        self.transformer = MyTransformerWithAttn(d_model=transformer_d_model, nhead=nhead, num_layers=transformer_num_layers)
        self.physicMLP = AdjustableMLP(transformer_d_model, physic_mlp_hidden_dims, 5*transformer_d_model, physic_mlp_num_layers)
        


        self.probMLP = AdjustableMLP(input_dim=probMLP_input_dims, hidden_dims=probMLP_hidden_dims, output_dim=mlp_output_dims, num_layers=probMLP_num_layers)  # 将 physic_output 投影到嵌入空间

        
        # 定义物理模型部分
        self.param_transformer =  MyTransformerWithAttn(d_model=transformer_d_model, nhead=nhead, num_layers=param_transformer_num_layers )
        self.paramMLP = AdjustableMLP(transformer_d_model,param_mlp_hidden_dims, 5*transformer_d_model, param_mlp_num_layers)
        self.norm = nn.LayerNorm(transformer_d_model)
        self.ion_attr_embed = AdjustableMLP(input_dim=len(ion_attr_list[0]), hidden_dims=ion_attr_embed_hidden_dims, output_dim=transformer_d_model, num_layers=ion_attr_embed_num_layers)  # 可学习离子嵌入器
        self.ion_encoder = MyTransformerWithAttn(d_model=transformer_d_model, nhead=nhead, num_layers=ion_encoder_num_layers)
        self.ion_postMLP = AdjustableMLP(input_dim=transformer_d_model, hidden_dims=ion_post_hidden_dims, output_dim=transformer_d_model, num_layers=ion_post_num_layers)  # 可学习离子嵌入器

        self.mlp_output_dims = mlp_output_dims
        self.my_cross_attn_transformer = MyCrossAttnTransformer(d_model=transformer_d_model, nhead=nhead, num_layers=cross_transformer_num_layers)


        self.param_compress = AdjustableMLP(transformer_d_model, param_embed_hidden_dims, 1, param_embed_num_layers)
        self.influence_compress = AdjustableMLP(transformer_d_model, physic_embed_hidden_dims, 1, physic_embed_num_layers)

        self.transformer_d_model = transformer_d_model
        # 对 physic_embed 添加残差和 norm
        self.prob_norm = nn.LayerNorm(transformer_d_model)

        # 对 ion_embeddings 添加残差和 norm
        self.ion_norm = nn.LayerNorm(transformer_d_model)


        self.envMLP = AdjustableMLP(envMLP_input_dim, env_mlp_hidden_dims, mlp_output_dims, env_mlp_num_layers)
        self.epMLP = AdjustableMLP(ep_input_dim, ep_mlp_hidden_dims, mlp_output_dims, ep_mlp_num_layers)

        self.Z_encoder =  MyTransformerWithAttn(d_model=transformer_d_model, nhead=nhead, num_layers=Z_encoder_num_layers )

        self.concMLP = AdjustableMLP(conc_input_dim, conc_mlp_hidden_dims, 1, conc_mlp_num_layers)

        # 1️⃣ 定义一个可学习的 cls_token (初始值可用正态分布)
        self.cls_token = nn.Parameter(torch.zeros(1, transformer_d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)




        # 真实频率列表（64个点）
        if num_freq_points==64:
            freq_values_hz = torch.tensor([
                19950, 15850, 12590, 10000, 7943, 6310, 5010, 3980, 3160, 2510, 1990, 1590, 1260, 1000,
                794.3, 631.0, 501.2, 398.1, 316.2, 251.2, 199.5, 158.5, 125.9, 100.0,
                79.43, 63.10, 50.12, 39.81, 31.62, 25.12, 19.95, 15.85, 12.59, 10.0,
                7.94, 6.31, 5.01, 3.98, 3.16, 2.51, 1.99, 1.59, 1.26, 1.0,
                0.7943, 0.6310, 0.5012, 0.3981, 0.3162, 0.2512, 0.1995, 0.1585, 0.1259, 0.1,
                0.07943, 0.06310, 0.05012, 0.03981, 0.03162, 0.02512, 0.01995, 0.01585, 0.01259, 0.01
            ], dtype=torch.float32)
        elif num_freq_points==63:
            freq_values_hz = torch.tensor([
            19950, 15850, 12590, 10000, 7943, 6310, 5010, 3980, 3160, 2510, 1990, 1590, 1260, 1000,
            794.3, 631.0, 501.2, 398.1, 316.2, 251.2, 199.5, 158.5, 125.9, 100.0,
            79.43, 63.10, 50.12, 39.81, 31.62, 25.12, 19.95, 15.85, 12.59, 10.0,
            7.94, 6.31, 5.01, 3.98, 3.16, 2.51, 1.99, 1.59, 1.26, 1.0,
            0.7943, 0.6310, 0.5012, 0.3981, 0.3162, 0.2512, 0.1995, 0.1585, 0.1259, 0.1,
            0.07943, 0.06310, 0.05012, 0.03981, 0.03162, 0.02512, 0.01995, 0.01585, 0.01259
            ], dtype=torch.float32)
        else:
            print("num_freq_points:",num_freq_points)
        # 归一化到 对数坐标系[0, 1]
        freq_values_log = torch.log10(freq_values_hz)  # log10变换
        freq_values_norm = (freq_values_log - freq_values_log.min()) / (freq_values_log.max() - freq_values_log.min())
        self.register_buffer("freq_values_tensor", freq_values_norm)     

        # 定义 frequency encoder MLP,将频率数据变为和MLP同样的大小，这样可以对阻抗数据进行频率编码
        self.freq_encoder = AdjustableMLP(1, freq_encoder_hidden_dims, mlp_output_dims, freq_encoder_num_layers)#(64，1)->(64,32)
        # 时间编码器
        self.time_embedding = AdjustableMLP(1, time_encoder_hidden_dims, mlp_output_dims, time_encoder_num_layers)  # nn.Embedding 默认只能处理离散索引（如 0、1、2、3），它并不会显式捕捉到这些时间点在物理上是“连续且间隔为2小时”的。那为什么还会用 nn.Embedding 表示时间？这是因为在很多场景下（特别是在 Transformer 等结构中），时间点只作为一个“位置标识符”存在，例如第几个时间步，它本身的绝对含义并不那么重要。




        
        ion_attr_tensor = torch.tensor(ion_attr_list_plus, dtype=torch.float32)
        self.register_buffer("ion_attr_tensor", ion_attr_tensor)


    def forward(self, volt_data, impe_data, env_params,electrolyzer_parameters,concentration):
        B, T, F, C = impe_data.shape  # (B,4,64,2)
        B2,T2,C2 = volt_data.shape
         # ==== 添加断言检查 ====
        assert B == B2, f"Batch size mismatch: impe_data B={B}, volt_data B={B2}"
        assert T == T2, f"Time steps mismatch: impe_data T={T}, volt_data T={T2}"

        batch_size = B
        num_time_points = T
        num_freq_points = F
        device = volt_data.device

        # ========== 电压特征 ==========
        volt_output = self.voltMLP(volt_data)  # (B,4,1)->(B,4,32)

        # ======== 时间编码(一次性计算) ========
        time_base_input = torch.arange(T, device=device).float().view(T, 1)   # (T,1)
        time_encoded_base = self.time_embedding(time_base_input)              # (T,32)

        # 扩展到 batch 维度 (B,T,32)
        time_encoded = time_encoded_base.unsqueeze(0).expand(B, T, -1)

        # 电压特征加时间编码
        volt_output = volt_output + time_encoded

        # ========== 阻抗特征 ==========
        impe_output = self.impeMLP(impe_data.view(B, T * F, -1))  # (B,256,32)
        impe_output = impe_output.view(B, T, F, -1)               # (B,4,64,32)

        # ======== 频率编码(一次性计算) ========
        freq_base_input = self.freq_values_tensor.view(F, 1)        # (64,1)
        freq_encoded_base = self.freq_encoder(freq_base_input)      # (64,32)
        freq_encoded = freq_encoded_base.unsqueeze(0).unsqueeze(0).expand(B, T, F, -1)

        # ======== 时间编码扩展到阻抗特征 ========
        time_encoded_impe = time_encoded.unsqueeze(2).expand(B, T, F, -1)    # (B,4,64,32)

        # 加入时间编码+频率编码
        impe_output = impe_output + time_encoded_impe + freq_encoded

        # reshape回原始形状
        impe_output = impe_output.view(B, T * F, -1)   # (B,256,32)


        # 提取环境参数
        temperature = env_params[:, 0].unsqueeze(1)      # (B,1)
        flow = env_params[:, 1].unsqueeze(1)             # (B,1)
        current = env_params[:, 2].unsqueeze(1)          # (B,1)


        # (1) 编码得到两个 32 维 token
        env_coding = self.envMLP(env_params)                 # (B, 32)
        ep_coding  = self.epMLP(electrolyzer_parameters)     # (B, 32)

        # (2) 扩展可学习 cls_token 到 batch
        B = env_coding.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)         # (B, 1, 32)

        # (3) 在“序列维”拼接成 3 个 token
        cls_inputs = torch.cat([
            env_coding.unsqueeze(1),                         # (B, 1, 32)
            ep_coding.unsqueeze(1),
            cls_token                                        # (B, 1, 32)
        ], dim=1)                                            # => (B, 3, 32)

        # (4) 通过 Transformer 编码器
        enc_out,Z_attn = self.Z_encoder(cls_inputs)                 # 可能返回 Tensor 或 (Tensor, attn) / (Tensor, list_of_attn)

        # (5) 取 CLS 位置（最后一个 token）
        cls_token_feat = enc_out[:, -1:, :]               # (B, 1,32)
        # print("cls_token_feat.shape",cls_token_feat.shape)

        # 拼接到volt_output和impe_output后面
        tokens = torch.cat([ volt_output, impe_output,cls_token_feat], dim=1)  # (B,1+4+256,32)
        transformer_output, attn_weights = self.transformer(tokens)  # (B, 261, 32)
 
        # 假设只用最后一层的transformer的注意力
        attn_map = attn_weights[-1]  # shape: (B, num_heads, tgt_len, src_len)
        cls_attn = attn_map[:, :, -1, :]  # (B, num_heads, 261)
        cls_attn_mean = cls_attn.mean(dim=1)    # (B, 261)  ← 平均所有头
        # 取最后一个输出 (blank input 的输出)
        last_output = transformer_output[:, -1, :] # Shape: (batch_size, feature_dim)

        
        # 提取阻抗频率部分注意力 (忽略电压4个token)
        freq_attn = cls_attn_mean[:, num_time_points:-1]  # (B, T*F)
        freq_attn = freq_attn.view(B, T, F)  # (B, T, F)


        # =============== 用于预测五个初始状态的物理参数 ===============
        # =============== 5️⃣ 使用主分支embedding预测初始物理参数 ===============

        # 1️⃣ 取电压第一个时间点的embedding (B,32)
        volt_first_feat = volt_output[:, 0, :].unsqueeze(1)  # (B,1,32)

        # 2️⃣ 取阻抗第一个时间点的embedding (B,64,32)
        impe_first_feat = impe_output[:, 0:num_freq_points, :]  # (B,64,32)

        # 3️⃣ 拼接 cls_token_feat、电压和阻抗
        param_tokens = torch.cat([ cls_token_feat,volt_first_feat, impe_first_feat], dim=1)  # (B,66,32)

        # 4️⃣ 送入参数 Transformer
        param_encoded,param_attn_weights = self.param_transformer(param_tokens)  # (B,66,32)

        # Transformer
        param_attn_map = param_attn_weights[-1]  # (B, nhead, N_tokens, N_tokens)
        param_attn_weights = param_attn_map[:, :, 0, :]  # (B, nhead, N_tokens)
        param_attn_mean = param_attn_weights.mean(dim=1)  # (B, N_tokens)
        
        # 提取阻抗频率部分注意力 (忽略电压1个token)
        freq_attn_param = param_attn_mean[:, 2:]  # (B, 1*F)
        freq_attn_param = freq_attn_param.view(B, F)  # (B, 1, F)
        
        first_cls_output = param_encoded[:, 0,:]  # (B, D)
        # print("first_cls_output.shape:",first_cls_output.shape)

        # 8. 通过 paramMLP 输出5个初始状态参数原始物理量（不做 clamp，不做 sigmoid）
        param_output = self.paramMLP(first_cls_output) # (B, 5*32)，允许其自由表达大幅环境变化
        # print("param_output.shape:",param_output.shape)
        param_output = param_output.view(B, 5, self.transformer_d_model)       # (B, 5, 32)
        # 每个参数独立压缩为标量
        param_values = self.param_compress(param_output) # (B, 5, 1)
        param_values = param_values.squeeze(-1)          # (B, 5)


        # ========== 2️⃣ 物理影响因子部分 ==========
        raw_physic_output = self.physicMLP(last_output)  # (B, 160)
        raw_physic_output = raw_physic_output.view(B, 5, self.transformer_d_model)




        # 每个影响因子独立压缩
        influence_values = self.influence_compress(raw_physic_output)  # (B, 5, 1)
        influence_values = influence_values.squeeze(-1)   
        # print("influence_values.shape:",influence_values.shape) 



        sigmoid = torch.nn.Sigmoid()
        
        # # ------------------- 行为影响因子（physic_output） -------------------
        theta_min, theta_max = 0.001, 0.999
        phi_min, phi_max     = 0.001, 0.999

        theta_ca = theta_min + (theta_max - theta_min) * sigmoid(influence_values[:, 0:1])
        phi_ca   = phi_min   + (phi_max   - phi_min)   * sigmoid(influence_values[:, 1:2])
        theta_an = theta_min + (theta_max - theta_min) * sigmoid(influence_values[:, 2:3])
        phi_an   = phi_min   + (phi_max   - phi_min)   * sigmoid(influence_values[:, 3:4])
        psi      = phi_min   + (phi_max   - phi_min)   * sigmoid(influence_values[:, 4:5])  # 同样限制


        # ------------------- 固有物性参数（param_output） -------------------
        sigma_min, sigma_max = 0.01, 2  # S/cm
        sigma_mem = sigma_min + (sigma_max - sigma_min) * sigmoid(param_values[:, 0:1])

        alpha_min, alpha_max = 0.01, 2
        alpha_ca = alpha_min + (alpha_max - alpha_min) * sigmoid(param_values[:, 1:2])
        alpha_an = alpha_min + (alpha_max - alpha_min) * sigmoid(param_values[:, 2:3])

        log_i0ca_min, log_i0ca_max = -9, 0
        log_i0an_min, log_i0an_max = -9, 0
        log_i0ca = log_i0ca_min + (log_i0ca_max - log_i0ca_min) * sigmoid(param_values[:, 3:4])
        log_i0an = log_i0an_min + (log_i0an_max - log_i0an_min) * sigmoid(param_values[:, 4:5])
        i_0ca = torch.pow(10, log_i0ca)
        i_0an = torch.pow(10, log_i0an)


        
        
        # ========== Cross Attention 融合影响因子和物性参数 ==========
        # ========== 3️⃣ Cross Attention 交互 ==========
        # Query: param_output, Key: raw_physic_output
        cross_output, cross_attn_weights = self.my_cross_attn_transformer(
            query=param_output,
            key_value=raw_physic_output
        )

        #预测浓度
        # 1) 形状转换 -> (B, 5*self.transformer_d_model)
        cross_output_pred = cross_output.flatten(start_dim=1)  # #(B, 5, self.transformer_d_model)->(B,5*self.transformer_d_model)

        # 2) 预测浓度
        conc_pred = self.concMLP(cross_output_pred)

        # 1️⃣ 投影
        # physic_embed = self.probMLP(param_output)  # (B, 5, 32)
        physic_embed_raw = self.probMLP(cross_output)  # (B, 5, 32)
        physic_embed = self.prob_norm(param_output + physic_embed_raw)  # ✅ 残差 + norm

        # 1️⃣ 输入 ion_attr_tensor（静态属性表），形状为 (6, 7)
        ion_attr_proj = self.ion_attr_embed(self.ion_attr_tensor)  # (6, 32)

        # 2️⃣ 添加 batch 维度，输入 Transformer
        ion_attr_proj = ion_attr_proj.unsqueeze(0)  # (1, 6, 32)

        # 3️⃣ 编码，返回编码结果和注意力权重
        ion_encoded, ion_attn_weights = self.ion_encoder(ion_attr_proj)  # (1, 6, 32), list of attn

        # 4️⃣ 去掉 batch 维度，作为每个离子的嵌入向量
        ion_embeddings_raw = self.ion_postMLP(ion_encoded.squeeze(0))   # (6, 32)
        ion_embeddings = self.ion_norm(ion_encoded.squeeze(0) + ion_embeddings_raw)  # ✅ 残差 + norm

        # 5️⃣ 点积：param_output -> physic_embed (B, 5, 32)
        #         ion_embeddings -> (6, 32)
        raw_scores = torch.matmul(physic_embed, ion_embeddings.T)  # (B, 5, 6)

        # 6️⃣ 聚合
        raw_prob_output = raw_scores.mean(dim=1)  # (B, 6)

        # 4️⃣ Softmax
        prob_output = torch.softmax(raw_prob_output, dim=1)
        

    


        # ---------- 电压计算：基于防NaN、稳定后的参数 ----------
        predicted_voltages = []
        for i in range(batch_size):
            pred_v, _, _, _, _ = calculate_predicted_voltage_plus(
                theta_ca[i], phi_ca[i], theta_an[i], phi_an[i], psi[i],
                temperature[i], flow[i], current[i],
                sigma_mem[i], alpha_ca[i], alpha_an[i], i_0ca[i], i_0an[i],
                electrolyzer_parameters= electrolyzer_parameters[i]  
            )
            predicted_voltages.append(pred_v)

        # 组合 batch 的电压结果
        predicted_voltage = torch.stack(predicted_voltages).unsqueeze(1)  # shape: (B, 1)

        wuxing = [sigma_mem,alpha_ca,alpha_an,i_0ca,i_0an]
        influence=[psi,theta_ca,theta_an,phi_ca,phi_an]

        return prob_output, predicted_voltage,conc_pred, wuxing,cls_attn_mean ,raw_prob_output,influence,param_attn_mean,freq_attn,freq_attn_param



# 定义完整模型
class Model_three_system_1117(nn.Module):#三系统架构
    def __init__(self, volt_input_dim, volt_mlp_hidden_dims,mlp_output_dims, volt_mlp_num_layers,
                 impe_input_dim,impe_mlp_hidden_dims,  impe_mlp_num_layers,
                 transformer_d_model, nhead, transformer_num_layers,param_transformer_num_layers,
                 physic_mlp_hidden_dims, physic_mlp_num_layers,
                 ion_attr_embed_hidden_dims,ion_attr_embed_num_layers,
                 ion_encoder_num_layers,
                 ion_post_hidden_dims,ion_post_num_layers,
                 probMLP_input_dims,probMLP_hidden_dims,probMLP_num_layers,
                 param_mlp_hidden_dims,param_mlp_num_layers,
                 freq_encoder_hidden_dims,freq_encoder_num_layers,
                 time_encoder_hidden_dims,time_encoder_num_layers,
                 cross_transformer_num_layers,
                 param_embed_hidden_dims,param_embed_num_layers,
                 physic_embed_hidden_dims,physic_embed_num_layers,
                 envMLP_input_dim, env_mlp_hidden_dims, env_mlp_num_layers,
                 ep_input_dim, ep_mlp_hidden_dims,  ep_mlp_num_layers,
                 Z_encoder_num_layers ,
                 conc_input_dim, conc_mlp_hidden_dims, conc_mlp_num_layers,
                 num_freq_points,num_time_points):
        
        
        super(Model_three_system_1117, self).__init__()
        self.voltMLP = AdjustableMLP(volt_input_dim, volt_mlp_hidden_dims, mlp_output_dims, volt_mlp_num_layers)
        self.impeMLP = AdjustableMLP(impe_input_dim, impe_mlp_hidden_dims, mlp_output_dims, impe_mlp_num_layers)
        self.transformer = MyTransformerWithAttn(d_model=transformer_d_model, nhead=nhead, num_layers=transformer_num_layers)
        self.physicMLP = AdjustableMLP(transformer_d_model, physic_mlp_hidden_dims, 5*transformer_d_model, physic_mlp_num_layers)
        


        self.probMLP = AdjustableMLP(input_dim=probMLP_input_dims, hidden_dims=probMLP_hidden_dims, output_dim=mlp_output_dims, num_layers=probMLP_num_layers)  # 将 physic_output 投影到嵌入空间

        
        # 定义物理模型部分
        self.param_transformer =  MyTransformerWithAttn(d_model=transformer_d_model, nhead=nhead, num_layers=param_transformer_num_layers )
        self.paramMLP = AdjustableMLP(transformer_d_model,param_mlp_hidden_dims, 5*transformer_d_model, param_mlp_num_layers)
        self.norm = nn.LayerNorm(transformer_d_model)
        self.ion_attr_embed = AdjustableMLP(input_dim=len(ion_attr_list_plus_1117[0]), hidden_dims=ion_attr_embed_hidden_dims, output_dim=transformer_d_model, num_layers=ion_attr_embed_num_layers)  # 可学习离子嵌入器
        self.ion_encoder = MyTransformerWithAttn(d_model=transformer_d_model, nhead=nhead, num_layers=ion_encoder_num_layers)
        self.ion_postMLP = AdjustableMLP(input_dim=transformer_d_model, hidden_dims=ion_post_hidden_dims, output_dim=transformer_d_model, num_layers=ion_post_num_layers)  # 可学习离子嵌入器

        self.mlp_output_dims = mlp_output_dims
        self.my_cross_attn_transformer = MyCrossAttnTransformer(d_model=transformer_d_model, nhead=nhead, num_layers=cross_transformer_num_layers)


        self.param_compress = AdjustableMLP(transformer_d_model, param_embed_hidden_dims, 1, param_embed_num_layers)
        self.influence_compress = AdjustableMLP(transformer_d_model, physic_embed_hidden_dims, 1, physic_embed_num_layers)

        self.transformer_d_model = transformer_d_model
        # 对 physic_embed 添加残差和 norm
        self.prob_norm = nn.LayerNorm(transformer_d_model)

        # 对 ion_embeddings 添加残差和 norm
        self.ion_norm = nn.LayerNorm(transformer_d_model)


        self.envMLP = AdjustableMLP(envMLP_input_dim, env_mlp_hidden_dims, mlp_output_dims, env_mlp_num_layers)
        self.epMLP = AdjustableMLP(ep_input_dim, ep_mlp_hidden_dims, mlp_output_dims, ep_mlp_num_layers)

        self.Z_encoder =  MyTransformerWithAttn(d_model=transformer_d_model, nhead=nhead, num_layers=Z_encoder_num_layers )

        self.conc_head = nn.Linear(5 * transformer_d_model, 1)

        # 1️⃣ 定义一个可学习的 cls_token (初始值可用正态分布)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)




        # 真实频率列表（64个点）
        if num_freq_points==64:
            freq_values_hz = torch.tensor([
                19950, 15850, 12590, 10000, 7943, 6310, 5010, 3980, 3160, 2510, 1990, 1590, 1260, 1000,
                794.3, 631.0, 501.2, 398.1, 316.2, 251.2, 199.5, 158.5, 125.9, 100.0,
                79.43, 63.10, 50.12, 39.81, 31.62, 25.12, 19.95, 15.85, 12.59, 10.0,
                7.94, 6.31, 5.01, 3.98, 3.16, 2.51, 1.99, 1.59, 1.26, 1.0,
                0.7943, 0.6310, 0.5012, 0.3981, 0.3162, 0.2512, 0.1995, 0.1585, 0.1259, 0.1,
                0.07943, 0.06310, 0.05012, 0.03981, 0.03162, 0.02512, 0.01995, 0.01585, 0.01259, 0.01
            ], dtype=torch.float32)
        elif num_freq_points==63:
            freq_values_hz = torch.tensor([
            19950, 15850, 12590, 10000, 7943, 6310, 5010, 3980, 3160, 2510, 1990, 1590, 1260, 1000,
            794.3, 631.0, 501.2, 398.1, 316.2, 251.2, 199.5, 158.5, 125.9, 100.0,
            79.43, 63.10, 50.12, 39.81, 31.62, 25.12, 19.95, 15.85, 12.59, 10.0,
            7.94, 6.31, 5.01, 3.98, 3.16, 2.51, 1.99, 1.59, 1.26, 1.0,
            0.7943, 0.6310, 0.5012, 0.3981, 0.3162, 0.2512, 0.1995, 0.1585, 0.1259, 0.1,
            0.07943, 0.06310, 0.05012, 0.03981, 0.03162, 0.02512, 0.01995, 0.01585, 0.01259
            ], dtype=torch.float32)
        else:
            print("num_freq_points:",num_freq_points)
        # 归一化到 对数坐标系[0, 1]
        freq_values_log = torch.log10(freq_values_hz)  # log10变换
        freq_values_norm = (freq_values_log - freq_values_log.min()) / (freq_values_log.max() - freq_values_log.min())
        self.register_buffer("freq_values_tensor", freq_values_norm)     

        # 定义 frequency encoder MLP,将频率数据变为和MLP同样的大小，这样可以对阻抗数据进行频率编码
        self.freq_encoder = AdjustableMLP(1, freq_encoder_hidden_dims, mlp_output_dims, freq_encoder_num_layers)#(64，1)->(64,32)
        # 时间编码器
        self.time_embedding = AdjustableMLP(1, time_encoder_hidden_dims, mlp_output_dims, time_encoder_num_layers)  # nn.Embedding 默认只能处理离散索引（如 0、1、2、3），它并不会显式捕捉到这些时间点在物理上是“连续且间隔为2小时”的。那为什么还会用 nn.Embedding 表示时间？这是因为在很多场景下（特别是在 Transformer 等结构中），时间点只作为一个“位置标识符”存在，例如第几个时间步，它本身的绝对含义并不那么重要。
        
        ion_attr_tensor = torch.tensor(ion_attr_list_plus_1117, dtype=torch.float32)
        self.register_buffer("ion_attr_tensor", ion_attr_tensor)
        
        self.ion_attr_dim = 7
        self.rule_dim = ion_attr_tensor.size(1) - self.ion_attr_dim  # 一般是 8
        self.num_ions = ion_attr_tensor.size(0)

        # 提取类别级 rule 原型 r_k（只用后 8 维）
        self.register_buffer("ion_rule_proto", ion_attr_tensor[:, self.ion_attr_dim:])  # (num_ions, rule_dim)
        
        # ========= 4. ★ 新增：rule_head + 层级多任务 heads =========
        # 从 last_output (B, transformer_d_model) ->  rule-space (B, rule_dim)
        self.rule_head = nn.Linear(transformer_d_model, self.rule_dim)

        # 高/低影响组（2类），以及高组 3 类、低组 3 类
        self.head_group = nn.Linear(transformer_d_model, 2)   # 0: low-impact, 1: high-impact
        self.head_high3 = nn.Linear(transformer_d_model, 3)   # [Ca, Na, Ni]
        self.head_low3  = nn.Linear(transformer_d_model, 3)   # [Cu, Fe, Cr]
        # 新增：推理时用的缓存
        self.ion_embeddings_eval = None

    def forward(self, volt_data, impe_data, env_params,electrolyzer_parameters,concentration):
        B, T, F, C = impe_data.shape  # (B,4,64,2)
        B2,T2,C2 = volt_data.shape
         # ==== 添加断言检查 ====
        assert B == B2, f"Batch size mismatch: impe_data B={B}, volt_data B={B2}"
        assert T == T2, f"Time steps mismatch: impe_data T={T}, volt_data T={T2}"

        batch_size = B
        num_time_points = T
        num_freq_points = F
        device = volt_data.device

        # ========== 电压特征 ==========
        volt_output = self.voltMLP(volt_data)  # (B,4,1)->(B,4,32)

        # ======== 时间编码(一次性计算) ========
        time_base_input = torch.arange(T, device=device).float().view(T, 1)   # (T,1)
        time_encoded_base = self.time_embedding(time_base_input)              # (T,32)

        # 扩展到 batch 维度 (B,T,32)
        time_encoded = time_encoded_base.unsqueeze(0).expand(B, T, -1)

        # 电压特征加时间编码
        volt_output = volt_output + time_encoded

        # ========== 阻抗特征 ==========
        impe_output = self.impeMLP(impe_data.view(B, T * F, -1))  # (B,256,32)
        impe_output = impe_output.view(B, T, F, -1)               # (B,4,64,32)

        # ======== 频率编码(一次性计算) ========
        freq_base_input = self.freq_values_tensor.view(F, 1)        # (64,1)
        freq_encoded_base = self.freq_encoder(freq_base_input)      # (64,32)
        freq_encoded = freq_encoded_base.unsqueeze(0).unsqueeze(0).expand(B, T, F, -1)

        # ======== 时间编码扩展到阻抗特征 ========
        time_encoded_impe = time_encoded.unsqueeze(2).expand(B, T, F, -1)    # (B,4,64,32)

        # 加入时间编码+频率编码
        impe_output = impe_output + time_encoded_impe + freq_encoded

        # reshape回原始形状
        impe_output = impe_output.view(B, T * F, -1)   # (B,256,32)


        # 提取环境参数
        temperature = env_params[:, 0].unsqueeze(1)      # (B,1)
        flow = env_params[:, 1].unsqueeze(1)             # (B,1)
        current = env_params[:, 2].unsqueeze(1)          # (B,1)


        # (1) 编码得到两个 32 维 token
        env_coding = self.envMLP(env_params)                 # (B, 32)
        ep_coding  = self.epMLP(electrolyzer_parameters)     # (B, 32)

        # (2) 扩展可学习 cls_token 到 batch
        B = env_coding.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)         # (B, 1, 32)

        # (3) 在“序列维”拼接成 3 个 token
        cls_inputs = torch.cat([
            env_coding.unsqueeze(1),                         # (B, 1, 32)
            ep_coding.unsqueeze(1),
            cls_token                                        # (B, 1, 32)
        ], dim=1)                                            # => (B, 3, 32)

        # (4) 通过 Transformer 编码器
        enc_out,Z_attn = self.Z_encoder(cls_inputs)                 # 可能返回 Tensor 或 (Tensor, attn) / (Tensor, list_of_attn)

        # (5) 取 CLS 位置（最后一个 token）
        cls_token_feat = enc_out[:, -1:, :]               # (B, 1,32)
        # print("cls_token_feat.shape",cls_token_feat.shape)

        # 拼接到volt_output和impe_output后面
        tokens = torch.cat([ volt_output, impe_output,cls_token_feat], dim=1)  # (B,1+4+256,32)
        transformer_output, attn_weights = self.transformer(tokens)  # (B, 261, 32)
 
        # 假设只用最后一层的transformer的注意力
        attn_map = attn_weights[-1]  # shape: (B, num_heads, tgt_len, src_len)
        cls_attn = attn_map[:, :, -1, :]  # (B, num_heads, 261)
        cls_attn_mean = cls_attn.mean(dim=1)    # (B, 261)  ← 平均所有头
        # 取最后一个输出 (blank input 的输出)
        last_output = transformer_output[:, -1, :] # Shape: (batch_size, feature_dim)

        
        # 提取阻抗频率部分注意力 (忽略电压4个token)
        freq_attn = cls_attn_mean[:, num_time_points:-1]  # (B, T*F)
        freq_attn = freq_attn.view(B, T, F)  # (B, T, F)


        # =============== 用于预测五个初始状态的物理参数 ===============
        # =============== 5️⃣ 使用主分支embedding预测初始物理参数 ===============

        # 1️⃣ 取电压第一个时间点的embedding (B,32)
        volt_first_feat = volt_output[:, 0, :].unsqueeze(1)  # (B,1,32)

        # 2️⃣ 取阻抗第一个时间点的embedding (B,64,32)
        impe_first_feat = impe_output[:, 0:num_freq_points, :]  # (B,64,32)

        # 3️⃣ 拼接 cls_token_feat、电压和阻抗
        param_tokens = torch.cat([ cls_token_feat,volt_first_feat, impe_first_feat], dim=1)  # (B,66,32)

        # 4️⃣ 送入参数 Transformer
        param_encoded,param_attn_weights = self.param_transformer(param_tokens)  # (B,66,32)

        # Transformer
        param_attn_map = param_attn_weights[-1]  # (B, nhead, N_tokens, N_tokens)
        param_attn_weights = param_attn_map[:, :, 0, :]  # (B, nhead, N_tokens)
        param_attn_mean = param_attn_weights.mean(dim=1)  # (B, N_tokens)
        
        # 提取阻抗频率部分注意力 (忽略电压1个token)
        freq_attn_param = param_attn_mean[:, 2:]  # (B, 1*F)
        freq_attn_param = freq_attn_param.view(B, F)  # (B, 1, F)
        
        first_cls_output = param_encoded[:, 0,:]  # (B, D)
        # print("first_cls_output.shape:",first_cls_output.shape)

        # 8. 通过 paramMLP 输出5个初始状态参数原始物理量（不做 clamp，不做 sigmoid）
        param_output = self.paramMLP(first_cls_output) # (B, 5*32)，允许其自由表达大幅环境变化
        # print("param_output.shape:",param_output.shape)
        param_output = param_output.view(B, 5, self.transformer_d_model)       # (B, 5, 32)
        # 每个参数独立压缩为标量
        param_values = self.param_compress(param_output) # (B, 5, 1)
        param_values = param_values.squeeze(-1)          # (B, 5)


        # ========== 2️⃣ 物理影响因子部分 ==========
        raw_physic_output = self.physicMLP(last_output)  # (B, 160)
        raw_physic_output = raw_physic_output.view(B, 5, self.transformer_d_model)




        # 每个影响因子独立压缩
        influence_values = self.influence_compress(raw_physic_output)  # (B, 5, 1)
        influence_values = influence_values.squeeze(-1)   
        # print("influence_values.shape:",influence_values.shape) 



        sigmoid = torch.nn.Sigmoid()
        
        # # ------------------- 行为影响因子（physic_output） -------------------
        theta_min, theta_max = 0.001, 0.999
        phi_min, phi_max     = 0.001, 0.999

        theta_ca = theta_min + (theta_max - theta_min) * sigmoid(influence_values[:, 0:1])
        phi_ca   = phi_min   + (phi_max   - phi_min)   * sigmoid(influence_values[:, 1:2])
        theta_an = theta_min + (theta_max - theta_min) * sigmoid(influence_values[:, 2:3])
        phi_an   = phi_min   + (phi_max   - phi_min)   * sigmoid(influence_values[:, 3:4])
        psi      = phi_min   + (phi_max   - phi_min)   * sigmoid(influence_values[:, 4:5])  # 同样限制


        # ------------------- 固有物性参数（param_output） -------------------
        sigma_min, sigma_max = 0.01, 2  # S/cm
        sigma_mem = sigma_min + (sigma_max - sigma_min) * sigmoid(param_values[:, 0:1])

        alpha_min, alpha_max = 0.01, 2# S/cm
        alpha_ca = alpha_min + (alpha_max - alpha_min) * sigmoid(param_values[:, 1:2])
        alpha_an = alpha_min + (alpha_max - alpha_min) * sigmoid(param_values[:, 2:3])

        log_i0ca_min, log_i0ca_max = -9, 0# S/cm
        log_i0an_min, log_i0an_max = -9, 0# S/cm
        log_i0ca = log_i0ca_min + (log_i0ca_max - log_i0ca_min) * sigmoid(param_values[:, 3:4])
        log_i0an = log_i0an_min + (log_i0an_max - log_i0an_min) * sigmoid(param_values[:, 4:5])
        i_0ca = torch.pow(10, log_i0ca)
        i_0an = torch.pow(10, log_i0an)


        
        
        # ========== Cross Attention 融合影响因子和物性参数 ==========
        # ========== 3️⃣ Cross Attention 交互 ==========
        # Query: param_output, Key: raw_physic_output
        cross_output, cross_attn_weights = self.my_cross_attn_transformer(
            query=param_output,
            key_value=raw_physic_output
        )

         # ******** 新增：用 cross_output 做“全局表示” ********
        # 对 5 个 token 做平均池化，得到 (B, D) 的整体表示
        cross_pooled = cross_output.mean(dim=1)  # (B, D)

        # 在 rule-space 中预测类别级规则原型
        rule_pred = self.rule_head(cross_pooled)      # (B, rule_dim)

        # 高/低影响组 & 组内 3 类离子
        group_logits = self.head_group(cross_pooled)  # (B, 2)
        high3_logits = self.head_high3(cross_pooled)  # (B, 3)
        low3_logits  = self.head_low3(cross_pooled)   # (B, 3)

        #预测浓度
        # 1) 形状转换 -> (B, 5*self.transformer_d_model)
        cross_output_pred = cross_output.flatten(start_dim=1)  # #(B, 5, self.transformer_d_model)->(B,5*self.transformer_d_model)

        # 2) 预测浓度
        conc_pred = self.conc_head(cross_output_pred)

        # 1️⃣ 投影
        # physic_embed = self.probMLP(param_output)  # (B, 5, 32)
        physic_embed_raw = self.probMLP(cross_output)  # (B, 5, 32)
        physic_embed = self.prob_norm(param_output + physic_embed_raw)  # ✅ 残差 + norm

        
        # 1️⃣ 输入 ion_attr_tensor（静态属性表），形状为 (6, 7)
        ion_attr_proj = self.ion_attr_embed(self.ion_attr_tensor)  # (6, 32)

        # 2️⃣ 添加 batch 维度，输入 Transformer
        ion_attr_proj = ion_attr_proj.unsqueeze(0)  # (1, 6, 32)

        # 3️⃣ 编码，返回编码结果和注意力权重
        ion_encoded, ion_attn_weights = self.ion_encoder(ion_attr_proj)  # (1, 6, 32), list of attn

        # 4️⃣ 去掉 batch 维度，作为每个离子的嵌入向量
        ion_embeddings_raw = self.ion_postMLP(ion_encoded.squeeze(0))   # (6, 32)
        ion_embeddings = self.ion_norm(ion_encoded.squeeze(0) + ion_embeddings_raw)  # ✅ 残差 + norm

        # 5️⃣ 点积：param_output -> physic_embed (B, 5, 32)
        #         ion_embeddings -> (6, 32)
        raw_scores = torch.matmul(physic_embed, ion_embeddings.T)  # (B, 5, 6)

        # 6️⃣ 聚合
        raw_prob_output = raw_scores.mean(dim=1)  # (B, 6)

        # 4️⃣ Softmax
        prob_output = torch.softmax(raw_prob_output, dim=1)
        

    


        # ---------- 电压计算：基于防NaN、稳定后的参数 ----------
        predicted_voltages = []
        for i in range(batch_size):
            pred_v, _, _, _, _ = calculate_predicted_voltage_plus(
                theta_ca[i], phi_ca[i], theta_an[i], phi_an[i], psi[i],
                temperature[i], flow[i], current[i],
                sigma_mem[i], alpha_ca[i], alpha_an[i], i_0ca[i], i_0an[i],
                electrolyzer_parameters= electrolyzer_parameters[i]  
            )
            predicted_voltages.append(pred_v)

        # 组合 batch 的电压结果
        predicted_voltage = torch.stack(predicted_voltages).unsqueeze(1)  # shape: (B, 1)

        wuxing = [sigma_mem,alpha_ca,alpha_an,i_0ca,i_0an]
        influence=[psi,theta_ca,theta_an,phi_ca,phi_an]

        # 在最后多 return rule_pred / group_logits / high3_logits / low3_logits
        return (
            prob_output,         # (B, 6)
            predicted_voltage,   # (B, 1)
            conc_pred,           # (B, 1)
            wuxing,              # list of tensors
            cls_attn_mean,       # (B, seq_len)
            raw_prob_output,     # (B, 6)
            influence,           # list of tensors
            param_attn_mean,     # (B, N_tokens)
            freq_attn,           # (B, T, F)
            freq_attn_param,     # (B, F)
            rule_pred,           # (B, rule_dim)
            group_logits,        # (B, 2)
            high3_logits,        # (B, 3)
            low3_logits          # (B, 3)
        )

class Model_MLP_ClassifierOnly(nn.Module):
    """
    纯 MLP 分类基线：
    - 所有输入展平后拼接 -> 主干 MLP -> 分类头（logits）
    - 仅保留分类任务（无电压/浓度/影响因子/物理量/注意力）
    """
    def __init__(
        self,
        num_time_points: int,
        num_freq_points: int,
        volt_channel_dim: int,   # volt_data 最后一维
        impe_channel_dim: int,   # impe_data 最后一维（如 2=实+虚）
        env_dim: int,
        ep_dim: int,
        conc_dim: int,
        num_classes: int = 6,
        # 主干与头部超参
        backbone_hidden=(1024, 512, 256),
        backbone_out_dim=256,
        backbone_use_bn=True,
        backbone_dropout=0.0,
        head_hidden=(128, 64),
        head_use_bn=False,
        head_dropout=0.0,
    ):
        super().__init__()
        self.T  = num_time_points
        self.F  = num_freq_points
        self.Cv = volt_channel_dim
        self.Ci = impe_channel_dim
        self.env_dim  = env_dim
        self.ep_dim   = ep_dim
        self.conc_dim = conc_dim

        # 展平+拼接后的总输入维度
        input_dim = (self.T * self.Cv) + (self.T * self.F * self.Ci) + self.env_dim + self.ep_dim + self.conc_dim

        # 主干
        self.backbone = AdjustableMLP(
            input_dim=input_dim,
            hidden_dims=backbone_hidden,
            output_dim=backbone_out_dim,
            use_bn=backbone_use_bn,
            dropout=backbone_dropout
        )

        # 分类头（输出 logits）
        self.classifier = AdjustableMLP(
            input_dim=backbone_out_dim,
            hidden_dims=head_hidden,
            output_dim=num_classes,
            use_bn=head_use_bn,
            dropout=head_dropout
        )

    def forward(self, volt_data, impe_data, env_params, electrolyzer_parameters, concentration):
        """
        输入：
            volt_data: (B, T, Cv)
            impe_data: (B, T, F, Ci)
            env_params: (B, E_env)
            electrolyzer_parameters: (B, E_ep)
            concentration: (B, E_conc)
        返回：
            logits: (B, num_classes)  —— 直接用于 nn.CrossEntropyLoss
        """
        B = volt_data.size(0)

        x = torch.cat([
            volt_data.reshape(B, self.T * self.Cv),
            impe_data.reshape(B, self.T * self.F * self.Ci),
            env_params.reshape(B, -1),
            electrolyzer_parameters.reshape(B, -1),
            concentration.reshape(B, -1),
        ], dim=1)

        feat = self.backbone(x)      # (B, D)
        logits = self.classifier(feat)# (B, num_classes)
        return logits