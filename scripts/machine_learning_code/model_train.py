import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
#
import yaml
from pathlib import Path
import torch
class BaseTrainer:
    def __init__(self, model, optimizer, device, model_save_folder, scheduler=None, save_every=100):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.model_save_folder = model_save_folder
        self.save_every = save_every
        os.makedirs(self.model_save_folder, exist_ok=True)

        # 记录各类损失的历史（用于保存 csv / 画图）
        self.loss_history = {}  # { "total": [], "ce": [], "voltage": [] }

        # ====== TensorBoard 日志器 ======
        # 日志目录：<model_save_folder>/runs
        self.log_dir = os.path.join(self.model_save_folder, "runs")
        os.makedirs(self.log_dir, exist_ok=True)

        # SummaryWriter 用于写入 TensorBoard 日志
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # 这里用 global_step 做“横轴”，当前设计是：每个 epoch 调一次 _record_loss，step++
        self.global_step = 0

    def compute_loss(self, batch):
        """
        子类必须重写该函数
        输入: batch -> (volt_data, impe_data, env_params, true_labels, true_voltages)
        输出: loss, loss_items(dict)
        """
        raise NotImplementedError

    def _record_loss(self, loss_items):
        """
        记录一个 epoch 的平均损失到内存，并同步写入 TensorBoard。
        loss_items: dict，比如 {"total": 1.23, "voltage": 0.45, ...}
        """
        # 认为每调用一次 _record_loss 就是一个“全局 step”（通常对应一个 epoch）
        self.global_step += 1

        for k, v in loss_items.items():
            # 1) 保存在本地 history（后面画图 / 存 csv 用）
            if k not in self.loss_history:
                self.loss_history[k] = []
            self.loss_history[k].append(v)

            # 2) 写入 TensorBoard
            if self.writer is not None:
                # 曲线名称统一加上前缀 "loss/"
                self.writer.add_scalar(f"loss/{k}", v, self.global_step)


    def train(self, train_loader, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            running_losses = {}

            for batch in train_loader:
                batch = [b.to(self.device) if torch.is_tensor(b) else b for b in batch]
                loss, loss_items = self.compute_loss(batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 累加 batch 损失
                for k, v in loss_items.items():
                    running_losses[k] = running_losses.get(k, 0) + v

            # 记录 epoch 平均损失
            avg_losses = {k: v / len(train_loader) for k, v in running_losses.items()}
            self._record_loss(avg_losses)

            # 额外把当前学习率写进 TensorBoard
            if self.writer is not None:
                lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("lr", lr, self.global_step)

            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()


            # 保存模型和损失曲线
            if (epoch + 1) % self.save_every == 0:
                self.save_model(epoch+1)
                self.save_losses(epoch+1)

        # 训练完成后保存最终模型与曲线
        self.save_model("final")
        self.save_losses("final")

    def save_model(self, epoch):
        save_path = os.path.join(self.model_save_folder, f"trained_model_epoch_{epoch}.pth")
        torch.save(self.model.state_dict(), save_path)
        print(f"✅ 模型已保存到 {save_path}")

    def save_losses(self, epoch):
        # 保存 CSV
        csv_path = os.path.join(self.model_save_folder, f"training_loss_epoch_{epoch}.csv")
        pd.DataFrame(self.loss_history).to_csv(csv_path, index=False)
        print(f"📁 损失值已保存到 {csv_path}")

        # 绘制损失曲线
        plt.figure()
        for k, v in self.loss_history.items():
            plt.plot(v, label=f"{k} Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curves")
        plt.legend()
        plt.savefig(os.path.join(self.model_save_folder, f"loss_epoch_{epoch}.png"))
        plt.close()


def compute_band_attention_loss(
    freq_attn: torch.Tensor,
    band_edges,
    target_band_dist: torch.Tensor = None,
    eps: float = 1e-8,
    reduction: str = "mean",
):
    """
    基于频段的注意力分布约束：

    - freq_attn: [B, T, num_freqs] 或 [B, num_freqs]，来自 freqAttn / freqAttnParam
    - band_edges: 长度为 K+1 的降序 list，例如 [20000, 1000, 100, 10, 1, 0.1, 0.0]
    - target_band_dist:
        * 若为 [K]，表示一个统一的目标频段分布（全 batch 公用）
        * 若为 [B, K]，表示每个样本一条目标频段分布（per-ion target）

    - reduction:
        * "mean": 返回一个标量（默认行为，兼容旧代码）
        * "none": 返回 per-sample loss，形状 [B]
    """
    if freq_attn is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 为了兼容，直接返回 0
        return torch.tensor(0.0, device=device) if reduction == "mean" else torch.zeros(0, device=device)

    # ===== 1. 统一成 [B, num_freqs] =====
    if freq_attn.dim() == 3:
        B, T, num_freqs = freq_attn.shape
        attn = freq_attn.sum(dim=1)  # [B, num_freqs]
    elif freq_attn.dim() == 2:
        B, num_freqs = freq_attn.shape
        attn = freq_attn
    else:
        raise ValueError(f"freq_attn shape {freq_attn.shape} not supported")

    device = attn.device

    # ===== 2. 你的 63 个频率点 =====
    freq_values_hz = torch.tensor([
        19950., 15850., 12590., 10000., 7943., 6310., 5010., 3980., 3160., 2510.,
        1990., 1590., 1260., 1000.,
        794.3, 631.0, 501.2, 398.1, 316.2, 251.2, 199.5, 158.5, 125.9, 100.0,
        79.43, 63.10, 50.12, 39.81, 31.62, 25.12, 19.95, 15.85, 12.59, 10.0,
        7.94, 6.31, 5.01, 3.98, 3.16, 2.51, 1.99, 1.59, 1.26, 1.0,
        0.7943, 0.6310, 0.5012, 0.3981, 0.3162, 0.2512, 0.1995, 0.1585, 0.1259, 0.1,
        0.07943, 0.06310, 0.05012, 0.03981, 0.03162, 0.02512, 0.01995, 0.01585, 0.01259
    ], dtype=torch.float32, device=device)

    if freq_values_hz.numel() != num_freqs:
        raise ValueError(f"Fixed freq table num_freqs={freq_values_hz.numel()} != attn num_freqs={num_freqs}")

    # ===== 3. 按 band_edges 聚合到各个频段 =====
    num_bands = len(band_edges) - 1
    band_energy_list = []

    for i in range(num_bands):
        high = band_edges[i]
        low = band_edges[i + 1]

        # (low, high] 区间
        mask = (freq_values_hz <= high) & (freq_values_hz > low)  # [num_freqs]
        if mask.sum() == 0:
            band_energy_list.append(torch.zeros(B, device=device))
        else:
            band_energy_list.append(attn[:, mask].sum(dim=-1))  # [B]

    band_energy = torch.stack(band_energy_list, dim=-1)  # [B, K]
    band_prob = band_energy / (band_energy.sum(dim=-1, keepdim=True) + eps)  # [B, K]

    # ===== 4. 目标频段分布 =====
    if target_band_dist is None:
        # 默认均匀分布
        target = torch.full((num_bands,), 1.0 / num_bands, device=device)
        target = target.view(1, -1).expand_as(band_prob)  # [B, K]
    else:
        if target_band_dist.dim() == 1:
            # [K] -> 归一化后扩展到 [B,K]
            target = target_band_dist.to(device)
            target = target / (target.sum() + eps)
            target = target.view(1, -1).expand_as(band_prob)  # [B, K]
        elif target_band_dist.dim() == 2:
            # [B,K] -> 每个样本自己的分布
            target = target_band_dist.to(device)
            target = target / (target.sum(dim=-1, keepdim=True) + eps)
            if target.shape != band_prob.shape:
                raise ValueError(
                    f"target_band_dist shape {target.shape} != band_prob {band_prob.shape}"
                )
        else:
            raise ValueError(f"target_band_dist dim={target_band_dist.dim()} not supported.")

    # ===== 5. MSE 约束，先得到 per-sample loss [B] =====
    # diff: [B, K]
    diff = band_prob - target
    per_sample = (diff ** 2).mean(dim=-1)  # [B]，对频段 K 维取均值

    if reduction == "none":
        # 返回 [B]，方便外部用 wA / wB 做样本加权
        return per_sample
    elif reduction == "mean":
        # 保持原来行为（一个标量）
        return per_sample.mean()
    else:
        raise ValueError(f"Unknown reduction type: {reduction}")



# 计算注意力集中损失
def compute_attention_focus_loss(freq_attn, top_k=3):
    """
    计算注意力集中损失，鼓励注意力集中在前 top_k 个频率点
    freq_attn: [B, T, F] 或 [B, F]
    """
    if freq_attn is None:
        return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')

    # 处理时间维度
    if freq_attn.dim() == 3:
        freq_attn = freq_attn.mean(dim=1)  # [B, F]
    freq_attn = torch.softmax(freq_attn, dim=-1)

    topk_values, _ = torch.topk(freq_attn, k=min(top_k, freq_attn.shape[-1]), dim=-1)
    focus_loss = 1.0 - topk_values.sum(dim=-1).mean()
    return focus_loss



class Trainer_ThreeSystem_1117(BaseTrainer):
    """
    Trainer_ThreeSystem_1117

    Pairwise training for the three-system model with:
    - voltage regression (physics-based voltage),
    - ion-type classification,
    - concentration regression,
    - rule-space prototype regression,
    - hierarchical group / sub-class classification (decision-tree like),
    - consistency constraints between paired samples (A, B),
    - monotonic decay constraints on physical parameters across time,
    - polarity constraints between anode / cathode exchange current densities.

    Total loss (per batch of pairs) is:

        L_total =
            α * (voltage loss for A + B)
          + β * (classification loss for A + B)
          + γ * (concentration loss for A + B)
          + λ_consistency * consistency_loss(A → B)
          + λ_rule        * rule_space_loss(A, B)
          + λ_group       * group_loss(A, B)
          + λ_tree        * tree_loss(A, B)
          + λ_polarity    * polarity_loss(A)
          + λ_monodec     * monotonicity_loss(A → B)
    """

    def __init__(
        self,
        model,
        optimizer,
        device,
        model_save_folder,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.1,
        lambda_rule: float = 0.01,
        lambda_group: float = 0.01,
        lambda_tree: float = 0.01,
        save_every: int = 100,
        lambda_band: float = 0.01,
        label_smoothing=0.1,

    ) -> None:
        """
        Args
        ----
        model : nn.Module
            Three-system model (1117 version), must output:
            prob, predV, conc, wuxing, cls_logits, rawProb, influence,
            paramAttn, freqAttn, freqAttnParam, rule, group_logits,
            high3_logits, low3_logits.
        optimizer : torch.optim.Optimizer
            Optimizer for all model parameters.
        device : torch.device or str
            Device used for training.
        model_save_folder : str or Path
            Directory to save model checkpoints and loss curves.
        alpha, beta, gamma : float
            Weights for voltage, classification, and concentration losses.
        lambda_rule : float
            Weight of rule-space regression loss (towards ion prototypes).
        lambda_group : float
            Weight of binary high/low group classification loss.
        lambda_tree : float
            Weight of intra-group 3-way classification loss (high3 + low3).
        save_every : int
            Save model and loss history every `save_every` epochs.
        """
        super().__init__(model, optimizer, device, model_save_folder, save_every=save_every)

        # Loss weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_rule = lambda_rule
        self.lambda_group = lambda_group
        self.lambda_tree = lambda_tree
        self.lambda_band = lambda_band

        # Classification and regression losses
        # 主分类（ion label）用 per-sample 形式，方便按阶段加权
        self.classify_loss_fn = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction='none'
        )


        self.mse_loss_fn = nn.MSELoss()
        self.mse_loss_fn_mean = nn.MSELoss(reduction='mean') # 用于需要整体平均的场景
        self.mse_loss_fn_none = nn.MSELoss(reduction='none') # 用于需要 per-sample 的场景

        # ====== Mapping from label index to ion type (example) ======

        # high_group_ids: ions in "high influence" group
        # low_group_ids : ions in "low influence" group
        # no_ion_id     : background / clean condition
        self.high_group_ids = torch.tensor([0, 1, 2], dtype=torch.long)  # example: Ca, Na, Ni
        self.low_group_ids = torch.tensor([3, 4, 5], dtype=torch.long)   # example: Fe, Cr, Cu
        self.no_ion_id = 6
        self.global_step = 0        # 用来在 add_scalar 里做 step

        # 1) 加载 band target
        band_yaml_path = Path(__file__).resolve().parent / "ion_band_targets.yaml"
        if band_yaml_path.exists():
            with open(band_yaml_path, "r", encoding="utf-8") as f:
                band_cfg = yaml.safe_load(f)

            ion_order_plus = band_cfg["ion_order_plus"]
            band_edges = band_cfg["band_edges"]
            targets_dict = band_cfg["targets"]

            # 和模型中 ion_order_plus 的顺序对齐
            self.band_edges = band_edges
            self.band_targets = []
            for ion in ion_order_plus:
                self.band_targets.append(targets_dict[ion])
            self.band_targets = torch.tensor(self.band_targets, dtype=torch.float32, device=device)  # (7, 6)
        else:
            self.band_edges = [20000, 1000, 100, 10, 1, 0.1, 0.0]
            self.band_targets = None

    # =========================================================================
    # Pair-wise training with rule & hierarchical tree structure
    # =========================================================================
    def compute_loss_pairs(
        self,
        batchA,
        batchB,
        dummy_mask=None,
        lambda_consistency: float = 1.0,
        eps: float = 1e-9,
        use_log_space: bool = True,
        lambda_monodec: float = 0.2,
        lambda_polarity: float = 0.5,
        use_conc_flagA: torch.Tensor = None,
        use_conc_flagB: torch.Tensor = None,
        stage_idA: torch.Tensor = None,
        stage_idB: torch.Tensor = None,
        weight_ratio=5
    ):
        """
        Compute loss for a pair of samples (A, B).


        Loss components:
        - voltage loss (MSE on predicted vs true voltage)
        - classification loss (ion type)
        - concentration loss
        - rule-space regression towards ion prototypes
        - group (high/low) classification
        - 3-way classification within high / low groups
        - consistency loss: physical parameters from A vs B (time ordering)
        - polarity loss: enforce i0_ca > i0_an (in log space)
        - monotonic decay loss: enforce physical parameters in B <= A (with slack)
        """
        # ------------------------------------------------------------------
        # 1) Forward pass for sample A
        # ------------------------------------------------------------------
        (
            probA,         # final probabilities (after softmax)                 (B, 7)
            predVA,        # predicted voltage sequence                          (B, T)
            concA,         # predicted concentration (regression or class logits)
            wuxingA,       # 5 physical parameters: (sigma_mem, alpha_ca, alpha_an, i0_ca, i0_an)
            clsA,          # not used directly here, reserved
            rawProbA,      # pre-softmax logits for ion classification           (B, 7)
            influenceA,    # 5 behavioral influence factors: (psi, theta_ca, theta_an, phi_ca, phi_an)
            paramAttnA,    # attention over physical parameters (unused here)
            freqAttnA,     # attention over frequencies (unused here)
            freqAttnParamA,# cross attention between freq and params (unused here)
            ruleA,         # rule-space embedding output                          (B, D_rule)
            groupA,        # logits for high/low group (2-way)                    (B, 2)
            high3A,        # logits for 3-way class inside high group             (B, 3)
            low3A,         # logits for 3-way class inside low group              (B, 3)
        ) = self.model(
            batchA[0].to(self.device),  # voltage data
            batchA[1].to(self.device),  # impedance data
            batchA[2].to(self.device),  # environment parameters
            batchA[5].to(self.device),  #electrolyzer parameters
            batchA[6].to(self.device),  # concentration label / auxiliary
        )

        """
        以下摘自forward
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
        """

        # ------------------------------------------------------------------
        # 2) Forward pass for sample B
        # ------------------------------------------------------------------
        (
            probB,
            predVB,
            concB,
            wuxingB,
            clsB,
            rawProbB,
            influenceB,
            paramAttnB,
            freqAttnB,
            freqAttnParamB,
            ruleB,
            groupB,
            high3B,
            low3B,
        ) = self.model(
            batchB[0].to(self.device),
            batchB[1].to(self.device),
            batchB[2].to(self.device),
            batchB[5].to(self.device),
            batchB[6].to(self.device),
        )

        # ------------------------------------------------------------------
        # 3) Prepare ground-truth labels / targets
        # ------------------------------------------------------------------
        labelA = batchA[3].to(self.device)                     # ion label for A       (B,)
        labelB = batchB[3].to(self.device)                     # ion label for B       (B,)
        trueVA = batchA[4].to(self.device).squeeze(-1)         # true voltage A        (B, T)
        trueVB = batchB[4].to(self.device).squeeze(-1)         # true voltage B        (B, T)
        conc_trueA = batchA[6].to(self.device)                 # true conc A           (B, ...)
        conc_trueB = batchB[6].to(self.device)                 # true conc B           (B, ...)
        # ------------------------------------------------------------------
        # 3.5) 阶段权重：剧烈变化阶段样本权重大，缓慢阶段样本权重小
        # ------------------------------------------------------------------
        if stage_idA is None or stage_idB is None:
            # 如果没传则默认全 1（兼容性考虑）
            wA = torch.ones_like(labelA, dtype=torch.float32, device=self.device)
            wB = torch.ones_like(labelB, dtype=torch.float32, device=self.device)
        else:
            stage_idA = stage_idA.to(self.device).view(-1)  # (B,)
            stage_idB = stage_idB.to(self.device).view(-1)  # (B,)

            # 例如：剧烈阶段权重 2.0，缓慢阶段权重 1.0，可按需要调大/调小
            base_weight = 1
            rapid_weight = weight_ratio*base_weight
            

            wA = torch.where(
                stage_idA == 1,
                torch.full_like(stage_idA, rapid_weight, dtype=torch.float32),
                torch.full_like(stage_idA, base_weight, dtype=torch.float32),
            ).to(self.device)

            wB = torch.where(
                stage_idB == 1,
                torch.full_like(stage_idB, rapid_weight, dtype=torch.float32),
                torch.full_like(stage_idB, base_weight, dtype=torch.float32),
            ).to(self.device)

        # pair 级别物理约束用的权重：A/B 谁是剧烈阶段按谁
        w_pair = torch.max(wA, wB)  # (B,)
        # ------------------------------------------------------------------
        # 4) Base losses: voltage / classification / concentration
        # ------------------------------------------------------------------
        # --- 电压损失：先得到 per-sample loss，再按阶段权重加权 ---
        # self.mse_loss_fn_none(predVA, trueVA) 形状为 (B, T)，对时间维取均值 → (B,)
        loss_voltageA_raw = self.mse_loss_fn_none(predVA, trueVA).mean(dim=1)  # (B,)
        loss_voltageB_raw = self.mse_loss_fn_none(predVB, trueVB).mean(dim=1)  # (B,)

        # 不再乘 wA / wB，任何阶段都必须满足电压拟合
        loss_voltageA = loss_voltageA_raw.mean()
        loss_voltageB = loss_voltageB_raw.mean()

        # --- 分类损失：per-sample CE，加阶段权重 ---
        ceA_raw = self.classify_loss_fn(rawProbA, labelA)  # (B,)
        ceB_raw = self.classify_loss_fn(rawProbB, labelB)  # (B,)

        loss_classA = (wA * ceA_raw).mean()
        loss_classB = (wB * ceB_raw).mean()

        # --- 浓度损失：使用 reduction='none' 获取每样本损失 ---
        # loss_concA_raw 形状: (B, 1)
        loss_concA_raw = self.mse_loss_fn_none(concA, conc_trueA)  # (B, 1)
        loss_concB_raw = self.mse_loss_fn_none(concB, conc_trueB)  # (B, 1)
        
        # 1. ppm 生效的样本才参与浓度 loss
        base_weightA = use_conc_flagA.float().to(self.device).view(-1, 1) * self.gamma  # (B, 1)
        base_weightB = use_conc_flagB.float().to(self.device).view(-1, 1) * self.gamma  # (B, 1)

        # 2. 叠加阶段权重（剧烈阶段浓度更重要）
        stage_weightA = wA.view(-1, 1)  # (B, 1)
        stage_weightB = wB.view(-1, 1)  # (B, 1)

        conc_sample_weightA = base_weightA * stage_weightA  # (B, 1)
        conc_sample_weightB = base_weightB * stage_weightB  # (B, 1)

        weighted_loss_concA = loss_concA_raw * conc_sample_weightA
        weighted_loss_concB = loss_concB_raw * conc_sample_weightB
        
        loss_concA = weighted_loss_concA.mean()
        loss_concB = weighted_loss_concB.mean()


        # 最终的 conc_term 已经是加权且平均过的
        conc_term = loss_concA + loss_concB


        
        
        loss_base = (
            self.alpha * (loss_voltageA + loss_voltageB)
            + self.beta * (loss_classA + loss_classB)
            + conc_term
        )


                # ------------------------------------------------------------------
        # 5) Rule-space regression loss (A / B)
        #     只在剧烈阶段 (stage == 1) 的样本上计算，其余阶段忽略
        # ------------------------------------------------------------------
        ion_rule_proto = self.model.ion_rule_proto.to(self.device)

        # A: rule-space regression towards prototype of true label
        rule_targetA = ion_rule_proto[labelA]          # (B, D_rule)
        rule_errA = self.mse_loss_fn_none(ruleA, rule_targetA).mean(dim=1)  # (B,)

        # B: rule-space regression
        rule_targetB = ion_rule_proto[labelB]          # (B, D_rule)
        rule_errB = self.mse_loss_fn_none(ruleB, rule_targetB).mean(dim=1)  # (B,)

        # === 根据 stage 决定哪些样本参与 rule loss ===
        if (stage_idA is None) or (stage_idB is None):
            # 兼容性：如果没传 stage，就退化为“全样本平均”
            loss_ruleA = rule_errA.mean()
            loss_ruleB = rule_errB.mean()
        else:
            # 只保留 stage == 1 的样本
            rapid_maskA = (stage_idA == 1)   # (B,) bool
            rapid_maskB = (stage_idB == 1)   # (B,) bool

            if rapid_maskA.any():
                # 只在剧烈阶段样本上取平均
                loss_ruleA = rule_errA[rapid_maskA].mean()
            else:
                # 当前 batch 没有剧烈阶段的 A 样本，则该项为 0
                loss_ruleA = torch.tensor(0.0, device=self.device)

            if rapid_maskB.any():
                loss_ruleB = rule_errB[rapid_maskB].mean()
            else:
                loss_ruleB = torch.tensor(0.0, device=self.device)

        loss_rule = loss_ruleA + loss_ruleB



                # ------------------------------------------------------------------
        # 6) Hierarchical group / sub-class losses (A / B)  —— 需要按阶段权重 wA, wB 加权
        # ------------------------------------------------------------------
        high_ids = self.high_group_ids.to(self.device)   # ion indices in "high" group
        low_ids = self.low_group_ids.to(self.device)     # ion indices in "low" group
        no_ion_id = self.no_ion_id                       # index for "no ion" / background

        # ===== A-side: group (high vs low) classification =====
        is_high_A = (labelA.unsqueeze(1) == high_ids.unsqueeze(0)).any(dim=1)
        is_low_A = (labelA.unsqueeze(1) == low_ids.unsqueeze(0)).any(dim=1)
        valid_group_A = (is_high_A | is_low_A) & (labelA != no_ion_id)

        if valid_group_A.any():
            # group_targetsA: 0 for low, 1 for high
            group_targetsA = torch.zeros_like(labelA)
            group_targetsA[is_high_A] = 1

            idxA = valid_group_A.nonzero(as_tuple=False).squeeze(1)   # 有效样本的索引
            logitsA = groupA[idxA]                                    # (N_valid, 2)
            targetsA = group_targetsA[idxA]                           # (N_valid,)
            weightsA = wA[idxA].float()                               # (N_valid,)

            # classify_loss_fn: reduction='none'，得到 (N_valid,) 的 per-sample loss
            per_sample_groupA = self.classify_loss_fn(logitsA, targetsA)  # (N_valid,)
            loss_groupA = (weightsA * per_sample_groupA).mean()
        else:
            loss_groupA = torch.tensor(0.0, device=self.device)

        # ===== A-side: 3-way classification inside high group =====
        if is_high_A.any():
            idx_highA = is_high_A.nonzero(as_tuple=False).squeeze(1)  # (N_high,)
            high_ids_expanded = high_ids.unsqueeze(0)                 # (1, 3)
            labels_highA = labelA[idx_highA].unsqueeze(1)             # (N_high, 1)
            # Find index k such that labels_highA == high_ids[k]
            high_targets3A = (labels_highA == high_ids_expanded).nonzero(as_tuple=False)[:, 1]
            logits_high3A = high3A[idx_highA]                         # (N_high, 3)
            weights_highA = wA[idx_highA].float()                     # (N_high,)

            per_sample_high3A = self.classify_loss_fn(logits_high3A, high_targets3A)  # (N_high,)
            loss_high3A = (weights_highA * per_sample_high3A).mean()
        else:
            loss_high3A = torch.tensor(0.0, device=self.device)

        # ===== A-side: 3-way classification inside low group =====
        if is_low_A.any():
            idx_lowA = is_low_A.nonzero(as_tuple=False).squeeze(1)    # (N_low,)
            low_ids_expanded = low_ids.unsqueeze(0)                   # (1, 3)
            labels_lowA = labelA[idx_lowA].unsqueeze(1)               # (N_low, 1)
            low_targets3A = (labels_lowA == low_ids_expanded).nonzero(as_tuple=False)[:, 1]
            logits_low3A = low3A[idx_lowA]                            # (N_low, 3)
            weights_lowA = wA[idx_lowA].float()                       # (N_low,)

            per_sample_low3A = self.classify_loss_fn(logits_low3A, low_targets3A)  # (N_low,)
            loss_low3A = (weights_lowA * per_sample_low3A).mean()
        else:
            loss_low3A = torch.tensor(0.0, device=self.device)

        # ===== B-side: group (high vs low) classification =====
        is_high_B = (labelB.unsqueeze(1) == high_ids.unsqueeze(0)).any(dim=1)
        is_low_B = (labelB.unsqueeze(1) == low_ids.unsqueeze(0)).any(dim=1)
        valid_group_B = (is_high_B | is_low_B) & (labelB != no_ion_id)

        if valid_group_B.any():
            group_targetsB = torch.zeros_like(labelB)
            group_targetsB[is_high_B] = 1

            idxB = valid_group_B.nonzero(as_tuple=False).squeeze(1)
            logitsB = groupB[idxB]
            targetsB = group_targetsB[idxB]
            weightsB = wB[idxB].float()

            per_sample_groupB = self.classify_loss_fn(logitsB, targetsB)
            loss_groupB = (weightsB * per_sample_groupB).mean()
        else:
            loss_groupB = torch.tensor(0.0, device=self.device)

        # ===== B-side: 3-way classification inside high group =====
        if is_high_B.any():
            idx_highB = is_high_B.nonzero(as_tuple=False).squeeze(1)
            high_ids_expanded = high_ids.unsqueeze(0)
            labels_highB = labelB[idx_highB].unsqueeze(1)
            high_targets3B = (labels_highB == high_ids_expanded).nonzero(as_tuple=False)[:, 1]
            logits_high3B = high3B[idx_highB]
            weights_highB = wB[idx_highB].float()

            per_sample_high3B = self.classify_loss_fn(logits_high3B, high_targets3B)
            loss_high3B = (weights_highB * per_sample_high3B).mean()
        else:
            loss_high3B = torch.tensor(0.0, device=self.device)

        # ===== B-side: 3-way classification inside low group =====
        if is_low_B.any():
            idx_lowB = is_low_B.nonzero(as_tuple=False).squeeze(1)
            low_ids_expanded = low_ids.unsqueeze(0)
            labels_lowB = labelB[idx_lowB].unsqueeze(1)
            low_targets3B = (labels_lowB == low_ids_expanded).nonzero(as_tuple=False)[:, 1]
            logits_low3B = low3B[idx_lowB]
            weights_lowB = wB[idx_lowB].float()

            per_sample_low3B = self.classify_loss_fn(logits_low3B, low_targets3B)
            loss_low3B = (weights_lowB * per_sample_low3B).mean()
        else:
            loss_low3B = torch.tensor(0.0, device=self.device)

        loss_group = loss_groupA + loss_groupB
        loss_tree = loss_high3A + loss_low3A + loss_high3B + loss_low3B


        # ------------------------------------------------------------------
        # 7) Consistency loss between (A, B) on physical parameters
        # ------------------------------------------------------------------
        def stack_params(wuxing):
            """Stack 5 physical parameters into a single tensor: (B, 5)."""
            sigma_mem, alpha_ca, alpha_an, i0_ca, i0_an = wuxing
            return torch.cat([sigma_mem, alpha_ca, alpha_an, i0_ca, i0_an], dim=1)

        def stack_infl(influence):
            """Stack 5 behavioral influence factors into a single tensor: (B, 5)."""
            psi, theta_ca, theta_an, phi_ca, phi_an = influence
            return torch.cat([psi, theta_ca, theta_an, phi_ca, phi_an], dim=1)

        pA = stack_params(wuxingA)       # (B, 5) physical parameters at time A
        mA = stack_infl(influenceA)      # (B, 5) influence factors at time A
        pB = stack_params(wuxingB)       # (B, 5) physical parameters at time B

        psi, theta_ca, theta_an, phi_ca, phi_an = mA.chunk(5, dim=1)
        sigma, a_ca, a_an, i0_ca, i0_an = pA.chunk(5, dim=1)

        if dummy_mask is None:
            # dummy_mask identifies padded / invalid pairs that are ignored in consistency loss
            dummy_mask = torch.zeros(pA.size(0), dtype=torch.bool, device=pA.device)

        if use_log_space:
            # Consistency in log-space: L(σ) + L(1-ψ) etc. vs log(pB)
            def clamp01(x):
                return torch.clamp(x, min=eps, max=1.0 - eps)

            L = lambda x: torch.log(torch.clamp(x, min=eps))

            lhs = torch.cat(
                [
                    L(sigma) + L(clamp01(1.0 - psi)),
                    L(a_ca) + L(clamp01(1-theta_ca)),
                    L(a_an) + L(clamp01(1-theta_an)),
                    L(i0_ca) + L(clamp01(1.0 - phi_ca)),
                    L(i0_an) + L(clamp01(1.0 - phi_an)),
                ],
                dim=1,
            )
            rhs = torch.log(torch.clamp(pB, min=eps))
            per_sample = ((lhs - rhs) ** 2).mean(dim=1)  # (B,)
        else:
            # Consistency in original space
            pA_next_pred = torch.cat(
                [
                    sigma * (1.0 - psi),
                    a_ca * (1-theta_ca),
                    a_an * (1-theta_an),
                    i0_ca * (1.0 - phi_ca),
                    i0_an * (1.0 - phi_an),
                ],
                dim=1,
            )
            per_sample = ((pA_next_pred - pB) ** 2).mean(dim=1)

        valid = (~dummy_mask).float()  # (B,)
        valid_float = valid.float()               # (B,)
        # 不再使用 w_pair，只对有效样本做平均
        denom = torch.clamp(valid_float.sum(), min=1.0)
        loss_consistency = (per_sample * valid_float).sum() / denom

                # ------------------------------------------------------------------
        # 7.5) Frequency-band attention regularization —— 区分阶段，乘 wA / wB
        # ------------------------------------------------------------------
        band_edges = [20000., 1000., 100., 10., 1., 0.1, 0.0]

        if getattr(self, "band_targets", None) is not None:
            # per-ion 目标频段分布
            targetA = self.band_targets[labelA]   # [B, K]
            targetB = self.band_targets[labelB]   # [B, K]

            # 这里返回 per-sample loss: [B]
            band_loss_A_raw = compute_band_attention_loss(
                freqAttnA,
                band_edges=band_edges,
                target_band_dist=targetA,
                reduction="none",
            )  # [B]
            band_loss_B_raw = compute_band_attention_loss(
                freqAttnB,
                band_edges=band_edges,
                target_band_dist=targetB,
                reduction="none",
            )  # [B]
        else:
            # 没有显式 target，则用默认均匀分布 / 熵正则等
            band_loss_A_raw = compute_band_attention_loss(
                freqAttnA,
                band_edges=band_edges,
                target_band_dist=None,
                reduction="none",
            )  # [B]
            band_loss_B_raw = compute_band_attention_loss(
                freqAttnB,
                band_edges=band_edges,
                target_band_dist=None,
                reduction="none",
            )  # [B]

        # 按阶段权重做样本加权：剧烈阶段 w 大，缓慢阶段 w 小
        band_loss_A = (wA * band_loss_A_raw).mean()
        band_loss_B = (wB * band_loss_B_raw).mean()
        loss_band = band_loss_A + band_loss_B



        # ------------------------------------------------------------------
        # 8) Combine base + consistency + rule + group + tree losses
        # ------------------------------------------------------------------
        total = (
            loss_base
            + lambda_consistency * loss_consistency
            + self.lambda_rule * loss_rule
            + self.lambda_group * loss_group
            + self.lambda_tree * loss_tree
        )

                # ------------------------------------------------------------------
        # 9) Polarity constraint: log(i0_ca) - log(i0_an) ≥ m_margin
        #     Enforces that cathode exchange current density > anode.
        #     —— 需要按 w_pair 加权（谁是剧烈阶段谁权重大）
        # ------------------------------------------------------------------
        m_margin = 2.0
        eps_small = 1e-12

        i0_ca_scalar = pA[:, [3]]  # i0_ca (B,1)
        i0_an_scalar = pA[:, [4]]  # i0_an (B,1)

        delta = torch.log(torch.clamp(i0_ca_scalar, min=eps_small)) - torch.log(
            torch.clamp(i0_an_scalar, min=eps_small)
        )  # (B,1)
        # per-sample penalty (B,)
        per_sample_polarity = torch.relu(m_margin - delta).view(-1)   # (B,)
        loss_i0_polarity = (w_pair * per_sample_polarity).mean()

        total = total + lambda_polarity * loss_i0_polarity

                # ------------------------------------------------------------------
        # 10) Monotonic-decay constraints: parameters at B should not increase vs A
        #     —— 需要按 w_pair 加权
        # ------------------------------------------------------------------
        valid_mask = (~dummy_mask)

        def mono_decrease_penalty(
            xA,
            xB,
            mask,
            stage_weight,
            slack: float = 0.0,
            eps: float = 1e-12,
            use_log: bool = True,
        ):
            """
            Penalty to enforce xB <= xA - slack (monotonic decay).
            If use_log=True, the constraint is applied in log-space.
            按照 stage_weight 对有效样本进行加权平均。
            """
            if not mask.any():
                return torch.tensor(0.0, device=xA.device)

            idx = mask.nonzero(as_tuple=False).squeeze(1)  # 有效样本索引
            xA_valid = xA[idx]
            xB_valid = xB[idx]
            w_valid = stage_weight[idx].float()            # (N_valid,)

            if use_log:
                lA = torch.log(torch.clamp(xA_valid, min=eps))
                lB = torch.log(torch.clamp(xB_valid, min=eps))
                per_sample = torch.relu(lB - lA - slack).view(-1)   # (N_valid,)
            else:
                per_sample = torch.relu(xB_valid - xA_valid - slack).view(-1)

            return (w_valid * per_sample).mean()

        # Different slack values per parameter type
        slack_sigma = 0.0
        slack_alpha = 0.1
        slack_i0 = 0.1

        loss_sigma_dec = mono_decrease_penalty(
            pA[:, [0]], pB[:, [0]], valid_mask, w_pair, slack=slack_sigma, use_log=True
        )
        loss_alpha_ca_dec = mono_decrease_penalty(
            pA[:, [1]], pB[:, [1]], valid_mask, w_pair, slack=slack_alpha, use_log=True
        )
        loss_alpha_an_dec = mono_decrease_penalty(
            pA[:, [2]], pB[:, [2]], valid_mask, w_pair, slack=slack_alpha, use_log=True
        )
        loss_i0_ca_dec = mono_decrease_penalty(
            pA[:, [3]], pB[:, [3]], valid_mask, w_pair, slack=slack_i0, use_log=True
        )
        loss_i0_an_dec = mono_decrease_penalty(
            pA[:, [4]], pB[:, [4]], valid_mask, w_pair, slack=slack_i0, use_log=True
        )

        loss_wuxing_diff = (
            loss_sigma_dec
            + loss_alpha_ca_dec
            + loss_alpha_an_dec
            + loss_i0_ca_dec
            + loss_i0_an_dec
        )
        total = total + lambda_monodec * loss_wuxing_diff

                # ✅ 新增：频段注意力分布约束
        total = total + self.lambda_band * loss_band


        # ------------------------------------------------------------------
        # 11) Collect scalar diagnostics for logging
        # ------------------------------------------------------------------
        log_dict = {
            "total": total.item(),
            "voltage": (loss_voltageA + loss_voltageB).item(),
            "classify": (loss_classA + loss_classB).item(),
            "conc": (loss_concA + loss_concB).item(),
            "consistency": loss_consistency.item(),
            "consist_valid": denom.item(),
            "consist_dummy": dummy_mask.float().sum().item(),
            "loss_i0_polarity": loss_i0_polarity.item(),
            "loss_wuxing_diff": loss_wuxing_diff.item(),
            "rule": loss_rule.item(),
            "group": loss_group.item(),
            "tree": loss_tree.item(),
            "band": loss_band.item(),  # 新增
        }

        return total, log_dict

    # =========================================================================
    # Training loop for pair-wise data (A, B)
    # =========================================================================
    def train_pairs(
        self,
        train_loader,
        num_epochs: int,
        lambda_consistency: float = 1.0,
        eps: float = 1e-9,
        use_log_space: bool = True,
        lambda_monodec: float = 0.2,
        lambda_polarity: float = 0.5,
        weight_ratio=5
        
    ) -> None:
        """
        Training loop using pair-wise data (batchA, batchB, dummy_mask).

        This does not affect the original single-sample `train()`; it is an
        additional routine specialized for consistency and monotonic constraints.

        Args
        ----
        train_loader :
            Iterable of (batchA, batchB, dummy_mask).
        num_epochs : int
            Number of epochs for pair-wise training.
        lambda_consistency, eps, use_log_space, lambda_monodec, lambda_polarity :
            Passed directly to `compute_loss_pairs`.
        """
        for epoch in range(num_epochs):
            self.model.train()
            running = {}

            for batchA, batchB, dummy_mask in train_loader:
                # Move data to device (if tensor-like)
                batchA = [b.to(self.device) if hasattr(b, "to") and callable(b.to) else b for b in batchA]
                batchB = [b.to(self.device) if hasattr(b, "to") and callable(b.to) else b for b in batchB]
                dummy_mask = dummy_mask.to(self.device)

                use_conc_flagA = batchA[7]
                use_conc_flagB = batchB[7]
                # 阶段 id（0 = 缓慢阶段, 1 = 剧烈变化阶段）
                stage_idA = batchA[8]  # (B,)
                stage_idB = batchB[8]  # (B,)

                # Compute loss and per-batch diagnostics
                loss, items = self.compute_loss_pairs(
                    batchA,
                    batchB,
                    dummy_mask=dummy_mask,
                    lambda_consistency=lambda_consistency,
                    eps=eps,
                    use_log_space=use_log_space,
                    lambda_monodec=lambda_monodec,
                    lambda_polarity=lambda_polarity,
                    use_conc_flagA=use_conc_flagA,
                    use_conc_flagB=use_conc_flagB,
                    stage_idA=stage_idA,
                    stage_idB=stage_idB,
                    weight_ratio=weight_ratio
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Accumulate losses for epoch-level averaging
                for k, v in items.items():
                    running[k] = running.get(k, 0.0) + v

            # Epoch-wise average
            avg = {k: v / len(train_loader) for k, v in running.items()}
            self._record_loss(avg)

            # Optional scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # Console print
            msg = " | ".join([f"{k}: {v:.4f}" for k, v in avg.items()])
            print(f"[Pairs] Epoch {epoch + 1}/{num_epochs} | {msg}")

            # Periodic checkpoint saving
            if (epoch + 1) % self.save_every == 0:
                self.save_model(f"{epoch + 1}")
                self.save_losses(f"{epoch + 1}")

        # Final save
        self.save_model("final")
        self.save_losses("final")
