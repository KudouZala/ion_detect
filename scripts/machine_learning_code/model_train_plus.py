import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
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


# ================== 示例子类 ==================

class Trainer_MLP_Only(BaseTrainer):
    def __init__(self, model, optimizer, device, model_save_folder, scheduler=None, class_weights=None, save_every=100):
        super().__init__(model, optimizer, device, model_save_folder, scheduler, save_every)
        self.ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    def compute_loss(self, batch):
        volt_data, impe_data, env_params, labels, _ = batch
        logits, raw_logits = self.model(volt_data, impe_data, env_params=None)
        loss = self.ce_loss_fn(raw_logits, labels)
        return loss, {"total": loss.item(), "ce": loss.item()}


class Trainer_Voltage_CE(BaseTrainer):
    def __init__(self, model, optimizer, device, model_save_folder, alpha=1.0, beta=1.0, save_every=100):
        super().__init__(model, optimizer, device, model_save_folder, save_every=save_every)
        self.alpha = alpha
        self.beta = beta
        self.ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.mse_loss_fn = nn.MSELoss()

    def compute_loss(self, batch):
        volt_data, impe_data, env_params, labels, true_voltages = batch
        prob_output, predicted_voltage, _, _, raw_prob_output, _ = self.model(volt_data, impe_data, env_params)
        loss_voltage = self.mse_loss_fn(predicted_voltage, true_voltages)
        loss_ce = self.ce_loss_fn(raw_prob_output, labels)
        total_loss = self.alpha * loss_voltage + self.beta * loss_ce
        return total_loss, {
            "total": total_loss.item(),
            "voltage": loss_voltage.item(),
            "ce": loss_ce.item()
        }

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


class Trainer_ThreeSystem_plus(BaseTrainer):
    def __init__(self, model, optimizer, device, model_save_folder, alpha=1.0, beta=1.0, gamma=0.1, save_every=100):
        """
        三系统训练器
        Args:
            model: PyTorch模型
            optimizer: 优化器
            device: torch.device
            model_save_folder: 保存路径
            alpha: 电压预测损失权重
            beta: 分类损失权重
            gamma: 注意力集中损失权重
            save_every: 每隔多少epoch保存一次
        """
        super().__init__(model, optimizer, device, model_save_folder, save_every=save_every)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.mse_loss_fn = nn.MSELoss()

    def compute_loss(self, batch):
        """
        计算总损失 = α * 电压损失 + β * 分类损失 + γ * 注意力集中损失
        """
        volt_data, impe_data, env_params, true_labels, true_voltages, electrolyzer_parameters,concentration = batch
        outputs = self.model(volt_data, impe_data, env_params,electrolyzer_parameters,concentration)
        # 解包模型输出
        (prob_output, predicted_voltage, wuxing,cls_attn_mean ,raw_prob_output,influence,param_attn_mean,freq_attn,freq_attn_param)= outputs


        # 1️⃣ 电压损失
        # 去掉多余的维度，保证 shape 一致
        # predicted_voltage = predicted_voltage.squeeze(-1)  # [8, 1]
        true_voltages = true_voltages.squeeze(-1)          # [8, 1]

        loss_voltage = self.mse_loss_fn(predicted_voltage, true_voltages)

        # 2️⃣ 分类损失
        loss_ce = self.ce_loss_fn(raw_prob_output, true_labels)

        # # 3️⃣ 注意力集中损失
        # loss_attn1 = compute_attention_focus_loss(freq_attn, top_k=3)
        # loss_attn2 = compute_attention_focus_loss(freq_attn_param, top_k=3)
        # loss_attn = (loss_attn1 + loss_attn2) / 2.0

        # 总损失
        total_loss = self.alpha * loss_voltage + self.beta * loss_ce 
        # total_loss = self.alpha * loss_voltage + self.beta * loss_ce + self.gamma * loss_attn

        return total_loss, {
            "total": total_loss.item(),
            "voltage": loss_voltage.item(),
            "ce": loss_ce.item(),
            # "attn": loss_attn.item()
        }
    

class Trainer_ThreeSystem(BaseTrainer):
    def __init__(self, model, optimizer, device, model_save_folder, alpha=1.0, beta=1.0, gamma=0.1, save_every=100):
        """
        三系统训练器
        Args:
            model: PyTorch模型
            optimizer: 优化器
            device: torch.device
            model_save_folder: 保存路径
            alpha: 电压预测损失权重
            beta: 分类损失权重
            gamma: 注意力集中损失权重
            save_every: 每隔多少epoch保存一次
        """
        super().__init__(model, optimizer, device, model_save_folder, save_every=save_every)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.mse_loss_fn = nn.MSELoss()

    def compute_loss(self, batch):
        """
        计算总损失 = α * 电压损失 + β * 分类损失 + γ * 注意力集中损失
        """
        volt_data, impe_data, env_params, true_labels, true_voltages, concentrations= batch
        outputs = self.model(volt_data, impe_data, env_params,concentrations)
        # 解包模型输出
        (prob_output, predicted_voltage, wuxing,cls_attn_mean ,raw_prob_output,influence,param_attn_mean,freq_attn,freq_attn_param)= outputs


        # 1️⃣ 电压损失
        # 去掉多余的维度，保证 shape 一致
        # predicted_voltage = predicted_voltage.squeeze(-1)  # [8, 1]
        true_voltages = true_voltages.squeeze(-1)          # [8, 1]

        loss_voltage = self.mse_loss_fn(predicted_voltage, true_voltages)

        # 2️⃣ 分类损失
        loss_ce = self.ce_loss_fn(raw_prob_output, true_labels)

        # # 3️⃣ 注意力集中损失
        # loss_attn1 = compute_attention_focus_loss(freq_attn, top_k=3)
        # loss_attn2 = compute_attention_focus_loss(freq_attn_param, top_k=3)
        # loss_attn = (loss_attn1 + loss_attn2) / 2.0

        # 总损失
        total_loss = self.alpha * loss_voltage + self.beta * loss_ce 
        # total_loss = self.alpha * loss_voltage + self.beta * loss_ce + self.gamma * loss_attn

        return total_loss, {
            "total": total_loss.item(),
            "voltage": loss_voltage.item(),
            "ce": loss_ce.item(),
            # "attn": loss_attn.item()
        }


class Trainer_ThreeSystem_1004(BaseTrainer):
    def __init__(self, model, optimizer, device, model_save_folder, alpha=1.0, beta=1.0, gamma=0.1, save_every=100):
        """
        三系统训练器
        Args:
            model: PyTorch模型
            optimizer: 优化器
            device: torch.device
            model_save_folder: 保存路径
            alpha: 电压预测损失权重
            beta: 分类损失权重
            gamma: 注意力集中损失权重
            save_every: 每隔多少epoch保存一次
        """
        super().__init__(model, optimizer, device, model_save_folder, save_every=save_every)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.classify_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.mse_loss_fn = nn.MSELoss()

    def compute_loss(self, batch):
        """
        计算总损失 = α * 电压损失 + β * 分类损失 + γ * 注意力集中损失
        """
        volt_data, impe_data, env_params, true_labels, true_voltages, electrolyzer_parameters,concentration = batch
        outputs = self.model(volt_data, impe_data, env_params,electrolyzer_parameters,concentration)
        # 解包模型输出
        (prob_output, predicted_voltage, conc_pred,wuxing,cls_attn_mean ,raw_prob_output,influence,param_attn_mean,freq_attn,freq_attn_param)= outputs


        # 1️⃣ 电压损失
        # 去掉多余的维度，保证 shape 一致
        # predicted_voltage = predicted_voltage.squeeze(-1)  # [8, 1]
        true_voltages = true_voltages.squeeze(-1)          # [8, 1]

        loss_voltage = self.mse_loss_fn(predicted_voltage, true_voltages)

        # 2️⃣ 分类损失
        loss_classify = self.classify_loss_fn(raw_prob_output, true_labels)

        # # 3️⃣ 注意力集中损失
        # loss_attn1 = compute_attention_focus_loss(freq_attn, top_k=3)
        # loss_attn2 = compute_attention_focus_loss(freq_attn_param, top_k=3)
        # loss_attn = (loss_attn1 + loss_attn2) / 2.0

        #浓度损失
        # 去掉多余的维度，保证 shape 一致
        # print("concentration.shape:",concentration.shape)
        # print("conc_pred.shape:",conc_pred.shape)
        loss_conc = self.mse_loss_fn(conc_pred, concentration)

        # 2️⃣ 分类损失
        loss_classify = self.classify_loss_fn(raw_prob_output, true_labels)

        # 总损失
        total_loss = self.alpha * loss_voltage + self.beta * loss_classify +self.gamma*loss_conc
        # total_loss = self.alpha * loss_voltage + self.beta * loss_classify + self.gamma * loss_attn

        return total_loss, {
            "total": total_loss.item(),
            "voltage": loss_voltage.item(),
            "classify": loss_classify.item(),
            "conc":loss_conc.item(),
            # "attn": loss_attn.item()
        }
    # 在 Trainer_ThreeSystem_1004 类里新增：
    # 在 model_train.py 的 Trainer_ThreeSystem_1004 类中新增：
    # model_train.py （仅展示 compute_loss_pairs 的函数签名与一致性处的修改）
    #volt_data, impe_data, env_param, label, true_voltage, electrolyzer_parameters,concentration
    def compute_loss_pairs(self, batchA, batchB, dummy_mask=None,
                        lambda_consistency=1.0, eps=1e-9, use_log_space=True,
                        lambda_polarity = 0.5,lambda_monodec = 0.2):
        """
        dummy_mask: [B] 的 bool 张量。True 表示“自配对”，一致性损失应屏蔽。
        """

        # ------- 前向 A / B：注意传 5 个输入 -------
        (probA, predVA, concA, wuxingA, clsA, rawProbA, influenceA, *_ ) = \
            self.model(batchA[0].to(self.device),
                    batchA[1].to(self.device),
                    batchA[2].to(self.device),
                    batchA[5].to(self.device),
                    batchA[6].to(self.device))

        (probB, predVB, concB, wuxingB, clsB, rawProbB, influenceB, *_ ) = \
            self.model(batchB[0].to(self.device),
                    batchB[1].to(self.device),
                    batchB[2].to(self.device),
                    batchB[5].to(self.device),
                    batchB[6].to(self.device))

        # ------- 原有电压/分类/浓度损失（不变） -------
        loss_voltageA = self.mse_loss_fn(predVA, batchA[4].to(self.device).squeeze(-1))
        loss_classA   = self.classify_loss_fn(rawProbA, batchA[3].to(self.device))
        loss_concA    = self.mse_loss_fn(concA, batchA[6].to(self.device))

        loss_voltageB = self.mse_loss_fn(predVB, batchB[4].to(self.device).squeeze(-1))
        loss_classB   = self.classify_loss_fn(rawProbB, batchB[3].to(self.device))
        loss_concB    = self.mse_loss_fn(concB, batchB[6].to(self.device))

        loss_base = self.alpha*(loss_voltageA+loss_voltageB) + \
                    self.beta *(loss_classA  +loss_classB  ) + \
                    self.gamma*(loss_concA   +loss_concB   )

        # ------- 一致性损失（带 mask 的逐样本计算） -------
        def stack_params(wuxing):
            sigma_mem, alpha_ca, alpha_an, i0_ca, i0_an = wuxing
            return torch.cat([sigma_mem, alpha_ca, alpha_an, i0_ca, i0_an], dim=1)  # (B,5)

        def stack_infl(influence):
            psi, theta_ca, theta_an, phi_ca, phi_an = influence
            return torch.cat([psi, theta_ca, theta_an, phi_ca, phi_an], dim=1)     # (B,5)

        pA = stack_params(wuxingA)    # (B,5)
        mA = stack_infl(influenceA)   # (B,5)
        pB = stack_params(wuxingB)    # (B,5)

        psi, theta_ca, theta_an, phi_ca, phi_an = mA.chunk(5, dim=1)
        sigma, a_ca, a_an, i0_ca, i0_an = pA.chunk(5, dim=1)

        if dummy_mask is None:
            dummy_mask = torch.zeros(pA.size(0), dtype=torch.bool, device=pA.device)

        if use_log_space:
            def clamp01(x):  # 限制到 (eps, 1-eps) 防 log(0) 和 log(负)
                return torch.clamp(x, min=eps, max=1.0 - eps)
            L = lambda x: torch.log(torch.clamp(x, min=eps))

            lhs = torch.cat([
                L(sigma)  + L(clamp01(1.0 - psi)),
                L(a_ca)   + L(clamp01(theta_ca)),
                L(a_an)   + L(clamp01(theta_an)),
                L(i0_ca)  + L(clamp01(1.0 - phi_ca)),
                L(i0_an)  + L(clamp01(1.0 - phi_an)),
            ], dim=1)
            rhs = torch.log(torch.clamp(pB, min=eps))

            # 逐样本 MSE（feature 维度取均值），再用 mask 做样本级平均
            per_sample = ((lhs - rhs) ** 2).mean(dim=1)  # [B]
        else:
            pA_next_pred = torch.cat([
                sigma  * (1.0 - psi),
                a_ca   * theta_ca,
                a_an   * theta_an,
                i0_ca  * (1.0 - phi_ca),
                i0_an  * (1.0 - phi_an),
            ], dim=1)
            per_sample = ((pA_next_pred - pB) ** 2).mean(dim=1)  # [B]

        valid = (~dummy_mask).float()  # True→0, False→1 之前先取反
        denom = torch.clamp(valid.sum(), min=1.0)  # 防全是 dummy
        loss_consistency = (per_sample * valid).sum() / denom



        total = loss_base + lambda_consistency * loss_consistency
        
        # === 交换电流“极别快慢”先验：log(i0_ca) - log(i0_an) ≥ m ===
        m_margin = 2.0   # 建议 1.5~3 试验；表示希望至少 ~10^m 数量级差
        eps = 1e-12

        i0_ca = pA[:, [3]]  # 你已有 pA = [σ, α_ca, α_an, i0_ca, i0_an]
        i0_an = pA[:, [4]]

        delta = torch.log(torch.clamp(i0_ca, min=eps)) - torch.log(torch.clamp(i0_an, min=eps))
        loss_i0_polarity = torch.relu(m_margin - delta).mean()
        total += lambda_polarity * loss_i0_polarity
        # ====== 单调下降约束：要求 B 的物性 ≤ A ======
        # 只对“有效相邻对”（非 dummy）施加
        valid_mask = (~dummy_mask)

        def mono_decrease_penalty(xA, xB, mask, slack=0.0, eps=1e-12, use_log=True):
            """
            目标：xB <= xA（允许 slack），否则惩罚。
            use_log=True 时：惩罚 ReLU( log(xB) - log(xA) - slack )
            """
            if not mask.any():
                return torch.tensor(0.0, device=xA.device)
            if use_log:
                lA = torch.log(torch.clamp(xA[mask], min=eps))
                lB = torch.log(torch.clamp(xB[mask], min=eps))
                return torch.relu(lB - lA - slack).mean()
            else:
                return torch.relu(xB[mask] - xA[mask] - slack).mean()

        # pA/pB 列顺序：0=σ_mem, 1=α_ca, 2=α_an, 3=i0_ca, 4=i0_an
        # 建议：σ、i0 用 log 域；α 也可用 log 域（在(0,1)内更稳），给一点松弛
        slack_sigma = 0.0
        slack_alpha = 0.01   # 给很小松弛避免浮点噪声
        slack_i0    = 0.0

        loss_sigma_dec = mono_decrease_penalty(pA[:, [0]], pB[:, [0]], valid_mask, slack=slack_sigma, use_log=True)
        loss_alpha_ca_dec = mono_decrease_penalty(pA[:, [1]], pB[:, [1]], valid_mask, slack=slack_alpha, use_log=True)
        loss_alpha_an_dec = mono_decrease_penalty(pA[:, [2]], pB[:, [2]], valid_mask, slack=slack_alpha, use_log=True)
        loss_i0_ca_dec = mono_decrease_penalty(pA[:, [3]], pB[:, [3]], valid_mask, slack=slack_i0,    use_log=True)
        loss_i0_an_dec = mono_decrease_penalty(pA[:, [4]], pB[:, [4]], valid_mask, slack=slack_i0,    use_log=True)
        loss_wuxing_diff = loss_sigma_dec + loss_alpha_ca_dec + loss_alpha_an_dec + loss_i0_ca_dec + loss_i0_an_dec

          # 先小点，训练稳定后再调
        total = total + lambda_monodec * loss_wuxing_diff


        return total, {
            "total": total.item(),
            "voltage": (loss_voltageA+loss_voltageB).item(),
            "classify": (loss_classA+loss_classB).item(),
            "conc": (loss_concA+loss_concB).item(),
            "consistency": loss_consistency.item(),
            "consist_valid": denom.item(),  # 参与一致性的样本数
            "consist_dummy": dummy_mask.float().sum().item(),
            "loss_i0_polarity":loss_i0_polarity.item(),
            "loss_wuxing_diff":loss_wuxing_diff.item(),
        }

    # === 加到 Trainer_ThreeSystem_1004 类中（与 compute_loss/compute_loss_pairs 同级） ===
    def train_pairs(self, train_loader, num_epochs, lambda_consistency=1.0, eps=1e-9, use_log_space=True,lambda_monodec = 0.2,lambda_polarity = 0.5):
        """
        使用成对数据 (A,B) 的训练循环。调用 compute_loss_pairs。
        不影响原本的单样本 train()。
        """
        for epoch in range(num_epochs):
            self.model.train()
            running = {}

            for batchA, batchB, dummy_mask in train_loader:
                batchA = [b.to(self.device) if hasattr(b, "to") and callable(b.to) else b for b in batchA]
                batchB = [b.to(self.device) if hasattr(b, "to") and callable(b.to) else b for b in batchB]
                dummy_mask = dummy_mask.to(self.device)

                loss, items = self.compute_loss_pairs(
                    batchA, batchB, dummy_mask=dummy_mask,
                    lambda_consistency=lambda_consistency, eps=1e-9, use_log_space=True,
                    lambda_polarity = lambda_monodec, lambda_monodec = lambda_monodec,
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 统计
                for k, v in items.items():
                    running[k] = running.get(k, 0.0) + v

            # Epoch-wise average
            avg = {k: v / len(train_loader) for k, v in running.items()}
            self._record_loss(avg)

            # 记录学习率
            if self.writer is not None:
                lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("lr", lr, self.global_step)

            # Optional scheduler step
            if self.scheduler is not None:
                self.scheduler.step()


            # 打印
            msg = " | ".join([f"{k}: {v:.4f}" for k, v in avg.items()])
            print(f"[Pairs] Epoch {epoch+1}/{num_epochs} | {msg}")

            # （按你的 save_every 逻辑保存）
            if (epoch + 1) % self.save_every == 0:
                self.save_model(f"{epoch+1}")
                self.save_losses(f"{epoch+1}")

        # 结尾保存
        self.save_model("final")
        self.save_losses("final")






class Trainer_MLP_ClassifierOnly(BaseTrainer):
    def __init__(self,
                 model,
                 optimizer,
                 device,
                 model_save_folder,
                 beta: float = 1.0,
                 save_every: int = 100,
                 label_smoothing: float = 0.1,
                 class_weights: torch.Tensor | None = None,
                 ):
        """
        纯分类训练器（只优化分类交叉熵）
        Args:
            model: PyTorch 模型（返回 logits）
            optimizer: 优化器
            device: torch.device
            model_save_folder: 保存目录
            beta: 分类损失系数（默认 1.0，留口子便于和旧脚本对齐）
            save_every: 每多少个 epoch 保存一次
            label_smoothing: CE 的标签平滑系数
            class_weights: 按类别加权的张量，形如 (num_classes,)
        """
        super().__init__(model, optimizer, device, model_save_folder, save_every=save_every)
        self.beta = beta
        self.ce_loss_fn = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None,
            label_smoothing=label_smoothing
        )

    # ---- 可选：稳健地从不同版本的 batch 中提取需要的 6 个字段 ----
    @staticmethod
    def _unpack_batch_for_classification(batch):
        """
        期望顺序（推荐）:
            (volt, impe, env, labels, ep, conc)
        兼容旧顺序:
            (volt, impe, env, labels, true_voltages, ep, conc)
        以及任何包含上述字段且长度>=6的情况（我们只取需要的6个）。
        """
        if len(batch) >= 7:
            # 旧管线: (0:volt,1:impe,2:env,3:labels,4:true_voltages,5:ep,6:conc)
            volt_data, impe_data, env_params, labels, _, ep, conc = batch[:7]
        elif len(batch) == 6:
            # 新管线: (0:volt,1:impe,2:env,3:labels,4:ep,5:conc)
            volt_data, impe_data, env_params, labels, ep, conc = batch
        else:
            raise ValueError(
                f"Batch length must be 6 or 7 for classification-only, but got {len(batch)}."
            )
        return volt_data, impe_data, env_params, labels, ep, conc

    def compute_loss(self, batch):
        """
        只计算分类损失：
            total = β * CE(logits, labels)
        返回:
            total_loss, {"total":..., "ce":..., "acc":...}
        """
        volt_data, impe_data, env_params, true_labels, electrolyzer_parameters, concentration = \
            self._unpack_batch_for_classification(batch)

        # ---- 放到 device（如果上游 DataLoader 没做）----
        volt_data = volt_data.to(self.device)
        impe_data = impe_data.to(self.device)
        env_params = env_params.to(self.device)
        true_labels = true_labels.to(self.device).long()
        electrolyzer_parameters = electrolyzer_parameters.to(self.device)
        concentration = concentration.to(self.device)

        # ---- 前向：模型只需输出 logits ----
        logits = self.model(
            volt_data, impe_data, env_params,
            electrolyzer_parameters, concentration
        )  # (B, num_classes)

        # ---- 分类损失 ----
        loss_ce = self.ce_loss_fn(logits, true_labels)
        total_loss = self.beta * loss_ce

        # ---- 简单指标：top-1 准确率 ----
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == true_labels).float().mean().item()

        return total_loss, {
            "total": total_loss.item(),
            "ce": loss_ce.item(),
            "acc": acc,
        }


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
        writer=None,       

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

        # Classification and regression losses
        self.classify_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.mse_loss_fn = nn.MSELoss()

        # ====== Mapping from label index to ion type (example) ======
        # Assume:
        #   0: Ca2+_ion
        #   1: Na+_ion
        #   2: Fe3+_ion
        #   3: Ni2+_ion
        #   4: Cr3+_ion
        #   5: Cu2+_ion
        #   6: no_ion
        #
        # high_group_ids: ions in "high influence" group
        # low_group_ids : ions in "low influence" group
        # no_ion_id     : background / clean condition
        self.high_group_ids = torch.tensor([0, 1, 3], dtype=torch.long)  # example: Ca, Na, Ni
        self.low_group_ids = torch.tensor([2, 4, 5], dtype=torch.long)   # example: Fe, Cr, Cu
        self.no_ion_id = 6
        self.writer = writer        # ✅ 记录下来
        self.global_step = 0        # 用来在 add_scalar 里做 step

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
    ):
        """
        Compute loss for a pair of samples (A, B).

        Each batch is a tuple/list:
            batchX = [
                volt_data,           # [0] voltage time series (B, T, 1)
                impe_data,           # [1] impedance spectra (B, T, F, 2)
                env_params,          # [2] environment parameters (B, 3) e.g. T, flow, current
                label,               # [3] ion type label (B,)
                true_voltage,        # [4] measured physical voltage (B, T)
                freq_values,         # [5] frequency values (B, F) or similar
                conc_label,          # [6] concentration label (B, ?) e.g. ppm or class
            ]

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
            batchA[5].to(self.device),  # frequency values or freq-related input
            batchA[6].to(self.device),  # concentration label / auxiliary
        )

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
        # 4) Base losses: voltage / classification / concentration
        # ------------------------------------------------------------------
        loss_voltageA = self.mse_loss_fn(predVA, trueVA)
        loss_classA = self.classify_loss_fn(rawProbA, labelA)
        loss_concA = self.mse_loss_fn(concA, conc_trueA)

        loss_voltageB = self.mse_loss_fn(predVB, trueVB)
        loss_classB = self.classify_loss_fn(rawProbB, labelB)
        loss_concB = self.mse_loss_fn(concB, conc_trueB)

        loss_base = (
            self.alpha * (loss_voltageA + loss_voltageB)
            + self.beta * (loss_classA + loss_classB)
            + self.gamma * (loss_concA + loss_concB)
        )

        # ------------------------------------------------------------------
        # 5) Rule-space regression loss (A / B)
        #     The model contains pre-defined ion rule prototypes:
        #     ion_rule_proto[label] ∈ R^D_rule
        # ------------------------------------------------------------------
        ion_rule_proto = self.model.ion_rule_proto.to(self.device)

        # A: rule-space regression towards prototype of true label
        rule_targetA = ion_rule_proto[labelA]          # (B, D_rule)
        loss_ruleA = self.mse_loss_fn(ruleA, rule_targetA)

        # B: rule-space regression
        rule_targetB = ion_rule_proto[labelB]
        loss_ruleB = self.mse_loss_fn(ruleB, rule_targetB)

        loss_rule = loss_ruleA + loss_ruleB

        # ------------------------------------------------------------------
        # 6) Hierarchical group / sub-class losses (A / B)
        # ------------------------------------------------------------------
        high_ids = self.high_group_ids.to(self.device)   # ion indices in "high" group
        low_ids = self.low_group_ids.to(self.device)     # ion indices in "low" group
        no_ion_id = self.no_ion_id                       # index for "no ion" / background

        # ===== A-side: group (high vs low) classification =====
        # Boolean masks for high / low / valid-group samples
        is_high_A = (labelA.unsqueeze(1) == high_ids.unsqueeze(0)).any(dim=1)
        is_low_A = (labelA.unsqueeze(1) == low_ids.unsqueeze(0)).any(dim=1)
        valid_group_A = (is_high_A | is_low_A) & (labelA != no_ion_id)

        if valid_group_A.any():
            # group_targetsA: 0 for low, 1 for high
            group_targetsA = torch.zeros_like(labelA)
            group_targetsA[is_high_A] = 1
            loss_groupA = self.classify_loss_fn(
                groupA[valid_group_A],               # logits for valid ions
                group_targetsA[valid_group_A],       # high(1)/low(0)
            )
        else:
            loss_groupA = torch.tensor(0.0, device=self.device)

        # ===== A-side: 3-way classification inside high group =====
        if is_high_A.any():
            high_ids_expanded = high_ids.unsqueeze(0)                    # (1, 3)
            labels_highA = labelA[is_high_A].unsqueeze(1)               # (B_high, 1)
            # Find index k such that labels_highA == high_ids[k]
            high_targets3A = (labels_highA == high_ids_expanded).nonzero(as_tuple=False)[:, 1]
            loss_high3A = self.classify_loss_fn(
                high3A[is_high_A],  # (B_high, 3)
                high_targets3A,     # (B_high,)
            )
        else:
            loss_high3A = torch.tensor(0.0, device=self.device)

        # ===== A-side: 3-way classification inside low group =====
        if is_low_A.any():
            low_ids_expanded = low_ids.unsqueeze(0)                      # (1, 3)
            labels_lowA = labelA[is_low_A].unsqueeze(1)                 # (B_low, 1)
            low_targets3A = (labels_lowA == low_ids_expanded).nonzero(as_tuple=False)[:, 1]
            loss_low3A = self.classify_loss_fn(
                low3A[is_low_A],   # (B_low, 3)
                low_targets3A,     # (B_low,)
            )
        else:
            loss_low3A = torch.tensor(0.0, device=self.device)

        # ===== B-side: group (high vs low) classification =====
        is_high_B = (labelB.unsqueeze(1) == high_ids.unsqueeze(0)).any(dim=1)
        is_low_B = (labelB.unsqueeze(1) == low_ids.unsqueeze(0)).any(dim=1)
        valid_group_B = (is_high_B | is_low_B) & (labelB != no_ion_id)

        if valid_group_B.any():
            group_targetsB = torch.zeros_like(labelB)
            group_targetsB[is_high_B] = 1
            loss_groupB = self.classify_loss_fn(
                groupB[valid_group_B],
                group_targetsB[valid_group_B],
            )
        else:
            loss_groupB = torch.tensor(0.0, device=self.device)

        # ===== B-side: 3-way classification inside high group =====
        if is_high_B.any():
            high_ids_expanded = high_ids.unsqueeze(0)
            labels_highB = labelB[is_high_B].unsqueeze(1)
            high_targets3B = (labels_highB == high_ids_expanded).nonzero(as_tuple=False)[:, 1]
            loss_high3B = self.classify_loss_fn(
                high3B[is_high_B],
                high_targets3B,
            )
        else:
            loss_high3B = torch.tensor(0.0, device=self.device)

        # ===== B-side: 3-way classification inside low group =====
        if is_low_B.any():
            low_ids_expanded = low_ids.unsqueeze(0)
            labels_lowB = labelB[is_low_B].unsqueeze(1)
            low_targets3B = (labels_lowB == low_ids_expanded).nonzero(as_tuple=False)[:, 1]
            loss_low3B = self.classify_loss_fn(
                low3B[is_low_B],
                low_targets3B,
            )
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
                    L(a_ca) + L(clamp01(theta_ca)),
                    L(a_an) + L(clamp01(theta_an)),
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
                    a_ca * theta_ca,
                    a_an * theta_an,
                    i0_ca * (1.0 - phi_ca),
                    i0_an * (1.0 - phi_an),
                ],
                dim=1,
            )
            per_sample = ((pA_next_pred - pB) ** 2).mean(dim=1)

        valid = (~dummy_mask).float()
        denom = torch.clamp(valid.sum(), min=1.0)
        loss_consistency = (per_sample * valid).sum() / denom

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
        # ------------------------------------------------------------------
        m_margin = 2.0
        eps_small = 1e-12

        i0_ca_scalar = pA[:, [3]]  # i0_ca (B,1)
        i0_an_scalar = pA[:, [4]]  # i0_an (B,1)

        delta = torch.log(torch.clamp(i0_ca_scalar, min=eps_small)) - torch.log(
            torch.clamp(i0_an_scalar, min=eps_small)
        )
        loss_i0_polarity = torch.relu(m_margin - delta).mean()
        total = total + lambda_polarity * loss_i0_polarity

        # ------------------------------------------------------------------
        # 10) Monotonic-decay constraints: parameters at B should not increase vs A
        # ------------------------------------------------------------------
        valid_mask = (~dummy_mask)

        def mono_decrease_penalty(xA, xB, mask, slack: float = 0.0, eps: float = 1e-12, use_log: bool = True):
            """
            Penalty to enforce xB <= xA - slack (monotonic decay).
            If use_log=True, the constraint is applied in log-space.
            """
            if not mask.any():
                return torch.tensor(0.0, device=xA.device)
            if use_log:
                lA = torch.log(torch.clamp(xA[mask], min=eps))
                lB = torch.log(torch.clamp(xB[mask], min=eps))
                return torch.relu(lB - lA - slack).mean()
            else:
                return torch.relu(xB[mask] - xA[mask] - slack).mean()

        # Different slack values per parameter type
        slack_sigma = 0.0
        slack_alpha = 0.01
        slack_i0 = 0.0

        loss_sigma_dec = mono_decrease_penalty(pA[:, [0]], pB[:, [0]], valid_mask, slack=slack_sigma, use_log=True)
        loss_alpha_ca_dec = mono_decrease_penalty(pA[:, [1]], pB[:, [1]], valid_mask, slack=slack_alpha, use_log=True)
        loss_alpha_an_dec = mono_decrease_penalty(pA[:, [2]], pB[:, [2]], valid_mask, slack=slack_alpha, use_log=True)
        loss_i0_ca_dec = mono_decrease_penalty(pA[:, [3]], pB[:, [3]], valid_mask, slack=slack_i0, use_log=True)
        loss_i0_an_dec = mono_decrease_penalty(pA[:, [4]], pB[:, [4]], valid_mask, slack=slack_i0, use_log=True)

        loss_wuxing_diff = (
            loss_sigma_dec
            + loss_alpha_ca_dec
            + loss_alpha_an_dec
            + loss_i0_ca_dec
            + loss_i0_an_dec
        )
        total = total + lambda_monodec * loss_wuxing_diff

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
