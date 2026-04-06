"""
model_new_train.py — 模块化训练器（消融实验用）

根据模型 forward 返回 dict 中存在的 key 自动决定计算哪些损失。
支持梯度裁剪、cosine 学习率调度、阶段权重等。

损失权重全部从 YAML cfg 读取，权重为 0 或 key 不存在时自动跳过。
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import yaml

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
except ImportError:
    accuracy_score = None
    precision_recall_fscore_support = None


class TrainerNew:
    """面向 IonDetectModel（dict 输出）的 pair-wise 训练器。"""

    def __init__(self, model, optimizer, device, model_save_folder, cfg):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.model_save_folder = str(model_save_folder)
        os.makedirs(self.model_save_folder, exist_ok=True)

        tcfg = cfg["train"]["trainer"]

        self.alpha = float(tcfg.get("alpha", 0.0))
        self.beta = float(tcfg.get("beta", 1.0))
        self.gamma = float(tcfg.get("gamma", 0.0))
        self.lambda_rule = float(tcfg.get("lambda_rule", 0.0))
        self.lambda_group = float(tcfg.get("lambda_group", 0.0))
        self.lambda_tree = float(tcfg.get("lambda_tree", 0.0))
        self.lambda_band = float(tcfg.get("lambda_band", 0.0))
        self.save_every = int(tcfg.get("save_every", 10))
        label_smoothing = float(tcfg.get("label_smoothing", 0.1))
        self.grad_clip = float(tcfg.get("grad_clip", 0.0))

        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, reduction="none"
        )
        self.mse_none = nn.MSELoss(reduction="none")

        self.high_group_ids = torch.tensor([0, 1, 2], dtype=torch.long)
        self.low_group_ids = torch.tensor([3, 4, 5], dtype=torch.long)
        self.no_ion_id = 6

        self.loss_history = {}
        self.global_step = 0

        log_dir = os.path.join(self.model_save_folder, "runs")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.eval_dir = os.path.join(self.model_save_folder, "eval_tables")
        os.makedirs(self.eval_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _record(self, items: dict):
        self.global_step += 1
        for k, v in items.items():
            self.loss_history.setdefault(k, []).append(v)
            self.writer.add_scalar(f"loss/{k}", v, self.global_step)

    def save_model(self, tag):
        path = os.path.join(
            self.model_save_folder, f"trained_model_epoch_{tag}.pth"
        )
        torch.save(self.model.state_dict(), path)
        print(f"✅ 模型已保存到 {path}")

    def save_losses(self, tag):
        csv_path = os.path.join(
            self.model_save_folder, f"training_loss_epoch_{tag}.csv"
        )
        pd.DataFrame(self.loss_history).to_csv(csv_path, index=False)
        print(f"📁 损失值已保存到 {csv_path}")

    # ------------------------------------------------------------------
    # 单侧前向 + 损失
    # ------------------------------------------------------------------
    def _forward_one_side(self, batch):
        """对一侧 batch 做前向，返回 (model_out, label, true_voltage, conc_true,
        use_conc_flag, stage_id)。"""
        volt = batch[0].to(self.device)
        impe = batch[1].to(self.device)
        env = batch[2].to(self.device)
        label = batch[3].to(self.device)
        true_v = batch[4].to(self.device).squeeze(-1)
        ep = batch[5].to(self.device)
        conc = batch[6].to(self.device)
        use_conc_flag = batch[7]
        stage_id = batch[8]

        out = self.model(volt, impe, env, ep, conc)
        return out, label, true_v, conc, use_conc_flag, stage_id

    # ------------------------------------------------------------------
    def compute_loss_pairs(
        self,
        batchA, batchB, dummy_mask,
        lambda_consistency=0.0,
        eps=1e-9,
        use_log_space=True,
        lambda_monodec=0.0,
        lambda_polarity=0.0,
        weight_ratio=3.0,
    ):
        outA, labelA, trueVA, concA, flagA, stageA = self._forward_one_side(batchA)
        outB, labelB, trueVB, concB, flagB, stageB = self._forward_one_side(batchB)

        # 阶段权重
        stageA = stageA.to(self.device).view(-1)
        stageB = stageB.to(self.device).view(-1)
        wA = torch.where(
            stageA == 1,
            torch.full_like(stageA, weight_ratio, dtype=torch.float32),
            torch.ones_like(stageA, dtype=torch.float32),
        )
        wB = torch.where(
            stageB == 1,
            torch.full_like(stageB, weight_ratio, dtype=torch.float32),
            torch.ones_like(stageB, dtype=torch.float32),
        )
        w_pair = torch.max(wA, wB)

        log_dict = {}
        total = torch.tensor(0.0, device=self.device)

        # ====== 分类损失 (CE) ======
        ceA = (wA * self.ce_loss(outA["logits"], labelA)).mean()
        ceB = (wB * self.ce_loss(outB["logits"], labelB)).mean()
        cls_term = self.beta * (ceA + ceB)
        total = total + cls_term
        log_dict["classify"] = (ceA + ceB).item()

        # ====== 电压损失 (optional) ======
        if self.alpha > 0 and "pred_voltage" in outA:
            pv_A = outA["pred_voltage"].view(-1)
            pv_B = outB["pred_voltage"].view(-1)
            vA = self.mse_none(pv_A, trueVA.view(-1)).mean()
            vB = self.mse_none(pv_B, trueVB.view(-1)).mean()
            v_term = self.alpha * (vA + vB)
            total = total + v_term
            log_dict["voltage"] = (vA + vB).item()

        # ====== 浓度损失 (optional) ======
        if self.gamma > 0 and "pred_conc" in outA:
            cA_raw = self.mse_none(outA["pred_conc"].view(-1), concA.view(-1))
            cB_raw = self.mse_none(outB["pred_conc"].view(-1), concB.view(-1))
            wc_A = flagA.float().to(self.device).view(-1) * self.gamma
            wc_B = flagB.float().to(self.device).view(-1) * self.gamma
            c_term = (cA_raw * wc_A * wA).mean() + \
                     (cB_raw * wc_B * wB).mean()
            total = total + c_term
            log_dict["conc"] = c_term.item()

        # ====== Consistency 损失 (optional) ======
        if lambda_consistency > 0 and "wuxing" in outA:
            total, consist_val = self._consistency_loss(
                outA, outB, dummy_mask, total, log_dict,
                lambda_consistency, eps, use_log_space,
            )

        # ====== Polarity 约束 (optional) ======
        if lambda_polarity > 0 and "wuxing" in outA:
            pA = self._stack_params(outA["wuxing"])
            m_margin = 2.0
            i0_ca = pA[:, [3]]
            i0_an = pA[:, [4]]
            delta = torch.log(i0_ca.clamp(min=eps)) - torch.log(i0_an.clamp(min=eps))
            pol_loss = (w_pair * torch.relu(m_margin - delta).view(-1)).mean()
            total = total + lambda_polarity * pol_loss
            log_dict["polarity"] = pol_loss.item()

        # ====== Monotonic decay (optional) ======
        if lambda_monodec > 0 and "wuxing" in outA:
            pA = self._stack_params(outA["wuxing"])
            pB = self._stack_params(outB["wuxing"])
            valid = (~dummy_mask.to(self.device)) if dummy_mask is not None else \
                torch.ones(pA.size(0), dtype=torch.bool, device=self.device)
            mono_loss = self._mono_decay(pA, pB, valid, w_pair, eps)
            total = total + lambda_monodec * mono_loss
            log_dict["monodec"] = mono_loss.item()

        # ====== Rule 损失 (optional) ======
        if self.lambda_rule > 0 and "rule_pred" in outA:
            proto = self.model.ion_rule_proto.to(self.device)
            rA = self.mse_none(outA["rule_pred"], proto[labelA]).mean(1)
            rB = self.mse_none(outB["rule_pred"], proto[labelB]).mean(1)
            rapidA = (stageA == 1)
            rapidB = (stageB == 1)
            rule_terms = []
            if rapidA.any():
                rule_terms.append(rA[rapidA].mean())
            if rapidB.any():
                rule_terms.append(rB[rapidB].mean())
            if rule_terms:
                rule_loss = sum(rule_terms)
                total = total + self.lambda_rule * rule_loss
                log_dict["rule"] = rule_loss.item()

        # ====== Group / Tree 损失 (optional) ======
        if self.lambda_group > 0 and "group_logits" in outA:
            g_loss = self._group_loss(outA, labelA, wA) + \
                     self._group_loss(outB, labelB, wB)
            total = total + self.lambda_group * g_loss
            log_dict["group"] = g_loss.item()

        if self.lambda_tree > 0 and "high3_logits" in outA:
            t_loss = self._tree_loss(outA, labelA, wA) + \
                     self._tree_loss(outB, labelB, wB)
            total = total + self.lambda_tree * t_loss
            log_dict["tree"] = t_loss.item()

        log_dict["total"] = total.item()
        return total, log_dict

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    @staticmethod
    def _stack_params(wuxing):
        return torch.cat(wuxing, dim=1)  # (B, 5)

    def _consistency_loss(self, outA, outB, dummy_mask, total, log_dict,
                          lam, eps, use_log):
        pA = self._stack_params(outA["wuxing"])
        mA = self._stack_params(outA["influence"])
        pB = self._stack_params(outB["wuxing"])
        sigma, a_ca, a_an, i0_ca, i0_an = pA.chunk(5, dim=1)
        psi, theta_ca, theta_an, phi_ca, phi_an = mA.chunk(5, dim=1)

        if dummy_mask is None:
            dummy_mask = torch.zeros(pA.size(0), dtype=torch.bool, device=pA.device)
        else:
            dummy_mask = dummy_mask.to(self.device)

        def clamp01(x):
            return x.clamp(min=eps, max=1.0 - eps)

        L = lambda x: torch.log(x.clamp(min=eps))

        if use_log:
            lhs = torch.cat([
                L(sigma) + L(clamp01(1.0 - psi)),
                L(a_ca) + L(clamp01(1.0 - theta_ca)),
                L(a_an) + L(clamp01(1.0 - theta_an)),
                L(i0_ca) + L(clamp01(1.0 - phi_ca)),
                L(i0_an) + L(clamp01(1.0 - phi_an)),
            ], dim=1)
            rhs = L(pB)
            per_sample = ((lhs - rhs) ** 2).mean(dim=1)
        else:
            pred = torch.cat([
                sigma * (1.0 - psi),
                a_ca * (1.0 - theta_ca),
                a_an * (1.0 - theta_an),
                i0_ca * (1.0 - phi_ca),
                i0_an * (1.0 - phi_an),
            ], dim=1)
            per_sample = ((pred - pB) ** 2).mean(dim=1)

        valid = (~dummy_mask).float()
        denom = valid.sum().clamp(min=1.0)
        consist = (per_sample * valid).sum() / denom

        total = total + lam * consist
        log_dict["consistency"] = consist.item()
        return total, consist

    def _mono_decay(self, pA, pB, valid_mask, w_pair, eps):
        if not valid_mask.any():
            return torch.tensor(0.0, device=self.device)
        idx = valid_mask.nonzero(as_tuple=False).squeeze(1)
        lA = torch.log(pA[idx].clamp(min=eps))
        lB = torch.log(pB[idx].clamp(min=eps))
        w = w_pair[idx].float()
        per_param = torch.relu(lB - lA).mean(dim=1)
        return (w * per_param).mean()

    def _group_loss(self, out, labels, weights):
        high_ids = self.high_group_ids.to(self.device)
        low_ids = self.low_group_ids.to(self.device)
        is_high = (labels.unsqueeze(1) == high_ids.unsqueeze(0)).any(dim=1)
        is_low = (labels.unsqueeze(1) == low_ids.unsqueeze(0)).any(dim=1)
        valid = (is_high | is_low) & (labels != self.no_ion_id)
        if not valid.any():
            return torch.tensor(0.0, device=self.device)
        targets = torch.zeros_like(labels)
        targets[is_high] = 1
        idx = valid.nonzero(as_tuple=False).squeeze(1)
        per = self.ce_loss(out["group_logits"][idx], targets[idx])
        return (weights[idx].float() * per).mean()

    def _tree_loss(self, out, labels, weights):
        high_ids = self.high_group_ids.to(self.device)
        low_ids = self.low_group_ids.to(self.device)
        loss = torch.tensor(0.0, device=self.device)

        is_high = (labels.unsqueeze(1) == high_ids.unsqueeze(0)).any(dim=1)
        if is_high.any():
            idx = is_high.nonzero(as_tuple=False).squeeze(1)
            t3 = (labels[idx].unsqueeze(1) == high_ids.unsqueeze(0)).nonzero(
                as_tuple=False
            )[:, 1]
            per = self.ce_loss(out["high3_logits"][idx], t3)
            loss = loss + (weights[idx].float() * per).mean()

        is_low = (labels.unsqueeze(1) == low_ids.unsqueeze(0)).any(dim=1)
        if is_low.any():
            idx = is_low.nonzero(as_tuple=False).squeeze(1)
            t3 = (labels[idx].unsqueeze(1) == low_ids.unsqueeze(0)).nonzero(
                as_tuple=False
            )[:, 1]
            per = self.ce_loss(out["low3_logits"][idx], t3)
            loss = loss + (weights[idx].float() * per).mean()

        return loss

    # ------------------------------------------------------------------
    # 验证
    # ------------------------------------------------------------------
    @staticmethod
    def _confusion_matrix_np(y_true, y_pred, num_classes):
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
                cm[int(t), int(p)] += 1
        return cm

    @staticmethod
    def _normalize_class_names(num_classes, class_names=None):
        if class_names is None:
            return [f"class_{i}" for i in range(num_classes)]
        names = list(class_names)
        if len(names) < num_classes:
            names.extend([f"class_{i}" for i in range(len(names), num_classes)])
        return names[:num_classes]

    def _save_eval_tables(self, epoch, y_true, y_pred, num_classes, class_names=None):
        class_names = self._normalize_class_names(num_classes, class_names)
        labels = list(range(num_classes))
        prec_cls, rec_cls, f1_cls, support_cls = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        cm = self._confusion_matrix_np(y_true, y_pred, num_classes)

        rows = []
        for i in range(num_classes):
            tp = int(cm[i, i])
            fn = int(cm[i, :].sum() - tp)
            fp = int(cm[:, i].sum() - tp)
            support = int(support_cls[i]) if i < len(support_cls) else int(cm[i, :].sum())
            row_wo_diag = cm[i, :].copy()
            row_wo_diag[i] = 0
            top_j = int(np.argmax(row_wo_diag)) if row_wo_diag.sum() > 0 else -1
            top_cnt = int(row_wo_diag[top_j]) if top_j >= 0 else 0
            top_ratio = (top_cnt / support) if support > 0 else 0.0
            rows.append(
                {
                    "epoch": int(epoch),
                    "class_id": i,
                    "class_name": class_names[i],
                    "support": support,
                    "precision": float(prec_cls[i]),
                    "recall": float(rec_cls[i]),
                    "f1": float(f1_cls[i]),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "top_confused_pred_id": top_j if top_j >= 0 else "",
                    "top_confused_pred_name": class_names[top_j] if top_j >= 0 else "",
                    "top_confused_count": top_cnt,
                    "top_confused_ratio": float(top_ratio),
                }
            )

        per_class_path = os.path.join(
            self.eval_dir, f"per_class_metrics_epoch_{int(epoch)}.csv"
        )
        pd.DataFrame(rows).to_csv(per_class_path, index=False, encoding="utf-8-sig")

        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_path = os.path.join(self.eval_dir, f"confusion_matrix_epoch_{int(epoch)}.csv")
        cm_df.to_csv(cm_path, encoding="utf-8-sig")

        pair_rows = []
        for i in range(num_classes):
            for j in range(num_classes):
                if i == j:
                    continue
                cnt = int(cm[i, j])
                if cnt <= 0:
                    continue
                pair_rows.append(
                    {
                        "epoch": int(epoch),
                        "true_id": i,
                        "true_name": class_names[i],
                        "pred_id": j,
                        "pred_name": class_names[j],
                        "count": cnt,
                        "ratio_in_true_class": float(cnt / max(int(cm[i, :].sum()), 1)),
                    }
                )
        pair_rows.sort(key=lambda x: x["count"], reverse=True)
        pair_path = os.path.join(
            self.eval_dir, f"confusion_pairs_epoch_{int(epoch)}.csv"
        )
        pd.DataFrame(pair_rows).to_csv(pair_path, index=False, encoding="utf-8-sig")
        return per_class_path, cm_path, pair_path

    def _eval_val_loader(self, val_loader, num_classes=7):
        if accuracy_score is None:
            return None
        self.model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for batchA, batchB, _ in val_loader:
                batchB = [
                    b.to(self.device) if hasattr(b, "to") and callable(b.to) else b
                    for b in batchB
                ]
                out = self.model(
                    batchB[0], batchB[1], batchB[2], batchB[5], batchB[6]
                )
                pred = out["prob"].argmax(dim=1).cpu().numpy()
                label = batchB[3].cpu().numpy()
                all_pred.append(pred)
                all_true.append(label)
        self.model.train()
        y_pred = np.concatenate(all_pred)
        y_true = np.concatenate(all_true)
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=range(num_classes),
            average="macro", zero_division=0,
        )
        return {
            "acc": float(acc),
            "prec": float(prec),
            "rec": float(rec),
            "f1": float(f1),
            "y_true": y_true,
            "y_pred": y_pred,
        }

    # ------------------------------------------------------------------
    # 训练循环
    # ------------------------------------------------------------------
    def train_pairs(
        self,
        train_loader,
        num_epochs,
        lambda_consistency=0.0,
        eps=1e-9,
        use_log_space=True,
        lambda_monodec=0.0,
        lambda_polarity=0.0,
        weight_ratio=3.0,
        val_loader=None,
        eval_every=10,
        num_classes=7,
        class_names=None,
    ):
        scheduler = None
        if hasattr(self, '_scheduler') and self._scheduler is not None:
            scheduler = self._scheduler

        best_f1 = 0.0

        for epoch in range(num_epochs):
            self.model.train()
            running = {}

            for batchA, batchB, dummy_mask in train_loader:
                batchA = [
                    b.to(self.device) if hasattr(b, "to") and callable(b.to) else b
                    for b in batchA
                ]
                batchB = [
                    b.to(self.device) if hasattr(b, "to") and callable(b.to) else b
                    for b in batchB
                ]
                dummy_mask = dummy_mask.to(self.device)

                loss, items = self.compute_loss_pairs(
                    batchA, batchB, dummy_mask,
                    lambda_consistency=lambda_consistency,
                    eps=eps,
                    use_log_space=use_log_space,
                    lambda_monodec=lambda_monodec,
                    lambda_polarity=lambda_polarity,
                    weight_ratio=weight_ratio,
                )

                self.optimizer.zero_grad()
                loss.backward()

                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.optimizer.step()

                for k, v in items.items():
                    running[k] = running.get(k, 0.0) + v

            avg = {k: v / len(train_loader) for k, v in running.items()}
            self._record(avg)

            if scheduler is not None:
                scheduler.step()

            msg = " | ".join(f"{k}: {v:.4f}" for k, v in avg.items())
            print(f"[New] Epoch {epoch+1}/{num_epochs} | {msg}")

            if val_loader is not None and eval_every > 0 and (epoch + 1) % eval_every == 0:
                metrics = self._eval_val_loader(val_loader, num_classes)
                if metrics is not None:
                    acc = metrics["acc"]
                    prec = metrics["prec"]
                    rec = metrics["rec"]
                    f1 = metrics["f1"]
                    print(f"--- 验证集 [epoch {epoch+1}] ---")
                    print(f"  准确率 (Accuracy):  {acc:.4f}")
                    print(f"  精确率 (Precision): {prec:.4f}")
                    print(f"  召回率 (Recall):    {rec:.4f}")
                    print(f"  F1 分数 (F1):       {f1:.4f}")
                    pc_path, cm_path, cp_path = self._save_eval_tables(
                        epoch=epoch + 1,
                        y_true=metrics["y_true"],
                        y_pred=metrics["y_pred"],
                        num_classes=num_classes,
                        class_names=class_names,
                    )
                    print(f"  📄 每类指标表: {pc_path}")
                    print(f"  📄 混淆矩阵表: {cm_path}")
                    print(f"  📄 错分去向表: {cp_path}")

                    if self.writer:
                        self.writer.add_scalar("val/accuracy", acc, self.global_step)
                        self.writer.add_scalar("val/precision_macro", prec, self.global_step)
                        self.writer.add_scalar("val/recall_macro", rec, self.global_step)
                        self.writer.add_scalar("val/f1", f1, self.global_step)

                    if f1 > best_f1:
                        best_f1 = f1
                        self.save_model("best")
                        print(f"  🏆 新最佳 F1={f1:.4f}，已保存 best 模型")

            if (epoch + 1) % self.save_every == 0:
                self.save_model(f"{epoch+1}")
                self.save_losses(f"{epoch+1}")

        self.save_model("final")
        self.save_losses("final")
        self.writer.close()
