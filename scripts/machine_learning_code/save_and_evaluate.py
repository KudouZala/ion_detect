import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt
import math
from pathlib import Path
from captum.attr import Saliency, IntegratedGradients
import json
from datasets import Dataset_1,Dataset_2
from models import Model_2
from tests import test_single_xlsx_and_generate_explanations
def save_test_results(model_save_folder, all_predicted_labels, all_true_labels_test, all_predicted_probabilities, class_names=None):
    """
    将测试用例的真实值、预测值和各个预测概率保存到 CSV 文件中。

    Args:
        model_save_folder (str): 模型保存的文件夹路径。
        all_predicted_labels (list or np.ndarray): 所有测试用例的预测标签。
        all_true_labels_test (list or np.ndarray): 所有测试用例的真实标签。
        all_predicted_probabilities (list or np.ndarray): 所有测试用例的预测概率，形状应为 (num_samples, num_classes)。
        class_names (list, optional): 类别名称列表。如果提供，将用作 CSV 文件的列名。默认为 None。
    """
    predicted_labels = np.array(all_predicted_labels)
    true_labels = np.array(all_true_labels_test)
    predicted_probabilities = np.array(all_predicted_probabilities)

    num_samples = predicted_labels.shape[0]
    results_data = {'True Label': true_labels, 'Predicted Label': predicted_labels}

    if predicted_probabilities.ndim > 1:
        num_classes = predicted_probabilities.shape[1]
        probability_columns = [f'Probability_Class_{i}' if class_names is None or i >= len(class_names) else f'Probability_{class_names[i]}' for i in range(num_classes)]
        for i, col_name in enumerate(probability_columns):
            results_data[col_name] = predicted_probabilities[:, i]
    else:
        results_data['Predicted Probability'] = predicted_probabilities

    results_df = pd.DataFrame(results_data)

    output_filepath = os.path.join(model_save_folder, 'test_results.csv')
    results_df.to_csv(output_filepath, index=False)
    print(f"测试结果已保存到: {output_filepath}")

def evaluate(model,model_save_folder,device,test_loader):
    # 在测试集上计算正确率
    model.eval()
    correct = 0
    total = 0
    all_predicted_labels = []
    all_true_labels_test = []
    all_predicted_probabilities = []

    with torch.no_grad():
        for volt_data, impe_data, env_params, true_labels, true_voltages in test_loader:
            volt_data = volt_data.to(device)
            impe_data = impe_data.to(device)
            env_params = env_params.to(device)
            true_labels = true_labels.to(device)
            true_voltages = true_voltages.to(device)
            prob_output, predicted_voltage, wuxing,cls_attn,raw_prob_output,yingxiang = model(volt_data, impe_data, env_params)
            _, predicted = torch.max(prob_output, 1)
            total += true_labels.size(0)
            correct += (predicted == true_labels).sum().item()

            all_predicted_labels.extend(predicted.cpu().numpy())
            all_true_labels_test.extend(true_labels.cpu().numpy())
            all_predicted_probabilities.extend(torch.softmax(prob_output, dim=1).cpu().numpy())

            # 输出每个 batch 的预测结果和正确结果 (可以根据需要注释掉)
            # print("Predicted labels:", predicted)
            # print("True labels:", true_labels)
            # print("Test Batch - Predicted Ion Poisoning Probabilities:")
            # print(torch.softmax(prob_output, dim=1))

    print(f"Test Accuracy: {correct / total * 100}%")
    save_test_results(model_save_folder, all_predicted_labels, all_true_labels_test, all_predicted_probabilities)


def evaluate_mlp_only(model,model_save_folder,device,test_loader):
    # 在测试集上计算正确率
    model.eval()
    correct = 0
    total = 0
    all_predicted_labels = []
    all_true_labels_test = []
    all_predicted_probabilities = []

    with torch.no_grad():
        for volt_data, impe_data, env_params, true_labels, true_voltages in test_loader:
            volt_data = volt_data.to(device)
            impe_data = impe_data.to(device)
            env_params = env_params.to(device)
            true_labels = true_labels.to(device)

            prob_output, raw_logits = model(volt_data, impe_data, env_params=None)
            _, predicted = torch.max(prob_output, 1)
            total += true_labels.size(0)
            correct += (predicted == true_labels).sum().item()

            all_predicted_labels.extend(predicted.cpu().numpy())
            all_true_labels_test.extend(true_labels.cpu().numpy())
            all_predicted_probabilities.extend(torch.softmax(prob_output, dim=1).cpu().numpy())

            # 输出每个 batch 的预测结果和正确结果 (可以根据需要注释掉)
            # print("Predicted labels:", predicted)
            # print("True labels:", true_labels)
            # print("Test Batch - Predicted Ion Poisoning Probabilities:")
            # print(torch.softmax(prob_output, dim=1))

    print(f"Test Accuracy: {correct / total * 100}%")
    save_test_results(model_save_folder, all_predicted_labels, all_true_labels_test, all_predicted_probabilities)