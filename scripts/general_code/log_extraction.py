import re
import pandas as pd

# 假设你的log内容存在log_text字符串中
with open("/home/cagalii/Application/train_machine_learning/最新_绘图和输出文件代码_更新位置/20250607c.log", "r") as f:
    log_text = f.read()


# 提取所有数值列表
grad_checks = re.findall(r"Grad check: tensor\(([\d\.eE+-]+)\)", log_text)
sigma_mems = re.findall(r"sigma_mem: ([\d\.eE+-]+)", log_text)
alpha_cas = re.findall(r"alpha_ca: ([\d\.eE+-]+)", log_text)
i_0cas = re.findall(r"i_0ca: ([\d\.eE+-]+)", log_text)

# 找出最长列长度
max_len = max(len(grad_checks), len(sigma_mems), len(alpha_cas), len(i_0cas))

# 补齐短列为None（或np.nan）
def pad_to(lst, length):
    return lst + [None] * (length - len(lst))

data = {
    "Grad Check": pad_to([float(x) for x in grad_checks], max_len),
    "sigma_mem": pad_to([float(x) for x in sigma_mems], max_len),
    "alpha_ca": pad_to([float(x) for x in alpha_cas], max_len),
    "i_0ca": pad_to([float(x) for x in i_0cas], max_len)
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 输出 CSV
df.to_csv("extracted_params.csv", index=False)

# 打印预览
print(df.head())
