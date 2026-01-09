import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 读取 CSV 文件
file_path = '/home/wry/application/ion_detect/scripts/ml_analysis_code/training_loss_epoch_15000.csv'
N = 10000  # 你想要读取的行数
data = pd.read_csv(file_path).head(N)


# 设置绘图
fig, ax1 = plt.subplots(figsize=(10, 6))

# 设置第一个 y 轴 - total 和 voltage_pred
ax1.plot(data.index, data['total'], label='Total Loss', color='tab:blue', linewidth=2)
ax1.plot(data.index, data['voltage_pred'], label='Voltage Prediction Loss', color='tab:orange', linewidth=2)

# 设置第一个 y 轴的标签
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (Total, Voltage Prediction)', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

# 创建第二个 y 轴 - classify 和 concentration_pred
ax2 = ax1.twinx()
ax2.plot(data.index, data['classify'], label='Classification Loss', color='tab:green', linewidth=2)
ax2.plot(data.index, data['concentration_pred'], label='Concentration Prediction Loss', color='tab:red', linewidth=2)

# 设置第二个 y 轴的标签
ax2.set_ylabel('Loss (Classification, Concentration Prediction)', color='black')
ax2.tick_params(axis='y', labelcolor='black')

# 添加图例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')


# 显示图形
plt.title('Training Loss by Epoch')
# 保存图片到当前文件夹
output_path = './training_loss_plot.png'  # 设置保存路径
plt.tight_layout()
plt.savefig(output_path)  # 保存为 PNG 格式
# plt.show()
