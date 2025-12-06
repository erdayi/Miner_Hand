import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mplcursors
import seaborn as sns

'''
    本代码旨在对比传统方法针对DexYCB迷你数据集不同阈值下各个评估指标效果
'''

# 设置 matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义阈值
thresholds = [0.0055, 0.0044, 0.0033, 0.0022, 0.0011]

# 定义各项评估指标的值
ari = [0.0819, 0.1620, 0.3621, 0.5310, 0.7562]
nmi = [0.7926, 0.8590, 0.9213, 0.9566, 0.9828]
homogeneity = [0.6565, 0.7529, 0.8541, 0.9168, 0.9663]
completeness = [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
v_measure = [0.7926, 0.8590, 0.9213, 0.9566, 0.9828]
fmi = [0.2082, 0.2979, 0.4708, 0.6016, 0.7799]
avg_purity = [0.5212, 0.6382, 0.7633, 0.8832, 0.9560]
avg_completeness = [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]

# 创建 DataFrame
data = {
    '阈值': thresholds,
    '调整兰德指数 (ARI)': ari,
    '标准化互信息 (NMI)': nmi,
    '同质性分数': homogeneity,
    '完整性分数': completeness,
    'V-measure分数': v_measure,
    'Fowlkes-Mallows分数': fmi,
    '平均簇纯度': avg_purity,
    '平均类别完整性': avg_completeness
}
df = pd.DataFrame(data)

# 创建画布
plt.figure(figsize=(12, 8))

# 使用 seaborn 的调色板获取颜色列表
colors = sns.color_palette("Set2", len(df.columns[1:]))

# 绘制各项指标的折线图
lines = []
for i, column in enumerate(df.columns[1:]):
    line, = plt.plot(df['阈值'], df[column], marker='o', label=column, color=colors[i], linewidth=2, markersize=8)
    lines.append(line)

# 设置图表标题和坐标轴标签
plt.title('聚类评估指标随阈值变化的折线图', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('阈值', fontsize=16)
plt.xticks(fontsize=12)
plt.ylabel('评估指标值', fontsize=16)
plt.yticks(fontsize=12)

# 设置图例
handles, labels = plt.gca().get_legend_handles_labels()
# 假设这些指标都是越高越好，添加向上箭头
arrow_up = u'\u2191'
new_labels = [label + f' {arrow_up}' for label in labels]
plt.legend(handles, new_labels, loc='best', fontsize=12, frameon=True, fancybox=True, shadow=True)

# 添加数据提示信息
cursor = mplcursors.cursor(lines, hover=True)

@cursor.connect("add")
def on_add(sel):
    index = sel.target.index
    label = sel.artist.get_label()
    x = df['阈值'][index]
    y = df[label][index]
    sel.annotation.set_text(f"{label}\n阈值: {x:.4f}\n指标值: {y:.4f}")
    sel.annotation.get_bbox_patch().set(fc='white', alpha=0.9, edgecolor='gray')
    sel.annotation.set_fontsize(12)

# 去除顶部和右侧边框
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 显示网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 显示图表
plt.tight_layout()
plt.show()