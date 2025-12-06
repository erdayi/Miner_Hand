import json
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import seaborn as sns

'''
    此代码用于分析聚类结果的簇内图片分布,并将结果可视化保存为图片
'''
# 全局字体设置（在seaborn样式设置前配置）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号乱码问题

def analyze_cluster_sizes(json_file_path):
    """分析聚类结果的规模分布"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        cluster_data = json.load(f)
    
    size_distribution = defaultdict(int)
    for cluster_id, samples in cluster_data.items():
        size_distribution[len(samples)] += 1
    
    # 打印统计结果
    print(f"\n分析文件: {os.path.basename(json_file_path)}")
    print("| 簇大小 | 簇数量 |")
    print("|--------|--------|")
    for size in sorted(size_distribution.keys()):
        print(f"| {size:<6} | {size_distribution[size]:<6} |")
    
    return size_distribution

def analyze_directory(directory, prefix, output_dir):
    """批量分析目录下的聚类结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 配置seaborn样式（注意会重置字体设置）
    sns.set(style="whitegrid", palette="pastel")
    
    # 重新应用字体设置（覆盖seaborn的样式重置）
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 获取目标文件列表
    files = [f for f in os.listdir(directory) 
            if f.startswith(prefix) and f.endswith('.json')]
    
    for filename in files:
        filepath = os.path.join(directory, filename)
        size_dist = analyze_cluster_sizes(filepath)
        
        # 可视化配置
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        
        # 从文件名解析参数
        params = filename.replace('clusters_mini_', '').replace('.json', '').split('_')
        try:
            title = (
                f"手部关键点聚类分布\n"
                f"样本数: {params[0]} | 相似度阈值: {params[1]}\n"
                f"最小样本: {params[2].replace('min','')} | XI参数: {params[4]}"
            )
        except IndexError:
            title = f"聚类分布 - {filename.replace('.json', '')}"
        
        # 绘制分布图
        sizes = sorted(size_dist.keys())
        counts = [size_dist[size] for size in sizes]
        
        # 使用seaborn绘制柱状图
        sns.barplot(
            x=sizes, 
            y=counts, 
            ax=ax,
            alpha=0.7,
            linewidth=1,
            edgecolor="navy"
        )
        
        # 图表装饰
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xlabel("聚类包含样本数", fontsize=12)
        ax.set_ylabel("聚类数量", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 保存输出
        output_path = os.path.join(output_dir, f"{filename.replace('.json', '')}_distribution.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='w')
        plt.close()
        print(f"生成可视化: {output_path}")

if __name__ == "__main__":
    # 配置路径参数
    DATA_DIR = "g:/GS/Tencent/Miner/data/HO3D/cluster_data/optics/shuffled"
    RESULT_PREFIX = ""
    OUTPUT_DIR = "g:/GS/Tencent/Miner/data/HO3D/cluster_data/visual_cluster_image_count"
    
    analyze_directory(DATA_DIR, RESULT_PREFIX, OUTPUT_DIR)