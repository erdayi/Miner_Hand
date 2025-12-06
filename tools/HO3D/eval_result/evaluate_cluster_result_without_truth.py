import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import os

'''
    本代码旨在评估未含聚类真值的乱序HO3D聚类结果各个指标的效果
'''

def extract_frame_info(image_path):
    """从图片路径中提取关键信息"""
    # 从路径中提取序列名和帧号
    parts = image_path.split(os.sep)
    sequence = parts[-3]  # 例如 'GPMF14'
    frame = int(parts[-1].split('.')[0])  # 例如 从 '0490.png' 提取 490
    return sequence, frame

def create_path_to_index_mapping(all_clusters):
    """创建图片路径到索引的映射"""
    path_to_index = {}
    index = 0
    # 遍历所有簇中的所有路径
    for cluster in all_clusters:
        for path in cluster:
            if path not in path_to_index:
                path_to_index[path] = index
                index += 1
    return path_to_index

def load_optics_results(json_path):
    """加载OPTICS聚类结果JSON数据"""
    with open(json_path) as f:
        data = json.load(f)
    return data['clusters']

def evaluate_clustering(distance_matrix, clusters, title):
    """基于距离矩阵的聚类评估方法"""
    # 创建路径到索引的映射
    path_to_index = create_path_to_index_mapping(clusters)
    total_samples = len(path_to_index)

    # 准备标签数组
    labels = -np.ones(total_samples)  # 初始化为-1表示未分类
    
    # 为每个簇分配标签
    for cluster_id, cluster in enumerate(clusters):
        for path in cluster:
            idx = path_to_index[path]
            labels[idx] = cluster_id

    # 提取有效样本
    valid_indices = np.where(labels != -1)[0]
    valid_labels = labels[valid_indices]
    valid_distance = distance_matrix[np.ix_(valid_indices, valid_indices)]

    # 计算簇内和簇间距离统计量
    intra_dists = []
    inter_dists = []
    cluster_means = []

    unique_labels = np.unique(valid_labels)
    for label in unique_labels:
        # 簇内距离
        mask = valid_labels == label
        cluster_dist = valid_distance[mask][:, mask]
        if len(cluster_dist) > 1:
            intra_dists.extend(cluster_dist[np.triu_indices(len(cluster_dist), k=1)])

        # 簇间距离
        other_mask = valid_labels != label
        inter_dist = valid_distance[mask][:, other_mask]
        if len(inter_dist) > 0:
            inter_dists.extend(inter_dist.flatten())

        # 簇中心
        cluster_means.append(np.mean(valid_distance[mask], axis=0))

    # 计算评估指标
    metrics = {
        'Silhouette Score': silhouette_score(valid_distance, valid_labels, metric='precomputed'),
        'Davies-Bouldin Index': davies_bouldin_score(valid_distance, valid_labels),
        'Calinski-Harabasz Index': calinski_harabasz_score(valid_distance, valid_labels),
        'Avg Intra-cluster Distance': np.mean(intra_dists) if intra_dists else 0,
        'Avg Inter-cluster Distance': np.mean(inter_dists) if inter_dists else 0,
        'Cluster Count': len(unique_labels),
        'Classified Samples': len(valid_labels),
        'Unclassified Samples': len(labels) - len(valid_labels)
    }

    # 可视化
    plt.figure(figsize=(18, 6))

    # 1. PCA降维可视化
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(valid_distance)

    plt.subplot(131)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=valid_labels, cmap='viridis', s=10)
    plt.title(f"{title} Clusters (PCA)")

    # 2. 聚类大小分布
    plt.subplot(132)
    cluster_sizes = [len(cluster) for cluster in clusters]
    plt.hist(cluster_sizes, bins=20)
    plt.title(f"{title} Cluster Size Distribution")
    plt.xlabel("Cluster Size")
    plt.ylabel("Count")

    # 3. 指标展示
    plt.subplot(133)
    plt.axis('off')
    metrics_text = "\n".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                              for k, v in metrics.items()])
    plt.text(0.1, 0.5, metrics_text, fontsize=12)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    return metrics

if __name__ == "__main__":
    # 使用示例
    distance_matrix_path = "data/HO3D/load_data/distance_matrix_interval_1_shuffled.npy"
    # optics_results_path = "data/HO3D/cluster_data/optics/shuffled/hand_clusters_optics_interval_1.json"
    optics_results_path = "data/HO3D/cluster_data/origin/shuffled/hand_clusters_0003_interval_1.json"

    # 加载数据
    distance_matrix = np.load(distance_matrix_path)
    optics_results = load_optics_results(optics_results_path)

    # 评估OPTICS聚类
    optics_results_metrics = evaluate_clustering(distance_matrix, optics_results, "OPTICS")

    print("\nOPTICS聚类评估结果:")
    for k, v in optics_results_metrics.items():
        print(f"{k}: {v}")