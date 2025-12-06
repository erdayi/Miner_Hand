import numpy as np
import json
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

'''
    本代码旨在评估HO3D迷你测试集未含聚类真值的各个指标的效果
'''


def load_optics_results(json_path):
    """加载OPTICS聚类结果JSON数据"""
    with open(json_path) as f:
        data = json.load(f)
    return data['clusters']  # 只返回clusters部分


def evaluate_clustering(distance_matrix, cluster_result, title):
    """
    基于距离矩阵的聚类评估方法
    参数:
        distance_matrix: 预计算的距离矩阵 (n_samples, n_samples)
        cluster_result: 聚类结果字典 {cluster_id: [sample_indices]}
        title: 聚类方法的标题
    """
    # 准备标签数组
    labels = -np.ones(distance_matrix.shape[0])  # 初始化为-1表示未分类
    for cluster_id, samples in cluster_result.items():
        for sample_idx in samples:
            labels[int(sample_idx)] = int(cluster_id)

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

    return metrics


if __name__ == "__main__":
    # 使用示例
    distance_matrix_path = "data/HO3D/load_data/distance_matrix_interval_5_shuffled.npy"
    optics_results_paths = [
        "data/HO3D/cluster_data/optics/shuffled/hand_clusters_optics_interval_5.json",
        "data/HO3D/cluster_data/origin/shuffled/hand_clusters_0003_interval_1.json",
        "data/HO3D/cluster_data/optics/shuffled/hand_clusters_0003_interval_5.json",
    ]

    # 加载距离矩阵
    distance_matrix = np.load(distance_matrix_path)

    # 评估多个OPTICS聚类结果
    for path in optics_results_paths:
        optics_results = load_optics_results(path)
        metrics = evaluate_clustering(distance_matrix, optics_results, "OPTICS")

        print(f"\nOPTICS聚类评估结果 ({path}):")
        for k, v in metrics.items():
            print(f"{k}: {v}")