import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

'''
    本代码旨在评估未含聚类真值的乱序HO3D聚类结果各个指标的效果
'''
def load_optics_results(json_path):
    """加载OPTICS聚类结果JSON数据"""
    with open(json_path) as f:
        data = json.load(f)
    return data


def load_initial_results(json_path):
    """加载初始方法聚类结果JSON数据"""
    with open(json_path) as f:
        data = json.load(f)
    return data


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
    cluster_sizes = [len(samples) for samples in cluster_result.values()]
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
    distance_matrix_path = "data/FreiHand/load_data/distance_matrix_all_min10_xi0.1.npy"
    optics_results_path = "data/FreiHand/cluster_data/clusters_optics_32560_0.0055_min10_xi0.1.json"
    # initial_results_path = "../../data/FreiHand/cluster_data/new_clusters_origin_32560_0.0055_clean.json"  # 假设的初始方法结果路径

    # 加载数据
    distance_matrix = np.load(distance_matrix_path)
    optics_results = load_optics_results(optics_results_path)
    # initial_results = load_initial_results(initial_results_path)

    # 评估OPTICS聚类
    optics_results_metrics = evaluate_clustering(distance_matrix, optics_results, "OPTICS")

    # # 评估初始方法聚类
    # initial_results_metrics = evaluate_clustering(distance_matrix, initial_results, "Initial Method")

    print("OPTICS聚类评估结果:")
    for k, v in optics_results_metrics.items():
        print(f"{k}: {v}")

    # print("\n初始方法聚类评估结果:")
    # for k, v in initial_results_metrics.items():
    #     print(f"{k}: {v}")