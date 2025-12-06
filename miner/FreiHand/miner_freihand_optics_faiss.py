import os
import json
import time
import numpy as np
import faiss
from sklearn.cluster import OPTICS

# 读取 FreiHand 数据集
def read_dataset(json_dir, max_samples):
    with open(json_dir, 'r') as f:
        data = json.load(f)
    dataset = []
    count = 0
    for item in data:
        if count >= max_samples:
            break
        # 直接加载三维坐标列表
        xyz = np.array(item)  # shape (num_points, 3)
        dataset.append(xyz)
        count += 1
    dataset = np.array(dataset, dtype=np.float32)
    return dataset, count

# 预处理：中心化，归一化，展平为一维向量，便于FAISS处理
def preprocess_dataset_for_faiss(dataset):
    dataset_centered = dataset - dataset.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(dataset_centered.reshape(dataset_centered.shape[0], -1), axis=1, keepdims=True) + 1e-8
    dataset_normalized = dataset_centered.reshape(dataset_centered.shape[0], -1) / norms
    return dataset_normalized.astype(np.float32)

# 利用FAISS GPU计算全距离矩阵
def compute_faiss_distance_matrix(dataset_vectors):
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(dataset_vectors.shape[1])
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(dataset_vectors)

    N = dataset_vectors.shape[0]
    MAX_K = 2048  # FAISS GPU限制
    distance_matrix = np.zeros((N, N), dtype=np.float32)

    # 分块计算
    for i in range(0, N, MAX_K):
        end_i = min(i+MAX_K, N)
        distances, indices = gpu_index.search(dataset_vectors[i:end_i], MAX_K)
        # 填充矩阵
        for batch_idx in range(end_i-i):
            global_idx = i + batch_idx
            for rank, j in enumerate(indices[batch_idx]):
                if j >= N:  # 处理最后一个分块
                    continue
                distance_matrix[global_idx, j] = distances[batch_idx, rank]

    # 构造对称矩阵
    distance_matrix = np.maximum(distance_matrix, distance_matrix.T)
    return distance_matrix

# 进行OPTICS聚类
def perform_clustering(distance_matrix, config):
    start_time = time.time()
    clustering_model = OPTICS(
        min_samples=config["clustering"]["min_samples"],
        metric='precomputed',
        cluster_method='xi',
        xi=config["clustering"]["xi"],
        n_jobs=-1
    )
    clustering_model.fit(distance_matrix)
    cluster_labels = clustering_model.labels_
    total_samples = len(cluster_labels)
    classified_samples = np.sum(cluster_labels != -1)
    unclassified_samples = np.sum(cluster_labels == -1)

    # 构造聚类结果列表，格式与原代码一致
    clusters = []
    unique_labels = set(cluster_labels)
    unique_labels.discard(-1)
    for label in unique_labels:
        indices = np.where(cluster_labels == label)[0].tolist()
        clusters.append(indices)

    elapsed_time = time.time() - start_time
    return clusters, total_samples, classified_samples, unclassified_samples, elapsed_time

# 对聚类结果进行阈值筛选和合并（示范逻辑）
def process_clusters(cluster_result, distance_matrix, config):
    start_time = time.time()
    threshold = config["threshold"]
    merge_threshold = config["merge_threshold"]

    # 1. 根据阈值筛选簇，排除均距离大于阈值的簇（示范实现）
    filtered_clusters = []
    for cluster in cluster_result:
        if len(cluster) < 2:
            continue
        # 计算簇内平均距离
        dists = []
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                dists.append(distance_matrix[cluster[i], cluster[j]])
        avg_dist = np.mean(dists) if dists else 0
        if avg_dist < threshold:
            filtered_clusters.append(cluster)

    # 2. 简单合并距离较近的簇（示范实现）
    merged_clusters = []
    used = set()
    for i, c1 in enumerate(filtered_clusters):
        if i in used:
            continue
        merged = set(c1)
        for j, c2 in enumerate(filtered_clusters):
            if j <= i or j in used:
                continue
            # 计算两个簇间最小距离
            min_dist = min(distance_matrix[p1, p2] for p1 in c1 for p2 in c2)
            if min_dist < merge_threshold:
                merged.update(c2)
                used.add(j)
        merged_clusters.append(sorted(list(merged)))
        used.add(i)

    elapsed_time = time.time() - start_time
    return merged_clusters, elapsed_time

# 保存聚类结果为JSON
def save_cluster_result(clusters, output_file):
    with open(output_file, 'w') as f:
        json.dump(clusters, f)
    print(f"聚类结果已保存到 {output_file}")

# 打印聚类统计信息
def print_statistics(total_samples, classified_samples, unclassified_samples, merged_cluster_result):
    print(f"总样本数: {total_samples}")
    print(f"聚类样本数: {classified_samples}")
    print(f"未分类样本数: {unclassified_samples}")
    print(f"聚类簇数: {len(merged_cluster_result)}")

# 主函数整合
def main_faiss_gpu():
    config = {
        "dataset": {
            "json_dir": r'data/FreiHand/origin_data/eval_xyz_list.json',
        },
        "max_samples": 3960,
        "distance_matrix_file": "data/FreiHand/load_data/distance_matrix_eval_faiss_gpu.npy",
        "clustering": {
            "min_samples": 8,
            "xi": 0.1
        },
        "threshold": 0.001,
        "merge_threshold": 0.001,
        "output_file": "data/FreiHand/cluster_data/clusters_optics_eval_faiss_gpu.json",
    }

    print("读取数据...")
    dataset, n_samples = read_dataset(config["dataset"]["json_dir"], config["max_samples"])
    print(f"样本数量: {n_samples}")

    print("预处理数据...")
    dataset_vectors = preprocess_dataset_for_faiss(dataset)

    if os.path.exists(config["distance_matrix_file"]):
        print("加载距离矩阵...")
        distance_matrix = np.load(config["distance_matrix_file"])
    else:
        print("计算距离矩阵 (FAISS GPU)...")
        distance_matrix = compute_faiss_distance_matrix(dataset_vectors)
        os.makedirs(os.path.dirname(config["distance_matrix_file"]), exist_ok=True)
        np.save(config["distance_matrix_file"], distance_matrix)
        print("距离矩阵已保存")

    print("执行OPTICS聚类...")
    cluster_result, total_samples, classified_samples, unclassified_samples, clustering_time = perform_clustering(
        distance_matrix, config)
    print(f"聚类耗时: {clustering_time:.2f} 秒")

    print("筛选与合并簇...")
    merged_cluster_result, filtering_time = process_clusters(cluster_result, distance_matrix, config)
    print(f"筛选合并耗时: {filtering_time:.2f} 秒")

    print_statistics(total_samples, classified_samples, unclassified_samples, merged_cluster_result)

    os.makedirs(os.path.dirname(config["output_file"]), exist_ok=True)
    save_cluster_result(merged_cluster_result, config["output_file"])

if __name__ == '__main__':
    main_faiss_gpu()
