 import json
import numpy as np
from scipy.linalg import orthogonal_procrustes
import time
import os
from joblib import Parallel, delayed
import multiprocessing
import re
from tqdm import tqdm
import gc

'''
FreiHand数据集手部姿态批量聚类工具

该工具使用HDBSCAN聚类算法对FreiHand数据集中的手部姿态进行批量聚类分析。
主要功能：
1. 将大数据集分成多个小批次（每批2万样本）
2. 对每个批次独立进行聚类
3. 分别保存每个批次的聚类结果
4. 提供进度跟踪和内存管理

使用方法：
python miner/FreiHand/miner_freihand_hdbscan_batch.py

参数说明：
- batch_size: 每批处理的样本数量，默认20000
- min_cluster_size: 最小聚类大小，默认8
- min_samples: 核心距离计算参数，默认5
- threshold: 聚类阈值，默认0.001
'''

# ===== 配置参数 =====
CONFIG = {
    # 数据集配置
    "dataset": {
        "name": "FreiHand",
        "json_dir": r'data/FreiHand/origin_data/xyz_list.json',
        "scale_enlarge": 1.25
    },
    # 批处理配置
    "batch": {
        "size": 20000,  # 每批处理的样本数量
        "distance_matrix_dir": "data/FreiHand/load_data/hdbscan/batch",  # 距离矩阵保存目录
        "cluster_result_dir": "data/FreiHand/cluster_data/hdbscan/batch"  # 聚类结果保存目录
    },
    # HDBSCAN聚类算法配置
    "clustering": {
        "min_cluster_size": 5,
        "min_samples": 3,
        "alpha": 0.3,
        "algorithm": "best",
        "core_dist_n_jobs": -1
    },
    # 后处理参数
    "distance_threshold": 0.003
}

def align_w_scale_vectorized(mtx1, mtx2):
    """向量化的对齐函数"""
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1
    return mtx2_t

def calculate_distance_chunk(chunk_data):
    """计算距离矩阵的一个块"""
    start_idx, end_idx, dataset = chunk_data
    n_samples = len(dataset)
    chunk_size = end_idx - start_idx
    chunk_matrix = np.zeros((chunk_size, n_samples))
    
    for i in range(chunk_size):
        idx = start_idx + i
        for j in range(n_samples):
            if idx < j:  # 只计算上三角部分
                xyz_aligned = align_w_scale_vectorized(dataset[idx], dataset[j])
                error = np.mean(np.linalg.norm(dataset[idx] - xyz_aligned, axis=1))
                chunk_matrix[i, j] = error
                chunk_matrix[i, j] = error  # 利用对称性
    
    return start_idx, chunk_matrix

def calculate_batch_distance_matrix(dataset, batch_idx, config):
    """计算单个批次的距离矩阵"""
    batch_size = config["batch"]["size"]
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(dataset))
    batch_data = dataset[start_idx:end_idx]
    
    n_samples = len(batch_data)
    n_cores = max(1, min(8, multiprocessing.cpu_count() - 1))
    chunk_size = max(1, n_samples // n_cores)
    
    # 创建任务块
    chunks = []
    for i in range(0, n_samples, chunk_size):
        end_chunk = min(i + chunk_size, n_samples)
        chunks.append((i, end_chunk, batch_data))
    
    # 并行计算
    results = Parallel(
        n_jobs=n_cores,
        max_nbytes='50M',
        prefer="processes",
        backend='loky',
        batch_size=10
    )(delayed(calculate_distance_chunk)(chunk) for chunk in chunks)
    
    # 合并结果
    distance_matrix = np.zeros((n_samples, n_samples))
    for start_idx, chunk_matrix in results:
        distance_matrix[start_idx:start_idx + chunk_matrix.shape[0]] = chunk_matrix
    
    # 利用对称性填充下三角部分
    distance_matrix = np.maximum(distance_matrix, distance_matrix.T)
    
    return distance_matrix

def save_batch_distance_matrix(distance_matrix, batch_idx, config):
    """保存批次距离矩阵"""
    distance_matrix_dir = config["batch"]["distance_matrix_dir"]
    os.makedirs(distance_matrix_dir, exist_ok=True)
    filename = os.path.join(distance_matrix_dir, f"distance_matrix_batch_{batch_idx}.npy")
    np.save(filename, distance_matrix)
    return filename

def load_batch_distance_matrix(batch_idx, config):
    """加载批次距离矩阵"""
    distance_matrix_dir = config["batch"]["distance_matrix_dir"]
    filename = os.path.join(distance_matrix_dir, f"distance_matrix_batch_{batch_idx}.npy")
    if os.path.exists(filename):
        return np.load(filename)
    return None

def hdbscan_clustering(distance_matrix, config):
    """使用HDBSCAN进行聚类"""
    try:
        import hdbscan
        clustering = hdbscan.HDBSCAN(
            metric='precomputed',
            min_cluster_size=config["clustering"]["min_cluster_size"],
            min_samples=config["clustering"]["min_samples"],
            alpha=config["clustering"]["alpha"],
            algorithm=config["clustering"]["algorithm"],
            core_dist_n_jobs=config["clustering"]["core_dist_n_jobs"]
        )
        cluster_labels = clustering.fit_predict(distance_matrix)
        return clustering, cluster_labels
    except ImportError:
        print("HDBSCAN未安装，使用OPTICS作为备选")
        from sklearn.cluster import OPTICS
        clustering = OPTICS(metric='precomputed', min_samples=config["clustering"]["min_samples"])
        clustering.fit(distance_matrix)
        return clustering, clustering.labels_

def filter_clusters(cluster_result, distance_matrix, threshold):
    """根据阈值筛选聚类结果"""
    final_cluster_result = {}
    
    for label, samples in cluster_result.items():
        if label == -1:  # 跳过噪声点
            continue
            
        valid_samples = []
        for sample in samples:
            distances = [distance_matrix[sample][other] for other in samples if other != sample]
            avg_distance = sum(distances) / len(distances) if distances else float('inf')
            
            if avg_distance <= threshold:
                valid_samples.append(sample)
        
        if valid_samples:
            final_cluster_result[label] = valid_samples

    # 对筛选后聚类结果的索引进行排序
    for label in final_cluster_result:
        final_cluster_result[label].sort()

    # 去除只有一个图片的簇
    outliers = cluster_result.get(-1, [])
    new_final_cluster_result = {}
    for label, cluster in final_cluster_result.items():
        if len(cluster) > 1:
            new_final_cluster_result[label] = cluster
        else:
            outliers.extend(cluster)
    
    return new_final_cluster_result, outliers

def save_batch_result(cluster_result, batch_idx, config):
    """保存批次聚类结果"""
    result_dir = config["batch"]["cluster_result_dir"]
    os.makedirs(result_dir, exist_ok=True)
    
    output_file = os.path.join(result_dir, f"hand_clusters_batch_{batch_idx}.json")
    
    final_result = {
        "clusters": [],
        "parameters": {
            "min_cluster_size": config["clustering"]["min_cluster_size"],
            "min_samples": config["clustering"]["min_samples"],
            "alpha": config["clustering"]["alpha"],
            "distance_threshold": config["distance_threshold"],
            "batch_idx": batch_idx
        }
    }
    
    # 处理聚类结果
    cluster_paths = []
    for label in sorted(cluster_result.keys()):
        if len(cluster_result[label]) > 1:  # 保留有效簇
            sorted_indices = sorted(cluster_result[label])
            cluster_paths.append(sorted_indices)
    
    final_result["clusters"] = cluster_paths
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)
    
    return output_file

def process_batch(dataset, batch_idx, config):
    """处理单个批次的数据"""
    print(f"\n处理批次 {batch_idx}...")
    
    # 计算或加载距离矩阵
    distance_matrix = load_batch_distance_matrix(batch_idx, config)
    if distance_matrix is None:
        print("计算距离矩阵...")
        distance_matrix = calculate_batch_distance_matrix(dataset, batch_idx, config)
        save_batch_distance_matrix(distance_matrix, batch_idx, config)
    
    # 执行聚类
    print("执行HDBSCAN聚类...")
    clustering, cluster_labels = hdbscan_clustering(distance_matrix, config)
    
    # 整理聚类结果
    cluster_result = {}
    for i, label in enumerate(cluster_labels):
        if label not in cluster_result:
            cluster_result[label] = []
        cluster_result[label].append(i)
    
    # 筛选聚类结果
    print("筛选聚类结果...")
    final_cluster_result, _ = filter_clusters(
        cluster_result, distance_matrix, config["distance_threshold"]
    )
    
    # 保存结果
    print("保存聚类结果...")
    output_file = save_batch_result(final_cluster_result, batch_idx, config)
    
    # 清理内存
    del distance_matrix
    del cluster_result
    del final_cluster_result
    gc.collect()
    
    return output_file

def main():
    """主函数：执行FreiHand数据集的批量聚类分析"""
    print("🚀 FreiHand HDBSCAN 批量聚类分析")
    print("🎯 处理海量数据集的批次聚类")
    print("=" * 60)
    
    try:
        # 读取数据集
        print("\n读取数据集...")
        with open(CONFIG["dataset"]["json_dir"], 'r') as f:
            dataset = json.load(f)
        
        # 处理数据
        processed_data = []
        for item in dataset:
            if isinstance(item, list) and len(item) > 0:
                processed_data.append(np.array(item))
        
        dataset = np.array(processed_data)
        total_samples = len(dataset)
        batch_size = CONFIG["batch"]["size"]
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        print(f"\n数据集信息：")
        print(f"总样本数：{total_samples}")
        print(f"批次大小：{batch_size}")
        print(f"批次数：{num_batches}")
        
        # 处理每个批次
        for batch_idx in range(num_batches):
            print(f"\n开始处理批次 {batch_idx + 1}/{num_batches}")
            output_file = process_batch(dataset, batch_idx, CONFIG)
            print(f"批次 {batch_idx + 1} 处理完成，结果保存至：{output_file}")
        
        print("\n✅ 所有批次处理完成！")
        
    except Exception as e:
        print(f"\n⚠️ 处理过程中出现错误: {str(e)}")
        return

if __name__ == "__main__":
    main()