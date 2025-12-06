import os
import json
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, List, Tuple
import math
from tqdm import tqdm

def calculate_cluster_metrics(clusters: Dict, distance_matrix: np.ndarray) -> Dict:
    """
    计算聚类指标
    
    Args:
        clusters: 聚类结果字典，key为簇标签，value为样本索引列表
        distance_matrix: 距离矩阵
    
    Returns:
        包含各项指标的字典
    """
    # 准备数据
    labels = np.zeros(len(distance_matrix))
    for label, samples in clusters.items():
        labels[samples] = label
    
    # 计算簇内距离 - 优化版本
    intra_distances = []
    for samples in clusters.values():
        if len(samples) > 1:
            # 使用矩阵切片一次性获取所有距离
            cluster_distances = distance_matrix[np.ix_(samples, samples)]
            # 计算上三角矩阵的平均值（不包括对角线）
            mask = np.triu_indices(len(samples), k=1)
            intra_distances.append(np.mean(cluster_distances[mask]))
    
    # 计算簇间距离 - 优化版本
    inter_distances = []
    cluster_centers = []
    for samples in clusters.values():
        if len(samples) > 0:
            # 使用矩阵切片一次性计算中心
            center = np.mean(distance_matrix[samples], axis=0)
            cluster_centers.append(center)
    
    if len(cluster_centers) > 1:
        # 使用矩阵运算一次性计算所有簇间距离
        centers_matrix = np.array(cluster_centers)
        for i in range(len(centers_matrix)):
            for j in range(i + 1, len(centers_matrix)):
                inter_distances.append(np.mean(np.abs(centers_matrix[i] - centers_matrix[j])))
    
    # 计算簇大小分布的熵
    cluster_sizes = [len(samples) for samples in clusters.values()]
    total_samples = sum(cluster_sizes)
    if total_samples > 0:
        probabilities = np.array(cluster_sizes) / total_samples
        entropy = -np.sum(probabilities * np.log2(probabilities))
    else:
        entropy = 0
    
    # 计算未分类样本比例
    unclassified_ratio = 1 - (sum(cluster_sizes) / len(distance_matrix))
    
    # 计算经典指标
    try:
        silhouette = silhouette_score(distance_matrix, labels)
    except:
        silhouette = 0
    
    try:
        davies_bouldin = davies_bouldin_score(distance_matrix, labels)
    except:
        davies_bouldin = float('inf')
    
    try:
        calinski_harabasz = calinski_harabasz_score(distance_matrix, labels)
    except:
        calinski_harabasz = 0
    
    return {
        "silhouette_score": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
        "intra_cluster_distance": np.mean(intra_distances) if intra_distances else 0,
        "inter_cluster_distance": np.mean(inter_distances) if inter_distances else 0,
        "cluster_entropy": entropy,
        "unclassified_ratio": unclassified_ratio,
        "total_clusters": len(clusters),
        "classified_samples": sum(cluster_sizes),
        "unclassified_samples": len(distance_matrix) - sum(cluster_sizes)
    }

def parse_parameters(filename: str) -> Dict:
    """
    从文件名解析参数
    
    Args:
        filename: 文件名，格式如 "clusters_min=5_samples=3_alpha=0.3_th=0.002.json"
    
    Returns:
        参数字典
    """
    params = {}
    try:
        # 去掉"clusters_"前缀和".json"后缀
        param_str = filename[9:-5]
        
        # 分割参数
        parts = param_str.split("_")
        for part in parts:
            if "=" in part:
                key, value = part.split("=")
                if key == "min":
                    params["min_cluster_size"] = int(value)
                elif key == "samples":
                    params["min_samples"] = int(value)
                elif key == "alpha":
                    params["alpha"] = float(value)
                elif key == "th":
                    params["distance_threshold"] = float(value)
    except Exception as e:
        print(f"Warning: Failed to parse parameters from filename {filename}: {str(e)}")
        # 返回默认参数
        params = {
            "min_cluster_size": 5,
            "min_samples": 3,
            "alpha": 0.3,
            "distance_threshold": 0.002
        }
    
    return params

def evaluate_parameter_combinations(
    results_dir: str,
    output_dir: str
) -> None:
    """
    评估所有参数组合的聚类结果
    
    Args:
        results_dir: 包含聚类结果的目录
        output_dir: 输出评估结果的目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有参数组合的结果
    all_results = []
    
    # 获取所有需要处理的文件
    files_to_process = [f for f in os.listdir(results_dir) 
                       if f.startswith("clusters_") and f.endswith(".json")]
    
    # 使用tqdm显示进度条
    for filename in tqdm(files_to_process, desc="Processing files"):
        # 解析参数
        params = parse_parameters(filename)
        
        # 加载聚类结果
        try:
            with open(os.path.join(results_dir, filename), "r") as f:
                result = json.load(f)
            
            # 计算指标
            metrics = calculate_cluster_metrics(
                {int(k): v for k, v in result["clusters"].items()},
                np.load("data/FreiHand/load_data/hdbscan/distance_matrix_32560.npy")
            )
            
            # 合并结果
            all_results.append({
                "parameters": params,
                "metrics": metrics
            })
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    if not all_results:
        print("No valid results found!")
        return
    
    print("Calculating scores...")
    
    # 计算综合得分
    for result in tqdm(all_results, desc="Calculating scores"):
        metrics = result["metrics"]
        
        # 归一化指标
        normalized_silhouette = (metrics["silhouette_score"] + 1) / 2  # 从[-1,1]映射到[0,1]
        normalized_davies = 1 / (1 + metrics["davies_bouldin"])  # 从[0,∞)映射到[0,1]
        normalized_calinski = min(1, metrics["calinski_harabasz"] / 1000)  # 从[0,∞)映射到[0,1]
        normalized_intra = 1 / (1 + metrics["intra_cluster_distance"])  # 从[0,∞)映射到[0,1]
        normalized_inter = min(1, metrics["inter_cluster_distance"] / 0.1)  # 从[0,∞)映射到[0,1]
        normalized_entropy = metrics["cluster_entropy"] / 10  # 假设最大熵为10
        normalized_unclassified = 1 - metrics["unclassified_ratio"]  # 从[0,1]映射到[0,1]
        
        # 计算综合得分
        result["score"] = (
            0.3 * normalized_silhouette +
            0.2 * normalized_davies +
            0.2 * normalized_calinski +
            0.1 * normalized_intra +
            0.1 * normalized_inter +
            0.05 * normalized_entropy +
            0.05 * normalized_unclassified
        )
    
    print("Saving results...")
    
    # 按综合得分排序
    all_results.sort(key=lambda x: x["score"], reverse=True)
    
    # 保存评估结果
    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)
    
    # 保存排序后的结果
    with open(os.path.join(output_dir, "ranked_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)
    
    # 保存前10个最佳参数组合
    top_10 = all_results[:10]
    with open(os.path.join(output_dir, "top_10_parameters.json"), "w") as f:
        json.dump(top_10, f, indent=4)
    
    print(f"Evaluation completed. Results saved to {output_dir}")

if __name__ == "__main__":
    results_dir = "data/FreiHand/cluster_data/hdbscan/param_test"
    output_dir = "data/FreiHand/cluster_data/hdbscan/evaluation"
    evaluate_parameter_combinations(results_dir, output_dir) 