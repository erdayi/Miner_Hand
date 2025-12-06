import json
import numpy as np
import time
import os
from datetime import datetime
import hdbscan
from typing import Dict, List, Tuple
from itertools import product

def generate_param_combinations(
    min_cluster_sizes: List[int] = [5, 8, 10],
    min_samples_list: List[int] = [3, 5, 7],
    alpha_values: List[float] = [0.3, 0.5, 0.7],
    distance_thresholds: List[float] = [0.001, 0.002, 0.003]
) -> List[Dict]:
    """
    生成参数组合列表
    
    Args:
        min_cluster_sizes: 最小簇大小列表
        min_samples_list: 最小样本数列表
        alpha_values: alpha值列表
        distance_thresholds: 距离阈值列表
    
    Returns:
        参数组合列表
    """
    param_combinations = []
    for min_size, min_samples, alpha, threshold in product(
        min_cluster_sizes, min_samples_list, alpha_values, distance_thresholds
    ):
        # 确保min_samples小于min_cluster_size
        if min_samples < min_size:
            param_combinations.append({
                "min_cluster_size": min_size,
                "min_samples": min_samples,
                "alpha": alpha,
                "distance_threshold": threshold
            })
    return param_combinations

def filter_clusters(cluster_result: Dict, distance_matrix: np.ndarray, threshold: float) -> Tuple[Dict, List]:
    """根据阈值筛选聚类结果"""
    final_cluster_result = {}
    outliers = []
    
    for label, samples in cluster_result.items():
        if label == -1:  # 跳过噪声点
            continue
            
        # 计算簇内样本间的平均距离
        valid_samples = []
        for sample in samples:
            distances = [distance_matrix[sample][other] for other in samples if other != sample]
            avg_distance = sum(distances) / len(distances) if distances else float('inf')
            
            # 如果平均距离小于阈值，保留该样本
            if avg_distance <= threshold:
                valid_samples.append(sample)
        
        if valid_samples:
            final_cluster_result[label] = valid_samples

    # 对筛选后聚类结果的索引进行排序
    for label in final_cluster_result:
        final_cluster_result[label].sort()

    # 去除只有一个图片的簇，并将其归为离群点
    outliers = cluster_result.get(-1, [])
    new_final_cluster_result = {}
    for label, cluster in final_cluster_result.items():
        if len(cluster) > 1:
            new_final_cluster_result[label] = cluster
        else:
            outliers.extend(cluster)
    
    return new_final_cluster_result, outliers

def merge_clusters(final_cluster_result: Dict, distance_matrix: np.ndarray, threshold: float) -> Dict:
    """合并相似的聚类"""
    merged_cluster_result = final_cluster_result.copy()
    labels = sorted(merged_cluster_result.keys())

    def calculate_cluster_distance(cluster1, cluster2):
        """计算两个簇之间的平均距离"""
        distances = []
        for sample1 in cluster1:
            for sample2 in cluster2:
                distances.append(distance_matrix[sample1][sample2])
        return sum(distances) / len(distances) if distances else float('inf')

    def should_merge(cluster1, cluster2):
        """判断两个簇是否应该合并"""
        avg_distance = calculate_cluster_distance(cluster1, cluster2)
        return avg_distance <= threshold

    # 计算所有簇对之间的距离
    cluster_distances = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            label1, label2 = labels[i], labels[j]
            cluster1 = merged_cluster_result[label1]
            cluster2 = merged_cluster_result[label2]
            distance = calculate_cluster_distance(cluster1, cluster2)
            cluster_distances.append((distance, label1, label2))
    
    # 按距离排序
    cluster_distances.sort()
    
    # 合并簇
    merged = set()
    for distance, label1, label2 in cluster_distances:
        if label1 in merged or label2 in merged:
            continue
            
        if should_merge(merged_cluster_result[label1], merged_cluster_result[label2]):
            # 合并簇
            merged_cluster_result[label1].extend(merged_cluster_result[label2])
            del merged_cluster_result[label2]
            merged.add(label1)
            merged.add(label2)
    
    # 重新排序簇标签
    new_merged_cluster_result = {}
    new_index = 0
    for cluster in merged_cluster_result.values():
        new_merged_cluster_result[new_index] = cluster
        new_index += 1
        
    return new_merged_cluster_result

def process_outliers(merged_cluster_result: Dict, outliers: List, distance_matrix: np.ndarray, threshold: float) -> Dict:
    """处理离群点"""
    if not outliers:
        return merged_cluster_result
        
    for outlier in outliers:
        best_cluster = None
        min_avg_distance = float('inf')
        
        # 找到最适合的簇
        for label, cluster in merged_cluster_result.items():
            distances = [distance_matrix[outlier][sample] for sample in cluster]
            avg_distance = sum(distances) / len(distances)
            
            if avg_distance < min_avg_distance and avg_distance <= threshold:
                min_avg_distance = avg_distance
                best_cluster = label
        
        # 如果找到合适的簇，添加离群点
        if best_cluster is not None:
            merged_cluster_result[best_cluster].append(outlier)
    
    return merged_cluster_result

def test_parameters(
    distance_matrix: np.ndarray,
    param_combinations: List[Dict],
    base_output_dir: str
) -> None:
    """测试不同的参数组合并保存结果"""
    # 确保输出目录存在
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 创建进度记录文件
    progress_file = os.path.join(base_output_dir, "progress.json")
    completed_params = set()
    
    # 加载已完成的参数组合
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                completed_params = set(json.load(f))
            print(f"已加载 {len(completed_params)} 个已完成的参数组合")
        except:
            print("进度文件损坏，将重新开始")
    
    # 测试每个参数组合
    for i, params in enumerate(param_combinations, 1):
        # 构建参数组合的唯一标识
        param_str = f"min{params['min_cluster_size']}_samples{params['min_samples']}_alpha{params['alpha']}_th{params['distance_threshold']}"
        output_file = os.path.join(base_output_dir, f"clusters_{param_str}.json")
        
        # 检查是否已经计算过这个参数组合
        if param_str in completed_params:
            print(f"\n跳过参数组合 {i}/{len(param_combinations)}: {param_str} (已存在)")
            continue
            
        print(f"\n测试参数组合 {i}/{len(param_combinations)}")
        print("=" * 50)
        
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 执行聚类
            clustering_time = time.time()
            clusterer = hdbscan.HDBSCAN(
                metric='precomputed',
                min_cluster_size=params['min_cluster_size'],
                min_samples=params['min_samples'],
                alpha=params['alpha'],
                algorithm='best',
                core_dist_n_jobs=-1
            )
            cluster_labels = clusterer.fit_predict(distance_matrix)
            clustering_time = time.time() - clustering_time
            
            # 后处理
            postprocess_time = time.time()
            
            # 整理初始聚类结果
            cluster_result = {}
            for i, label in enumerate(cluster_labels):
                if label not in cluster_result:
                    cluster_result[label] = []
                cluster_result[label].append(i)
            
            # 执行与原始代码相同的后处理步骤
            final_cluster_result, outliers = filter_clusters(
                cluster_result, distance_matrix, params['distance_threshold']
            )
            
            merged_cluster_result = merge_clusters(
                final_cluster_result, distance_matrix, params['distance_threshold']
            )
            
            final_result = process_outliers(
                merged_cluster_result, outliers, distance_matrix, params['distance_threshold']
            )
            
            # 计算统计信息
            total_samples = len(cluster_labels)
            classified_samples = sum(1 for label in cluster_labels if label != -1)
            unclassified_samples = sum(1 for label in cluster_labels if label == -1)
            
            # 计算簇大小统计
            cluster_sizes = [len(samples) for samples in final_result.values()]
            if cluster_sizes:
                min_size = min(cluster_sizes)
                max_size = max(cluster_sizes)
                avg_size = sum(cluster_sizes) / len(cluster_sizes)
            else:
                min_size = max_size = avg_size = 0
            
            # 计算聚类分布
            small_clusters = sum(1 for size in cluster_sizes if size <= 10)
            medium_clusters = sum(1 for size in cluster_sizes if 11 <= size <= 50)
            large_clusters = sum(1 for size in cluster_sizes if size > 50)
            
            postprocess_time = time.time() - postprocess_time
            
            # 准备结果
            result = {
                "parameters": params,
                "statistics": {
                    "total_samples": total_samples,
                    "classified_samples": classified_samples,
                    "unclassified_samples": unclassified_samples,
                    "total_clusters": len(cluster_sizes),
                    "cluster_size_stats": {
                        "min_size": min_size,
                        "max_size": max_size,
                        "avg_size": round(avg_size, 1)
                    },
                    "cluster_distribution": {
                        "small_clusters": small_clusters,
                        "medium_clusters": medium_clusters,
                        "large_clusters": large_clusters
                    }
                },
                "timing": {
                    "clustering_time": round(clustering_time, 2),
                    "postprocess_time": round(postprocess_time, 2),
                    "total_time": round(time.time() - start_time, 2)
                },
                "clusters": {str(k): v for k, v in final_result.items()}
            }
            
            # 保存结果
            with open(output_file, "w") as f:
                json.dump(result, f, indent=4)
            
            # 更新进度
            completed_params.add(param_str)
            # 将集合转换为排序后的列表
            sorted_params = sorted(list(completed_params))
            # 保存进度文件，使用indent=4实现换行格式
            with open(progress_file, "w") as f:
                json.dump(sorted_params, f, indent=4)
            
            # 打印统计信息
            print(f"\n参数组合: {param_str}")
            print(f"聚类时间: {clustering_time:.2f}秒")
            print(f"后处理时间: {postprocess_time:.2f}秒")
            print(f"总时间: {time.time() - start_time:.2f}秒")
            print("\n聚类统计:")
            print(f"总样本数: {total_samples}")
            print(f"分类样本数: {classified_samples}")
            print(f"未分类样本数: {unclassified_samples}")
            print(f"总簇数: {len(cluster_sizes)}")
            print("\n簇大小统计:")
            print(f"最小簇大小: {min_size}")
            print(f"最大簇大小: {max_size}")
            print(f"平均簇大小: {avg_size:.1f}")
            print("\n簇分布:")
            print(f"小簇(≤10): {small_clusters}")
            print(f"中簇(11-50): {medium_clusters}")
            print(f"大簇(>50): {large_clusters}")
            
        except Exception as e:
            print(f"处理参数组合 {param_str} 时出错：{str(e)}")
            import traceback
            print(traceback.format_exc())
            continue

def main():
    try:
        print("开始加载距离矩阵...")
        # 加载距离矩阵
        distance_matrix_file = "data/FreiHand/load_data/hdbscan/distance_matrix_32560.npy"
        distance_matrix = np.load(distance_matrix_file)
        print(f"距离矩阵加载完成，形状: {distance_matrix.shape}")
        
        # 定义参数范围 min_samples > min_cluster_size，会导致没有有效组合
        min_cluster_sizes = [4, 6, 8, 10]  # 最小簇大小
        min_samples_list = [3, 5, 7, 9]    # 最小样本数
        alpha_values = [0.3, 0.5, 0.7]  # alpha值
        distance_thresholds = [0.003]  # 距离阈值

        print("生成参数组合...")
        # 生成参数组合
        param_combinations = generate_param_combinations(
            min_cluster_sizes=min_cluster_sizes,
            min_samples_list=min_samples_list,
            alpha_values=alpha_values,
            distance_thresholds=distance_thresholds
        )
        
        if not param_combinations:
            print("警告：没有生成有效的参数组合！请检查参数设置。")
            print(f"当前参数设置：")
            print(f"min_cluster_sizes: {min_cluster_sizes}")
            print(f"min_samples_list: {min_samples_list}")
            print(f"alpha_values: {alpha_values}")
            print(f"distance_thresholds: {distance_thresholds}")
            return
            
        print(f"生成了 {len(param_combinations)} 个参数组合")
        
        # 设置输出目录
        base_output_dir = "data/FreiHand/cluster_data/hdbscan/param_test"
        
        print("开始执行参数测试...")
        # 执行参数测试
        test_parameters(distance_matrix, param_combinations, base_output_dir)
        print("参数测试完成！")
        
    except Exception as e:
        print(f"程序执行出错：{str(e)}")
        import traceback
        print("详细错误信息：")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()