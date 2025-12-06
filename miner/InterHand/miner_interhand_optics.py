'''
InterHand2.6M数据集手部姿态聚类工具

该工具使用OPTICS聚类算法对InterHand2.6M数据集中的手部姿态进行聚类分析。
主要功能：
1. 加载预处理后的手部3D关节点数据和图像路径
2. 计算手部姿态之间的相似度（使用正交对齐方法）
3. 使用OPTICS算法进行聚类
4. 对聚类结果进行后处理（阈值筛选、簇合并、离群点处理）
5. 输出聚类结果和统计信息

使用方法：
python miner/InterHand/miner_interhand_optics.py --hand_type left --threshold 0.0003
python miner/InterHand/miner_interhand_optics.py --hand_type right --threshold 0.0003


参数说明：
--hand_type: 手部类型，可选 'left' 或 'right'
--threshold: 聚类阈值，值越小聚类越严格
--merge_threshold: 合并簇的阈值，值越小合并越严格
--min_samples: OPTICS聚类的最小样本数
--xi: OPTICS聚类的xi参数

输出：
1. 聚类结果JSON文件，包含：
   - 每个簇的图像路径列表
   - 使用的聚类参数信息
2. 命令行统计信息，包含：
   - 样本统计（总数、分类数、未分类数）
   - 时间统计（距离矩阵计算、聚类时间）
   - 簇大小统计（最大、最小、平均）
'''

import json
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import OPTICS
from tqdm import tqdm
import time
import os
import argparse
from pathlib import Path
import re

def align_w_scale(mtx1, mtx2):
    """
    使用正交对齐方法进行匹配，并返回对齐后的结果。
    :param mtx1: 第一个矩阵，代表手部关键点数据
    :param mtx2: 第二个矩阵，代表手部关键点数据
    :return: 对齐后的第二个矩阵
    """
    # 计算两个矩阵的均值
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    # 减去均值进行中心化
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # 计算矩阵的范数，并添加一个小的常数避免除零错误
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    # 归一化矩阵
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # 使用正交普罗克拉斯提斯分析计算旋转矩阵和缩放因子
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)
    # 应用旋转和缩放
    mtx2_t = np.dot(mtx2_t, R.T) * s
    # 恢复到原始尺度和位置
    mtx2_t = mtx2_t * s1 + t1
    return mtx2_t

def save_distance_matrix(matrix, filename):
    """保存距离矩阵到 .npy 文件"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.save(filename, matrix)

def load_distance_matrix(filename):
    """从 .npy 文件加载距离矩阵"""
    return np.load(filename)

def validate_distance_matrix(matrix, n_samples):
    """验证距离矩阵维度"""
    if matrix.shape != (n_samples, n_samples):
        raise ValueError(f"距离矩阵维度错误，应为({n_samples}, {n_samples})，实际为{matrix.shape}")

def index_to_path(index_list, path_file, hand_type):
    """将索引列表转换为图像路径列表"""
    try:
        with open(path_file, 'r') as f:
            paths_data = json.load(f)
        return [paths_data[hand_type][i] for i in index_list]
    except Exception as e:
        print(f"路径转换错误: {str(e)}")
        return []

def load_or_calculate_distance_matrix(dataset, distance_matrix_file):
    """加载或计算距离矩阵"""
    error_matrix_calculation_time = 0
    if os.path.exists(distance_matrix_file):
        print("加载已保存的误差矩阵...")
        start_time = time.time()
        try:
            # 从文件中加载距离矩阵
            distance_matrix = load_distance_matrix(distance_matrix_file)
            # 验证矩阵维度是否与数据集匹配
            n_samples = len(dataset)
            validate_distance_matrix(distance_matrix, n_samples)
            # 计算加载耗时
            error_matrix_calculation_time = time.time() - start_time
            print(f"加载耗时: {error_matrix_calculation_time:.2f} 秒")
        except Exception as e:
            print(f"加载矩阵失败: {str(e)}")
            distance_matrix = None
    else:
        distance_matrix = None

    if distance_matrix is None:
        print("计算误差矩阵...")
        start_time = time.time()
        n_samples = len(dataset)
        distance_matrix = np.zeros((n_samples, n_samples))
        for i in tqdm(range(n_samples), desc="计算误差矩阵"):
            for j in range(i + 1, n_samples):
                # 计算对齐后的矩阵
                xyz_aligned = align_w_scale(dataset[i], dataset[j])
                # 计算对齐误差
                error = np.mean(np.linalg.norm(dataset[i] - xyz_aligned, axis=1))
                # 填充距离矩阵
                distance_matrix[i, j] = error
                distance_matrix[j, i] = error
        save_distance_matrix(distance_matrix, distance_matrix_file)
        error_matrix_calculation_time = time.time() - start_time
        print(f"误差矩阵计算耗时: {error_matrix_calculation_time:.2f} 秒，已保存")

    return distance_matrix, error_matrix_calculation_time

def optics_clustering(distance_matrix, min_samples, xi):
    """使用 OPTICS 进行聚类"""
    start_time = time.time()
    print("开始聚类...")
    # 初始化 OPTICS 聚类器，使用预计算的距离矩阵
    clustering = OPTICS(metric='precomputed', min_samples=min_samples, xi=xi)
    # 拟合数据进行聚类
    clustering.fit(distance_matrix)
    # 计算聚类耗时
    clustering_time = time.time() - start_time
    print(f"聚类耗时: {clustering_time:.2f} 秒")
    return clustering, clustering_time

def filter_clusters(cluster_result, distance_matrix, threshold):
    """根据阈值筛选聚类结果"""
    start_time = time.time()
    final_cluster_result = {}
    for label, samples in cluster_result.items():
        valid_samples = []
        for sample in samples:
            valid = True
            for other_sample in samples:
                if distance_matrix[sample][other_sample] > threshold:
                    valid = False
                    break
            if valid:
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
    final_cluster_result = new_final_cluster_result

    # 计算阈值筛选耗时
    threshold_filtering_time = time.time() - start_time
    print(f"阈值筛选耗时: {threshold_filtering_time:.2f} 秒")
    return final_cluster_result, outliers, threshold_filtering_time

def merge_clusters(final_cluster_result, distance_matrix, merge_threshold):
    """合并相似的聚类"""
    merged_cluster_result = final_cluster_result.copy()
    labels = sorted(merged_cluster_result.keys())

    def should_merge(cluster1, cluster2):
        for sample1 in cluster1:
            for sample2 in cluster2:
                if distance_matrix[sample1][sample2] > merge_threshold:
                    return False
        return True

    def merge_clusters_internal():
        if not labels:
            return merged_cluster_result
        base_label = labels[0]
        base_cluster = merged_cluster_result[base_label]
        for i in range(1, len(labels)):
            other_label = labels[i]
            other_cluster = merged_cluster_result[other_label]
            if should_merge(base_cluster, other_cluster):
                base_cluster.extend(other_cluster)
                del merged_cluster_result[other_label]
        # 重新排序簇标签
        new_merged_cluster_result = {}
        new_index = 0
        for cluster in merged_cluster_result.values():
            new_merged_cluster_result[new_index] = cluster
            new_index += 1
        return new_merged_cluster_result

    merged_cluster_result = merge_clusters_internal()
    return merged_cluster_result

def process_outliers(merged_cluster_result, outliers, distance_matrix, merge_threshold):
    """处理离群点"""
    for outlier in outliers:
        for label, cluster in merged_cluster_result.items():
            can_add = True
            for sample in cluster:
                if distance_matrix[outlier][sample] > merge_threshold:
                    can_add = False
                    break
            if can_add:
                merged_cluster_result[label].append(outlier)
                break
    return merged_cluster_result

def extract_frame_idx(view_path):
    """从视角路径中提取帧号"""
    match = re.search(r'image(\d+)', view_path)
    if match:
        return int(match.group(1))
    return 0

def extract_sequence_name(view_path):
    """从视角路径中提取序列名"""
    return view_path.split('/')[0]

def sort_cluster_paths(paths):
    """对簇内的图像路径进行排序"""
    # 为每个路径创建包含序列名和帧号的元组
    path_info = [(path, extract_sequence_name(path), extract_frame_idx(path)) for path in paths]
    # 按序列名和帧号排序
    path_info.sort(key=lambda x: (x[1], x[2]))
    # 返回排序后的路径列表
    return [info[0] for info in path_info]

def save_cluster_result(cluster_result, path_file, save_path, hand_type, config):
    """保存聚类结果为JSON格式"""
    start_time = time.time()
    final_result = {
        "clusters": [],
        "parameters": {
            "min_samples": config["clustering"]["min_samples"],
            "xi": config["clustering"]["xi"],
            "threshold": config["threshold"],
            "merge_threshold": config["merge_threshold"],
            "hand_type": hand_type
        }
    }
    
    # 处理聚类结果
    for label in sorted(cluster_result.keys()):
        paths = index_to_path(cluster_result[label], path_file, hand_type)
        if len(paths) > 1:  # 保留有效簇
            # 对簇内的路径进行排序
            sorted_paths = sort_cluster_paths(paths)
            final_result["clusters"].append(sorted_paths)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存结果
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)
    
    json_storage_time = time.time() - start_time
    print(f"聚类结果已保存到 {save_path}，包含 {len(final_result['clusters'])} 个有效簇")
    print(f"JSON 文件存储耗时: {json_storage_time:.2f} 秒")
    return json_storage_time

def print_statistics(total_samples, classified_samples, unclassified_samples, merged_cluster_result, min_samples, error_matrix_calculation_time, clustering_time):
    """打印统计信息"""
    print("\n统计信息:")
    print(f"总样本数: {total_samples}")
    print(f"分类样本数: {classified_samples} ({classified_samples / total_samples * 100:.2f}%)")
    print(f"未分类样本数: {unclassified_samples} ({unclassified_samples / total_samples * 100:.2f}%)")
    print(f"最终簇数: {len(merged_cluster_result)}")
    print(f"最小样本数(min_samples): {min_samples}")
    print(f"\n时间统计:")
    print(f"距离矩阵计算时间: {error_matrix_calculation_time:.2f} 秒")
    print(f"聚类时间: {clustering_time:.2f} 秒")
    print(f"总耗时: {error_matrix_calculation_time + clustering_time:.2f} 秒")
    
    # 计算每个簇的大小
    cluster_sizes = [len(cluster) for cluster in merged_cluster_result.values()]
    if cluster_sizes:
        print(f"\n簇大小统计:")
        print(f"最大簇大小: {max(cluster_sizes)}")
        print(f"最小簇大小: {min(cluster_sizes)}")
        print(f"平均簇大小: {sum(cluster_sizes) / len(cluster_sizes):.2f}")
    else:
        print("\n没有有效的簇")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="InterHand2.6M数据集手部姿态OPTICS聚类工具")
    
    # 数据相关参数
    parser.add_argument('--data_dir', type=str, default="data/InterHand/origin_data", 
                        help="数据目录路径")
    parser.add_argument('--hand_type', type=str, default="left",
                        choices=["left", "right"],
                        help="手部类型：左手(left)或右手(right)")
    
    # 聚类相关参数
    parser.add_argument('--threshold', type=float, default=0.0003, 
                        help="聚类阈值，值越小聚类越严格")
    parser.add_argument('--merge_threshold', type=float, default=0.0003, 
                        help="合并簇的阈值，值越小合并越严格")
    parser.add_argument('--min_samples', type=int, default=5, 
                        help="OPTICS聚类的最小样本数")
    parser.add_argument('--xi', type=float, default=0.05, 
                        help="OPTICS聚类的xi参数")
    
    # 输出相关参数
    parser.add_argument('--output_dir', type=str, 
                        default="data/InterHand/cluster_data/optics", 
                        help="输出目录路径")
    parser.add_argument('--load_dir', type=str, 
                        default="data/InterHand/load_data", 
                        help="距离矩阵加载/保存目录")
    
    return parser.parse_args()

def main():
    """主函数，整合处理逻辑"""
    # 解析命令行参数
    args = parse_args()
    
    # 使用Path处理路径
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    load_dir = Path(args.load_dir)
    
    # 构建配置
    config = {
        "data": {
            "joints": str(data_dir / "val_joint_3d_shuffled.json"),
            "paths": str(data_dir / "val_image_paths_shuffled.json"),
            "save_path": str(output_dir / f"hand_clusters_optics_{args.hand_type}_min{args.min_samples}_xi{args.xi}_th{args.threshold}_merge{args.merge_threshold}.json")
        },
        "distance_matrix": {
            "file": str(load_dir / f"distance_matrix_{args.hand_type}_shuffled.npy")
        },
        "clustering": {
            "min_samples": args.min_samples,
            "xi": args.xi
        },
        "threshold": args.threshold,  # 阈值筛选参数
        "merge_threshold": args.merge_threshold  # 合并簇的阈值
    }
    
    print("\n配置信息:")
    print(f"  - 关节点数据: {config['data']['joints']}")
    print(f"  - 图像路径数据: {config['data']['paths']}")
    print(f"  - 距离矩阵文件: {config['distance_matrix']['file']}")
    print(f"  - 聚类阈值: {config['threshold']}")
    print(f"  - 合并阈值: {config['merge_threshold']}")
    print(f"  - OPTICS参数: min_samples={config['clustering']['min_samples']}, xi={config['clustering']['xi']}")
    print(f"  - 输出路径: {config['data']['save_path']}")

    # 加载数据
    print("\n加载手部关节点数据...")
    with open(config["data"]["joints"], 'r') as f:
        joints_data = json.load(f)
    hand_data = np.array(joints_data[args.hand_type])
    print(f"加载完成，数据形状: {hand_data.shape}")

    # 加载或计算距离矩阵
    distance_matrix, error_matrix_calculation_time = load_or_calculate_distance_matrix(
        hand_data, config["distance_matrix"]["file"]
    )

    # 使用 OPTICS 进行聚类
    clustering, clustering_time = optics_clustering(
        distance_matrix,
        config["clustering"]["min_samples"],
        config["clustering"]["xi"]
    )

    # 提取聚类结果
    cluster_labels = clustering.labels_
    # 计算聚类的数量
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"聚类完成，共发现 {num_clusters} 个簇。")

    # 统计数量
    total_samples = len(cluster_labels)
    classified_samples = 0
    unclassified_samples = 0

    # 遍历聚类标签，统计分类和未分类样本数量
    for label in cluster_labels:
        if label == -1:
            unclassified_samples += 1
        else:
            classified_samples += 1

    # 整理聚类结果（未筛选）
    cluster_result = {}
    for i, label in enumerate(cluster_labels):
        if label not in cluster_result:
            cluster_result[label] = []
        cluster_result[label].append(i)

    # 根据阈值筛选聚类结果
    final_cluster_result, outliers, threshold_filtering_time = filter_clusters(
        cluster_result, distance_matrix, config["threshold"]
    )

    # 聚类后合并
    merged_cluster_result = merge_clusters(final_cluster_result, distance_matrix, config["merge_threshold"])

    # 处理离群点
    merged_cluster_result = process_outliers(
        merged_cluster_result, outliers, distance_matrix, config["merge_threshold"]
    )

    # 打印统计信息
    print_statistics(total_samples, classified_samples, unclassified_samples, merged_cluster_result, 
                    config["clustering"]["min_samples"], error_matrix_calculation_time, clustering_time)

    # 保存结果
    save_cluster_result(
        merged_cluster_result,
        config["data"]["paths"],
        config["data"]["save_path"],
        args.hand_type,
        config
    )

    print("\n所有操作完成，OPTICS聚类结果已保存")

if __name__ == "__main__":
    main()