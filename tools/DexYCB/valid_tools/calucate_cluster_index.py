import json
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score
import numpy as np
from collections import defaultdict
import os

'''
    本代码用于评估聚类结果(需真实聚类类别)
    输入：
        1. 聚类结果json文件
        2. 真实分类结果json文件
    输出：
        聚类评估指标
            - 计算调整兰德指数 (ARI)
            - 计算标准化互信息 (NMI)
            - 计算同质性分数和完整性分数
            - 计算簇纯度和类别完整性
'''

def create_label_mapping(cluster_data, class_data, sequence_indices=None):
    """
    创建样本到标签的映射
    :param cluster_data: 聚类结果
    :param class_data: 真实分类结果
    :param sequence_indices: 要处理的序列索引列表，如果为None则处理所有序列
    :return: (cluster_labels, true_labels, all_samples)
    """
    # 获取所有样本路径
    all_samples = set()
    
    # 处理聚类标签
    for i, cluster in enumerate(cluster_data.get("left_clusters", [])):
        if sequence_indices is None or i in sequence_indices:
            all_samples.update(cluster)
    for i, cluster in enumerate(cluster_data.get("right_clusters", [])):
        if sequence_indices is None or i in sequence_indices:
            all_samples.update(cluster)
    
    # 创建样本到标签的映射
    sample_to_cluster = {}
    sample_to_true = {}
    
    # 处理聚类标签
    for i, cluster in enumerate(cluster_data.get("left_clusters", [])):
        if sequence_indices is None or i in sequence_indices:
            for sample in cluster:
                sample_to_cluster[sample] = i
    for i, cluster in enumerate(cluster_data.get("right_clusters", [])):
        if sequence_indices is None or i in sequence_indices:
            for sample in cluster:
                sample_to_cluster[sample] = i + len(cluster_data.get("left_clusters", []))
    
    # 处理真实标签
    for i, cluster in enumerate(class_data.get("left_clusters", [])):
        if sequence_indices is None or i in sequence_indices:
            for sample in cluster:
                sample_to_true[sample] = i
    for i, cluster in enumerate(class_data.get("right_clusters", [])):
        if sequence_indices is None or i in sequence_indices:
            for sample in cluster:
                sample_to_true[sample] = i + len(class_data.get("left_clusters", []))
    
    # 转换为标签数组
    cluster_labels = []
    true_labels = []
    for sample in all_samples:
        cluster_labels.append(sample_to_cluster.get(sample, -1))
        true_labels.append(sample_to_true.get(sample, -1))
    
    return np.array(cluster_labels), np.array(true_labels), list(all_samples)

def calculate_cluster_metrics(cluster_data, class_data, sequence_indices=None):
    """
    计算聚类评估指标
    :param cluster_data: 聚类结果
    :param class_data: 真实分类结果
    :param sequence_indices: 要处理的序列索引列表，如果为None则处理所有序列
    :return: 评估指标字典
    """
    # 创建标签映射
    cluster_labels, true_labels, all_samples = create_label_mapping(cluster_data, class_data, sequence_indices)
    
    # 计算各项指标
    metrics = {
        'adjusted_rand_score': adjusted_rand_score(true_labels, cluster_labels),
        'normalized_mutual_info': normalized_mutual_info_score(true_labels, cluster_labels),
        'homogeneity': homogeneity_score(true_labels, cluster_labels),
        'completeness': completeness_score(true_labels, cluster_labels)
    }
    
    # 计算簇的纯度
    cluster_purities = calculate_cluster_purity(cluster_labels, true_labels)
    metrics['average_purity'] = np.mean(cluster_purities)
    
    # 计算簇的完整性
    cluster_completeness = calculate_cluster_completeness(cluster_labels, true_labels)
    metrics['average_completeness'] = np.mean(cluster_completeness)
    
    return metrics

def calculate_cluster_purity(cluster_labels, true_labels):
    """
    计算每个簇的纯度
    :param cluster_labels: 聚类标签
    :param true_labels: 真实标签
    :return: 每个簇的纯度列表
    """
    purities = []
    for cluster in np.unique(cluster_labels):
        mask = cluster_labels == cluster
        if np.sum(mask) == 0:
            continue
        true_labels_in_cluster = true_labels[mask]
        most_common = np.bincount(true_labels_in_cluster).max()
        purity = most_common / len(true_labels_in_cluster)
        purities.append(purity)
    return purities

def calculate_cluster_completeness(cluster_labels, true_labels):
    """
    计算每个真实类别的完整性
    :param cluster_labels: 聚类标签
    :param true_labels: 真实标签
    :return: 每个真实类别的完整性列表
    """
    completeness = []
    for true_class in np.unique(true_labels):
        mask = true_labels == true_class
        if np.sum(mask) == 0:
            continue
        cluster_labels_in_class = cluster_labels[mask]
        most_common = np.bincount(cluster_labels_in_class).max()
        comp = most_common / len(cluster_labels_in_class)
        completeness.append(comp)
    return completeness

def print_metrics(metrics):
    """打印评估指标"""
    print("\n================= 聚类评估结果 =================")
    print(f"调整兰德指数 (ARI): {metrics['adjusted_rand_score']:.4f}")
    print(f"标准化互信息 (NMI): {metrics['normalized_mutual_info']:.4f}")
    print(f"同质性分数: {metrics['homogeneity']:.4f}")
    print(f"完整性分数: {metrics['completeness']:.4f}")
    print(f"平均簇纯度: {metrics['average_purity']:.4f}")
    print(f"平均类别完整性: {metrics['average_completeness']:.4f}")

if __name__ == "__main__":
    # 配置文件路径
    CLUSTER_JSON = "data/minidata/new/hand_clusters_optics_0.0055.json"
    CLASSIFICATION_JSON = "data/minidata/new/classification_total_result.json"
    
    # 设置要处理的序列索引
    sequence_indices = list(range(0, 100, 2))  # 每隔一个取一个索引，从0到98

    # 加载数据
    print("正在加载数据...")
    with open(CLUSTER_JSON, 'r', encoding='utf-8') as f:
        cluster_data = json.load(f)
    with open(CLASSIFICATION_JSON, 'r', encoding='utf-8') as f:
        class_data = json.load(f)

    # 计算评估指标
    print("正在计算评估指标...")
    metrics = calculate_cluster_metrics(cluster_data, class_data, sequence_indices)
    
    # 打印结果
    print_metrics(metrics)