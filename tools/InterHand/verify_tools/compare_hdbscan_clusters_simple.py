'''
聚类结果比较工具

该工具用于比较真实聚类结果和HDBSCAN聚类结果的差异：
1. 真实聚类结果 (val_multi_view_true_clusters_sampled_5.json)
2. HDBSCAN聚类结果 (hand_clusters_hdbscan_*.json)

主要功能：
1. 合并左右手的HDBSCAN聚类结果
2. 将聚类结果转换为标签列表
3. 计算聚类评估指标：
   - Adjusted Rand Index (ARI)
   - Adjusted Mutual Information (AMI)
4. 找出聚类不一致的样本
5. 将错误样本保存到JSON文件

使用方法：
python tools/InterHand/verify_tools/compare_clusters.py

输出：
1. 左右手的聚类评估指标
2. 聚类错误的样本列表 (cluster_mismatch.json)
'''

import json
import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import os

def load_clusters(file_path):
    """加载聚类结果文件"""
    with open(file_path, 'r') as f:
        return json.load(f)

def merge_hdbscan_clusters(left_file, right_file):
    """合并左右手的HDBSCAN聚类结果"""
    left_data = load_clusters(left_file)
    right_data = load_clusters(right_file)
    
    merged_data = {
        "left_clusters": left_data["clusters"],
        "right_clusters": right_data["clusters"]
    }
    
    # 保存合并后的结果
    output_path = 'data/InterHand/cluster_data/hdbscan/val_multi_view_hdbscan_clusters_sampled_5.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    return merged_data

def convert_to_labels(clusters):
    """将聚类结果转换为标签列表"""
    # 创建一个字典来映射文件路径到索引
    path_to_idx = {}
    idx = 0
    
    # 首先收集所有唯一的文件路径
    for cluster in clusters:
        for path in cluster:
            if path not in path_to_idx:
                path_to_idx[path] = idx
                idx += 1
    
    # 创建标签列表
    labels = np.full(len(path_to_idx), -1)  # -1表示噪声点
    
    # 为每个聚类分配标签
    for cluster_idx, cluster in enumerate(clusters):
        for path in cluster:
            labels[path_to_idx[path]] = cluster_idx
            
    return labels

def compare_clusters(true_clusters, pred_clusters):
    """比较两个聚类结果"""
    # 转换为标签列表
    true_labels = convert_to_labels(true_clusters)
    pred_labels = convert_to_labels(pred_clusters)
    
    # 确保两个标签列表长度相同
    min_length = min(len(true_labels), len(pred_labels))
    true_labels = true_labels[:min_length]
    pred_labels = pred_labels[:min_length]
    
    # 计算评估指标
    ari = adjusted_rand_score(true_labels, pred_labels)
    ami = adjusted_mutual_info_score(true_labels, pred_labels)
    
    return {
        'adjusted_rand_index': ari,
        'adjusted_mutual_info': ami
    }

def find_mismatches(true_clusters, pred_clusters):
    """找出聚类不一致的样本，返回列表[{path, true_cluster, pred_cluster}]"""
    # 收集所有唯一路径
    all_paths = set()
    for cluster in true_clusters:
        all_paths.update(cluster)
    for cluster in pred_clusters:
        all_paths.update(cluster)
    all_paths = list(all_paths)

    # 构建路径到标签的映射
    true_path2label = {}
    for cluster_idx, cluster in enumerate(true_clusters):
        for path in cluster:
            true_path2label[path] = cluster_idx
    pred_path2label = {}
    for cluster_idx, cluster in enumerate(pred_clusters):
        for path in cluster:
            pred_path2label[path] = cluster_idx

    mismatches = []
    for path in all_paths:
        true_label = true_path2label.get(path, -1)
        pred_label = pred_path2label.get(path, -1)
        if true_label != pred_label:
            mismatches.append({
                "path": path,
                "true_cluster": true_label,
                "pred_cluster": pred_label
            })
    return mismatches

def main():
    # 合并HDBSCAN聚类结果
    left_hdbscan_path = 'data/InterHand/cluster_data/hdbscan/hand_clusters_hdbscan_left_min8_th0.0003_merge0.0003.json'
    right_hdbscan_path = 'data/InterHand/cluster_data/hdbscan/hand_clusters_hdbscan_right_min8_th0.0003_merge0.0003.json'
    hdbscan_clusters = merge_hdbscan_clusters(left_hdbscan_path, right_hdbscan_path)
    
    # 加载真实聚类结果
    true_clusters_path = 'data/InterHand/cluster_data/origin/val_multi_view_true_clusters_sampled_5.json'
    true_clusters = load_clusters(true_clusters_path)
    
    # 比较左右手的聚类结果
    left_metrics = compare_clusters(true_clusters['left_clusters'], hdbscan_clusters['left_clusters'])
    right_metrics = compare_clusters(true_clusters['right_clusters'], hdbscan_clusters['right_clusters'])
    
    print("Left hand clustering comparison:")
    print(f"Adjusted Rand Index: {left_metrics['adjusted_rand_index']:.4f}")
    print(f"Adjusted Mutual Information: {left_metrics['adjusted_mutual_info']:.4f}")
    print("\nRight hand clustering comparison:")
    print(f"Adjusted Rand Index: {right_metrics['adjusted_rand_index']:.4f}")
    print(f"Adjusted Mutual Information: {right_metrics['adjusted_mutual_info']:.4f}")

    # 输出聚类错误的样本
    left_mismatches = find_mismatches(true_clusters['left_clusters'], hdbscan_clusters['left_clusters'])
    right_mismatches = find_mismatches(true_clusters['right_clusters'], hdbscan_clusters['right_clusters'])
    mismatch_output = {
        "left_errors": left_mismatches,
        "right_errors": right_mismatches
    }
    mismatch_path = 'data/InterHand/cluster_data/hdbscan/cluster_mismatch.json'
    with open(mismatch_path, 'w') as f:
        json.dump(mismatch_output, f, indent=2)
    print(f"\nMismatch samples saved to {mismatch_path}")

if __name__ == "__main__":
    main() 