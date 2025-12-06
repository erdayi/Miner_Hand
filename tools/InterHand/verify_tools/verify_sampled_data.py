'''
验证采样数据一致性工具

该工具用于验证两个采样后的数据文件中的图片路径是否完全对应。
主要功能：
1. 加载采样后的多视角聚类结果和打乱后的数据
2. 提取所有图片路径并比较
3. 检查是否有遗漏或多余的路径
4. 输出详细的验证报告

使用方法：
python tools/InterHand/origin_deal/verify_sampled_data.py
'''

import json
from pathlib import Path
from collections import defaultdict

# 定义路径
CLUSTERS_FILE = 'data/InterHand/cluster_data/origin/val_multi_view_true_clusters_sampled_5.json'
SHUFFLED_FILE = 'data/InterHand/origin_data/val_image_paths_shuffled.json'

def load_data():
    """加载数据文件"""
    print("加载数据文件...")
    with open(CLUSTERS_FILE, 'r') as f:
        clusters_data = json.load(f)
    with open(SHUFFLED_FILE, 'r') as f:
        shuffled_data = json.load(f)
    return clusters_data, shuffled_data

def extract_paths_from_clusters(clusters_data):
    """从聚类数据中提取所有图片路径"""
    paths = {
        'left': set(),
        'right': set()
    }
    
    for hand_type in ['left', 'right']:
        for cluster in clusters_data[f'{hand_type}_clusters']:
            paths[hand_type].update(cluster)
    
    return paths

def verify_paths_consistency(clusters_paths, shuffled_paths):
    """验证路径一致性"""
    print("\n验证路径一致性...")
    
    for hand_type in ['left', 'right']:
        print(f"\n{hand_type}手验证结果:")
        
        # 转换为集合以便比较
        clusters_set = clusters_paths[hand_type]
        shuffled_set = set(shuffled_paths[hand_type])
        
        # 检查数量
        print(f"聚类数据中的路径数量: {len(clusters_set)}")
        print(f"打乱数据中的路径数量: {len(shuffled_set)}")
        
        # 检查是否有遗漏的路径
        missing_in_shuffled = clusters_set - shuffled_set
        if missing_in_shuffled:
            print(f"\n在打乱数据中缺失的路径 (前5个):")
            for path in list(missing_in_shuffled)[:5]:
                print(f"- {path}")
        
        # 检查是否有多余的路径
        extra_in_shuffled = shuffled_set - clusters_set
        if extra_in_shuffled:
            print(f"\n在打乱数据中多余的路径 (前5个):")
            for path in list(extra_in_shuffled)[:5]:
                print(f"- {path}")
        
        # 检查完全匹配
        if clusters_set == shuffled_set:
            print("\n✓ 路径完全匹配!")
        else:
            print("\n✗ 路径不完全匹配!")

def main():
    # 加载数据
    clusters_data, shuffled_data = load_data()
    
    # 从聚类数据中提取路径
    clusters_paths = extract_paths_from_clusters(clusters_data)
    
    # 验证路径一致性
    verify_paths_consistency(clusters_paths, shuffled_data)
    
    print("\n验证完成!")

if __name__ == "__main__":
    main() 