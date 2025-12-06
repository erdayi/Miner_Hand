'''
路径一致性验证工具

该工具用于验证两个JSON文件中的图片路径是否完全一致(基于真实聚类结果 和 打乱后的数据)
1. 聚类结果文件 (val_multi_view_true_clusters_sampled_5.json)
2. 打乱后的数据文件 (val_image_paths_shuffled.json)

主要功能：
1. 从聚类结果中提取所有图片路径
2. 从打乱数据中提取所有图片路径
3. 比较两个文件中的路径是否完全一致
4. 如果有不一致，显示具体哪些路径不一致

使用方法：
python tools/InterHand/verify_tools/compare_paths.py

输出：
1. 两个文件中的路径数量
2. 不一致的路径数量（如果有）
3. 不一致路径的示例（如果有）
4. 一致性验证结果
'''

import json
import os

def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_paths_from_clusters(clusters_data):
    """从聚类结果中提取所有路径"""
    paths = set()
    for cluster in clusters_data['left_clusters']:
        paths.update(cluster)
    for cluster in clusters_data['right_clusters']:
        paths.update(cluster)
    return paths

def main():
    # 加载文件
    paths_file = 'data/InterHand/origin_data/val_image_paths_shuffled.json'
    clusters_file = 'data/InterHand/cluster_data/origin/val_multi_view_true_clusters_sampled_5.json'
    
    print("加载文件...")
    paths_data = load_json(paths_file)
    clusters_data = load_json(clusters_file)
    
    # 从聚类结果中提取路径
    print("提取聚类结果中的路径...")
    cluster_paths = extract_paths_from_clusters(clusters_data)
    
    # 从paths文件中提取路径
    print("提取paths文件中的路径...")
    paths_file_paths = set()
    for hand_type in ['left', 'right']:
        paths_file_paths.update(paths_data[hand_type])
    
    # 比较路径
    print("\n比较结果:")
    print(f"聚类结果中的路径数量: {len(cluster_paths)}")
    print(f"Paths文件中的路径数量: {len(paths_file_paths)}")
    
    # 检查聚类结果中的路径是否都在paths文件中
    missing_in_paths = cluster_paths - paths_file_paths
    if missing_in_paths:
        print(f"\n在聚类结果中存在但在paths文件中不存在的路径数量: {len(missing_in_paths)}")
        print("前5个示例:")
        for path in list(missing_in_paths)[:5]:
            print(f"  {path}")
    
    # 检查paths文件中的路径是否都在聚类结果中
    missing_in_clusters = paths_file_paths - cluster_paths
    if missing_in_clusters:
        print(f"\n在paths文件中存在但在聚类结果中不存在的路径数量: {len(missing_in_clusters)}")
        print("前5个示例:")
        for path in list(missing_in_clusters)[:5]:
            print(f"  {path}")
    
    if not missing_in_paths and not missing_in_clusters:
        print("\n✅ 两个文件中的路径完全一致！")

if __name__ == "__main__":
    main() 