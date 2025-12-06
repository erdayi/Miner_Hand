'''
验证两个脚本导出的图片路径是否完全一致

该脚本用于比较：
1. export_val_image_path.py 导出的 val_image_paths.json
2. export_val_multi_view_clusters.py 导出的 val_multi_view_true_clusters.json

比较内容：
1. 每种手型的图片路径数量
2. 图片路径是否完全一致
3. 不一致的路径详情
'''

import json
from pathlib import Path

# 定义文件路径
IMAGE_PATHS_FILE = 'data/InterHand/origin_data/val_image_paths.json'
CLUSTERS_FILE = 'data/InterHand/cluster_data/origin/val_multi_view_true_clusters.json'

def load_data():
    """加载数据文件"""
    print("加载数据文件...")
    with open(IMAGE_PATHS_FILE, 'r') as f:
        image_paths = json.load(f)
    with open(CLUSTERS_FILE, 'r') as f:
        clusters = json.load(f)
    return image_paths, clusters

def extract_paths_from_clusters(clusters):
    """从聚类结果中提取所有图片路径"""
    paths = {
        'left': set(),
        'right': set()
    }
    
    for hand_type in ['left', 'right']:
        for cluster in clusters[f'{hand_type}_clusters']:
            paths[hand_type].update(cluster)
    
    return paths

def compare_paths(image_paths, cluster_paths):
    """比较两种数据中的图片路径"""
    print("\n比较结果:")
    
    for hand_type in ['left', 'right']:
        print(f"\n{hand_type}手:")
        
        # 转换为集合以便比较
        image_paths_set = set(image_paths[hand_type])
        cluster_paths_set = cluster_paths[hand_type]
        
        # 比较数量
        print(f"图片路径文件中的路径数量: {len(image_paths_set)}")
        print(f"聚类文件中的路径数量: {len(cluster_paths_set)}")
        
        # 找出差异
        only_in_image_paths = image_paths_set - cluster_paths_set
        only_in_clusters = cluster_paths_set - image_paths_set
        
        print(f"仅在图片路径文件中存在的路径数量: {len(only_in_image_paths)}")
        print(f"仅在聚类文件中存在的路径数量: {len(only_in_clusters)}")
        
        # 输出一些示例
        if only_in_image_paths:
            print("\n仅在图片路径文件中存在的路径示例:")
            for path in list(only_in_image_paths)[:5]:
                print(f"- {path}")
        
        if only_in_clusters:
            print("\n仅在聚类文件中存在的路径示例:")
            for path in list(only_in_clusters)[:5]:
                print(f"- {path}")

def main():
    # 加载数据
    image_paths, clusters = load_data()
    
    # 从聚类结果中提取路径
    cluster_paths = extract_paths_from_clusters(clusters)
    
    # 比较路径
    compare_paths(image_paths, cluster_paths)

if __name__ == "__main__":
    main()