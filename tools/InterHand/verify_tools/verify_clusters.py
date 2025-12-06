"""
验证InterHand2.6M数据集的多视角聚类结果与图片路径、3D坐标数据的一致性

该脚本用于验证以下三个文件中的数据是否一致：
1. val_multi_view_true_clusters.json: 多视角聚类结果
2. val_image_paths.json: 图片路径文件
3. val_joint_3d.json: 3D坐标文件

验证内容包括：
1. 检查聚类中的图片是否都在图片路径文件中
2. 检查图片路径文件中的图片是否都在聚类中
3. 检查聚类中的图片是否都有对应的3D坐标
4. 统计并比较三种数据中的图片数量

输出结果包括：
- 每种手型（左手/右手）的图片数量统计
- 数据不一致的警告信息（如果有）
- 验证结果的直观显示（✓/✗）

使用方法：
python tools/InterHand/valid_tools/verify_clusters.py
"""

import os
import json
from pathlib import Path
from collections import defaultdict

# 定义路径
CLUSTER_FILE = 'data/InterHand/cluster_data/origin/val_multi_view_true_clusters.json'
IMAGE_PATH_FILE = 'data/InterHand/origin_data/val_image_paths.json'
JOINT_3D_FILE = 'data/InterHand/origin_data/val_joint_3d.json'

def load_data():
    """加载所有数据文件"""
    print("加载数据文件...")
    with open(CLUSTER_FILE, 'r') as f:
        cluster_data = json.load(f)
    with open(IMAGE_PATH_FILE, 'r') as f:
        image_paths = json.load(f)
    with open(JOINT_3D_FILE, 'r') as f:
        joint_3d = json.load(f)
    
    return cluster_data, image_paths, joint_3d

def verify_data(cluster_data, image_paths, joint_3d):
    """验证数据一致性"""
    print("\n开始验证数据...")
    
    # 统计信息
    stats = {
        'left': {
            'cluster_images': set(),
            'path_images': set(image_paths['left']),
            'joint_3d_count': len(joint_3d['left'])
        },
        'right': {
            'cluster_images': set(),
            'path_images': set(image_paths['right']),
            'joint_3d_count': len(joint_3d['right'])
        }
    }
    
    # 收集聚类中的所有图片
    for hand_type in ['left', 'right']:
        clusters = cluster_data[f'{hand_type}_clusters']
        for cluster in clusters:
            stats[hand_type]['cluster_images'].update(cluster)
    
    # 验证每种手型的数据
    for hand_type in ['left', 'right']:
        print(f"\n{hand_type}手验证结果:")
        print(f"聚类中的图片数量: {len(stats[hand_type]['cluster_images'])}")
        print(f"图片路径文件中的数量: {len(stats[hand_type]['path_images'])}")
        print(f"3D坐标文件中的数量: {stats[hand_type]['joint_3d_count']}")
        
        # 检查聚类中的图片是否都在图片路径文件中
        cluster_only = stats[hand_type]['cluster_images'] - stats[hand_type]['path_images']
        if cluster_only:
            print(f"警告：聚类中有{len(cluster_only)}张图片不在图片路径文件中")
            print("前5个示例:")
            for img in list(cluster_only)[:5]:
                print(f"  - {img}")
        
        # 检查图片路径文件中的图片是否都在聚类中
        path_only = stats[hand_type]['path_images'] - stats[hand_type]['cluster_images']
        if path_only:
            print(f"警告：图片路径文件中有{len(path_only)}张图片不在聚类中")
            print("前5个示例:")
            for img in list(path_only)[:5]:
                print(f"  - {img}")
        
        # 检查数量是否一致
        if (len(stats[hand_type]['cluster_images']) == len(stats[hand_type]['path_images']) == 
            stats[hand_type]['joint_3d_count']):
            print("✓ 所有数量一致")
        else:
            print("✗ 数量不一致")
            
        # 检查聚类中的图片是否都有对应的3D坐标
        if len(stats[hand_type]['cluster_images']) == stats[hand_type]['joint_3d_count']:
            print("✓ 聚类中的图片都有对应的3D坐标")
        else:
            print("✗ 聚类中的图片数量与3D坐标数量不一致")

def main():
    # 加载数据
    cluster_data, image_paths, joint_3d = load_data()
    
    # 验证数据
    verify_data(cluster_data, image_paths, joint_3d)
    
    print("\n验证完成!")

if __name__ == "__main__":
    main() 