import os
import json
import random
from pathlib import Path
import numpy as np
import re

'''
    03-InterHand2.6M eval打乱数据工具 基于真实聚类结果 进行打乱 用于后续的聚类
'''
# 定义路径
INPUT_CLUSTER_FILE = 'data/InterHand/cluster_data/origin/val_multi_view_true_clusters_sampled_5.json'
INPUT_PATH_FILE = 'data/InterHand/origin_data/val_image_paths.json'
INPUT_JOINT_FILE = 'data/InterHand/origin_data/val_joint_3d.json'
OUTPUT_PATH_FILE = 'data/InterHand/origin_data/val_image_paths_shuffled.json'
OUTPUT_JOINT_FILE = 'data/InterHand/origin_data/val_joint_3d_shuffled.json'

def extract_paths_from_clusters(clusters_data):
    """从聚类结果中提取所有路径"""
    paths = []
    for cluster in clusters_data['left_clusters']:
        paths.extend(cluster)
    for cluster in clusters_data['right_clusters']:
        paths.extend(cluster)
    return paths

def load_data():
    """加载数据"""
    print("加载数据文件...")
    with open(INPUT_CLUSTER_FILE, 'r') as f:
        clusters_data = json.load(f)
    with open(INPUT_PATH_FILE, 'r') as f:
        image_paths = json.load(f)
    with open(INPUT_JOINT_FILE, 'r') as f:
        joint_3d_list = json.load(f)
    return clusters_data, image_paths, joint_3d_list

def organize_data(clusters_data, image_paths, joint_3d_list):
    """组织数据"""
    # 从聚类结果中提取所有路径
    print("从聚类结果中提取路径...")
    all_paths = extract_paths_from_clusters(clusters_data)
    
    # 创建路径到索引的映射（用原始 image_paths 顺序）
    path_to_index = {}
    for hand_type in ['left', 'right']:
        for i, path in enumerate(image_paths[hand_type]):
            path_to_index[path] = (hand_type, i)
    
    # 组织数据
    organized_data = {
        'left': [],
        'right': []
    }
    
    # 统计信息
    total_paths = len(all_paths)
    processed_paths = 0
    skipped_paths = 0
    
    for path in all_paths:
        processed_paths += 1
        if processed_paths % 1000 == 0:
            print(f"处理进度: {processed_paths}/{total_paths}")
        
        if path in path_to_index:
            hand_type, index = path_to_index[path]
            organized_data[hand_type].append({
                'image_path': path,
                'joint_coords': joint_3d_list[hand_type][index]
            })
        else:
            skipped_paths += 1
            if skipped_paths <= 5:
                print(f"\n跳过路径 (未找到对应的关节点数据):")
                print(f"路径: {path}")
    
    print(f"\n处理完成!")
    print(f"总路径数量: {total_paths}")
    print(f"处理路径数量: {processed_paths}")
    print(f"跳过路径数量: {skipped_paths}")
    
    return organized_data

def shuffle_data(organized_data):
    """打乱数据顺序"""
    shuffled_data = {
        'left': [],
        'right': []
    }
    
    for hand_type in ['left', 'right']:
        # 创建索引列表
        indices = list(range(len(organized_data[hand_type])))
        # 打乱索引
        random.shuffle(indices)
        
        # 按打乱后的索引重新组织数据
        shuffled_data[hand_type] = [organized_data[hand_type][i] for i in indices]
    
    return shuffled_data

def main():
    # 加载数据
    clusters_data, image_paths, joint_3d_list = load_data()
    
    # 组织数据
    print("组织数据...")
    organized_data = organize_data(clusters_data, image_paths, joint_3d_list)
    
    # 打乱数据
    print("打乱数据顺序...")
    shuffled_data = shuffle_data(organized_data)
    
    # 准备输出数据
    output_image_paths = {
        'left': [item['image_path'] for item in shuffled_data['left']],
        'right': [item['image_path'] for item in shuffled_data['right']]
    }
    
    output_joint_3d = {
        'left': [item['joint_coords'] for item in shuffled_data['left']],
        'right': [item['joint_coords'] for item in shuffled_data['right']]
    }
    
    # 输出统计信息
    print(f"\n处理完成!")
    print(f"左手图片数量: {len(output_image_paths['left'])}")
    print(f"右手图片数量: {len(output_image_paths['right'])}")
    print(f"左手3D关节点数量: {len(output_joint_3d['left'])}")
    print(f"右手3D关节点数量: {len(output_joint_3d['right'])}")
    
    # 保存结果
    with open(OUTPUT_PATH_FILE, 'w') as f:
        json.dump(output_image_paths, f, indent=2)
    with open(OUTPUT_JOINT_FILE, 'w') as f:
        json.dump(output_joint_3d, f, indent=2)
    
    print(f"\n数据已保存到:")
    print(f"图片路径: {OUTPUT_PATH_FILE}")
    print(f"关节点数据: {OUTPUT_JOINT_FILE}")

if __name__ == "__main__":
    main() 