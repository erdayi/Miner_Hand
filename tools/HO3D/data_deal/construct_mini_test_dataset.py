import os
import csv
import json
import pickle
import numpy as np
import re
from collections import defaultdict

def count_rgb_images(base_dir):
    """统计HO3D_v2训练集中RGB图片的数量，只统计有五个前缀相同且RGB图片数量一致的文件夹"""
    
    # 获取所有子文件夹
    all_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    # 按前缀分组
    prefix_groups = defaultdict(list)
    for folder in all_folders:
        # 提取前缀（所有非数字字符）
        prefix = ''.join([c for c in folder if not c.isdigit()])
        prefix_groups[prefix].append(folder)
    
    # 只保留有5个子文件夹的前缀组
    valid_groups = {prefix: folders for prefix, folders in prefix_groups.items() if len(folders) == 5}
    
    # 进一步筛选RGB图片数量一致的组
    consistent_groups = {}
    
    print("===== 检查RGB图片数量一致性 =====")
    for prefix, folders in valid_groups.items():
        # 统计每个文件夹中的RGB图片数量
        folder_counts = {}
        for folder in folders:
            rgb_folder = os.path.join(base_dir, folder, 'rgb')
            
            # 检查rgb文件夹是否存在
            if not os.path.exists(rgb_folder):
                print(f"{folder}: rgb文件夹不存在，跳过")
                continue
            
            # 统计图片数量
            image_count = 0
            for file in os.listdir(rgb_folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_count += 1
            
            folder_counts[folder] = image_count
        
        # 检查是否所有文件夹的图片数量都一致
        if len(folder_counts) == 5:  # 确保所有5个文件夹都有rgb目录
            counts = list(folder_counts.values())
            if all(count == counts[0] for count in counts):
                print(f"前缀 {prefix}: 所有5个文件夹的RGB图片数量一致，每个文件夹 {counts[0]} 张图片")
                consistent_groups[prefix] = folders
            else:
                print(f"前缀 {prefix}: RGB图片数量不一致，跳过 - {folder_counts}")
    
    # 统计结果
    results = []
    total_count = 0
    
    print("\n===== HO3D_v2 训练集RGB图片统计（仅包含图片数量一致的组） =====")
    
    # 遍历每个有效的前缀组
    for prefix, folders in consistent_groups.items():
        for folder in sorted(folders):
            rgb_folder = os.path.join(base_dir, folder, 'rgb')
            
            # 统计图片数量
            image_count = 0
            for file in os.listdir(rgb_folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_count += 1
            
            # 打印结果
            print(f"{folder}: {image_count}张图片")
            
            # 保存结果
            results.append((folder, image_count))
            total_count += image_count
    
    # 打印总计
    print(f"总计: {len(consistent_groups)}个前缀组，{len(results)}个文件夹，{total_count}张图片")
    
    return consistent_groups

def generate_ground_truth_clusters(base_dir, output_file, consistent_groups, interval=5):
    """生成真实聚类标签，基于相同前缀和帧号的图像属于同一个手势的假设"""
    
    # 生成聚类
    clusters = []
    
    for prefix, folders in consistent_groups.items():
        # 对文件夹按数字后缀排序
        sorted_folders = sorted(folders, key=lambda x: int(re.search(r'\d+', x).group()))
        
        # 获取第一个文件夹中的图片列表作为参考
        reference_folder = sorted_folders[0]
        rgb_folder = os.path.join(base_dir, reference_folder, 'rgb')
        
        if not os.path.exists(rgb_folder):
            print(f"警告: {rgb_folder} 不存在，跳过")
            continue
        
        # 获取所有图片文件并排序
        rgb_files = [f for f in os.listdir(rgb_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        rgb_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        
        # 每隔interval帧选取一个
        selected_files = rgb_files[::interval]
        
        for rgb_file in selected_files:
            # 为每个选定的帧创建一个聚类
            cluster = []
            
            for folder in sorted_folders:
                img_path = os.path.join(base_dir, folder, 'rgb', rgb_file)
                if os.path.exists(img_path):
                    # 使用绝对路径
                    abs_path = os.path.abspath(img_path).replace('/', '\\')
                    cluster.append(abs_path)
            
            # 只添加包含所有5个视角的聚类
            if len(cluster) == 5:
                clusters.append(cluster)
    
    # 保存结果
    result = {"clusters": clusters}
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    print(f"生成了 {len(clusters)} 个真实聚类，已保存到 {output_file}")
    return result

def construct_mini_dataset(base_dir, output_dir, interval=5):
    """构建迷你数据集,每隔interval帧选取一个,生成img_paths.json和hand_joint_3d.json
    只处理有五个前缀相同且RGB图片数量一致的文件夹，同时生成真实聚类标签"""
    
    # 获取符合条件的前缀组（五个子文件夹前缀一致且RGB图片数量一致）
    consistent_groups = count_rgb_images(base_dir)
    
    # 准备数据
    img_paths = []
    hand_joint_3d = []
    
    print("\n===== 构建迷你数据集 =====")
    print(f"采样间隔: 每{interval}帧选取一帧")
    
    # 遍历每个有效的前缀组
    for prefix, folders in consistent_groups.items():
        for folder in sorted(folders):
            rgb_folder = os.path.join(base_dir, folder, 'rgb')
            meta_folder = os.path.join(base_dir, folder, 'meta')
            
            # 检查rgb和meta文件夹是否存在
            if not os.path.exists(rgb_folder) or not os.path.exists(meta_folder):
                print(f"{folder}: rgb或meta文件夹不存在，跳过")
                continue
            
            # 获取所有图片文件并排序
            rgb_files = [f for f in os.listdir(rgb_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            rgb_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
            
            # 每隔interval帧选取一个
            selected_files = rgb_files[::interval]
            
            for rgb_file in selected_files:
                # 图片路径
                img_path = os.path.join('train', folder, 'rgb', rgb_file)
                img_paths.append(img_path)
                
                # 对应的meta文件
                meta_file = os.path.splitext(rgb_file)[0] + '.pkl'
                meta_path = os.path.join(meta_folder, meta_file)
                
                if not os.path.exists(meta_path):
                    print(f"警告: {meta_path} 不存在，跳过")
                    img_paths.pop()  # 移除刚刚添加的图片路径
                    continue
                
                # 读取meta文件
                try:
                    with open(meta_path, 'rb') as f:
                        meta_data = pickle.load(f)
                    
                    # 提取handJoints3D
                    joints = meta_data.get('handJoints3D')
                    if joints is not None:
                        # 将numpy数组转换为列表
                        joints_list = joints.tolist() if isinstance(joints, np.ndarray) else joints
                        hand_joint_3d.append(joints_list)
                    else:
                        print(f"警告: {meta_path} 中没有handJoints3D数据，跳过")
                        img_paths.pop()  # 移除刚刚添加的图片路径
                except Exception as e:
                    print(f"错误: 无法读取 {meta_path}: {e}")
                    img_paths.pop()  # 移除刚刚添加的图片路径
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为JSON文件，文件名中包含interval参数
    img_paths_file = os.path.join(output_dir, f'img_paths_interval_{interval}.json')
    hand_joint_3d_file = os.path.join(output_dir, f'hand_joint_3d_interval_{interval}.json')
    
    with open(img_paths_file, 'w') as f:
        json.dump(img_paths, f, indent=2)
    
    with open(hand_joint_3d_file, 'w') as f:
        json.dump(hand_joint_3d, f, indent=2)
    
    print(f"处理完成! 共选取了 {len(img_paths)} 个样本")
    print(f"图片路径已保存到: {img_paths_file}")
    print(f"手部3D关节已保存到: {hand_joint_3d_file}")
    
    # 生成真实聚类标签
    ground_truth_dir = os.path.join(os.path.dirname(output_dir), 'cluster_data', 'ground_truth')
    ground_truth_file = os.path.join(ground_truth_dir, f'hand_clusters_interval_{interval}.json')
    generate_ground_truth_clusters(base_dir, ground_truth_file, consistent_groups, interval)

if __name__ == "__main__":
    # 设置数据集路径
    dataset_dir = r"F:\GS\Dataset\HO3D_v2\train"
    output_dir = r"G:\GS\Tencent\Miner\data\HO3D\origin_data"
    
    # 构建迷你数据集，每隔5帧选取一个，同时生成真实聚类标签
    construct_mini_dataset(dataset_dir, output_dir, interval=5)