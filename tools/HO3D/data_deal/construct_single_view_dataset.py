import os
import json
import pickle
import numpy as np
import re
from collections import defaultdict

def count_rgb_images(base_dir, split='train'):
    """统计HO3D_v2数据集中RGB图片的数量，处理单视角和多视角数据
    
    Args:
        base_dir: 数据集根目录
        split: 'train' 或 'evaluation'
    """
    # 构建完整路径
    split_dir = os.path.join(base_dir, split)
    
    # 获取所有子文件夹
    all_folders = [f for f in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, f))]
    
    # 按前缀分组
    prefix_groups = defaultdict(list)
    for folder in all_folders:
        prefix = ''.join([c for c in folder if not c.isdigit()])
        prefix_groups[prefix].append(folder)
    
    # 处理每个前缀组，只保留一个文件夹
    selected_groups = {}
    for prefix, folders in prefix_groups.items():
        sorted_folders = sorted(folders)
        selected_groups[prefix] = [sorted_folders[0]]
        
        if len(folders) > 1:
            print(f"\n{split}集 - 前缀 {prefix} 有 {len(folders)} 个文件夹:")
            print(f"  保留: {sorted_folders[0]}")
            print(f"  跳过: {', '.join(sorted_folders[1:])}")
    
    # 统计结果
    results = []
    total_count = 0
    
    print(f"\n===== HO3D_v2 {split}集RGB图片统计 =====")
    print("\n文件夹列表：")
    print("-" * 50)
    
    for prefix, folders in sorted(selected_groups.items()):
        folder = folders[0]
        rgb_folder = os.path.join(split_dir, folder, 'rgb')
        
        if not os.path.exists(rgb_folder):
            print(f"{folder}: rgb文件夹不存在，跳过")
            continue
        
        image_count = 0
        for file in os.listdir(rgb_folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_count += 1
        
        print(f"{folder}: {image_count}张图片")
        results.append((folder, image_count))
        total_count += image_count
    
    print("-" * 50)
    print(f"总计: {len(selected_groups)}个前缀组，{len(results)}个文件夹，{total_count}张图片")
    
    return selected_groups, total_count

def construct_single_view_dataset(base_dir, output_dir, interval=3):
    """构建数据集,每隔interval帧选取一个,生成img_paths.json和hand_joint_3d.json
    
    Args:
        base_dir: 原始数据集目录
        output_dir: 输出目录
        interval: 采样间隔
    """
    # 获取train和evaluation的数据组
    train_groups, train_total = count_rgb_images(base_dir, 'train')
    eval_groups, eval_total = count_rgb_images(base_dir, 'evaluation')
    
    # 准备数据
    img_paths = []
    hand_joint_3d = []
    
    print("\n===== 构建数据集 =====")
    print(f"采样间隔: 每{interval}帧选取一帧")
    
    # 处理train和evaluation数据
    for split, groups in [('train', train_groups), ('evaluation', eval_groups)]:
        print(f"\n处理{split}集...")
        
        for prefix, folders in groups.items():
            folder = folders[0]
            rgb_folder = os.path.join(base_dir, split, folder, 'rgb')
            meta_folder = os.path.join(base_dir, split, folder, 'meta')
            
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
                img_path = os.path.join(split, folder, 'rgb', rgb_file)
                
                # 对应的meta文件
                meta_file = os.path.splitext(rgb_file)[0] + '.pkl'
                meta_path = os.path.join(meta_folder, meta_file)
                
                if not os.path.exists(meta_path):
                    print(f"警告: {meta_path} 不存在，跳过")
                    continue
                
                try:
                    with open(meta_path, 'rb') as f:
                        meta_data = pickle.load(f)
                    
                    joints = meta_data.get('handJoints3D')
                    if joints is None:
                        print(f"警告: {meta_path} 中没有handJoints3D数据，跳过")
                        continue
                    
                    # 确保关节点数据是numpy数组
                    if not isinstance(joints, np.ndarray):
                        joints = np.array(joints)
                    
                    # 验证关节点数据的shape
                    if joints.shape != (21, 3):
                        print(f"警告: {meta_path} 中的关节点数据shape不正确 ({joints.shape})，应为(21, 3)，跳过")
                        continue
                    
                    # 验证数据是否包含NaN或无穷大
                    if np.isnan(joints).any() or np.isinf(joints).any():
                        print(f"警告: {meta_path} 中的关节点数据包含NaN或无穷大，跳过")
                        continue
                    
                    # 数据验证通过，添加到列表
                    img_paths.append(img_path)
                    hand_joint_3d.append(joints.tolist())
                    
                except Exception as e:
                    print(f"错误: 无法读取 {meta_path}: {e}")
                    continue
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为JSON文件
    img_paths_file = os.path.join(output_dir, f'img_paths_interval_{interval}_order.json')
    hand_joint_3d_file = os.path.join(output_dir, f'hand_joint_3d_interval_{interval}_order.json')
    
    with open(img_paths_file, 'w') as f:
        json.dump(img_paths, f, indent=2)
    
    with open(hand_joint_3d_file, 'w') as f:
        json.dump(hand_joint_3d, f, indent=2)
    
    print(f"\n处理完成!")
    print(f"训练集: {train_total}张图片")
    print(f"评估集: {eval_total}张图片")
    print(f"总共选取了 {len(img_paths)} 个有效样本")
    print(f"图片路径已保存到: {img_paths_file}")
    print(f"手部3D关节已保存到: {hand_joint_3d_file}")
    
    # 验证保存的数据
    print("\n验证保存的数据...")
    try:
        with open(hand_joint_3d_file, 'r') as f:
            saved_joints = json.load(f)
        
        # 验证所有样本的shape
        all_valid = True
        for i, joints in enumerate(saved_joints):
            if len(joints) != 21 or len(joints[0]) != 3:
                print(f"错误: 样本 {i} 的shape不正确: {len(joints)}x{len(joints[0])}")
                all_valid = False
        
        if all_valid:
            print("✅ 所有样本的关节点数据格式正确 (21x3)")
    except Exception as e:
        print(f"验证失败: {e}")

if __name__ == "__main__":
    # 设置数据集路径
    dataset_dir = r"F:\GS\Dataset\HO3D_v2"  # 注意这里改为根目录
    output_dir = r"G:\GS\Tencent\Miner\data\HO3D\origin_data\single_view"
    
    # 构建数据集，每隔5帧选取一个
    construct_single_view_dataset(dataset_dir, output_dir, interval=1) 