'''
02-InterHand2.6M eval真实聚类导出工具

InterHand2.6M数据集多视角聚类导出工具

该工具用于从InterHand2.6M数据集中提取和组织多视角手部图像数据。
主要功能：
1. 加载原始数据集信息（图像标注和3D关节点数据）
2. 提取和组织多视角数据，确保每个视角都有有效的3D关节点数据
3. 按序列和帧号对数据进行排序
4. 进行跨帧采样（每5帧取1帧）
5. 导出多视角聚类结果，每个聚类包含同一帧的多个视角图像

数据组织方式：
- 按序列(sequence) -> 帧(frame) -> 手型(hand_type) -> 视角(views)进行组织
- 只保留至少有2个视角的帧
- 确保每个视角都有至少15个有效的3D关节点
- 每5帧取1帧进行采样

输出：
1. 多视角聚类结果JSON文件，包含：
   - left_clusters: 左手多视角聚类结果
   - right_clusters: 右手多视角聚类结果
2. 统计信息，包含：
   - 每种手型的聚类数量
   - 每种手型的总视角数
   - 平均每聚类的视角数

使用方法：
python tools/InterHand/origin_deal/export_val_multi_view_clusters.py --skip_interval 5

注意：
- 需要确保BASE_DIR指向正确的InterHand2.6M数据集路径
- 输出目录会自动创建
- 处理大量数据时可能需要较长时间
'''

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

# 定义路径
BASE_DIR = Path('I:/gs/raw/InterHand2.6M_5fps_batch1')
IMAGE_DIR = BASE_DIR / 'images' / 'val' / 'Capture0'
OUTPUT_DIR = 'data/InterHand/cluster_data/origin'
DATA_FILE = BASE_DIR / 'annotations' / 'val' / 'InterHand2.6M_val_data.json'
JOINT_3D_FILE = BASE_DIR / 'annotations' / 'val' / 'InterHand2.6M_val_joint_3d.json'

def load_data():
    """加载数据集信息"""
    print("加载数据文件...")
    with open(DATA_FILE, 'r') as f:
        data_info = json.load(f)
    with open(JOINT_3D_FILE, 'r') as f:
        joint_3d_data = json.load(f)
    
    # 创建图片信息字典
    image_dict = {img['id']: img for img in data_info['images']}
    annotation_dict = {ann['image_id']: ann for ann in data_info['annotations']}
    
    # 创建3D关节点数据的快速查找字典
    joint_3d_dict = {}
    for capture_id, capture_data in joint_3d_data.items():
        for frame_idx, frame_data in capture_data.items():
            key = f"{capture_id}_{frame_idx}"
            joint_3d_dict[key] = frame_data
    
    return joint_3d_dict, image_dict, annotation_dict

def extract_joint_coords(joint_data, hand_type):
    """从joint_data中提取指定手类型的21个关键点的3D坐标"""
    if not joint_data:
        return None
    
    world_coords = joint_data['world_coord']
    joint_valid = joint_data['joint_valid']
    
    # 确保数据格式正确
    if len(world_coords) != 42 or len(joint_valid) != 42:
        return None
    
    # 根据hand_type选择对应的21个点
    if hand_type == 'left':
        coords = world_coords[21:]  # 后21个点是左手
        valid = joint_valid[21:]    # 后21个有效性标记
    elif hand_type == 'right':
        coords = world_coords[:21]  # 前21个点是右手
        valid = joint_valid[:21]    # 前21个有效性标记
    else:
        return None
    
    # 检查有效关节点的数量
    valid_count = sum(1 for v in valid if v[0] == 1.0)
    
    # 如果有效关节点数量太少，返回None
    if valid_count < 15:
        return None
    
    return True  # 只返回是否有效

def organize_multi_view_data(joint_3d_dict, image_dict, annotation_dict):
    """组织多视角数据"""
    # 统计信息
    total_annotations = len(annotation_dict)
    processed_annotations = 0
    skipped_annotations = 0
    
    # 收集所有数据
    all_data = []
    for image_id, image_info in image_dict.items():
        processed_annotations += 1
        if processed_annotations % 1000 == 0:
            print(f"处理进度: {processed_annotations}/{total_annotations}")
            
        if image_id not in annotation_dict:
            skipped_annotations += 1
            continue
            
        annotation = annotation_dict[image_id]
        hand_type = annotation['hand_type']
        
        # 只处理单手的图片
        if hand_type not in ['left', 'right']:
            skipped_annotations += 1
            continue
            
        # 检查hand_type_valid
        if annotation['hand_type_valid'] != 1:
            skipped_annotations += 1
            continue
            
        # 获取序列和帧信息
        sequence = image_info['seq_name']
        frame = image_info['frame_idx']
        
        # 检查3D关节点数据
        key = f"{image_info['capture']}_{frame}"
        if key in joint_3d_dict:
            frame_data = joint_3d_dict[key]
            if extract_joint_coords(frame_data, hand_type):
                all_data.append({
                    'seq_name': sequence,
                    'frame_idx': frame,
                    'hand_type': hand_type,
                    'view': image_info['file_name']
                })
            else:
                skipped_annotations += 1
        else:
            skipped_annotations += 1
    
    # 按序列名和帧号排序
    all_data.sort(key=lambda x: (x['seq_name'], x['frame_idx']))
    
    # 按序列组织数据
    sequence_data = {
        'left': defaultdict(list),
        'right': defaultdict(list)
    }
    
    for item in all_data:
        sequence_data[item['hand_type']][item['seq_name']].append(item)
    
    print(f"\n处理完成!")
    print(f"总标注数量: {total_annotations}")
    print(f"处理标注数量: {processed_annotations}")
    print(f"跳过标注数量: {skipped_annotations}")
    
    return sequence_data

def skip_frames(sequence_data, skip_interval):
    """跨帧采样数据"""
    sampled_data = {
        'left': [],
        'right': []
    }
    
    for hand_type in ['left', 'right']:
        # 获取所有序列名并按字典序排序
        seq_names = sorted(sequence_data[hand_type].keys())
        
        for seq_name in seq_names:
            frames = sequence_data[hand_type][seq_name]
            # 按帧号排序
            frames.sort(key=lambda x: x['frame_idx'])
            # 跨帧采样
            for i in range(0, len(frames), skip_interval):
                sampled_data[hand_type].append(frames[i])
    
    return sampled_data

def export_multi_view_clusters(sampled_data, skip_interval):
    """导出多视角聚类结果"""
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 统计信息
    stats = {
        'left': {'clusters': 0, 'total_views': 0},
        'right': {'clusters': 0, 'total_views': 0}
    }
    
    # 处理数据
    result = {
        'left_clusters': [],
        'right_clusters': []
    }
    
    # 按序列和帧组织数据
    organized_data = {
        'left': defaultdict(lambda: defaultdict(list)),
        'right': defaultdict(lambda: defaultdict(list))
    }
    
    # 组织数据
    for hand_type in ['left', 'right']:
        for item in sampled_data[hand_type]:
            organized_data[hand_type][item['seq_name']][item['frame_idx']].append(item['view'])
    
    # 生成聚类
    for hand_type in ['left', 'right']:
        # 获取所有序列名并按字典序排序
        seq_names = sorted(organized_data[hand_type].keys())
        
        for sequence in seq_names:
            # 获取该序列的所有帧并按帧号排序
            frames = sorted(organized_data[hand_type][sequence].keys())
            for frame in frames:
                views = organized_data[hand_type][sequence][frame]
                    # 只保留至少有两个视角的帧
                if len(views) >= 2:
                        result[f'{hand_type}_clusters'].append(views)
    
    # 更新统计信息
    for hand_type in ['left', 'right']:
        stats[hand_type]['clusters'] = len(result[f'{hand_type}_clusters'])
        stats[hand_type]['total_views'] = sum(len(views) for views in result[f'{hand_type}_clusters'])
    
    # 保存聚类结果
    output_file = os.path.join(OUTPUT_DIR, f'val_multi_view_true_clusters_sampled_{skip_interval}.json')
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n统计信息:")
    for hand_type in ['left', 'right']:
        print(f"\n{hand_type}手:")
        print(f"聚类数量: {stats[hand_type]['clusters']}")
        print(f"总视角数: {stats[hand_type]['total_views']}")
        if stats[hand_type]['clusters'] > 0:
            print(f"平均每聚类视角数: {stats[hand_type]['total_views'] / stats[hand_type]['clusters']:.2f}")
    print(f"\n聚类结果已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="对多视角聚类结果进行跨帧采样")
    parser.add_argument('--skip_interval', type=int, default=5,
                      help="跨帧采样的间隔（默认：5）")
    args = parser.parse_args()
    
    # 加载数据
    joint_3d_dict, image_dict, annotation_dict = load_data()
    
    # 组织数据
    print("组织数据...")
    sequence_data = organize_multi_view_data(joint_3d_dict, image_dict, annotation_dict)
    
    # 跨帧采样
    print(f"执行跨帧采样 (间隔: {args.skip_interval})...")
    sampled_data = skip_frames(sequence_data, args.skip_interval)
    
    # 导出多视角聚类结果
    print("导出多视角聚类结果...")
    export_multi_view_clusters(sampled_data, args.skip_interval)
    
    print("\n处理完成!")

if __name__ == "__main__":
    main() 