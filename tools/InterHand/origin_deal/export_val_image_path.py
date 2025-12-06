import os
import json
from pathlib import Path
import re

'''
    01-导出val图片路径和关节点数据
'''
# 定义路径
BASE_DIR = Path('I:/gs/raw/InterHand2.6M_5fps_batch1')
IMAGE_DIR = BASE_DIR / 'images' / 'val' / 'Capture0'
OUTPUT_PATH_FILE = 'data/InterHand/origin_data/val_image_paths.json'
OUTPUT_JOINT_FILE = 'data/InterHand/origin_data/val_joint_3d.json'
JOINT_3D_FILE = BASE_DIR / 'annotations' / 'val' / 'InterHand2.6M_val_joint_3d.json'
DATA_FILE = BASE_DIR / 'annotations' / 'val' / 'InterHand2.6M_val_data.json'

# 确保输出目录存在
os.makedirs(os.path.dirname(OUTPUT_PATH_FILE), exist_ok=True)

def load_data_files():
    """加载所有必要的数据文件"""
    print("加载数据文件...")
    with open(JOINT_3D_FILE, 'r') as f:
        joint_3d_data = json.load(f)
    with open(DATA_FILE, 'r') as f:
        data_info = json.load(f)
    
    # 创建查找字典
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
        print(f"警告：关节点数据格式不正确，期望42个点，实际得到{len(world_coords)}个点")
        return None
    
    # 根据hand_type选择对应的21个点
    if hand_type == 'left':
        coords = world_coords[21:]  # 后21个点是左手
        valid = joint_valid[21:]    # 后21个有效性标记
    elif hand_type == 'right':
        coords = world_coords[:21]  # 前21个点是右手
        valid = joint_valid[:21]    # 前21个有效性标记
    else:
        print(f"警告：未知的手类型 {hand_type}")
        return None
    
    # 检查有效关节点的数量
    valid_count = sum(1 for v in valid if v[0] == 1.0)
    print(f"有效关节点数量: {valid_count}/21")
    
    # 如果有效关节点数量太少，返回None
    if valid_count < 15:
        print(f"警告：有效关节点数量过少 ({valid_count}/21)")
        return None
    
    # 对于无效的关节点，将其坐标设置为0
    for i in range(len(coords)):
        if valid[i][0] == 0.0:
            coords[i] = [0, 0, 0]
    
    return coords

def main():
    # 加载数据文件
    joint_3d_dict, image_dict, annotation_dict = load_data_files()
    
    # 初始化结果字典
    image_paths = {
        'left': [],
        'right': []
    }
    joint_3d_list = {
        'left': [],
        'right': []
    }
    
    # 用于排序的数据结构
    sorted_data = {
        'left': [],
        'right': []
    }
    
    # 统计信息
    total_annotations = len(annotation_dict)
    processed_annotations = 0
    skipped_annotations = 0
    
    # 遍历所有标注
    for image_id, annotation in annotation_dict.items():
        processed_annotations += 1
        if processed_annotations % 1000 == 0:
            print(f"处理进度: {processed_annotations}/{total_annotations}")
        
        # 获取图片信息
        if image_id not in image_dict:
            skipped_annotations += 1
            continue
        
        image_info = image_dict[image_id]
        hand_type = annotation['hand_type']
        
        # 只处理单手的图片
        if hand_type not in ['left', 'right']:
            skipped_annotations += 1
            continue
            
        # 检查hand_type_valid
        if annotation['hand_type_valid'] != 1:
            skipped_annotations += 1
            continue
            
        # 构建查找键
        key = f"{image_info['capture']}_{image_info['frame_idx']}"
        
        if key in joint_3d_dict:
            frame_data = joint_3d_dict[key]
            
            # 提取3D关节点坐标
            joint_coords = extract_joint_coords(frame_data, hand_type)
            
            if joint_coords is not None:
                # 保存图片路径和关节点数据，同时保存排序信息
                sorted_data[hand_type].append({
                    'image_path': image_info['file_name'],
                    'joint_coords': joint_coords,
                    'seq_name': image_info['seq_name'],
                    'frame_idx': image_info['frame_idx'],
                    'camera': image_info['camera']
                })
                
                # 打印前几个样本的信息
                if len(sorted_data[hand_type]) <= 5:
                    print(f"\n样本 {len(sorted_data[hand_type])} ({hand_type}手):")
                    print(f"图片路径: {image_info['file_name']}")
                    print(f"序列名称: {image_info['seq_name']}")
                    print(f"帧号: {image_info['frame_idx']}")
                    print(f"相机: {image_info['camera']}")
                    print(f"有效关节点数量: {sum(1 for v in joint_coords if v != [0,0,0])}/21")
                    print(f"标注ID: {annotation['id']}")
                    print(f"图片ID: {image_id}")
        else:
            skipped_annotations += 1
            if skipped_annotations <= 5:  # 只打印前几个跳过的样本
                print(f"\n跳过样本 (未找到3D关节点数据):")
                print(f"图片ID: {image_id}")
                print(f"图片路径: {image_info['file_name']}")
                print(f"Capture: {image_info['capture']}")
                print(f"Frame: {image_info['frame_idx']}")
                print(f"标注ID: {annotation['id']}")
    
    # 对每种手型的数据进行排序
    for hand_type in ['left', 'right']:
        # 按序列名和帧号排序
        sorted_data[hand_type].sort(key=lambda x: (x['seq_name'], x['frame_idx']))
        
        # 提取排序后的图片路径和关节点数据
        image_paths[hand_type] = [item['image_path'] for item in sorted_data[hand_type]]
        joint_3d_list[hand_type] = [item['joint_coords'] for item in sorted_data[hand_type]]
    
    # 输出统计信息
    print(f"\n处理完成!")
    print(f"总标注数量: {total_annotations}")
    print(f"处理标注数量: {processed_annotations}")
    print(f"跳过标注数量: {skipped_annotations}")
    print(f"左手图片数量: {len(image_paths['left'])}")
    print(f"右手图片数量: {len(image_paths['right'])}")
    print(f"左手3D关节点数量: {len(joint_3d_list['left'])}")
    print(f"右手3D关节点数量: {len(joint_3d_list['right'])}")
    
    # 保存结果
    with open(OUTPUT_PATH_FILE, 'w') as f:
        json.dump(image_paths, f, indent=2)
    with open(OUTPUT_JOINT_FILE, 'w') as f:
        json.dump(joint_3d_list, f, indent=2)
    
    print(f"\n数据已保存到:")
    print(f"图片路径: {OUTPUT_PATH_FILE}")
    print(f"关节点数据: {OUTPUT_JOINT_FILE}")

if __name__ == "__main__":
    main()