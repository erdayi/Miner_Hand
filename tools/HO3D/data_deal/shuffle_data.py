import json
import os
import random
import numpy as np
import re

'''
    此代码主要用于打乱数据顺序，但保持图片路径和3D关节点数据的对应关系
    主要原因是按顺序聚类会影响初始方法的聚类效果,不利于真实的聚类效果评测
'''
def shuffle_data(img_paths_file, joints_file, output_dir, suffix='shuffled'):
    """
    读取图片路径和3D关节点数据，打乱顺序但保持对应关系，并保存到新文件
    
    参数:
    img_paths_file: 图片路径JSON文件
    joints_file: 3D关节点JSON文件
    output_dir: 输出目录
    suffix: 输出文件后缀
    """
    # 读取图片路径
    print(f"读取图片路径文件: {img_paths_file}")
    with open(img_paths_file, 'r') as f:
        img_paths = json.load(f)
    
    # 读取3D关节点数据
    print(f"读取3D关节点文件: {joints_file}")
    with open(joints_file, 'r') as f:
        joints_3d = json.load(f)
    
    # 确保两个数据长度一致
    assert len(img_paths) == len(joints_3d), "图片路径和3D关节点数据长度不一致！"
    print(f"数据总数: {len(img_paths)}")
    
    # 创建索引列表并打乱
    indices = list(range(len(img_paths)))
    random.shuffle(indices)
    
    # 根据打乱的索引重新排列数据
    shuffled_img_paths = [img_paths[i] for i in indices]
    shuffled_joints_3d = [joints_3d[i] for i in indices]
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取原始文件名（不含路径和扩展名）
    img_paths_basename = os.path.basename(img_paths_file)
    joints_basename = os.path.basename(joints_file)

    # 去掉_order后缀
    img_paths_base = re.sub(r'_order$', '', os.path.splitext(img_paths_basename)[0])
    joints_base = re.sub(r'_order$', '', os.path.splitext(joints_basename)[0])

    # 构建输出文件路径
    img_paths_output = os.path.join(output_dir, f"{img_paths_base}_{suffix}.json")
    joints_output = os.path.join(output_dir, f"{joints_base}_{suffix}.json")
    
    # 保存打乱后的数据
    with open(img_paths_output, 'w') as f:
        json.dump(shuffled_img_paths, f, indent=2)
    print(f"打乱后的图片路径已保存到: {img_paths_output}")
    
    with open(joints_output, 'w') as f:
        json.dump(shuffled_joints_3d, f, indent=2)
    print(f"打乱后的3D关节点数据已保存到: {joints_output}")
    
    # 验证打乱是否成功
    different_count = sum(1 for i, j in enumerate(indices) if i != j)
    print(f"打乱成功率: {different_count / len(indices) * 100:.2f}%")
    
    return img_paths_output, joints_output

if __name__ == "__main__":
    # 设置文件路径
    img_paths_file = r"data\HO3D\origin_data\single_view\img_paths_interval_1_order.json"
    joints_file = r"data\HO3D\origin_data\single_view\hand_joint_3d_interval_1_order.json"
    output_dir = r"data\HO3D\origin_data\single_view"
    # 执行打乱操作
    shuffle_data(img_paths_file, joints_file, output_dir)