'''
   本代码旨在随机打乱图片路径和关节数据的顺序，同时保持它们的一一对应关系
'''
import json
import numpy as np
from typing import Tuple, List
import os

def shuffle_paired_data(
    img_paths_json: str, 
    joint_data_json: str, 
    output_img_paths_json: str = None,
    output_joint_data_json: str = None,
    seed: int = None
) -> Tuple[List, List]:
    """
    打乱图片路径和关节数据的顺序，同时保持它们的一一对应关系
    
    参数:
    img_paths_json: str - 图片路径JSON文件路径
    joint_data_json: str - 关节数据JSON文件路径
    output_img_paths_json: str - 输出的打乱后图片路径JSON文件路径（可选）
    output_joint_data_json: str - 输出的打乱后关节数据JSON文件路径（可选）
    seed: int - 随机种子（可选）
    
    返回:
    Tuple[List, List] - 打乱后的(图片路径列表, 关节数据列表)
    """
    # 设置随机种子（如果提供）
    if seed is not None:
        np.random.seed(seed)
    
    # 加载数据
    try:
        with open(img_paths_json, 'r', encoding='utf-8') as f:
            img_paths = json.load(f)
        with open(joint_data_json, 'r', encoding='utf-8') as f:
            joint_data = json.load(f)
    except Exception as e:
        raise Exception(f"加载数据时出错: {str(e)}")
    
    # 验证数据长度是否匹配
    if len(img_paths) != len(joint_data):
        raise ValueError(f"图片路径数量({len(img_paths)})与关节数据数量({len(joint_data)})不匹配")
    
    # 生成随机排列的索引
    indices = np.arange(len(img_paths))
    np.random.shuffle(indices)
    
    # 使用相同的随机索引重排两个列表
    shuffled_img_paths = [img_paths[i] for i in indices]
    shuffled_joint_data = [joint_data[i] for i in indices]
    
    # 如果提供了输出路径，保存打乱后的数据
    if output_img_paths_json:
        os.makedirs(os.path.dirname(output_img_paths_json), exist_ok=True)
        with open(output_img_paths_json, 'w', encoding='utf-8') as f:
            json.dump(shuffled_img_paths, f, indent=4)
        print(f"已保存打乱后的图片路径到: {output_img_paths_json}")
    
    if output_joint_data_json:
        os.makedirs(os.path.dirname(output_joint_data_json), exist_ok=True)
        with open(output_joint_data_json, 'w', encoding='utf-8') as f:
            json.dump(shuffled_joint_data, f, indent=4)
        print(f"已保存打乱后的关节数据到: {output_joint_data_json}")
    
    return shuffled_img_paths, shuffled_joint_data

# 使用示例
def main():
    # 输入文件路径
    left_img_paths = 'data/minidata/new/disorder/left_hand_img_paths_clean.json'
    left_joint_data = 'data/minidata/new/disorder/left_hand_joint_3d_clean.json'
    right_img_paths = 'data/minidata/new/disorder/right_hand_img_paths_clean.json'
    right_joint_data = 'data/minidata/new/disorder/right_hand_joint_3d_clean.json'
    
    # 输出文件路径
    output_left_img = 'data/minidata/new/disorder/left_hand_img_paths_shuffled.json'
    output_left_joint = 'data/minidata/new/disorder/left_hand_joint_3d_shuffled.json'
    output_right_img = 'data/minidata/new/disorder/right_hand_img_paths_shuffled.json'
    output_right_joint = 'data/minidata/new/disorder/right_hand_joint_3d_shuffled.json'
    
    # 设置随机种子以确保结果可重现
    seed = 42
    
    # 打乱左手数据
    print("正在打乱左手数据...")
    shuffle_paired_data(
        left_img_paths, 
        left_joint_data,
        output_left_img,
        output_left_joint,
        seed=seed
    )
    
    # 打乱右手数据
    print("正在打乱右手数据...")
    shuffle_paired_data(
        right_img_paths, 
        right_joint_data,
        output_right_img,
        output_right_joint,
        seed=seed
    )
    
    print("数据打乱完成！")

if __name__ == "__main__":
    main()