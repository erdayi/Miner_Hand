import json
import numpy as np
import time
from tqdm import tqdm
from scipy.linalg import orthogonal_procrustes
import os
import argparse
from pathlib import Path

'''
    此代码使用传统锚点样本方法对InterHand2.6M数据集聚类,区分左右手,将结果保存为JSON文件

    # 处理左手数据
    python miner_interhand_origin.py --hand_type left --interval 5 --threshold 0.0003

    # 处理右手数据
    python miner_interhand_origin.py --hand_type right --interval 5 --threshold 0.0003
'''

# 定义基础路径
BASE_DIR = Path('I:/gs/raw/InterHand2.6M_5fps_batch1')
IMAGE_DIR = BASE_DIR / 'images' / 'val'

def align_w_scale(mtx1, mtx2):
    """使用正交对齐方法进行匹配，并返回对齐后的结果"""
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1
    return mtx2_t

def mine_hand_data(dataset, anchor_idx_list, threshold):
    """使用锚点样本方法挖掘相似手势，返回索引级匹配结果"""
    aligned_err_res = {}
    processed_samples = set()
    start_time = time.time()

    for anchor_idx in tqdm(anchor_idx_list, desc="挖掘锚点样本"):
        if anchor_idx in processed_samples:
            continue

        anchor_xyz = dataset[anchor_idx]
        matched_samples = []  # 存储匹配样本索引

        for idx in range(len(dataset)):
            if idx == anchor_idx or idx in processed_samples:
                continue

            xyz_aligned = align_w_scale(anchor_xyz, dataset[idx])
            error = np.mean(np.linalg.norm(anchor_xyz - xyz_aligned, axis=1))
            if error < threshold:
                matched_samples.append(idx)
                processed_samples.add(idx)

        if matched_samples:
            aligned_err_res[anchor_idx] = matched_samples
        processed_samples.add(anchor_idx)

    print(f"挖掘总耗时: {time.time() - start_time:.2f} 秒")
    return aligned_err_res, processed_samples

def index_to_path(index_list, path_file, hand_type):
    """将索引列表转换为绝对图像路径列表"""
    try:
        with open(path_file, 'r') as f:
            relative_paths = json.load(f)
        absolute_paths = []
        for i in index_list:
            rel_path = relative_paths[hand_type][i]
            # 统一处理路径格式
            rel_path = rel_path.replace('\\', '/').lstrip('/')  # 统一为正斜杠并去除前导/
            # 使用 Path 对象处理路径拼接
            abs_path = str(IMAGE_DIR / rel_path)
            absolute_paths.append(abs_path)
        return absolute_paths
    except Exception as e:
        print(f"路径转换错误: {str(e)}")
        return []

def save_cluster_result(cluster_result, path_file, save_path, hand_type):
    """保存聚类结果为JSON格式"""
    final_result = {"clusters": []}
    
    # 处理聚类结果
    for anchor_idx, member_indices in cluster_result.items():
        # 合并锚点和匹配样本（去重）
        cluster_indices = [anchor_idx] + member_indices
        unique_indices = list(set(cluster_indices))  # 确保无重复索引
        cluster_paths = index_to_path(unique_indices, path_file, hand_type)
        if cluster_paths:
            final_result["clusters"].append(cluster_paths)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存结果
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)
    
    print(f"聚类结果已保存到 {save_path}，包含 {len(final_result['clusters'])} 个有效簇")

def print_statistics(dataset, aligned_err_res, processed_samples):
    """打印统计信息"""
    total_samples = len(dataset)
    classified_samples = len(processed_samples)
    num_anchors = len(aligned_err_res)
    
    # 计算所有涉及的样本（包括锚点和匹配样本）
    if aligned_err_res:
        involved_samples = set().union(*[set([k]+v) for k, v in aligned_err_res.items()])
        involved_count = len(involved_samples)
    else:
        involved_count = 0
    
    print(f"总样本数: {total_samples}")
    print(f"分类样本数: {classified_samples} ({classified_samples / total_samples * 100:.2f}%)")
    print(f"聚类数: {num_anchors}")
    print(f"涉及样本数: {involved_count} ({involved_count / total_samples * 100:.2f}%)")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="InterHand2.6M数据集手部姿态聚类工具")
    
    # 数据相关参数
    parser.add_argument('--data_dir', type=str, default="data/InterHand/origin_data", 
                        help="数据目录路径")
    parser.add_argument('--hand_type', type=str, default="left", 
                        choices=["left", "right"], 
                        help="手部类型：左手(left)或右手(right)")
    parser.add_argument('--interval', type=int, default=1, 
                        help="数据采样间隔")
    parser.add_argument('--num_samples', type=int, default=3000,
                        help="要处理的图片数量，None表示处理所有图片")
    
    # 聚类相关参数
    parser.add_argument('--threshold', type=float, default=0.0003, 
                        help="聚类阈值，值越小聚类越严格")
    parser.add_argument('--output_dir', type=str, 
                        default="data/InterHand/cluster_data/origin", 
                        help="输出目录路径")
    
    return parser.parse_args()

def main():
    """主函数，整合处理逻辑"""
    # 解析命令行参数
    args = parse_args()
    
    # 构建输入输出路径
    data_path_prefix = args.data_dir
    output_path_prefix = os.path.join(args.output_dir, args.hand_type)
    
    # 格式化阈值为科学计数法字符串（例如：0.0003 -> 0003）
    threshold_str = f"{args.threshold:.4f}".replace("0.", "")
    
    # 构建配置
    config = {
        "data": {
            "joints": f"{data_path_prefix}/val_joint_3d.json",
            "paths": f"{data_path_prefix}/val_image_paths.json"
        },
        "save_path": f"{output_path_prefix}/hand_clusters_{threshold_str}_interval_{args.interval}.json",
        "threshold": args.threshold,
        "hand_type": args.hand_type,
        "num_samples": args.num_samples
    }
    
    print(f"配置信息:")
    print(f"  - 关节点数据: {config['data']['joints']}")
    print(f"  - 图像路径数据: {config['data']['paths']}")
    print(f"  - 手部类型: {config['hand_type']}")
    print(f"  - 聚类阈值: {config['threshold']}")
    print(f"  - 输出路径: {config['save_path']}")
    print(f"  - 处理图片数量: {config['num_samples'] if config['num_samples'] else '全部'}")

    # 加载手部关节点数据
    print("\n加载手部关节点数据...")
    with open(config["data"]["joints"], 'r') as f:
        joint_data = json.load(f)
    hand_data = np.array(joint_data[args.hand_type])
    
    # 如果指定了样本数量，则进行采样
    if config["num_samples"] is not None:
        if config["num_samples"] > len(hand_data):
            print(f"警告：请求的样本数量({config['num_samples']})大于总样本数({len(hand_data)})，将使用所有样本")
        else:
            # 随机采样指定数量的样本
            indices = np.random.choice(len(hand_data), config["num_samples"], replace=False)
            hand_data = hand_data[indices]
            print(f"已随机采样 {len(hand_data)} 个样本")
    
    print(f"加载完成，数据形状: {hand_data.shape}")
    
    # 执行聚类
    print("\n开始处理手部数据...")
    cluster_result, processed_samples = mine_hand_data(
        hand_data, list(range(len(hand_data))), config["threshold"]
    )
    print("手部数据处理完成")
    
    # 保存带路径的聚类结果
    save_cluster_result(
        cluster_result,
        config["data"]["paths"],
        config["save_path"],
        args.hand_type
    )
    
    # 打印统计信息
    print("\n统计信息:")
    print_statistics(hand_data, cluster_result, processed_samples)

if __name__ == "__main__":
    main()