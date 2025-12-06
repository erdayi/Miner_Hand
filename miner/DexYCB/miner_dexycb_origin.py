import json
import numpy as np
import time
from tqdm import tqdm
from scipy.linalg import orthogonal_procrustes
import os

'''
    此代码使用传统锚点样本方法对DexYCB迷你测试集聚类,将结果合并于一个json文件中,构建了对应簇中锚点样本的照片路径
'''
BASE_IMAGE_PATH = "F:\\DexYCB\\dataset"  # 用户指定的基础路径

def align_w_scale(mtx1, mtx2):
    """保持原有对齐算法不变"""
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
    """保持原有挖掘逻辑，返回索引级匹配结果"""
    aligned_err_res = {}
    processed_samples = set()
    start_time = time.time()

    for anchor_idx in tqdm(anchor_idx_list, desc="Mining Anchors"):
        if anchor_idx in processed_samples:
            continue

        anchor_xyz = dataset[anchor_idx]
        matched_samples = []  # 存储匹配样本索引（去除误差值）

        for idx in range(len(dataset)):
            if idx == anchor_idx or idx in processed_samples:
                continue

            xyz_aligned = align_w_scale(anchor_xyz, dataset[idx])
            error = np.mean(np.linalg.norm(anchor_xyz - xyz_aligned, axis=1))
            if error < threshold:
                matched_samples.append(idx)
                processed_samples.add(idx)

        if matched_samples:
            aligned_err_res[anchor_idx] = matched_samples  # 直接存储索引列表
        processed_samples.add(anchor_idx)

    print(f"Total mining time: {time.time() - start_time:.2f} seconds")
    return aligned_err_res, processed_samples

def index_to_path(index_list, path_file):
    """将索引列表转换为绝对图像路径列表（增强路径拼接逻辑）"""
    try:
        with open(path_file, 'r') as f:
            relative_paths = json.load(f)
        absolute_paths = []
        for i in index_list:
            rel_path = relative_paths[i]
            # 处理相对路径可能存在的前导/和路径分隔符问题
            rel_path = rel_path.lstrip('/').replace('/', '\\')  # 统一为反斜杠并去除前导/
            # 使用os.path.join进行安全拼接（处理盘符和目录分隔）
            abs_path = os.path.join(BASE_IMAGE_PATH, rel_path)
            absolute_paths.append(abs_path)
        return absolute_paths
    except Exception as e:
        print(f"路径转换错误: {str(e)}")
        return []
def save_combined_cluster(left_res, right_res, left_path_file, right_path_file, save_path):
    """
    保存左右手聚类结果为统一JSON
    :param left_res: 左手索引级聚类结果 {锚点索引: 匹配索引列表}
    :param right_res: 右手索引级聚类结果
    :param left_path_file: 左手图像路径文件
    :param right_path_file: 右手图像路径文件
    """
    final_result = {"left_clusters": [], "right_clusters": []}

    # 处理左手聚类
    for anchor_idx, member_indices in left_res.items():
        # 合并锚点和匹配样本（去重）
        cluster_indices = [anchor_idx] + member_indices
        unique_indices = list(set(cluster_indices))  # 确保无重复索引
        cluster_paths = index_to_path(unique_indices, left_path_file)
        if cluster_paths:
            final_result["left_clusters"].append(cluster_paths)

    # 处理右手聚类
    for anchor_idx, member_indices in right_res.items():
        cluster_indices = [anchor_idx] + member_indices
        unique_indices = list(set(cluster_indices))
        cluster_paths = index_to_path(unique_indices, right_path_file)
        if cluster_paths:
            final_result["right_clusters"].append(cluster_paths)

    # 保存结果
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)
    print(f"聚类结果已保存到 {save_path}，包含左右手图像路径")

def print_statistics(dataset, aligned_err_res, processed_samples):
    """保持原有统计逻辑"""
    total_samples = len(dataset)
    classified_samples = len(processed_samples)
    num_anchors = len(aligned_err_res)
    involved_samples = set().union(*[set([k]+v) for k, v in aligned_err_res.items()])
    
    print(f"总样本数: {total_samples}")
    print(f"分类样本数: {classified_samples}")
    print(f"聚类数: {num_anchors}")
    print(f"涉及样本数: {len(involved_samples)}")

def main():
    """主函数，整合路径处理逻辑"""
    config = {
        "data": {
            "left_joints": "data/minidata/new/disorder/left_hand_joint_3d_shuffled.json",
            "right_joints": "data/minidata/new/disorder/right_hand_joint_3d_shuffled.json",
            "left_paths": "data/minidata/new/disorder/left_hand_img_paths_shuffled.json",
            "right_paths": "data/minidata/new/disorder/right_hand_img_paths_shuffled.json"
        },
        "save_path": "data/minidata/new/disorder/hand_clusters_origin_0.0033.json",
        "threshold": 0.0033
    }

    # 加载左右手数据
    left_data = np.array(json.load(open(config["data"]["left_joints"])))
    right_data = np.array(json.load(open(config["data"]["right_joints"])))
    
    # 执行聚类
    print("开始处理左手数据...")
    left_res, left_processed = mine_hand_data(
        left_data, list(range(len(left_data))), config["threshold"]
    )
    print("左手数据处理完成")
    
    print("开始处理右手数据...")
    right_res, right_processed = mine_hand_data(
        right_data, list(range(len(right_data))), config["threshold"]
    )
    print("右手数据处理完成")
    
    # 保存带路径的聚类结果
    save_combined_cluster(
        left_res, right_res,
        config["data"]["left_paths"], config["data"]["right_paths"],
        config["save_path"]
    )
    
    # 打印统计信息（可选）
    print("\n左手统计:")
    print_statistics(left_data, left_res, left_processed)
    print("\n右手统计:")
    print_statistics(right_data, right_res, right_processed)

if __name__ == "__main__":
    main()    