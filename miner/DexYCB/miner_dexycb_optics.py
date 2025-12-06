import json
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import OPTICS
from tqdm import tqdm
import time
import os

'''
聚类算法对DexYCB迷你测试集聚类

文件输入路径
"left_joints": "data/minidata/new/left_hand_joint_3d_clean.json",
"right_joints": "data/minidata/new/right_hand_joint_3d_clean.json",
"left_paths": "data/minidata/new/left_hand_img_paths_clean.json",
"right_paths": "data/minidata/new/right_hand_img_paths_clean.json",

聚类结果保存路径
"save_path": "data/minidata/new/hand_clusters_optics_0.0055_clean.json"

'''
BASE_IMAGE_PATH = "F:\\DexYCB\\dataset"  # Windows双反斜杠格式基础路径

def align_w_scale(mtx1, mtx2):
    """
    使用正交对齐方法进行匹配，并返回对齐后的结果。
    :param mtx1: 第一个矩阵，代表手部关键点数据
    :param mtx2: 第二个矩阵，代表手部关键点数据
    :return: 对齐后的第二个矩阵
    """
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

def save_distance_matrix(matrix, filename):
    """保存距离矩阵到 .npy 文件"""
    np.save(filename, matrix)

def load_distance_matrix(filename):
    """从 .npy 文件加载距离矩阵"""
    return np.load(filename)

def validate_distance_matrix(matrix, n_samples):
    """验证距离矩阵维度"""
    if matrix.shape != (n_samples, n_samples):
        raise ValueError(f"距离矩阵维度错误，应为({n_samples}, {n_samples})，实际为{matrix.shape}")

def index_to_path(index_list, path_file):
    """将索引列表转换为绝对图像路径列表"""
    try:
        with open(path_file, 'r') as f:
            relative_paths = json.load(f)
        absolute_paths = []
        for i in index_list:
            rel_path = relative_paths[i].lstrip('/')  # 去除前导斜杠
            abs_path = os.path.join(BASE_IMAGE_PATH, rel_path.replace('/', '\\'))  # 统一为反斜杠
            absolute_paths.append(abs_path)
        return absolute_paths
    except Exception as e:
        print(f"路径转换错误: {str(e)}")
        return []

def perform_clustering(dataset, distance_matrix_file):
    """执行聚类并返回索引级聚类结果"""
    # 加载或计算距离矩阵
    if os.path.exists(distance_matrix_file):
        print("加载已保存的误差矩阵...")
        start_time = time.time()
        try:
            distance_matrix = load_distance_matrix(distance_matrix_file)
            n_samples = len(dataset)
            validate_distance_matrix(distance_matrix, n_samples)
            error_matrix_calculation_time = time.time() - start_time
            print(f"加载耗时: {error_matrix_calculation_time:.2f} 秒")
        except Exception as e:
            print(f"加载矩阵失败: {str(e)}")
            distance_matrix = None
    else:
        distance_matrix = None

    if distance_matrix is None:
        print("计算误差矩阵...")
        start_time = time.time()
        n_samples = len(dataset)
        distance_matrix = np.zeros((n_samples, n_samples))
        for i in tqdm(range(n_samples), desc="计算误差矩阵"):
            for j in range(i + 1, n_samples):
                xyz_aligned = align_w_scale(dataset[i], dataset[j])
                error = np.mean(np.linalg.norm(dataset[i] - xyz_aligned, axis=1))
                distance_matrix[i, j] = distance_matrix[j, i] = error
        save_distance_matrix(distance_matrix, distance_matrix_file)
        error_matrix_calculation_time = time.time() - start_time
        print(f"误差矩阵计算耗时: {error_matrix_calculation_time:.2f} 秒，已保存")

    # OPTICS聚类
    print("开始聚类...")
    start_time = time.time()
    clustering = OPTICS(metric='precomputed', min_samples=8, xi=0.05)
    clustering.fit(distance_matrix)
    cluster_labels = clustering.labels_
    clustering_time = time.time() - start_time
    print(f"聚类耗时: {clustering_time:.2f} 秒")

    # 整理聚类结果
    cluster_result = {}
    for idx, label in enumerate(cluster_labels):
        if label != -1:  # 忽略离群点
            if label not in cluster_result:
                cluster_result[label] = []
            cluster_result[label].append(idx)
    
    # 按标签排序并过滤单样本簇
    sorted_clusters = {k: sorted(v) for k, v in sorted(cluster_result.items()) if len(v) > 1}
    return sorted_clusters  # 返回{label: sorted_index_list}

def save_combined_cluster(left_clusters, right_clusters, left_path_file, right_path_file, save_path):
    """保存左右手聚类结果为统一JSON（传统方法格式）"""
    final_result = {"left_clusters": [], "right_clusters": []}
    
    # 处理左手聚类
    for label in sorted(left_clusters.keys()):
        paths = index_to_path(left_clusters[label], left_path_file)
        if len(paths) > 1:  # 保留有效簇
            final_result["left_clusters"].append(paths)
    
    # 处理右手聚类
    for label in sorted(right_clusters.keys()):
        paths = index_to_path(right_clusters[label], right_path_file)
        if len(paths) > 1:
            final_result["right_clusters"].append(paths)
    
    # 保存结果
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)
    print(f"聚类结果已保存到 {save_path}，包含左右手图像路径")

if __name__ == "__main__":
    config = {
        "data": {
            "left_joints": "data/minidata/new/disorder/left_hand_joint_3d_shuffled.json",
            "right_joints": "data/minidata/new/disorder/right_hand_joint_3d_shuffled.json",
            "left_paths": "data/minidata/new/disorder/left_hand_img_paths_shuffled.json",
            "right_paths": "data/minidata/new/disorder/right_hand_img_paths_shuffled.json",
            "save_path": "data/minidata/new/disorder/hand_clusters_optics_0.0055_shuffled.json"
        },
        "distance_matrix": {
            "left": "data/minidata/new/disorder/left_distance_matrix_shuffled_0.005.npy",
            "right": "data/minidata/new/disorder/right_distance_matrix_shuffled_0.005.npy"
        }
    }

    # 加载数据
    left_data = np.array(json.load(open(config["data"]["left_joints"])))
    right_data = np.array(json.load(open(config["data"]["right_joints"])))

    # 左手聚类
    print("开始处理左手数据...")
    left_clusters = perform_clustering(left_data, config["distance_matrix"]["left"])
    print(f"左手聚类完成，有效簇数: {len(left_clusters)}")

    # 右手聚类
    print("\n开始处理右手数据...")
    right_clusters = perform_clustering(right_data, config["distance_matrix"]["right"])
    print(f"右手聚类完成，有效簇数: {len(right_clusters)}")

    # 合并保存
    save_combined_cluster(
        left_clusters, right_clusters,
        config["data"]["left_paths"], config["data"]["right_paths"],
        config["data"]["save_path"]
    )

    print("所有操作完成，OPTICS聚类结果已按传统方法格式保存")