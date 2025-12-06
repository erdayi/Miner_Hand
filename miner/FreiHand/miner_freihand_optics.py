import json
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import OPTICS
from tqdm import tqdm
import time
import os

'''
    本方法基于OPTICS算法对FreiHand相似手势进行聚类，聚类结果保存为 clusters_optics_32560_0.003_min8_xi0.05.json
'''

def align_w_scale(mtx1, mtx2):
    """
    使用正交对齐方法进行匹配，并返回对齐后的结果。
    :param mtx1: 第一个矩阵，代表手部关键点数据
    :param mtx2: 第二个矩阵，代表手部关键点数据
    :return: 对齐后的第二个矩阵
    """
    # 计算两个矩阵的均值
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    # 减去均值进行中心化
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # 计算矩阵的范数，并添加一个小的常数避免除零错误
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    # 归一化矩阵
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # 使用正交普罗克拉斯提斯分析计算旋转矩阵和缩放因子
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)
    # 应用旋转和缩放
    mtx2_t = np.dot(mtx2_t, R.T) * s
    # 恢复到原始尺度和位置
    mtx2_t = mtx2_t * s1 + t1
    return mtx2_t


def save_distance_matrix(matrix, filename):
    """
    保存距离矩阵到 .npy 文件。
    :param matrix: 距离矩阵
    :param filename: 保存的文件名
    """
    np.save(filename, matrix)


def load_distance_matrix(filename):
    """
    从 .npy 文件加载距离矩阵。
    :param filename: 要加载的文件名
    :return: 加载的距离矩阵
    """
    return np.load(filename)


def validate_distance_matrix(matrix, n_samples):
    """
    验证距离矩阵是否与当前数据集兼容。
    :param matrix: 距离矩阵
    :param n_samples: 当前数据集的样本数量
    :raises ValueError: 如果矩阵维度与数据集不匹配
    """
    if matrix.shape != (n_samples, n_samples):
        raise ValueError("距离矩阵维度与当前数据集不匹配")


def read_dataset(json_path, max_samples):
    """读取数据集"""
    start_time = time.time()
    try:
        # 打开 JSON 文件并加载数据
        with open(json_path, 'r') as f:
            dataset = json.load(f)
        # 限制最大样本数
        dataset = np.array(dataset[:max_samples])
        n_samples = len(dataset)
        # 计算数据读取耗时
        data_reading_time = time.time() - start_time
        print(f"数据读取耗时: {data_reading_time:.2f} 秒")
        return dataset, n_samples
    except Exception as e:
        print(f"读取数据失败: {str(e)}")
        exit()


def calculate_distance_matrix(dataset, distance_matrix_file):
    """计算距离矩阵"""
    n_samples = len(dataset)
    start_time = time.time()
    # 初始化距离矩阵
    distance_matrix = np.zeros((n_samples, n_samples))

    # 双重循环计算每对样本之间的距离
    for i in tqdm(range(n_samples), desc="计算误差矩阵"):
        for j in range(i + 1, n_samples):
            # 计算对齐后的矩阵
            xyz_aligned = align_w_scale(dataset[i], dataset[j])
            # 计算对齐误差
            error = np.mean(np.linalg.norm(dataset[i] - xyz_aligned, axis=1))
            # 填充距离矩阵
            distance_matrix[i, j] = error
            distance_matrix[j, i] = error

    # 计算误差矩阵计算耗时
    error_matrix_calculation_time = time.time() - start_time
    print(f"误差矩阵计算耗时: {error_matrix_calculation_time:.2f} 秒")
    # 保存距离矩阵到文件
    save_distance_matrix(distance_matrix, distance_matrix_file)
    print("误差矩阵已保存")
    return distance_matrix, error_matrix_calculation_time


def optics_clustering(distance_matrix, min_samples, xi):
    """使用 OPTICS 进行聚类"""
    start_time = time.time()
    print("开始聚类...")
    # 初始化 OPTICS 聚类器，使用预计算的距离矩阵
    clustering = OPTICS(metric='precomputed', min_samples=min_samples, xi=xi)
    # 拟合数据进行聚类
    clustering.fit(distance_matrix)
    # 计算聚类耗时
    clustering_time = time.time() - start_time
    print(f"聚类耗时: {clustering_time:.2f} 秒")
    return clustering, clustering_time


def filter_clusters(cluster_result, distance_matrix, threshold):
    """根据阈值筛选聚类结果"""
    start_time = time.time()
    final_cluster_result = {}
    for label, samples in cluster_result.items():
        valid_samples = []
        for sample in samples:
            valid = True
            for other_sample in samples:
                if distance_matrix[sample][other_sample] > threshold:
                    valid = False
                    break
            if valid:
                valid_samples.append(sample)
        if valid_samples:
            final_cluster_result[label] = valid_samples

    # 对筛选后聚类结果的索引进行排序
    for label in final_cluster_result:
        final_cluster_result[label].sort()

    # 去除只有一个图片的簇，并将其归为离群点
    outliers = cluster_result.get(-1, [])
    new_final_cluster_result = {}
    for label, cluster in final_cluster_result.items():
        if len(cluster) > 1:
            new_final_cluster_result[label] = cluster
        else:
            outliers.extend(cluster)
    final_cluster_result = new_final_cluster_result

    # 计算阈值筛选耗时
    threshold_filtering_time = time.time() - start_time
    print(f"阈值筛选耗时: {threshold_filtering_time:.2f} 秒")
    return final_cluster_result, outliers, threshold_filtering_time


def merge_clusters(final_cluster_result, distance_matrix, merge_threshold):
    """合并相似的聚类"""
    merged_cluster_result = final_cluster_result.copy()
    labels = sorted(merged_cluster_result.keys())

    def should_merge(cluster1, cluster2):
        for sample1 in cluster1:
            for sample2 in cluster2:
                if distance_matrix[sample1][sample2] > merge_threshold:
                    return False
        return True

    def merge_clusters_internal():
        if not labels:
            return merged_cluster_result
        base_label = labels[0]
        base_cluster = merged_cluster_result[base_label]
        for i in range(1, len(labels)):
            other_label = labels[i]
            other_cluster = merged_cluster_result[other_label]
            if should_merge(base_cluster, other_cluster):
                base_cluster.extend(other_cluster)
                del merged_cluster_result[other_label]
        # 重新排序簇标签
        new_merged_cluster_result = {}
        new_index = 0
        for cluster in merged_cluster_result.values():
            new_merged_cluster_result[new_index] = cluster
            new_index += 1
        return new_merged_cluster_result

    merged_cluster_result = merge_clusters_internal()
    return merged_cluster_result


def process_outliers(merged_cluster_result, outliers, distance_matrix, merge_threshold):
    """处理离群点"""
    for outlier in outliers:
        for label, cluster in merged_cluster_result.items():
            can_add = True
            for sample in cluster:
                if distance_matrix[outlier][sample] > merge_threshold:
                    can_add = False
                    break
            if can_add:
                merged_cluster_result[label].append(outlier)
                break
    return merged_cluster_result


def save_cluster_result(cluster_result, output_path):
    """保存聚类结果到JSON文件"""
    start_time = time.time()
    # 对聚类结果的索引进行排序
    for label in cluster_result:
        cluster_result[label].sort()

    # 对聚类结果的键进行排序
    sorted_cluster_result = {k: cluster_result[k] for k in sorted(cluster_result)}
    cluster_result_str_keys = {str(k): v for k, v in sorted_cluster_result.items()}

    # 计算 JSON 文件存储耗时
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_result_str_keys, f, indent=4)

    json_storage_time = time.time() - start_time
    print(f"JSON 文件存储耗时: {json_storage_time:.2f} 秒")
    return json_storage_time


def load_or_calculate_distance_matrix(config):
    """加载或计算距离矩阵"""
    error_matrix_calculation_time = 0
    if os.path.exists(config["distance_matrix_file"]):
        print("加载已保存的误差矩阵...")
        start_time = time.time()
        try:
            # 从文件中加载距离矩阵
            distance_matrix = load_distance_matrix(config["distance_matrix_file"])
            # 获取矩阵的样本数量
            n_samples = distance_matrix.shape[0]
            # 验证矩阵维度是否与数据集匹配
            validate_distance_matrix(distance_matrix, n_samples)
            # 计算加载耗时
            error_matrix_calculation_time = time.time() - start_time
            print(f"加载耗时: {error_matrix_calculation_time:.2f} 秒")
            dataset = None  # 避免重复读取数据
        except Exception as e:
            print(f"加载矩阵失败: {str(e)}")
            distance_matrix = None
    else:
        distance_matrix = None

    if distance_matrix is None:
        # 读取数据集
        dataset, n_samples = read_dataset(config["dataset"]["json_dir"], config["max_samples"])
        # 计算 PA 对齐误差矩阵
        distance_matrix, error_matrix_calculation_time = calculate_distance_matrix(
            dataset, config["distance_matrix_file"]
        )
    else:
        dataset = None  # 避免重复读取数据

    return distance_matrix, n_samples, error_matrix_calculation_time


def perform_clustering(distance_matrix, config):
    """执行聚类并返回结果"""
    # 使用 OPTICS 进行聚类
    clustering, clustering_time = optics_clustering(
        distance_matrix,
        config["clustering"]["min_samples"],
        config["clustering"]["xi"]
    )

    # 提取聚类结果
    cluster_labels = clustering.labels_
    # 计算聚类的数量
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"聚类完成，共发现 {num_clusters} 个簇。")

    # 统计数量
    total_samples = len(cluster_labels)
    classified_samples = 0
    unclassified_samples = 0

    # 遍历聚类标签，统计分类和未分类样本数量
    for label in cluster_labels:
        if label == -1:
            unclassified_samples += 1
        else:
            classified_samples += 1

    # 整理聚类结果（未筛选）
    cluster_result = {}
    for i, label in enumerate(cluster_labels):
        if label not in cluster_result:
            cluster_result[label] = []
        cluster_result[label].append(i)

    return cluster_result, total_samples, classified_samples, unclassified_samples, clustering_time


def process_clusters(cluster_result, distance_matrix, config):
    """处理聚类结果，包括筛选、合并和处理离群点"""
    # 根据阈值筛选聚类结果
    final_cluster_result, outliers, threshold_filtering_time = filter_clusters(
        cluster_result, distance_matrix, config["threshold"]
    )

    # 聚类后合并
    merged_cluster_result = merge_clusters(final_cluster_result, distance_matrix, config["merge_threshold"])

    # 处理离群点
    merged_cluster_result = process_outliers(
        merged_cluster_result, outliers, distance_matrix, config["merge_threshold"]
    )

    return merged_cluster_result, threshold_filtering_time


def print_statistics(total_samples, classified_samples, unclassified_samples, merged_cluster_result):
    """打印统计结果"""
    # 计算筛选后的总样本数
    filtered_total = sum(len(samples) for samples in merged_cluster_result.values())

    # 输出统计结果
    print("\n统计结果：")
    print(f"总照片数：{total_samples}")
    print(f"分类的照片数（未筛选）：{classified_samples}")
    print(f"未分类的照片数（未筛选）：{unclassified_samples}")
    print(f"筛选后分类的照片数：{filtered_total}")
    print(f"减少样本数：{total_samples - filtered_total}")


def main():
    # 配置参数
    config = {
        # 数据集配置
        "dataset": {
            "name": "FreiHand",
            "json_dir": r'data/FreiHand/origin_data/eval_xyz_list.json',
            "scale_enlarge": 1.25
        },
        # 距离矩阵配置
        "distance_matrix_file": "data/FreiHand/load_data/distance_matrix_eval_min8_xi0.1.npy",
        # 样本数量配置
        "max_samples": 3960,  # 最大处理样本数
        # 聚类算法配置
        "clustering": {
            "min_samples": 8,
            "xi": 0.1
        },
        # 筛选阈值配置
        "threshold": 0.001,
        # 合并阈值配置
        "merge_threshold": 0.001,
        # 输出文件配置
        "output_file": "data/FreiHand/cluster_data/clusters_optics_eval_0.003_min8_xi0.1.json",
        "unfiltered_output_file": "data/FreiHand/cluster_data/unfiltered_eval_clusters.json"
    }

    # 初始化总耗时
    total_time = 0

    # 加载或计算距离矩阵
    distance_matrix, n_samples, error_matrix_calculation_time = load_or_calculate_distance_matrix(config)
    total_time += error_matrix_calculation_time

    # 执行聚类
    (cluster_result, total_samples,
     classified_samples, unclassified_samples,
     clustering_time) = perform_clustering(distance_matrix, config)
    total_time += clustering_time

    # 保存未筛选的聚类结果
    save_cluster_result(cluster_result, config["unfiltered_output_file"])

    # 处理聚类结果
    merged_cluster_result, threshold_filtering_time = process_clusters(
        cluster_result, distance_matrix, config
    )
    total_time += threshold_filtering_time

    # 保存筛选后的 JSON 文件
    json_storage_time = save_cluster_result(merged_cluster_result, config["output_file"])
    total_time += json_storage_time

    # 计算总耗时
    print(f"总耗时: {total_time:.2f} 秒")

    # 打印统计结果
    print_statistics(total_samples, classified_samples, unclassified_samples, merged_cluster_result)

    print(f"完成，聚类结果已保存为 {config['output_file']}")


if __name__ == "__main__":
    main()