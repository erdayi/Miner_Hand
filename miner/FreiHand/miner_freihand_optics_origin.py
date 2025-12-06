import json
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import OPTICS
from tqdm import tqdm
import time
import os

'''
    本方法基于OPTICS算法对FreiHand相似手势进行聚类，
    聚类结果示例保存为 clusters_optics_32560_0.003_min8_xi0.05.json (根据配置变化)
'''

def align_w_scale(mtx1, mtx2):
    """
    使用正交对齐方法（Orthogonal Procrustes analysis）进行匹配，并返回对齐后的结果。
    此方法考虑了平移、旋转和缩放。
    :param mtx1: 第一个矩阵 (N, D)，代表一组D维关键点数据
    :param mtx2: 第二个矩阵 (N, D)，代表另一组D维关键点数据
    :return: 对齐到mtx1参考系下的mtx2版本
    """
    # 计算两个矩阵的均值 (中心点)
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    # 中心化：减去各自的均值
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # 归一化尺度：计算各自中心化后的范数，并用其进行归一化
    # 添加一个小的常数1e-8避免除零错误
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # 使用scipy的orthogonal_procrustes计算旋转矩阵R和缩放因子s
    # R使得 mtx2_t @ R.T * s 尽可能接近 mtx1_t
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)
    
    # 应用旋转和缩放，将mtx2_t对齐到mtx1_t
    # 注意：orthogonal_procrustes 返回的 s 是 mtx2_t 的缩放因子，
    # 使得 sum ||mtx1_t - s * mtx2_t @ R||^2 最小化
    # 这里我们希望将 mtx2 变换到 mtx1 的空间，所以 mtx2_t 是被变换的
    mtx2_aligned_normalized = np.dot(mtx2_t, R.T) * s 

    # 恢复到原始尺度和位置 (相对于mtx1的原始尺度和位置)
    # 将对齐和缩放后的mtx2_t乘以s1 (mtx1的尺度)，再加上t1 (mtx1的中心)
    mtx2_aligned = mtx2_aligned_normalized * s1 + t1
    return mtx2_aligned


def save_distance_matrix(matrix, filename):
    """
    保存距离矩阵到 .npy 文件。
    :param matrix: 距离矩阵
    :param filename: 保存的文件名
    """
    np.save(filename, matrix)
    print(f"距离矩阵已保存至: {filename}")


def load_distance_matrix(filename):
    """
    从 .npy 文件加载距离矩阵。
    :param filename: 要加载的文件名
    :return: 加载的距离矩阵
    """
    print(f"从 {filename} 加载距离矩阵...")
    return np.load(filename)


def validate_distance_matrix(matrix, n_samples):
    """
    验证距离矩阵是否与当前数据集兼容。
    :param matrix: 距离矩阵
    :param n_samples: 当前数据集的样本数量
    :raises ValueError: 如果矩阵维度与数据集不匹配
    """
    if matrix.shape != (n_samples, n_samples):
        raise ValueError(f"距离矩阵维度 {matrix.shape} 与当前数据集样本数 {n_samples} 不匹配")
    print("距离矩阵维度验证通过。")


def read_dataset(json_path, max_samples):
    """读取数据集中的关键点坐标"""
    start_time = time.time()
    try:
        with open(json_path, 'r') as f:
            # 假设JSON结构是 [{keypoints_xyz: [...]}, ...]
            # 或者直接是 [[...], [...], ...] 列表的列表
            # 根据实际的FreiHand JSON结构调整
            # 如果json直接是关键点列表: data = json.load(f)
            # 如果是 {'keypoints_xyz': ...}: keypoints = [np.array(item['keypoints_xyz']) for item in json.load(f)]
            # 从您之前的代码 miner_freihand_optics_origin.py 推断，应该是 item['keypoints_xyz']
            # 但当前代码直接用 dataset = json.load(f)，意味着json文件本身就是N*21*3的列表
            dataset_raw = json.load(f)
        
        # 确保dataset_raw中的每个元素都转换为numpy array
        dataset = [np.array(item) for item in dataset_raw]

        if max_samples is not None and max_samples > 0:
            dataset = dataset[:max_samples]
        
        dataset = np.array(dataset) # 转换为Numpy数组以进行后续处理
        n_samples = len(dataset)
        
        data_reading_time = time.time() - start_time
        print(f"数据读取完成，共 {n_samples} 个样本，耗时: {data_reading_time:.2f} 秒")
        return dataset, n_samples
    except FileNotFoundError:
        print(f"错误: JSON 文件未找到于路径 {json_path}")
        exit(1)
    except json.JSONDecodeError:
        print(f"错误: JSON 文件 {json_path} 格式无效")
        exit(1)
    except Exception as e:
        print(f"读取数据时发生未知错误: {str(e)}")
        exit(1)


def calculate_distance_matrix(dataset, distance_matrix_file):
    """计算基于对齐误差的距离矩阵"""
    n_samples = len(dataset)
    start_time = time.time()
    distance_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)

    for i in tqdm(range(n_samples), desc="计算对齐误差矩阵"):
        for j in range(i + 1, n_samples): # 仅计算上三角部分，因为矩阵对称
            xyz_aligned_j_to_i = align_w_scale(dataset[i], dataset[j])
            # 计算 dataset[i] 和 对齐后的 dataset[j] 之间的均方根误差或平均欧氏距离
            # error = np.sqrt(np.mean(np.sum((dataset[i] - xyz_aligned_j_to_i)**2, axis=1)))
            error = np.mean(np.linalg.norm(dataset[i] - xyz_aligned_j_to_i, axis=1))
            distance_matrix[i, j] = error
            distance_matrix[j, i] = error # 矩阵对称

    error_matrix_calculation_time = time.time() - start_time
    print(f"误差矩阵计算完成，耗时: {error_matrix_calculation_time:.2f} 秒")
    save_distance_matrix(distance_matrix, distance_matrix_file)
    return distance_matrix, error_matrix_calculation_time


def optics_clustering(distance_matrix, min_samples, xi):
    """使用 OPTICS 进行聚类"""
    start_time = time.time()
    print(f"开始 OPTICS 聚类 (min_samples={min_samples}, xi={xi})...")
    # metric='precomputed' 因为我们提供了距离矩阵
    clustering_model = OPTICS(metric='precomputed', min_samples=min_samples, xi=xi, n_jobs=-1) # 使用所有可用CPU核心
    clustering_model.fit(distance_matrix)
    clustering_time = time.time() - start_time
    print(f"OPTICS 聚类完成，耗时: {clustering_time:.2f} 秒")
    return clustering_model, clustering_time


def filter_clusters(cluster_result_raw, distance_matrix, threshold):
    """
    根据簇内最大距离阈值筛选聚类结果。
    移除簇内任意两点间距离大于阈值的样本，并处理仅含单个样本的簇。
    :param cluster_result_raw: 原始聚类结果 {label: [sample_indices]}
    :param distance_matrix: 距离矩阵
    :param threshold: 簇内样本间允许的最大距离
    :return: (筛选后的聚类结果, 离群点索引列表, 耗时)
    """
    start_time = time.time()
    print(f"开始筛选聚类结果 (阈值: {threshold})...")
    filtered_clusters = {}
    outliers = list(cluster_result_raw.get(-1, [])) # 初始离群点

    for label, samples in cluster_result_raw.items():
        if label == -1: # 跳过原始离群点，已加入outliers
            continue

        if len(samples) < 2: # 单个样本的簇直接归为离群点
            outliers.extend(samples)
            continue

        # 检查簇内样本是否都满足阈值条件
        # 一个更严格的筛选：簇内所有样本对的距离都必须 <= threshold
        # 当前代码的逻辑是：对于一个样本，如果它与簇内其他所有样本的距离都 <= threshold，则保留它
        # 这可能导致最终簇不完全满足“任意两点距离<=threshold”
        # 以下是原逻辑的实现：
        valid_samples_in_cluster = []
        for i in range(len(samples)):
            sample_i_idx = samples[i]
            is_sample_i_valid = True
            for j in range(len(samples)):
                if i == j:
                    continue
                sample_j_idx = samples[j]
                if distance_matrix[sample_i_idx, sample_j_idx] > threshold:
                    is_sample_i_valid = False
                    break
            if is_sample_i_valid:
                valid_samples_in_cluster.append(sample_i_idx)
        
        if len(valid_samples_in_cluster) > 1: # 筛选后至少保留2个样本
            filtered_clusters[label] = sorted(valid_samples_in_cluster)
        elif len(valid_samples_in_cluster) == 1: # 筛选后只剩1个，归为离群点
            outliers.append(valid_samples_in_cluster[0])
        # 如果筛选后0个，则整个原始簇的样本都间接变成了离群点（或未被选中）

    # 重新编号筛选后的簇，确保标签连续从0开始
    final_cluster_result_renumbered = {}
    new_label_idx = 0
    for old_label in sorted(filtered_clusters.keys()): # 按旧标签排序以保证一定程度的稳定性
        final_cluster_result_renumbered[new_label_idx] = filtered_clusters[old_label]
        new_label_idx += 1

    threshold_filtering_time = time.time() - start_time
    print(f"阈值筛选完成，耗时: {threshold_filtering_time:.2f} 秒")
    return final_cluster_result_renumbered, sorted(list(set(outliers))), threshold_filtering_time


def merge_clusters(cluster_result_to_merge, distance_matrix, merge_threshold):
    """
    合并相似的聚类。
    当前策略：迭代地将其他簇合并到当前处理的第一个簇（基准簇）中，如果它们之间足够相似。
    注意：这种策略不是全局最优的成对合并，而是贪婪地以第一个簇为核心进行吸收。
    :param cluster_result_to_merge: 待合并的聚类结果 {label: [sample_indices]}
    :param distance_matrix: 距离矩阵
    :param merge_threshold: 簇间合并的距离阈值
    :return: 合并后的聚类结果
    """
    start_time = time.time()
    print(f"开始合并相似聚类 (合并阈值: {merge_threshold})...")
    if not cluster_result_to_merge: # 如果没有簇可以合并
        print("没有簇可供合并。耗时: 0.00 秒")
        return {}

    # 使用可修改的列表进行迭代和修改
    # 初始时，每个簇都是一个独立的单元
    # 我们将尝试将后续的簇合并到前面的簇中
    # 为了简化，这里采用一种迭代方式：不断尝试合并，直到没有更多合并发生
    
    merged_clusters_list = [cluster_result_to_merge[label] for label in sorted(cluster_result_to_merge.keys())]

    while True:
        made_a_merge_in_pass = False
        i = 0
        while i < len(merged_clusters_list):
            j = i + 1
            while j < len(merged_clusters_list):
                cluster1 = merged_clusters_list[i]
                cluster2 = merged_clusters_list[j]
                
                # 检查两个簇是否应该合并：簇1中每个点到簇2中每个点的平均距离（或最大距离）
                # 原代码的should_merge逻辑：任意两点距离 > merge_threshold 则不合并
                # 这里我们定义 should_merge_pairwise
                can_merge_pairwise = True
                if not cluster1 or not cluster2: #任一簇为空则不能合并
                    can_merge_pairwise = False
                else:
                    for s1_idx in cluster1:
                        for s2_idx in cluster2:
                            if distance_matrix[s1_idx, s2_idx] > merge_threshold:
                                can_merge_pairwise = False
                                break
                        if not can_merge_pairwise:
                            break
                
                if can_merge_pairwise:
                    merged_clusters_list[i].extend(cluster2)
                    merged_clusters_list[i] = sorted(list(set(merged_clusters_list[i]))) #去重并排序
                    merged_clusters_list.pop(j) # 移除被合并的簇 cluster2
                    made_a_merge_in_pass = True
                    # 由于列表修改，保持j在当前位置，下一个被检查的还是索引j（因为元素前移了）
                else:
                    j += 1 # cluster2不能合并到cluster1，检查下一个cluster2
            i += 1 # 处理下一个cluster1
        
        if not made_a_merge_in_pass:
            break # 如果一轮下来没有任何合并，则结束

    # 重新构建字典并编号
    final_merged_dict = {}
    for idx, cluster_samples in enumerate(merged_clusters_list):
        final_merged_dict[idx] = cluster_samples
    
    merge_time = time.time() - start_time
    print(f"合并聚类完成，耗时: {merge_time:.2f} 秒")
    return final_merged_dict


def process_outliers(current_clusters, outliers_list, distance_matrix, merge_threshold):
    """
    尝试将离群点分配到现有最相似的簇中。
    :param current_clusters: 当前聚类结果 {label: [sample_indices]}
    :param outliers_list: 离群点索引列表
    :param distance_matrix: 距离矩阵
    :param merge_threshold: 离群点归入簇的距离阈值
    :return: 更新后的聚类结果
    """
    start_time = time.time()
    print(f"开始处理离群点 (数量: {len(outliers_list)}, 合并阈值: {merge_threshold})...")
    if not current_clusters: # 如果没有簇，离群点无法归入
        print("没有目标簇可供离群点归入。耗时: 0.00 秒")
        # 所有离群点仍然是离群点，可以考虑创建一个新的离群点簇
        # if outliers_list:
        #    current_clusters[-1] = sorted(list(set(outliers_list)))
        return current_clusters

    unassigned_outliers = []
    for outlier_idx in tqdm(outliers_list, desc="处理离群点"):
        best_cluster_label = -1
        min_avg_dist_to_cluster = float('inf')

        for label, cluster_samples in current_clusters.items():
            if not cluster_samples: continue # 跳过空簇

            # 检查离群点到该簇所有样本的距离是否都小于阈值
            can_add_to_this_cluster = True
            current_max_dist = 0 # 或者用平均距离
            distances_to_cluster = []
            for sample_idx_in_cluster in cluster_samples:
                dist = distance_matrix[outlier_idx, sample_idx_in_cluster]
                distances_to_cluster.append(dist)
                if dist > merge_threshold:
                    can_add_to_this_cluster = False
                    break
            
            if can_add_to_this_cluster:
                # 如果满足条件，计算平均距离，选择平均距离最小的簇
                avg_dist = np.mean(distances_to_cluster) if distances_to_cluster else float('inf')
                if avg_dist < min_avg_dist_to_cluster:
                    min_avg_dist_to_cluster = avg_dist
                    best_cluster_label = label
        
        if best_cluster_label != -1:
            current_clusters[best_cluster_label].append(outlier_idx)
            current_clusters[best_cluster_label].sort()
        else:
            unassigned_outliers.append(outlier_idx)
    
    # 可以选择将未分配的离群点放入一个特殊的-1标签簇
    if unassigned_outliers:
        if -1 in current_clusters:
            current_clusters[-1].extend(unassigned_outliers)
            current_clusters[-1] = sorted(list(set(current_clusters[-1])))
        else:
            current_clusters[-1] = sorted(list(set(unassigned_outliers)))

    outlier_processing_time = time.time() - start_time
    print(f"离群点处理完成，{len(unassigned_outliers)} 个仍未分配。耗时: {outlier_processing_time:.2f} 秒")
    return current_clusters


def save_cluster_result(cluster_data_to_save, output_filepath):
    """保存聚类结果到JSON文件"""
    start_time = time.time()
    print(f"开始保存聚类结果至 {output_filepath}...")
    # 确保簇内样本索引排序，簇标签（键）也排序
    # 将所有标签转为字符串，因为JSON的键必须是字符串
    # 确保簇标签是整数且排序
    final_data_to_save = {}
    # 将-1标签（离群点）放在最后
    sorted_labels = sorted([k for k in cluster_data_to_save.keys() if k != -1])
    if -1 in cluster_data_to_save:
        sorted_labels.append(-1)

    for label in sorted_labels:
        samples = sorted(list(set(cluster_data_to_save[label]))) # 排序并去重
        if samples: #只保存非空簇
             final_data_to_save[str(label)] = samples

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_data_to_save, f, indent=4)
        json_storage_time = time.time() - start_time
        print(f"聚类结果成功保存。耗时: {json_storage_time:.2f} 秒")
        return json_storage_time
    except IOError as e:
        print(f"错误: 无法写入JSON文件 {output_filepath}. 原因: {e}")
        return 0 # 表示保存失败或耗时为0


def load_or_calculate_distance_matrix(config):
    """加载或计算距离矩阵"""
    dist_matrix_filepath = config["distance_matrix_file"]
    json_data_path = config["dataset"]["json_dir"]
    max_samples_to_process = config["max_samples"]
    error_matrix_calc_time = 0
    dataset_for_dist_calc = None
    num_samples_in_matrix = 0

    if os.path.exists(dist_matrix_filepath):
        print(f"发现已保存的误差矩阵: {dist_matrix_filepath}")
        load_start_time = time.time()
        try:
            distance_matrix = load_distance_matrix(dist_matrix_filepath)
            # 验证矩阵是否与配置中的max_samples（如果适用）大致匹配
            # 如果max_samples与矩阵维度不符，可能需要重新计算或警告
            num_samples_in_matrix = distance_matrix.shape[0]
            if max_samples_to_process is not None and max_samples_to_process > 0 and num_samples_in_matrix != max_samples_to_process:
                print(f"警告: 加载的距离矩阵样本数 ({num_samples_in_matrix}) 与配置的最大样本数 ({max_samples_to_process}) 不符。将使用矩阵的样本数。")
                # 或者决定重新计算：
                # print("将重新计算距离矩阵以匹配配置。")
                # distance_matrix = None 
            else:
                 validate_distance_matrix(distance_matrix, num_samples_in_matrix) # 与自身维度验证
            
            error_matrix_calc_time = time.time() - load_start_time
            print(f"误差矩阵加载完成，耗时: {error_matrix_calc_time:.2f} 秒")
        except Exception as e:
            print(f"加载预计算的距离矩阵失败: {str(e)}. 将重新计算。")
            distance_matrix = None
    else:
        print(f"未找到预计算的距离矩阵文件: {dist_matrix_filepath}. 将进行计算。")
        distance_matrix = None

    if distance_matrix is None:
        dataset_for_dist_calc, num_samples_in_matrix = read_dataset(json_data_path, max_samples_to_process)
        if num_samples_in_matrix == 0:
            print("错误：数据集中没有样本可供处理。")
            exit(1)
        distance_matrix, error_matrix_calc_time = calculate_distance_matrix(
            dataset_for_dist_calc, dist_matrix_filepath
        )
    
    # 如果是从文件加载的，num_samples_in_matrix 已经设置
    # 如果是新计算的，num_samples_in_matrix 也已经从 read_dataset 返回
    # 确保 n_samples 反映实际使用的样本数
    actual_n_samples = distance_matrix.shape[0]
    return distance_matrix, actual_n_samples, error_matrix_calc_time


def perform_clustering_pipeline(distance_matrix, config):
    """执行聚类流程并返回结果"""
    clustering_model, clustering_time = optics_clustering(
        distance_matrix,
        config["clustering"]["min_samples"],
        config["clustering"]["xi"]
    )

    cluster_labels_from_optics = clustering_model.labels_
    num_raw_clusters = len(set(cluster_labels_from_optics)) - (1 if -1 in cluster_labels_from_optics else 0)
    print(f"OPTICS初步聚类完成，发现 {num_raw_clusters} 个簇 (不包括离群点)。")

    total_samples_count = len(cluster_labels_from_optics)
    classified_samples_count = np.sum(cluster_labels_from_optics != -1)
    unclassified_samples_count = np.sum(cluster_labels_from_optics == -1)

    # 整理原始聚类结果
    raw_cluster_dict = {}
    for i, label in enumerate(cluster_labels_from_optics):
        raw_cluster_dict.setdefault(label, []).append(i)
    
    # 保存未筛选的聚类结果 (可选)
    if "unfiltered_output_file" in config and config["unfiltered_output_file"]:
        save_cluster_result(raw_cluster_dict, config["unfiltered_output_file"])
        print(f"未筛选的原始聚类结果已保存至: {config['unfiltered_output_file']}")

    return raw_cluster_dict, total_samples_count, classified_samples_count, unclassified_samples_count, clustering_time


def process_and_refine_clusters(raw_clusters, distance_matrix, config):
    """处理聚类结果，包括筛选、合并和处理离群点"""
    processing_start_time = time.time()

    # 1. 根据阈值筛选聚类结果
    filtered_clusters_dict, outliers_after_filter, filter_time = filter_clusters(
        raw_clusters, distance_matrix, config["threshold"]
    )
    print(f"筛选后剩下 {len(filtered_clusters_dict)} 个簇和 {len(outliers_after_filter)} 个离群点。")

    # 2. 合并相似簇
    # 注意：这里的 merge_clusters 实现已被修改为更通用的迭代合并策略
    merged_clusters_dict = merge_clusters(filtered_clusters_dict, distance_matrix, config["merge_threshold"])
    print(f"合并后剩下 {len(merged_clusters_dict)} 个簇。")

    # 3. 处理离群点 (包括之前筛选下来的和OPTICS本身的离群点)
    # 将 merge_clusters 后可能产生的空簇的样本（如果逻辑允许）也视为离群点，或确保 merge 不产生空簇
    # 当前 merge_clusters 和 filter_clusters 应该不会主动产生空簇，但标签可能不连续
    # outliers_after_filter 是 filter_clusters 返回的离群点
    # raw_clusters[-1] 是OPTICS的原始离群点 (如果存在)
    all_outliers_to_process = list(outliers_after_filter) # 从筛选步骤来的
    # 如果原始聚类有-1标签（OPTICS离群点），也加入处理列表
    # if -1 in raw_clusters:
    #    all_outliers_to_process.extend(raw_clusters[-1])
    # all_outliers_to_process = sorted(list(set(all_outliers_to_process))) #去重并排序
    # 注意：filter_clusters 已经包含了原始的-1离群点

    final_refined_clusters = process_outliers(
        merged_clusters_dict, all_outliers_to_process, distance_matrix, config["merge_threshold"]
    )
    
    total_processing_time = time.time() - processing_start_time + filter_time # filter_time 已包含在内
    print(f"聚类后处理总耗时: {total_processing_time:.2f} 秒")
    return final_refined_clusters, total_processing_time


def print_final_statistics(total_initial_samples, initial_classified, initial_unclassified, final_clusters_result):
    """打印最终的统计结果"""
    final_classified_count = 0
    final_unclassified_count = 0
    num_final_clusters = 0

    for label, samples in final_clusters_result.items():
        if not samples: continue # 跳过空簇
        if label == -1:
            final_unclassified_count += len(samples)
        else:
            final_classified_count += len(samples)
            num_final_clusters += 1
            
    print("\n--- 最终统计结果 ---")
    print(f"总处理样本数: {total_initial_samples}")
    print(f"OPTICS初始分类样本数: {initial_classified}")
    print(f"OPTICS初始离群点数: {initial_unclassified}")
    print(f"最终有效簇数量 (不含离群点簇): {num_final_clusters}")
    print(f"最终分类样本总数: {final_classified_count}")
    print(f"最终离群点总数: {final_unclassified_count}")
    if total_initial_samples > 0:
        reduction_percentage = ((total_initial_samples - final_classified_count) / total_initial_samples) * 100
        print(f"相较于总样本，非分类/离群点占比: {reduction_percentage:.2f}%")


def main():
    # 配置参数
    config = {
        "dataset": {
            "name": "FreiHand",
            "json_dir": r'data/FreiHand/origin_data/eval_xyz_list.json', # 示例路径，请修改为您的实际路径
            # "scale_enlarge": 1.25 # 此参数在当前代码中未使用
        },
        "distance_matrix_file": "data/FreiHand/load_data/distance_matrix_eval_optics.npy", # 示例路径
        "max_samples": 3960,  # 最大处理样本数, None或0表示处理全部
        "clustering": {
            "min_samples": 8, # OPTICS的min_samples参数
            "xi": 0.1         # OPTICS的xi参数，用于簇提取
        },
        "threshold": 0.001,       # 簇内筛选阈值
        "merge_threshold": 0.001, # 簇间合并及离群点分配阈值
        "output_file": "data/FreiHand/cluster_data/clusters_optics_eval_final.json", # 示例路径
        "unfiltered_output_file": "data/FreiHand/cluster_data/clusters_optics_eval_raw.json" # 可选，原始OPTICS输出
    }

    # 确保输出目录存在
    for path_key in ["distance_matrix_file", "output_file", "unfiltered_output_file"]:
        if config[path_key]:
            dir_name = os.path.dirname(config[path_key])
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
                print(f"创建目录: {dir_name}")

    overall_start_time = time.time()
    total_calc_time = 0

    # 1. 加载或计算距离矩阵
    distance_matrix, n_actual_samples, dist_matrix_time = load_or_calculate_distance_matrix(config)
    total_calc_time += dist_matrix_time
    if n_actual_samples == 0:
        return # 没有样本，提前退出

    # 2. 执行OPTICS聚类
    raw_clusters, total_s, classified_s, unclassified_s, cluster_t = perform_clustering_pipeline(distance_matrix, config)
    total_calc_time += cluster_t

    # 3. 后处理聚类结果 (筛选、合并、离群点处理)
    final_clusters, post_processing_t = process_and_refine_clusters(
        raw_clusters, distance_matrix, config
    )
    total_calc_time += post_processing_t

    # 4. 保存最终筛选和处理后的聚类结果
    if final_clusters:
        save_time = save_cluster_result(final_clusters, config["output_file"])
        total_calc_time += save_time
        print(f"\n最终聚类结果已保存至: {config['output_file']}")
    else:
        print("\n没有有效的最终聚类结果可供保存。")

    overall_end_time = time.time()
    print(f"\n--- 总耗时统计 ---")
    print(f"各计算步骤累计耗时: {total_calc_time:.2f} 秒")
    print(f"脚本总运行耗时: {overall_end_time - overall_start_time:.2f} 秒")

    # 5. 打印统计结果
    print_final_statistics(total_s, classified_s, unclassified_s, final_clusters)


if __name__ == "__main__":
    main()
