import json
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import OPTICS
from tqdm import tqdm
import time
import os
from multiprocessing import Pool
import functools
import psutil

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

def align_w_scale_batch(mtx1, mtx_list, means1, means_list, norms1, norms_list):
    """
    批量对齐多个矩阵到同一个目标矩阵
    返回对齐后的矩阵列表
    """
    aligned_batch = []
    mtx1_t = (mtx1 - means1) / norms1
    
    for i, mtx2 in enumerate(mtx_list):
        t2 = means_list[i]
        s2 = norms_list[i]
        mtx2_t = (mtx2 - t2) / s2
        
        # 使用正交普罗克拉斯提斯分析计算旋转矩阵
        R, _ = orthogonal_procrustes(mtx1_t, mtx2_t)
        
        # 应用旋转并恢复到原始尺度和位置
        mtx2_aligned = np.dot(mtx2_t, R.T) * s2 * norms1 + means1
        aligned_batch.append(mtx2_aligned)
    
    return np.array(aligned_batch)

def save_distance_matrix(matrix, filename):
    """
    保存距离矩阵到 .npy 文件。
    :param matrix: 距离矩阵
    :param filename: 保存的文件名
    """
    np.save(filename, matrix)
    print(f"距离矩阵已保存至 {filename}")

def load_distance_matrix(filename):
    """
    从 .npy 文件加载距离矩阵。
    :param filename: 要加载的文件名
    :return: 加载的距离矩阵
    """
    if os.path.exists(filename):
        print(f"加载距离矩阵: {filename}")
        return np.load(filename)
    else:
        print(f"文件不存在: {filename}")
        return None

def validate_distance_matrix(matrix, n_samples):
    """
    验证距离矩阵是否与当前数据集兼容。
    :param matrix: 距离矩阵
    :param n_samples: 当前数据集的样本数量
    :raises ValueError: 如果矩阵维度与数据集不匹配
    """
    if matrix.shape != (n_samples, n_samples):
        raise ValueError(f"距离矩阵维度 {matrix.shape} 与当前数据集样本数 {n_samples} 不匹配")

def read_dataset(json_path, max_samples):
    """读取数据集"""
    start_time = time.time()
    try:
        # 打开 JSON 文件并加载数据
        print(f"读取数据集: {json_path}")
        with open(json_path, 'r') as f:
            dataset = json.load(f)
        # 限制最大样本数
        dataset = np.array(dataset[:max_samples])
        n_samples = len(dataset)
        # 计算数据读取耗时
        data_reading_time = time.time() - start_time
        print(f"数据读取完成，耗时: {data_reading_time:.2f} 秒，样本数: {n_samples}")
        return dataset, n_samples
    except Exception as e:
        print(f"读取数据失败: {str(e)}")
        exit()

def compute_batch(args):
    """
    计算距离矩阵的一个批次（并行工作单元）
    每个批次处理多个行，每行使用向量化方法计算与其他行的距离
    """
    batch_indices, dataset, means, norms = args
    n_samples = len(dataset)
    batch_size = len(batch_indices)
    batch_results = np.zeros((batch_size, n_samples))
    
    for i, row_idx in enumerate(batch_indices):
        # 获取当前样本
        mtx1 = dataset[row_idx]
        t1 = means[row_idx]
        s1 = norms[row_idx]
        
        # 计算与当前行之后的所有行的距离（上三角部分）
        after_indices = np.arange(row_idx + 1, n_samples)
        
        if len(after_indices) > 0:
            # 批量获取待比较的样本
            mtx_list = [dataset[j] for j in after_indices]
            means_list = [means[j] for j in after_indices]
            norms_list = [norms[j] for j in after_indices]
            
            # 批量对齐
            aligned_batch = align_w_scale_batch(mtx1, mtx_list, t1, means_list, s1, norms_list)
            
            # 向量化计算误差
            errors = np.mean(np.linalg.norm(
                np.expand_dims(mtx1, axis=0) - aligned_batch, 
                axis=2
            ), axis=1)
            
            # 填充结果
            batch_results[i, after_indices] = errors
    
    return batch_indices, batch_results

def calculate_distance_matrix_hybrid(dataset, distance_matrix_file, n_processes=None, batch_size=100):
    """
    混合使用向量化和并行计算的距离矩阵计算方法
    """
    n_samples = len(dataset)
    print(f"开始计算距离矩阵，样本数: {n_samples}")
    start_time = time.time()
    
    # 预计算所有样本的均值和范数，避免重复计算
    print("预计算样本统计量...")
    means = np.array([data.mean(axis=0) for data in dataset])
    norms = np.array([np.linalg.norm(data - means[i], axis=0).sum() + 1e-8 for i, data in enumerate(dataset)])
    
    # 生成批次索引
    batches = []
    for i in range(0, n_samples, batch_size):
        batch_indices = np.arange(i, min(i + batch_size, n_samples))
        batches.append((batch_indices, dataset, means, norms))
    
    # 获取系统CPU和内存信息
    total_cpus = os.cpu_count()
    available_cpus = len(psutil.Process().cpu_affinity()) if hasattr(psutil.Process(), 'cpu_affinity') else total_cpus
    total_memory = psutil.virtual_memory().total / (1024**3)  # GB
    available_memory = psutil.virtual_memory().available / (1024**3)  # GB
    
    print(f"系统信息: CPU核心数={total_cpus}, 可用核心数={available_cpus}, 总内存={total_memory:.2f}GB, 可用内存={available_memory:.2f}GB")
    used_processes = available_cpus if n_processes is None else n_processes
    print(f"使用{used_processes}个进程并行计算距离矩阵，批次大小={batch_size}")
    
    # 使用进程池并行计算
    with Pool(processes=used_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(compute_batch, batches),
            total=len(batches),
            desc="并行计算误差矩阵"
        ))
    
    # 填充结果矩阵（上三角部分）
    print("填充结果矩阵...")
    distance_matrix = np.zeros((n_samples, n_samples))
    for batch_indices, batch_results in results:
        for i, row_idx in enumerate(batch_indices):
            distance_matrix[row_idx, row_idx+1:] = batch_results[i, row_idx+1:]
    
    # 利用对称性填充下三角部分
    print("填充对称部分...")
    distance_matrix = distance_matrix + distance_matrix.T
    
    # 计算误差矩阵计算耗时
    error_matrix_calculation_time = time.time() - start_time
    print(f"误差矩阵计算完成，耗时: {error_matrix_calculation_time:.2f} 秒")
    
    # 保存距离矩阵
    save_distance_matrix(distance_matrix, distance_matrix_file)
    
    # 内存使用统计
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024**3)  # GB
    print(f"当前内存使用: {memory_usage:.2f}GB")
    
    return distance_matrix, error_matrix_calculation_time

def optics_clustering(distance_matrix, min_samples, xi):
    """使用 OPTICS 进行聚类"""
    start_time = time.time()
    print("开始 OPTICS 聚类...")
    
    # 获取系统内存信息
    total_memory = psutil.virtual_memory().total / (1024**3)  # GB
    available_memory = psutil.virtual_memory().available / (1024**3)  # GB
    print(f"聚类前内存状态: 总内存={total_memory:.2f}GB, 可用内存={available_memory:.2f}GB")
    
    # 初始化 OPTICS 聚类器，使用预计算的距离矩阵
    clustering = OPTICS(metric='precomputed', min_samples=min_samples, xi=xi)
    
    # 执行聚类
    clustering.fit(distance_matrix)
    
    # 计算聚类耗时
    clustering_time = time.time() - start_time
    print(f"聚类完成，耗时: {clustering_time:.2f} 秒")
    
    # 聚类后的内存使用
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024**3)  # GB
    print(f"聚类后内存使用: {memory_usage:.2f}GB")
    
    return clustering, clustering_time

def filter_clusters(cluster_result, distance_matrix, threshold):
    """根据阈值筛选聚类结果"""
    start_time = time.time()
    print(f"开始聚类筛选，阈值={threshold}")
    
    final_cluster_result = {}
    for label, samples in cluster_result.items():
        valid_samples = []
        
        # 预先计算每个样本与其他样本的最大距离
        max_distances = np.zeros(len(samples))
        for i, sample in enumerate(samples):
            max_dist = 0
            for other_sample in samples:
                if sample != other_sample:
                    dist = distance_matrix[sample][other_sample]
                    if dist > max_dist:
                        max_dist = dist
            max_distances[i] = max_dist
        
        # 筛选符合条件的样本
        for i, sample in enumerate(samples):
            if max_distances[i] <= threshold:
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
    print(f"阈值筛选完成，耗时: {threshold_filtering_time:.2f} 秒")
    return final_cluster_result, outliers, threshold_filtering_time

def merge_clusters(final_cluster_result, distance_matrix, merge_threshold):
    """合并相似的聚类"""
    print(f"开始聚类合并，阈值={merge_threshold}")
    start_time = time.time()
    
    merged_cluster_result = final_cluster_result.copy()
    labels = sorted(merged_cluster_result.keys())

    def should_merge(cluster1, cluster2):
        # 优化：只检查每个簇的最远样本对
        max_dist = 0
        for sample1 in cluster1:
            for sample2 in cluster2:
                dist = distance_matrix[sample1][sample2]
                if dist > max_dist:
                    max_dist = dist
                if max_dist > merge_threshold:
                    return False
        return True

    # 聚类合并逻辑
    if not labels:
        return merged_cluster_result
        
    for i in range(len(labels)):
        if i >= len(labels):  # 可能在合并过程中标签列表变短
            break
        base_label = labels[i]
        base_cluster = merged_cluster_result[base_label]
        
        j = i + 1
        while j < len(labels):
            other_label = labels[j]
            other_cluster = merged_cluster_result[other_label]
            
            if should_merge(base_cluster, other_cluster):
                # 合并两个簇
                base_cluster.extend(other_cluster)
                del merged_cluster_result[other_label]
                labels.pop(j)  # 从标签列表中移除已合并的簇
            else:
                j += 1

    # 重新排序簇标签
    new_merged_cluster_result = {}
    new_index = 0
    for cluster in merged_cluster_result.values():
        new_merged_cluster_result[new_index] = cluster
        new_index += 1

    # 计算合并耗时
    merge_time = time.time() - start_time
    print(f"聚类合并完成，耗时: {merge_time:.2f} 秒")
    return new_merged_cluster_result

def process_outliers(merged_cluster_result, outliers, distance_matrix, merge_threshold):
    """处理离群点"""
    print(f"开始处理离群点，阈值={merge_threshold}")
    start_time = time.time()
    
    # 计算每个离群点与各聚类的最大距离
    for outlier in outliers:
        best_fit_label = -1
        min_max_distance = float('inf')
        
        for label, cluster in merged_cluster_result.items():
            max_distance = 0
            for sample in cluster:
                dist = distance_matrix[outlier][sample]
                if dist > max_distance:
                    max_distance = dist
            
            if max_distance <= merge_threshold and max_distance < min_max_distance:
                min_max_distance = max_distance
                best_fit_label = label
        
        # 如果找到合适的聚类，将离群点加入
        if best_fit_label != -1:
            merged_cluster_result[best_fit_label].append(outlier)
    
    # 计算处理离群点耗时
    process_time = time.time() - start_time
    print(f"离群点处理完成，耗时: {process_time:.2f} 秒")
    return merged_cluster_result

def save_cluster_result(cluster_result, output_path):
    """保存聚类结果到JSON文件"""
    start_time = time.time()
    print(f"保存聚类结果到 {output_path}")
    
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
    print(f"JSON 文件存储完成，耗时: {json_storage_time:.2f} 秒")
    return json_storage_time

def load_or_calculate_distance_matrix(config):
    """加载或计算距离矩阵"""
    error_matrix_calculation_time = 0
    
    # 检查是否存在已保存的距离矩阵
    if os.path.exists(config["distance_matrix_file"]):
        print(f"尝试加载已保存的误差矩阵: {config['distance_matrix_file']}")
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
            print(f"距离矩阵加载成功，耗时: {error_matrix_calculation_time:.2f} 秒")
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
        distance_matrix, error_matrix_calculation_time = config["calculate_distance_matrix_func"](
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

    # 计算聚类数量
    num_clusters = len(merged_cluster_result) if -1 not in merged_cluster_result else len(merged_cluster_result) - 1

    # 输出统计结果
    print("\n统计结果：")
    print(f"总样本数：{total_samples}")
    print(f"分类的样本数（未筛选）：{classified_samples} ({classified_samples/total_samples*100:.2f}%)")
    print(f"未分类的样本数（未筛选）：{unclassified_samples} ({unclassified_samples/total_samples*100:.2f}%)")
    print(f"筛选后分类的样本数：{filtered_total} ({filtered_total/total_samples*100:.2f}%)")
    print(f"减少样本数：{total_samples - filtered_total} ({(total_samples - filtered_total)/total_samples*100:.2f}%)")
    print(f"最终聚类数量：{num_clusters}")

    # 输出每个聚类的大小
    print("\n聚类大小分布：")
    cluster_sizes = [len(samples) for samples in merged_cluster_result.values()]
    for i, size in enumerate(sorted(cluster_sizes, reverse=True)):
        print(f"聚类 {i+1}: {size} 个样本")
        if i >= 9:  # 只显示前10个最大的聚类
            break
    if len(cluster_sizes) > 10:
        print(f"... 以及其他 {len(cluster_sizes)-10} 个聚类")

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
        "distance_matrix_file": "data/FreiHand/load_data/new_distance_matrix_eval_min8_xi0.1.npy",
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
        "output_file": "data/FreiHand/cluster_data/16_200_clusters_optics_eval_0.003_min8_xi0.1.json",
        "unfiltered_output_file": "data/FreiHand/cluster_data/unfiltered_eval_clusters.json",
        # 距离矩阵计算函数配置
        "calculate_distance_matrix_func": functools.partial(
            calculate_distance_matrix_hybrid,
            n_processes=16,  # 根据CPU核心数调整
            batch_size=200  # 根据内存情况调整
        )
    }

    # 打印系统信息
    print(f"系统信息: CPU核心数={os.cpu_count()}, 内存={psutil.virtual_memory().total/(1024**3):.2f}GB")
    print(f"当前工作目录: {os.getcwd()}")
    
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