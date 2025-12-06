import json
import numpy as np
from scipy.linalg import orthogonal_procrustes
import time
import os
from joblib import Parallel, delayed
import multiprocessing
import re

'''
HO3D数据集单视角手部姿态聚类工具

该工具使用HDBSCAN聚类算法对HO3D数据集中的单视角手部姿态进行聚类分析。
主要功能：
1. 加载HO3D数据集中的手部3D关节点数据
2. 计算手部姿态之间的相似度（使用正交对齐方法）
3. 使用HDBSCAN算法进行聚类
4. 对聚类结果进行后处理（阈值筛选、簇合并、离群点处理）
5. 输出聚类结果和统计信息

使用方法：
python miner/HO3D/miner_single_view_hdbscan.py

参数说明：
- min_cluster_size: 最小聚类大小，默认8
- min_samples: 核心距离计算参数，默认5
- threshold: 聚类阈值，默认0.001
- merge_threshold: 合并簇的阈值，默认0.001

输出：
1. 聚类结果JSON文件，包含：
   - 每个簇的样本索引列表
   - 使用的聚类参数信息
2. 命令行统计信息，包含：
   - 样本统计（总数、分类数、未分类数）
   - 时间统计（距离矩阵计算、聚类时间）
   - 簇大小统计（最大、最小、平均）
'''

# ===== 配置参数 =====
CONFIG = {
    # 数据集配置
    "dataset": {
        "name": "HO3D",
        "joints_file": r'data/HO3D/origin_data/single_view/hand_joint_3d_interval_1_shuffled.json',
        "paths_file": r'data/HO3D/origin_data/single_view/img_paths_interval_1_shuffled.json',
        "scale_enlarge": 1.25
    },
    # 距离矩阵配置
    "distance_matrix_file": "data/HO3D/load_data/hdbscan/single_view/distance_matrix.npy",
    # HDBSCAN聚类算法配置
    "clustering": {
        "min_cluster_size": 10,    # 增加最小簇大小
        "min_samples": 5,          # 增加最小样本数
        "alpha": 0.3              # 降低alpha使聚类更严格
    },
    # 筛选阈值配置
    "threshold": 0.008,           # 稍微降低阈值
    # 合并阈值配置
    "merge_threshold": 0.008,     # 稍微降低合并阈值
    # 输出文件配置
    "output_file": "data/HO3D/cluster_data/hdbscan/single_view/hand_clusters_hdbscan_min{min_cluster_size}_samples{min_samples}_alpha{alpha}_th{threshold}.json",
    # 最大样本数限制
    "max_samples": None  # 设置为None表示不限制样本数
}

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

def save_distance_matrix(matrix, filename):
    """保存距离矩阵到 .npy 文件"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
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
    """读取HO3D数据集
    
    Args:
        json_path: HO3D数据集JSON文件路径
        max_samples: 最大样本数量
        
    Returns:
        processed_data: 处理后的手部关键点数据
        n_samples: 实际加载的样本数量
    """
    start_time = time.time()
    try:
        # 打开 JSON 文件并加载数据
        print(f"读取数据集: {json_path}")
        with open(json_path, 'r') as f:
            dataset = json.load(f)
        
        # 处理数据
        processed_data = []
        for item in dataset:
            if isinstance(item, list) and len(item) > 0:
                processed_data.append(np.array(item))
        
        # 限制最大样本数
        if max_samples is not None:
            processed_data = processed_data[:max_samples]
        n_samples = len(processed_data)
        
        if n_samples == 0:
            raise ValueError(f"没有找到有效的手关键点数据，请检查数据文件: {json_path}")
        
        # 计算数据读取耗时
        data_reading_time = time.time() - start_time
        print(f"数据读取耗时: {data_reading_time:.2f} 秒")
        print(f"成功加载 {n_samples} 个样本")
        
        return np.array(processed_data), n_samples
    except Exception as e:
        print(f"读取数据失败: {str(e)}")
        raise


def align_w_scale_vectorized(mtx1, mtx2):
    """
    向量化的对齐函数，使用NumPy广播机制优化计算
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


def calculate_distance_chunk(chunk_data):
    """
    计算距离矩阵的一个块
    """
    start_idx, end_idx, dataset = chunk_data
    n_samples = len(dataset)
    chunk_size = end_idx - start_idx
    chunk_matrix = np.zeros((chunk_size, n_samples))
    
    for i in range(chunk_size):
        idx = start_idx + i
        for j in range(n_samples):
            if idx < j:  # 只计算上三角部分
                xyz_aligned = align_w_scale_vectorized(dataset[idx], dataset[j])
                error = np.mean(np.linalg.norm(dataset[idx] - xyz_aligned, axis=1))
                chunk_matrix[i, j] = error
                chunk_matrix[i, j] = error  # 利用对称性
    
    return start_idx, chunk_matrix


def save_progress(distance_matrix, progress_file, processed_pairs, total_pairs):
    """保存计算进度"""
    progress_data = {
        'matrix': distance_matrix,
        'processed_pairs': processed_pairs,
        'total_pairs': total_pairs,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    np.save(progress_file, progress_data)
    print(f"   - 进度已保存: {processed_pairs}/{total_pairs} ({processed_pairs/total_pairs*100:.1f}%)")


def load_progress(progress_file):
    """加载计算进度"""
    try:
        progress_data = np.load(progress_file, allow_pickle=True).item()
        return progress_data['matrix'], progress_data['processed_pairs'], progress_data['total_pairs']
    except Exception as e:
        print(f"加载进度失败: {str(e)}")
        return None, 0, 0


def calculate_distance_matrix(dataset, distance_matrix_file):
    """优化的距离矩阵计算函数，支持进度保存和恢复"""
    n_samples = len(dataset)
    start_time = time.time()
    
    # 进度文件路径
    progress_file = distance_matrix_file.replace('.npy', '_progress.npy')
    
    # 检查是否有未完成的进度
    if os.path.exists(progress_file):
        print("\n🔄 发现未完成的计算进度，正在恢复...")
        distance_matrix, processed_pairs, total_pairs = load_progress(progress_file)
        if distance_matrix is not None:
            print(f"   - 已恢复进度: {processed_pairs}/{total_pairs} ({processed_pairs/total_pairs*100:.1f}%)")
            print("   - 继续计算...")
        else:
            distance_matrix = np.zeros((n_samples, n_samples))
            processed_pairs = 0
            total_pairs = (n_samples * (n_samples - 1)) // 2
    else:
        distance_matrix = np.zeros((n_samples, n_samples))
        processed_pairs = 0
        total_pairs = (n_samples * (n_samples - 1)) // 2
    
    # 减少并行进程数量，避免内存问题
    n_cores = max(1, min(8, multiprocessing.cpu_count() - 1))
    
    # 计算每个进程处理的样本数
    chunk_size = max(1, n_samples // n_cores)
    chunks = []
    
    # 创建任务块
    for i in range(0, n_samples, chunk_size):
        end_idx = min(i + chunk_size, n_samples)
        chunks.append((i, end_idx, dataset))
    
    print(f"\n🔄 计算模式: 并行计算")
    print(f"   - 使用 {n_cores} 个CPU核心")
    print(f"   - 每个核心处理约 {chunk_size} 个样本")
    print(f"   - 总样本数: {n_samples}")
    
    try:
        # 使用真正的多进程并行计算
        results = Parallel(
            n_jobs=n_cores,
            max_nbytes='50M',  # 限制每个进程的内存使用
            prefer="processes",
            backend='loky',
            batch_size=10  # 增加批处理大小
        )(
            delayed(calculate_distance_chunk)(chunk) for chunk in chunks
        )
        
        # 合并结果
        for start_idx, chunk_matrix in results:
            distance_matrix[start_idx:start_idx + chunk_matrix.shape[0]] = chunk_matrix
        
        # 利用对称性填充下三角部分
        distance_matrix = np.maximum(distance_matrix, distance_matrix.T)
        
        # 计算耗时
        error_matrix_calculation_time = time.time() - start_time
        print(f"\n✅ 并行计算成功完成")
        print(f"   - 计算耗时: {error_matrix_calculation_time:.2f} 秒")
        print(f"   - 平均每个样本耗时: {error_matrix_calculation_time/n_samples:.4f} 秒")
        
        # 保存距离矩阵
        save_distance_matrix(distance_matrix, distance_matrix_file)
        print("   - 距离矩阵已保存")
        
        # 删除进度文件
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print("   - 进度文件已清理")
        
        # 保存计算时间记录
        time_record = {
            "computation_time": error_matrix_calculation_time,
            "n_cores": n_cores,
            "n_samples": n_samples,
            "computation_mode": "parallel_processes",
            "avg_time_per_sample": error_matrix_calculation_time/n_samples,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        time_file = distance_matrix_file.replace('.npy', '_time.json')
        with open(time_file, 'w') as f:
            json.dump(time_record, f, indent=4)
        
        return distance_matrix, error_matrix_calculation_time
        
    except Exception as e:
        print(f"\n⚠️ 并行计算失败: {str(e)}")
        print("🔄 切换到单线程计算模式...")
        
        # 回退到单线程计算
        last_progress = 0
        last_save_time = time.time()
        save_interval = 300  # 每5分钟保存一次进度
        
        print(f"   - 总样本数: {n_samples}")
        print(f"   - 需要计算的样本对: {total_pairs}")
        print("   - 开始计算...")
        
        try:
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    # 检查是否已经计算过
                    if distance_matrix[i, j] == 0:
                        xyz_aligned = align_w_scale_vectorized(dataset[i], dataset[j])
                        error = np.mean(np.linalg.norm(dataset[i] - xyz_aligned, axis=1))
                        distance_matrix[i, j] = error
                        distance_matrix[j, i] = error
                    
                    # 更新进度
                    processed_pairs += 1
                    progress = (processed_pairs * 100) // total_pairs
                    
                    # 显示进度
                    if progress > last_progress:
                        print(f"   - 计算进度: {progress}% ({processed_pairs}/{total_pairs})")
                        last_progress = progress
                    
                    # 定期保存进度
                    current_time = time.time()
                    if current_time - last_save_time >= save_interval:
                        save_progress(distance_matrix, progress_file, processed_pairs, total_pairs)
                        last_save_time = current_time
                
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断，正在保存进度...")
            save_progress(distance_matrix, progress_file, processed_pairs, total_pairs)
            print("进度已保存，可以稍后继续计算")
            raise
        
        # 计算耗时
        error_matrix_calculation_time = time.time() - start_time
        print(f"\n✅ 单线程计算完成")
        print(f"   - 计算耗时: {error_matrix_calculation_time:.2f} 秒")
        print(f"   - 平均每个样本耗时: {error_matrix_calculation_time/n_samples:.4f} 秒")
        
        # 保存距离矩阵
        save_distance_matrix(distance_matrix, distance_matrix_file)
        print("   - 距离矩阵已保存")
        
        # 删除进度文件
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print("   - 进度文件已清理")
        
        # 保存计算时间记录
        time_record = {
            "computation_time": error_matrix_calculation_time,
            "n_cores": 1,
            "n_samples": n_samples,
            "computation_mode": "single_thread",
            "avg_time_per_sample": error_matrix_calculation_time/n_samples,
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        time_file = distance_matrix_file.replace('.npy', '_time.json')
        with open(time_file, 'w') as f:
            json.dump(time_record, f, indent=4)
        
        return distance_matrix, error_matrix_calculation_time


def hdbscan_clustering(distance_matrix, min_cluster_size, min_samples=None, cluster_selection_epsilon=0.0):
    """
    使用HDBSCAN算法进行聚类
    
    Args:
        distance_matrix: 距离矩阵
        min_cluster_size: 最小簇大小
        min_samples: 核心距离计算参数，如果为None则使用min_cluster_size
        cluster_selection_epsilon: 簇选择参数
        
    Returns:
        cluster_result: 聚类结果
    """
    import hdbscan
    
    # 如果min_samples为None，使用min_cluster_size
    if min_samples is None:
        min_samples = min_cluster_size
    
    # 构建HDBSCAN参数
    hdbscan_params = {
        'metric': 'precomputed',
        'min_cluster_size': min_cluster_size,
        'min_samples': min_samples,
        'cluster_selection_epsilon': cluster_selection_epsilon,
        'algorithm': 'best',
        'core_dist_n_jobs': -1
    }
    
    # 执行聚类
    clusterer = hdbscan.HDBSCAN(**hdbscan_params)
    cluster_labels = clusterer.fit_predict(distance_matrix)
    
    # 构建聚类结果
    cluster_result = {
        'labels': cluster_labels.tolist(),
        'parameters': hdbscan_params,
        'clusters': {}
    }
    
    # 统计每个簇的样本
    for i, label in enumerate(cluster_labels):
        if label not in cluster_result['clusters']:
            cluster_result['clusters'][label] = []
        cluster_result['clusters'][label].append(i)
    
    return cluster_result


def filter_clusters(cluster_result, distance_matrix, threshold):
    """根据阈值筛选聚类结果"""
    start_time = time.time()
    final_cluster_result = {}
    
    print("\n🔍 开始筛选聚类...")
    print(f"初始簇数量: {len(cluster_result)}")
    
    for label, samples in cluster_result.items():
        if label == -1:  # 跳过噪声点
            continue
            
        print(f"\n处理簇 {label}:")
        print(f"  样本数量: {len(samples)}")
        
        valid_samples = []
        for sample in samples:
            valid = True
            for other_sample in samples:
                if distance_matrix[sample][other_sample] > threshold:
                    valid = False
                    break
            if valid:
                valid_samples.append(sample)
                
        print(f"  有效样本数量: {len(valid_samples)}")
        
        if len(valid_samples) >= 2:  # 只保留至少包含2个样本的簇
            final_cluster_result[label] = valid_samples
            print(f"  ✅ 保留该簇")
        else:
            print(f"  ❌ 丢弃该簇")

    # 对筛选后聚类结果的索引进行排序
    for label in final_cluster_result:
        final_cluster_result[label].sort()

    # 计算阈值筛选耗时
    threshold_filtering_time = time.time() - start_time
    print(f"\n阈值筛选完成:")
    print(f"  耗时: {threshold_filtering_time:.2f} 秒")
    print(f"  最终簇数量: {len(final_cluster_result)}")
    
    return final_cluster_result, [], threshold_filtering_time


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


def index_to_path(index_list, path_file, hand_type):
    """将索引列表转换为图像路径列表"""
    try:
        with open(path_file, 'r') as f:
            paths_data = json.load(f)
        return [paths_data[hand_type][i] for i in index_list]
    except Exception as e:
        print(f"路径转换错误: {str(e)}")
        return []


def extract_frame_idx(view_path):
    """从视角路径中提取帧号"""
    match = re.search(r'image(\d+)', view_path)
    if match:
        return int(match.group(1))
    return 0


def extract_sequence_name(view_path):
    """从视角路径中提取序列名"""
    return view_path.split('/')[0]


def sort_cluster_paths(paths):
    """对簇内的图像路径进行排序"""
    # 为每个路径创建包含序列名和帧号的元组
    path_info = [(path, extract_sequence_name(path), extract_frame_idx(path)) for path in paths]
    # 按序列名和帧号排序
    path_info.sort(key=lambda x: (x[1], x[2]))
    # 返回排序后的路径列表
    return [info[0] for info in path_info]


def save_cluster_result(cluster_result, save_path, config):
    """保存聚类结果为JSON格式
    
    Args:
        cluster_result: 聚类结果字典，键为簇标签，值为样本索引列表
        save_path: 保存路径
        config: 配置参数字典
        
    Returns:
        json_storage_time: JSON文件存储耗时
    """
    start_time = time.time()
    print(f"\n💾 正在保存聚类结果...")
    print(f"目标路径: {save_path}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(save_path)
    if not os.path.exists(output_dir):
        print(f"创建输出目录: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    final_result = {
        "clusters": [],
        "parameters": {
            "min_cluster_size": config["clustering"]["min_cluster_size"],
            "min_samples": config["clustering"]["min_samples"],
            "threshold": config["threshold"],
            "merge_threshold": config["merge_threshold"]
        }
    }
    
    # 处理聚类结果
    print("处理聚类数据...")
    cluster_paths = []
    for label in sorted(cluster_result.keys()):
        if len(cluster_result[label]) > 1:  # 保留有效簇
            # 对簇内的索引进行排序
            sorted_indices = sorted(cluster_result[label])
            cluster_paths.append(sorted_indices)
    
    final_result["clusters"] = cluster_paths
    
    # 保存结果
    try:
        print(f"写入文件: {save_path}")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=4, ensure_ascii=False)
        
        # 验证文件是否成功保存
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            print(f"✅ 成功保存聚类结果")
            print(f"   - 文件大小: {file_size / 1024:.1f}KB")
            print(f"   - 包含 {len(final_result['clusters'])} 个有效簇")
        else:
            raise RuntimeError("文件保存失败")
            
    except Exception as e:
        print(f"❌ 保存聚类结果时出错: {str(e)}")
        raise
    
    json_storage_time = time.time() - start_time
    print(f"JSON 文件存储耗时: {json_storage_time:.2f} 秒")
    return json_storage_time


def load_or_calculate_distance_matrix(config):
    """加载或计算距离矩阵"""
    error_matrix_calculation_time = 0
    distance_matrix_file = config["distance_matrix_file"]
    
    # 确保目录存在
    os.makedirs(os.path.dirname(distance_matrix_file), exist_ok=True)
    
    if os.path.exists(distance_matrix_file):
        print(f"\n📂 发现已存在的距离矩阵文件: {distance_matrix_file}")
        print("正在加载距离矩阵...")
        start_time = time.time()
        try:
            distance_matrix = load_distance_matrix(distance_matrix_file)
            error_matrix_calculation_time = time.time() - start_time
            print(f"距离矩阵加载耗时: {error_matrix_calculation_time:.2f} 秒")
            print(f"距离矩阵形状: {distance_matrix.shape}")
            return distance_matrix, distance_matrix.shape[0], error_matrix_calculation_time
        except Exception as e:
            print(f"加载距离矩阵失败: {str(e)}")
            print("将重新计算距离矩阵...")
    else:
        print(f"\n📂 未找到距离矩阵文件，将重新计算")
    
    # 读取数据集
    dataset, n_samples = read_dataset(config["dataset"]["joints_file"], config["max_samples"])
    # 计算距离矩阵
    distance_matrix, error_matrix_calculation_time = calculate_distance_matrix(
        dataset, distance_matrix_file
    )
    
    return distance_matrix, n_samples, error_matrix_calculation_time


def perform_hdbscan_clustering(distance_matrix, config):
    """
    执行HDBSCAN聚类
    
    Args:
        distance_matrix: 距离矩阵
        config: 配置参数
        
    Returns:
        cluster_result: 聚类结果
    """
    print("开始HDBSCAN聚类...")
    start_time = time.time()
    
    # 获取聚类参数
    min_cluster_size = config['clustering']['min_cluster_size']
    min_samples = config['clustering']['min_samples']
    alpha = config['clustering']['alpha']
    
    # 执行聚类
    cluster_result = hdbscan_clustering(
        distance_matrix,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=alpha
    )
    
    # 打印聚类结果
    n_clusters = len([k for k in cluster_result['clusters'].keys() if k != -1])
    print(f"HDBSCAN聚类完成，共发现 {n_clusters} 个簇。")
    
    clustering_time = time.time() - start_time
    print(f"HDBSCAN聚类耗时: {clustering_time:.2f} 秒")
    
    return cluster_result


def process_clusters(cluster_result, distance_matrix, config):
    """处理聚类结果，包括筛选、合并和处理离群点"""
    print("\n🔄 处理聚类结果...")
    print(f"原始簇数量: {len(cluster_result['clusters'])}")
    
    # 根据阈值筛选聚类结果
    final_cluster_result, outliers, threshold_filtering_time = filter_clusters(
        cluster_result['clusters'], distance_matrix, config["threshold"]
    )
    
    print(f"筛选后簇数量: {len(final_cluster_result)}")

    # 聚类后合并
    merged_cluster_result = merge_clusters(final_cluster_result, distance_matrix, config["merge_threshold"])
    print(f"合并后簇数量: {len(merged_cluster_result)}")

    # 处理离群点
    merged_cluster_result = process_outliers(
        merged_cluster_result, outliers, distance_matrix, config["merge_threshold"]
    )
    print(f"最终簇数量: {len(merged_cluster_result)}")

    return merged_cluster_result, threshold_filtering_time


def print_cluster_summary(total_samples, classified_samples, unclassified_samples, merged_cluster_result):
    """打印聚类结果总结"""
    # 计算筛选后的总样本数
    filtered_total = sum(len(samples) for samples in merged_cluster_result.values())
    
    print("\n📊 聚类结果总结：")
    print("=" * 50)
    print(f"总样本数：{total_samples}")
    print(f"发现聚类数：{len(merged_cluster_result)}")
    print(f"聚类样本数：{classified_samples}")
    print(f"噪声样本数：{unclassified_samples}")
    print(f"最终有效样本数：{filtered_total}")
    
    if merged_cluster_result:
        cluster_sizes = [len(samples) for samples in merged_cluster_result.values()]
        print(f"\n聚类大小统计：")
        print(f"  最小聚类大小：{min(cluster_sizes)}")
        print(f"  最大聚类大小：{max(cluster_sizes)}")
        print(f"  平均聚类大小：{sum(cluster_sizes)/len(cluster_sizes):.1f}")
        
        # 显示聚类大小分布
        small_clusters = sum(1 for size in cluster_sizes if size <= 10)
        medium_clusters = sum(1 for size in cluster_sizes if 11 <= size <= 50)
        large_clusters = sum(1 for size in cluster_sizes if size > 50)
        
        print(f"\n聚类分布：")
        print(f"  小聚类(≤10个样本)：{small_clusters}")
        print(f"  中聚类(11-50个样本)：{medium_clusters}")
        print(f"  大聚类(>50个样本)：{large_clusters}")


def get_config(hand_type):
    """根据手部类型获取配置"""
    config = CONFIG.copy()
    # 更新距离矩阵文件路径
    config["distance_matrix_file"] = config["distance_matrix_file"].format(hand_type=hand_type)
    # 更新输出文件路径
    config["output_file"] = config["output_file"].format(
        hand_type=hand_type,
        min_cluster_size=config["clustering"]["min_cluster_size"],
        threshold=config["threshold"],
        merge_threshold=config["merge_threshold"]
    )
    config["unfiltered_output_file"] = config["unfiltered_output_file"].format(hand_type=hand_type)
    return config


def main():
    """主函数：执行HO3D数据集的手部姿态聚类分析"""
    print("🚀 HO3D HDBSCAN 手势聚类分析")
    print("🎯 高效的层次密度聚类算法")
    print("=" * 60)

    # 处理数据
    print("\n处理数据...")
    process_data(CONFIG)


def process_data(config):
    """处理HO3D数据集
    
    Args:
        config: 配置参数字典，包含数据集路径、聚类参数等
    """
    # 初始化总耗时和结果变量
    total_time = 0
    merged_cluster_result = {}  # 初始化为空字典
    total_samples = 0
    classified_samples = 0
    unclassified_samples = 0

    try:
        # 加载或计算距离矩阵
        distance_matrix, n_samples, error_matrix_calculation_time = load_or_calculate_distance_matrix(config)
        total_time += error_matrix_calculation_time
        total_samples = n_samples

        # 执行HDBSCAN聚类
        cluster_result = perform_hdbscan_clustering(distance_matrix, config)
        total_time += error_matrix_calculation_time

        # 统计分类和未分类样本数
        labels = cluster_result['labels']
        classified_samples = sum(1 for label in labels if label != -1)
        unclassified_samples = sum(1 for label in labels if label == -1)

        # 处理聚类结果
        merged_cluster_result, threshold_filtering_time = process_clusters(
            cluster_result, distance_matrix, config
        )
        total_time += threshold_filtering_time

        # 保存筛选后的结果
        output_file = config["output_file"].format(
            min_cluster_size=config["clustering"]["min_cluster_size"],
            min_samples=config["clustering"]["min_samples"],
            alpha=config["clustering"]["alpha"],
            threshold=config["threshold"]
        )
        
        json_storage_time = save_cluster_result(
            merged_cluster_result,
            output_file,
            config
        )
        total_time += json_storage_time

        # 清理内存
        del distance_matrix
        del cluster_result
        import gc
        gc.collect()
        
        # 打印聚类结果总结
        print_cluster_summary(total_samples, classified_samples, unclassified_samples, merged_cluster_result)
        
        print(f"\n✅ 聚类完成！结果已保存为：")
        print(f"   📁 筛选后结果: {output_file}")

    except Exception as e:
        print(f"\n⚠️ 处理过程中出现错误: {str(e)}")
        return

    # 计算总耗时
    print(f"\n⏱️  总处理时间: {total_time:.2f} 秒")


if __name__ == "__main__":
    main() 