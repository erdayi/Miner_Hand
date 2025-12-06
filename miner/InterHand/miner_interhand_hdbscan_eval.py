import json
import numpy as np
from scipy.linalg import orthogonal_procrustes
import time
import os
from joblib import Parallel, delayed
import multiprocessing
import re
from scipy import sparse
import numba
from numba import cuda
import math
import sys
from concurrent.futures import ThreadPoolExecutor
import shutil
import hdbscan

# 检查是否有可用的psutil
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️ psutil未安装，内存监控功能将不可用")

# 检查是否有可用的GPU
try:
    import cupy as cp
    HAS_GPU = True
    print("✅ GPU加速可用")
except ImportError:
    HAS_GPU = False
    print("⚠️ GPU加速不可用，将使用CPU模式")

'''
InterHand2.6M数据集手部姿态聚类工具

该工具使用HDBSCAN聚类算法对InterHand2.6M数据集中的手部姿态进行聚类分析。
主要功能：
1. 加载预处理后的手部3D关节点数据和图像路径
2. 计算手部姿态之间的相似度（使用正交对齐方法）
3. 使用HDBSCAN算法进行聚类
4. 对聚类结果进行后处理（阈值筛选、簇合并、离群点处理）
5. 输出聚类结果和统计信息

使用方法：
python miner/InterHand/miner_interhand_hdbscan.py

参数说明：
- min_cluster_size: 最小聚类大小，默认8
- min_samples: 核心距离计算参数，默认5
- threshold: 聚类阈值，默认0.0003
- merge_threshold: 合并簇的阈值，默认0.0003

输出：
1. 聚类结果JSON文件，包含：
   - 每个簇的图像路径列表
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
        "name": "InterHand",
        "json_dir": r'data/InterHand/origin_data/val_joint_3d.json',
        "paths_file": r'data/InterHand/origin_data/val_image_paths.json',
        "scale_enlarge": 1.25
    },
    # 距离矩阵配置
    "distance_matrix_file": "F:/GS/miner/matrix/data/InterHand/load_data/hdbscan/eval_distance_matrix_{hand_type}.npy",
    # 样本数量配置
    "max_samples": 120000,  # 减少最大样本数以降低内存使用
    # HDBSCAN聚类算法配置
    "clustering": {
        "min_cluster_size": 8,  # 核心参数：最小聚类大小（必需）
        # 可选参数（如需调优可启用）：
        "min_samples": 5,       # 核心距离计算参数（默认等于min_cluster_size）
        "cluster_selection_epsilon": 0.0  # 聚类合并阈值（默认0.0不合并）
    },
    # 筛选阈值配置
    "threshold": 0.0003,
    # 合并阈值配置
    "merge_threshold": 0.0003,
    # 输出文件配置
    "output_file": "data/InterHand/cluster_data/hdbscan/eval_hand_clusters_hdbscan_{hand_type}_min{min_cluster_size}_th{threshold}_merge{merge_threshold}.json",
    "unfiltered_output_file": "data/InterHand/cluster_data/hdbscan/eval_unfiltered_hdbscan_{hand_type}_clusters.json"
}

@numba.jit(nopython=True, parallel=True)
def align_w_scale_numba(mtx1, mtx2):
    """使用Numba加速的对齐函数，避免使用不支持的NumPy函数"""
    # 计算均值 - 手动实现替代np.mean(mtx1, axis=0)
    t1 = np.zeros(mtx1.shape[1])
    for j in range(mtx1.shape[1]):
        s = 0.0
        for i in range(mtx1.shape[0]):
            s += mtx1[i, j]
        t1[j] = s / mtx1.shape[0]
    
    t2 = np.zeros(mtx2.shape[1])
    for j in range(mtx2.shape[1]):
        s = 0.0
        for i in range(mtx2.shape[0]):
            s += mtx2[i, j]
        t2[j] = s / mtx2.shape[0]
    
    # 中心化
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2
    
    # 计算范数
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    
    # 归一化
    mtx1_t /= s1
    mtx2_t /= s2
    
    # 计算旋转矩阵
    H = mtx1_t.T @ mtx2_t
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # 应用旋转和缩放
    mtx2_t = mtx2_t @ R.T * s1 + t1
    return mtx2_t

@cuda.jit
def calculate_distance_kernel(dataset, result, start_i, start_j, block_size_i, block_size_j):
    """CUDA核函数，优化内存访问和计算效率"""
    # 使用共享内存来缓存数据
    shared_data = cuda.shared.array(shape=(32, 32, 3), dtype=numba.float32)
    
    i, j = cuda.grid(2)
    if i < block_size_i and j < block_size_j:
        idx_i = start_i + i
        idx_j = start_j + j
        if idx_i < idx_j:
            # 使用共享内存优化数据访问
            error = 0.0
            # 分块处理数据以减少内存访问
            for k_start in range(0, dataset.shape[2], 32):
                k_end = min(k_start + 32, dataset.shape[2])
                # 加载数据到共享内存
                for k in range(k_start, k_end):
                    if k < dataset.shape[2]:
                        shared_data[k - k_start, 0, 0] = dataset[idx_i, k, 0]
                        shared_data[k - k_start, 0, 1] = dataset[idx_i, k, 1]
                        shared_data[k - k_start, 0, 2] = dataset[idx_i, k, 2]
                        shared_data[k - k_start, 1, 0] = dataset[idx_j, k, 0]
                        shared_data[k - k_start, 1, 1] = dataset[idx_j, k, 1]
                        shared_data[k - k_start, 1, 2] = dataset[idx_j, k, 2]
                
                cuda.syncthreads()
                
                # 计算误差
                for k in range(k_end - k_start):
                    diff_x = shared_data[k, 0, 0] - shared_data[k, 1, 0]
                    diff_y = shared_data[k, 0, 1] - shared_data[k, 1, 1]
                    diff_z = shared_data[k, 0, 2] - shared_data[k, 1, 2]
                    error += diff_x * diff_x + diff_y * diff_y + diff_z * diff_z
                
                cuda.syncthreads()
            
            result[i, j] = math.sqrt(error / dataset.shape[2])

def calculate_block_distance_gpu(dataset, start_i, end_i, start_j, end_j):
    """使用GPU计算块距离矩阵，优化内存管理和批处理"""
    # 计算实际块大小
    block_size_i = end_i - start_i
    block_size_j = end_j - start_j
    
    # 优化CUDA线程配置
    threads_per_block = (32, 32)  # 增加到32x32以提高并行度
    blocks_per_grid_i = (block_size_i + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_j = (block_size_j + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_i, blocks_per_grid_j)
    
    # 确保数据类型为float32
    dataset = dataset.astype(np.float32)
    
    # 将数据转移到GPU，使用完整块大小创建结果矩阵
    dataset_gpu = cp.asarray(dataset)
    result_gpu = cp.zeros((block_size_i, block_size_j), dtype=cp.float32)
    
    # 启动核函数，传递实际块大小
    calculate_distance_kernel[blocks_per_grid, threads_per_block](
        dataset_gpu, result_gpu, start_i, start_j, block_size_i, block_size_j
    )
    
    # 将结果转回CPU
    result = cp.asnumpy(result_gpu)
    
    # 处理对称部分，根据实际块大小调整
    if block_size_i == block_size_j:
        # 正方形块，直接对称
        result = np.maximum(result, result.T)
    else:
        # 长方形块，创建完整对称矩阵
        full_result = np.zeros((block_size_i, block_size_i), dtype=np.float32)
        full_result[:block_size_i, :block_size_j] = result
        full_result[:block_size_j, :block_size_i] = result.T
        result = full_result
    
    return result

def get_gpu_info():
    """获取GPU信息"""
    if not HAS_GPU:
        return "GPU不可用"
    
    try:
        # 获取GPU设备信息
        device = cp.cuda.Device(0)
        # 使用nvidia-smi命令获取GPU信息
        import subprocess
        nvidia_smi = subprocess.check_output("nvidia-smi", shell=True).decode()
        # 解析GPU名称
        import re
        gpu_name = re.search(r"NVIDIA GeForce RTX \d+", nvidia_smi)
        device_name = gpu_name.group(0) if gpu_name else "Unknown"
        
        # 获取内存信息
        device_memory = cp.cuda.runtime.memGetInfo()
        total_memory = device_memory[0] / (1024**3)  # 转换为GB
        free_memory = device_memory[1] / (1024**3)
        
        return {
            "name": device_name,
            "total_memory": f"{total_memory:.1f}GB",
            "free_memory": f"{free_memory:.1f}GB",
            "compute_capability": device.compute_capability
        }
    except Exception as e:
        print(f"获取GPU信息时出错: {str(e)}")
        return {
            "name": "Unknown",
            "total_memory": "Unknown",
            "free_memory": "Unknown",
            "compute_capability": "Unknown"
        }

def check_disk_space(path, required_space_gb):
    """检查指定路径的磁盘空间是否足够"""
    try:
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024**3)  # 转换为GB
        if free_gb < required_space_gb:
            print(f"⚠️ 磁盘空间不足！")
            print(f"   - 需要: {required_space_gb:.1f}GB")
            print(f"   - 可用: {free_gb:.1f}GB")
            return False
        return True
    except Exception as e:
        print(f"⚠️ 检查磁盘空间时出错: {str(e)}")
        return False

def calculate_required_space(n_samples, dtype_size=4):
    """计算所需磁盘空间（GB）"""
    # 距离矩阵大小 = n_samples * n_samples * dtype_size (bytes)
    matrix_size = n_samples * n_samples * dtype_size
    # 转换为GB并添加20%的缓冲空间
    required_gb = (matrix_size / (1024**3)) * 1.2
    return required_gb

def calculate_distance_matrix(dataset, distance_matrix_file):
    n_samples = len(dataset)
    start_time = time.time()
    
    # 计算所需磁盘空间
    required_space = calculate_required_space(n_samples)
    if not check_disk_space(os.path.dirname(distance_matrix_file), required_space):
        raise RuntimeError("磁盘空间不足，无法继续计算")
    
    print(f"\n💾 磁盘空间检查通过")
    print(f"   - 需要空间: {required_space:.1f}GB")
    
    # 创建临时目录用于存储分块数据
    temp_dir = os.path.join(os.path.dirname(distance_matrix_file), 'temp_blocks')
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 动态计算块大小
        block_size = calculate_dynamic_block_size(n_samples)
    n_blocks = (n_samples + block_size - 1) // block_size
    
    print(f"\n🔄 开始分块计算距离矩阵")
    print(f"   - 总样本数: {n_samples}")
    print(f"   - 块大小: {block_size}")
    print(f"   - 块数量: {n_blocks}")
    
    # 分批处理数据
    for i in range(n_blocks):
        start_i = i * block_size
        end_i = min((i + 1) * block_size, n_samples)
        block_size_i = end_i - start_i
        
            # 预加载i块数据
            block_i = dataset[start_i:end_i]
            
            # 并行处理j块
            with ThreadPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
                futures = []
        for j in range(i, n_blocks):
            start_j = j * block_size
            end_j = min((j + 1) * block_size, n_samples)
            block_size_j = end_j - start_j
            
                    # 为每个块创建临时文件
                    temp_file = os.path.join(temp_dir, f'block_{i}_{j}.npy')
                    futures.append(executor.submit(
                        process_block, 
                        i, j, block_i, dataset[start_j:end_j],
                        start_i, end_i, start_j, end_j,
                        temp_file, HAS_GPU
                    ))
                
                # 监控内存使用
                monitor_memory_usage(futures)
                
                # 显示进度
            progress = (i + 1) * 100 // n_blocks
            print(f"\r计算进度: {progress}% ({i + 1}/{n_blocks})", end="")
        
        print("\n\n🔄 合并分块数据...")
        
        # 创建分块存储的距离矩阵文件
        matrix_file = distance_matrix_file + '.tmp'
        matrix_mmap = np.memmap(matrix_file, dtype=np.float32, mode='w+', 
                               shape=(n_samples, n_samples))
        
        # 分块合并数据
        for i in range(n_blocks):
            start_i = i * block_size
            end_i = min((i + 1) * block_size, n_samples)
            
            for j in range(i, n_blocks):
                start_j = j * block_size
                end_j = min((j + 1) * block_size, n_samples)
                
                temp_file = os.path.join(temp_dir, f'block_{i}_{j}.npy')
                if os.path.exists(temp_file):
                    # 加载块数据
                    block_data = np.load(temp_file)
                    
                    # 写入到内存映射文件
                    matrix_mmap[start_i:end_i, start_j:end_j] = block_data
                    if i != j:
                        matrix_mmap[start_j:end_j, start_i:end_i] = block_data.T
                    
                    # 删除临时文件
                    os.remove(temp_file)
            
            # 定期刷新写入磁盘
            matrix_mmap.flush()
            print(f"\r合并进度: {(i + 1) * 100 // n_blocks}%", end="")
        
        print("\n✅ 距离矩阵计算完成")
                
        # 保存最终结果
        np.save(distance_matrix_file, matrix_mmap)
        
        # 清理临时文件
        del matrix_mmap
        os.remove(matrix_file)
        
        return distance_matrix_file, time.time() - start_time
        
    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def process_block(i, j, block_i, block_j, 
                 start_i, end_i, start_j, end_j,
                 temp_file, use_gpu):
    """处理单个数据块并保存到临时文件"""
                        block_size_i = end_i - start_i
                        block_size_j = end_j - start_j
    
    if use_gpu:
        # 优化后的GPU计算
        block_matrix = gpu_block_distance(block_i, block_j)
    else:
        # 优化内存的CPU计算
                        block_matrix = np.zeros((block_size_i, block_size_j), dtype=np.float32)
                        for ii in range(block_size_i):
                            for jj in range(block_size_j):
                if start_i + ii < start_j + jj:
                    aligned = align_and_scale(block_i[ii], block_j[jj])
                    error = np.mean(np.linalg.norm(block_i[ii] - aligned, axis=1))
                                    block_matrix[ii, jj] = error
                        
    # 保存块数据到临时文件
    np.save(temp_file, block_matrix)
    return i, j

def calculate_dynamic_block_size(n_samples):
    """根据可用内存动态计算块大小"""
    # 获取系统内存
    total_mem = psutil.virtual_memory().total if HAS_PSUTIL else 64 * 1024**3
    
    # 估算所需内存
    base_mem = 500 * 1024**2  # 基础内存500MB
    per_sample = 2 * 1024  # 每样本2KB
    
    # 计算最大可能块大小
    max_block = int(((total_mem - base_mem) / per_sample) ** 0.5)
    
    # 设置合理上限
    return min(max_block, 4000, n_samples)

def monitor_memory_usage(futures):
    """监控内存使用并在必要时调整"""
    if not HAS_PSUTIL:
        return
    
    while any(not f.done() for f in futures):
        mem = psutil.virtual_memory()
        if mem.percent > 85:  # 内存使用超过85%
            # 警告并等待
            print(f"⚠️ 内存使用过高 ({mem.percent}%)，暂停新任务...")
            time.sleep(5)
        time.sleep(1)

def gpu_block_distance(block_i, block_j):
    """优化GPU内存使用的块计算"""
    # 将数据分片传输到GPU
    result = cp.zeros((len(block_i), len(block_j)), dtype=cp.float32)
    
    # 分片处理避免一次性占用太多GPU内存
    sub_size = 512  # 子块大小
    for i in range(0, len(block_i), sub_size):
        sub_i = cp.asarray(block_i[i:i+sub_size])
        for j in range(0, len(block_j), sub_size):
            sub_j = cp.asarray(block_j[j:j+sub_size])
            
            # 调用优化后的核函数
            compute_subblock_kernel(sub_i, sub_j, result[i:i+sub_size, j:j+sub_size])
    
    return cp.asnumpy(result)

@numba.jit(nopython=True, fastmath=True)
def align_and_scale(mtx1, mtx2):
    """优化的对齐函数，减少临时变量"""
    # 使用更高效的计算方式
    t1 = np.mean(mtx1, axis=0)
    t2 = np.mean(mtx2, axis=0)
    
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2
    
    norm1 = np.sqrt(np.sum(mtx1_t**2))
    norm2 = np.sqrt(np.sum(mtx2_t**2))
    
    if norm1 < 1e-8 or norm2 < 1e-8:
        return mtx2
    
    mtx1_t /= norm1
    mtx2_t /= norm2
    
    # 简化SVD计算
    H = mtx1_t.T @ mtx2_t
    U, s, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    return (mtx2_t @ R.T) * norm1 + t1

def memory_guard(threshold=0.85):
    """内存使用守卫"""
    if not HAS_PSUTIL:
        return
    
    mem = psutil.virtual_memory()
    if mem.percent > threshold * 100:
        print(f"🚨 内存使用率过高 ({mem.percent}%)，建议：")
        print(f"  - 减少样本量 (当前: {n_samples})")
        print(f"  - 增加块大小 (当前: {block_size})")
        print(f"  - 关闭其他内存密集型应用")
        
        # 尝试释放内存
        import gc
        gc.collect()
        if HAS_GPU:
            cp.get_default_memory_pool().free_all_blocks()
        
        mem_after = psutil.virtual_memory()
        if mem_after.percent > threshold * 100:
            print("❌ 内存释放不足，考虑终止程序")
            return False
    return True

# 在计算循环中调用
if not memory_guard():
    sys.exit("内存不足，终止程序")

def filter_clusters(cluster_result, distance_matrix_file, threshold):
    """根据阈值筛选聚类结果"""
    start_time = time.time()
    final_cluster_result = {}
    
    # 加载距离矩阵
    matrix_data = np.load(distance_matrix_file, mmap_mode='r')
    
    # 使用线程池并行处理簇
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
        futures = []
        
        # 提交所有簇的处理任务
    for label, samples in cluster_result.items():
            if label == -1:  # 跳过离群点
                continue
            future = executor.submit(process_cluster_batch, samples, matrix_data, threshold)
            futures.append((label, future))
        
        # 收集结果
        for label, future in futures:
            try:
                valid_samples = future.result()
        if valid_samples:
            final_cluster_result[label] = valid_samples
            except Exception as e:
                print(f"处理簇 {label} 时出错: {str(e)}")

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

def process_cluster_batch(samples, matrix_data, threshold, batch_size=1000):
    """批量处理簇的筛选"""
    valid_samples = []
    n_samples = len(samples)
    
    # 分批处理样本
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch_samples = samples[i:batch_end]
        
        # 获取当前批次样本与其他样本的距离矩阵
        distances = matrix_data[batch_samples][:, samples]
        
        # 使用向量化操作检查每个样本是否满足阈值条件
        valid_mask = np.all(distances <= threshold, axis=1)
        valid_samples.extend(np.array(batch_samples)[valid_mask])
        
        # 显示进度
        progress = (batch_end * 100) // n_samples
        print(f"\r处理进度: {progress}%", end="")
    
    print()  # 换行
    return valid_samples

def process_batch(clustering, batch_matrix, batch_idx, batch_results):
    """处理单个批次的聚类"""
    try:
        # 执行聚类
        batch_labels = clustering.fit_predict(batch_matrix)
        
        # 调整标签以避免批次间的冲突
        if batch_idx > 0 and batch_results:  # 确保batch_results不为空
            max_label = max(max(labels) for labels in batch_results if len(labels) > 0)
            batch_labels[batch_labels != -1] += max_label + 1
        
        print(f"批次 {batch_idx + 1} 完成，发现 {len(set(batch_labels)) - (1 if -1 in batch_labels else 0)} 个簇")
        
        # 清理内存
        del batch_matrix
        import gc
        gc.collect()
        
        return batch_labels
        
    except Exception as e:
        print(f"\n⚠️ 处理批次 {batch_idx + 1} 时出现错误: {str(e)}")
        print("尝试使用备用方法...")
        
        try:
            # 备用方法：使用较小的块大小
            sub_batch_size = 10000
            sub_batches = (len(batch_matrix) + sub_batch_size - 1) // sub_batch_size
            sub_batch_results = []
            
            for sub_idx in range(sub_batches):
                sub_start = sub_idx * sub_batch_size
                sub_end = min(sub_start + sub_batch_size, len(batch_matrix))
                
                # 提取子批次数据
                sub_matrix = batch_matrix[sub_start:sub_end, sub_start:sub_end]
                
                # 执行聚类
                sub_labels = clustering.fit_predict(sub_matrix)
                
                # 调整标签
                if sub_idx > 0 and sub_batch_results:  # 确保sub_batch_results不为空
                    max_sub_label = max(max(labels) for labels in sub_batch_results if len(labels) > 0)
                    sub_labels[sub_labels != -1] += max_sub_label + 1
                
                sub_batch_results.append(sub_labels)
                
                # 清理内存
                del sub_matrix
                import gc
                gc.collect()
            
            # 合并子批次结果
            batch_labels = np.concatenate(sub_batch_results)
            
            # 调整标签
            if batch_idx > 0 and batch_results:  # 确保batch_results不为空
                max_label = max(max(labels) for labels in batch_results if len(labels) > 0)
                batch_labels[batch_labels != -1] += max_label + 1
            
            print(f"批次 {batch_idx + 1} 完成（使用备用方法），发现 {len(set(batch_labels)) - (1 if -1 in batch_labels else 0)} 个簇")
            
            return batch_labels
            
        except Exception as e2:
            print(f"备用方法也失败: {str(e2)}")
            return None

def batch_hdbscan_clustering(distance_matrix_file, batch_size=40000, min_cluster_size=8, min_samples=None, cluster_selection_epsilon=0.0):
    """分批进行HDBSCAN聚类，然后合并结果"""
    print(f"\n🔄 开始分批聚类 (批次大小: {batch_size})")
    
    # 加载距离矩阵的元数据
    matrix_data = np.load(distance_matrix_file, mmap_mode='r')
    n_samples = matrix_data.shape[0]
    
    # 计算批次数
    n_batches = (n_samples + batch_size - 1) // batch_size
    print(f"总样本数: {n_samples}, 批次数: {n_batches}")
    
    # 存储每个批次的聚类结果
    batch_results = []
    
    # 创建HDBSCAN聚类器
    clustering = hdbscan.HDBSCAN(
        metric='precomputed',
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        algorithm='best',
        core_dist_n_jobs=-1
    )
    
    # 使用线程池并行处理批次
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
        futures = []
        
        # 提交所有批次任务
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            
            print(f"\n提交批次 {batch_idx + 1}/{n_batches} (样本 {start_idx}-{end_idx})")
            
            # 提取当前批次的距离矩阵
            batch_matrix = matrix_data[start_idx:end_idx, start_idx:end_idx].astype(np.float64)
            
            # 提交聚类任务
            future = executor.submit(process_batch, clustering, batch_matrix, batch_idx, batch_results)
            futures.append(future)
        
        # 等待所有批次完成
        for future in futures:
            try:
                batch_labels = future.result()
                if batch_labels is not None:
                    batch_results.append(batch_labels)
            except Exception as e:
                print(f"处理批次时出错: {str(e)}")
    
    # 检查是否有有效的批次结果
    if not batch_results:
        raise RuntimeError("所有批次处理都失败了")
    
    # 合并批次结果
    print("\n🔄 合并批次结果...")
    final_labels = np.concatenate(batch_results)
    
    # 处理批次间的连接
    print("处理批次间的连接...")
    for batch_idx in range(n_batches - 1):
        current_end = (batch_idx + 1) * batch_size
        next_start = current_end
        
        # 获取批次边界样本的标签
        current_labels = batch_results[batch_idx]
        next_labels = batch_results[batch_idx + 1]
        
        # 计算批次间的距离
        overlap_size = 100  # 重叠区域大小
        if current_end + overlap_size > n_samples:
            overlap_size = n_samples - current_end
        
        if overlap_size > 0:
            try:
                # 获取边界区域的距离矩阵
                boundary_matrix = matrix_data[current_end-overlap_size:current_end, 
                                            next_start:next_start+overlap_size].astype(np.float64)
                
                # 使用CONFIG中的阈值进行簇合并
                threshold = CONFIG["merge_threshold"]  # 使用配置中的合并阈值
                for i in range(overlap_size):
                    for j in range(overlap_size):
                        if boundary_matrix[i, j] < threshold:
                            current_label = current_labels[-overlap_size+i]
                            next_label = next_labels[j]
                            if current_label != -1 and next_label != -1:
                                # 合并簇
                                final_labels[final_labels == next_label] = current_label
            except Exception as e:
                print(f"处理批次 {batch_idx + 1} 和 {batch_idx + 2} 之间的连接时出错: {str(e)}")
                continue
    
    return final_labels

def hdbscan_clustering(distance_matrix, min_cluster_size, min_samples=None, cluster_selection_epsilon=0.0):
    """使用 HDBSCAN 进行聚类，支持分块处理大型距离矩阵"""
    start_time = time.time()
    print("开始HDBSCAN聚类...")
    
    try:
        import hdbscan
        from scipy import sparse
        
        # 如果距离矩阵是文件路径，使用分批处理
        if isinstance(distance_matrix, str):
            print("使用分批处理大型距离矩阵...")
            # 使用分批聚类
            cluster_labels = batch_hdbscan_clustering(
                distance_matrix,
                batch_size=20000,  # 每批2万张
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon
            )
            
            # 创建聚类对象（用于返回）
            clustering = hdbscan.HDBSCAN(
                metric='precomputed',
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon
            )
            clustering.labels_ = cluster_labels
            
        else:
            # 对于小型距离矩阵，使用原始方法
            clustering = hdbscan.HDBSCAN(
                metric='precomputed',
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                algorithm='best',
                core_dist_n_jobs=-1
            )
            cluster_labels = clustering.fit_predict(distance_matrix)
        
        # 计算聚类耗时
        clustering_time = time.time() - start_time
        print(f"HDBSCAN聚类耗时: {clustering_time:.2f} 秒")
        
        return clustering, cluster_labels, clustering_time
        
    except ImportError:
        print("❌ HDBSCAN未安装！")
        print("请安装HDBSCAN: pip install hdbscan")
        print("回退到OPTICS聚类...")
        
        # 回退到OPTICS
        clustering = OPTICS(metric='precomputed', min_samples=min_samples or min_cluster_size, xi=cluster_selection_epsilon)
        clustering.fit(distance_matrix)
        cluster_labels = clustering.labels_
        
        clustering_time = time.time() - start_time
        print(f"OPTICS聚类耗时: {clustering_time:.2f} 秒")
        
        return clustering, cluster_labels, clustering_time

def merge_clusters(final_cluster_result, distance_matrix_file, merge_threshold):
    """合并相似的聚类"""
    merged_cluster_result = final_cluster_result.copy()
    labels = sorted(merged_cluster_result.keys())
    
    # 加载距离矩阵
    matrix_data = np.load(distance_matrix_file, mmap_mode='r')

    def should_merge(cluster1, cluster2):
        for sample1 in cluster1:
            for sample2 in cluster2:
                if matrix_data[sample1][sample2] > merge_threshold:
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

def process_outliers(merged_cluster_result, outliers, distance_matrix_file, merge_threshold):
    """处理离群点"""
    # 加载距离矩阵
    matrix_data = np.load(distance_matrix_file, mmap_mode='r')
    
    for outlier in outliers:
        for label, cluster in merged_cluster_result.items():
            can_add = True
            for sample in cluster:
                if matrix_data[outlier][sample] > merge_threshold:
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

def ensure_json_files_exist(config, hand_type):
    """确保必要的JSON文件存在"""
    # 检查并创建路径文件
    paths_file = config["dataset"]["paths_file"]
    if not os.path.exists(paths_file):
        print(f"\n📝 创建路径文件: {paths_file}")
        # 创建示例路径数据
        paths_data = {
            "left": [],
            "right": []
        }
        # 确保目录存在
        os.makedirs(os.path.dirname(paths_file), exist_ok=True)
        # 保存文件
        with open(paths_file, 'w', encoding='utf-8') as f:
            json.dump(paths_data, f, indent=4, ensure_ascii=False)
        print(f"✅ 路径文件创建成功")

    # 检查并创建输出目录
    output_dir = os.path.dirname(config["output_file"].format(
        hand_type=hand_type,
        min_cluster_size=config["clustering"]["min_cluster_size"],
        threshold=config["threshold"],
        merge_threshold=config["merge_threshold"]
    ))
    if not os.path.exists(output_dir):
        print(f"\n📁 创建输出目录: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"✅ 输出目录创建成功")

def process_hand_data(config, hand_type):
    """处理单个手部数据"""
    # 初始化总耗时和结果变量
    total_time = 0
    merged_cluster_result = {}  # 初始化为空字典
    total_samples = 0
    classified_samples = 0
    unclassified_samples = 0

    try:
        # 确保必要的JSON文件存在
        print("\n🔍 检查必要的文件...")
        ensure_json_files_exist(config, hand_type)

        # 加载或计算距离矩阵
        distance_matrix_file, n_samples, error_matrix_calculation_time = load_or_calculate_distance_matrix(config, hand_type)
        total_time += error_matrix_calculation_time

        # 执行HDBSCAN聚类
        (cluster_result, total_samples,
         classified_samples, unclassified_samples,
         clustering_time) = perform_hdbscan_clustering(distance_matrix_file, config)
        total_time += clustering_time

        # 处理聚类结果
        print("\n🔄 开始阈值筛选...")
        merged_cluster_result, threshold_filtering_time = process_clusters(
            cluster_result, distance_matrix_file, config
        )
        total_time += threshold_filtering_time

        # 保存筛选后的结果
        print("\n💾 准备保存聚类结果...")
        output_file = config["output_file"].format(
            hand_type=hand_type,
            min_cluster_size=config["clustering"]["min_cluster_size"],
            threshold=config["threshold"],
            merge_threshold=config["merge_threshold"]
        )
        
        # 检查输出路径
        print(f"输出文件路径: {output_file}")
        if not os.path.exists(os.path.dirname(output_file)):
            print(f"创建输出目录: {os.path.dirname(output_file)}")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存结果
        json_storage_time = save_cluster_result(
            merged_cluster_result,
            config["dataset"]["paths_file"],
            output_file,
            hand_type,
            config
        )
        total_time += json_storage_time

        # 清理内存
        import gc
        gc.collect()
        
        # 打印聚类结果总结
        print_cluster_summary(total_samples, classified_samples, unclassified_samples, merged_cluster_result)
        
        print(f"\n✅ 聚类完成！结果已保存为：")
        print(f"   📁 筛选后结果: {output_file}")
        print(f"   📁 完整结果: {config['unfiltered_output_file'].format(hand_type=hand_type)}")

    except Exception as e:
        print(f"\n⚠️ 处理过程中出现错误: {str(e)}")
        print("继续处理其他数据...")
        return

    # 计算总耗时
    print(f"\n⏱️  总处理时间: {total_time:.2f} 秒")

def load_or_calculate_distance_matrix(config, hand_type):
    """加载或计算距离矩阵"""
    error_matrix_calculation_time = 0
    distance_matrix_file = config["distance_matrix_file"].format(hand_type=hand_type)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(distance_matrix_file), exist_ok=True)
    
    if os.path.exists(distance_matrix_file):
        print(f"\n📂 发现已存在的距离矩阵文件: {distance_matrix_file}")
        print("正在加载距离矩阵...")
        start_time = time.time()
        try:
            # 返回文件路径而不是加载矩阵
            error_matrix_calculation_time = time.time() - start_time
            print(f"距离矩阵加载耗时: {error_matrix_calculation_time:.2f} 秒")
            return distance_matrix_file, None, error_matrix_calculation_time
        except Exception as e:
            print(f"加载距离矩阵失败: {str(e)}")
            print("将重新计算距离矩阵...")
    else:
        print(f"\n📂 未找到距离矩阵文件，将重新计算")

        # 读取数据集
        dataset, n_samples = read_dataset(config["dataset"]["json_dir"], config["max_samples"], hand_type)
        # 计算距离矩阵
    distance_matrix_file, error_matrix_calculation_time = calculate_distance_matrix(
            dataset, distance_matrix_file
        )

    return distance_matrix_file, n_samples, error_matrix_calculation_time

def perform_hdbscan_clustering(distance_matrix_file, config):
    """执行HDBSCAN聚类并返回结果"""
    # 从配置中提取参数
    clustering_config = config["clustering"]
    min_cluster_size = clustering_config["min_cluster_size"]
    
    # 获取可选参数（如果存在）
    min_samples = clustering_config.get("min_samples")
    cluster_selection_epsilon = clustering_config.get("cluster_selection_epsilon", 0.0)
    
    # 使用 HDBSCAN 进行聚类
    clustering, cluster_labels, clustering_time = hdbscan_clustering(
        distance_matrix_file,
        min_cluster_size,
        min_samples,
        cluster_selection_epsilon
    )

    # 计算聚类的数量
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"HDBSCAN聚类完成，共发现 {num_clusters} 个簇。")

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

def process_clusters(cluster_result, distance_matrix_file, config):
    """处理聚类结果，包括筛选、合并和处理离群点"""
    # 根据阈值筛选聚类结果
    final_cluster_result, outliers, threshold_filtering_time = filter_clusters(
        cluster_result, distance_matrix_file, config["threshold"]
    )

    # 聚类后合并
    merged_cluster_result = merge_clusters(final_cluster_result, distance_matrix_file, config["merge_threshold"])

    # 处理离群点
    merged_cluster_result = process_outliers(
        merged_cluster_result, outliers, distance_matrix_file, config["merge_threshold"]
    )

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
    print("🚀 InterHand HDBSCAN 手势聚类分析")
    print("🎯 优化的内存使用和并行计算")
    print("=" * 60)

    # 处理左手数据
    print("\n处理左手数据...")
    process_hand_data(CONFIG, "left")

    # 处理右手数据
    print("\n处理右手数据...")
    process_hand_data(CONFIG, "right")

def read_dataset(json_path, max_samples, hand_type):
    """读取InterHand数据集"""
    start_time = time.time()
    try:
        # 打开 JSON 文件并加载数据
        with open(json_path, 'r') as f:
            dataset = json.load(f)
        
        # 获取指定手型的数据
        if hand_type not in dataset:
            raise ValueError(f"数据文件中没有找到 {hand_type} 手的数据")
        
        hand_data = dataset[hand_type]
        if not isinstance(hand_data, list):
            raise ValueError(f"{hand_type} 手的数据格式错误")
        
        # 处理数据
        processed_data = []
        for item in hand_data:
            if isinstance(item, list) and len(item) > 0:
                processed_data.append(np.array(item))
        
        # 限制最大样本数
        processed_data = processed_data[:max_samples]
        n_samples = len(processed_data)
        
        if n_samples == 0:
            raise ValueError(f"没有找到有效的 {hand_type} 手关键点数据，请检查数据文件: {json_path}")
        
        # 计算数据读取耗时
        data_reading_time = time.time() - start_time
        print(f"数据读取耗时: {data_reading_time:.2f} 秒")
        print(f"成功加载 {n_samples} 个 {hand_type} 手样本")
        
        return np.array(processed_data), n_samples
    except Exception as e:
        print(f"读取数据失败: {str(e)}")
        exit()

def save_distance_matrix(matrix, filename):
    """保存距离矩阵到 .npy 文件"""
    # 确保输出目录存在
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

def save_cluster_result(cluster_result, path_file, save_path, hand_type, config):
    """保存聚类结果为JSON格式"""
    start_time = time.time()
    print(f"\n💾 正在保存聚类结果...")
    print(f"目标路径: {save_path}")
    
    # 检查路径文件是否存在
    if not os.path.exists(path_file):
        print(f"⚠️ 路径文件不存在: {path_file}")
        print("创建新的路径文件...")
        paths_data = {
            "left": [],
            "right": []
        }
        os.makedirs(os.path.dirname(path_file), exist_ok=True)
        with open(path_file, 'w', encoding='utf-8') as f:
            json.dump(paths_data, f, indent=4, ensure_ascii=False)
        print(f"✅ 路径文件创建成功")
    
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
            "merge_threshold": config["merge_threshold"],
            "hand_type": hand_type
        }
    }
    
    # 处理聚类结果
    print("处理聚类数据...")
    cluster_paths = []
    for label in sorted(cluster_result.keys()):
        paths = index_to_path(cluster_result[label], path_file, hand_type)
        if len(paths) > 1:  # 保留有效簇
            # 对簇内的路径进行排序
            sorted_paths = sort_cluster_paths(paths)
            cluster_paths.append(sorted_paths)
    
    # 对簇进行排序
    def get_cluster_key(cluster):
        # 获取簇中第一个路径的序列名和帧号作为排序键
        first_path = cluster[0]
        return (extract_sequence_name(first_path), extract_frame_idx(first_path))
    
    # 按序列名和帧号对簇进行排序
    cluster_paths.sort(key=get_cluster_key)
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

if __name__ == "__main__":
    main() 