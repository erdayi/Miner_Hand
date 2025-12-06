import json
import numpy as np
from scipy.linalg import orthogonal_procrustes
import time
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, array, struct
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StructType, StructField
import hdbscan
from typing import List, Tuple
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== 配置参数 =====
CONFIG = {
    # Spark配置
    "spark": {
        "app_name": "InterHand HDBSCAN Clustering",
        "master": "local[*]",  # 使用所有可用核心
        "driver_memory": "8g",
        "executor_memory": "8g",
        "executor_cores": 4,
        "num_executors": 1,
        "max_result_size": "4g",
        "python_worker_timeout": "600",
        "spark.network.timeout": "600s",
        "spark.executor.heartbeatInterval": "60s",
        "spark.python.worker.memory": "4g",
        "spark.python.worker.reuse": "true",
        "spark.python.worker.keepalive": "true",
        "spark.python.worker.timeout": "600",
        "spark.python.worker.port": "0",
        "spark.python.worker.port.retries": "100",
        "spark.python.worker.port.timeout": "600",
        "spark.python.worker.port.range": "10000-20000",
        "spark.local.dir": "/tmp/spark-temp",
        "spark.driver.extraJavaOptions": "-Djava.io.tmpdir=/tmp/spark-temp",
        "spark.executor.extraJavaOptions": "-Djava.io.tmpdir=/tmp/spark-temp",
        "spark.task.maxFailures": "2",
        "spark.speculation": "true",
        "spark.memory.fraction": "0.8",
        "spark.memory.storageFraction": "0.5",
        "spark.shuffle.file.buffer": "1m",
        "spark.shuffle.compress": "true",
        "spark.shuffle.spill.compress": "true",
        "spark.broadcast.compress": "true",
        "spark.rdd.compress": "true",
        "spark.io.compression.codec": "lz4",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.kryoserializer.buffer.max": "1g",
        "spark.kryo.registrationRequired": "false",
        "spark.sql.shuffle.partitions": "200",
        "spark.default.parallelism": "200",
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.sql.execution.arrow.pyspark.fallback.enabled": "true",
        "spark.executor.memoryOverhead": "2g",
        "spark.driver.memoryOverhead": "2g",
        "spark.memory.offHeap.enabled": "true",
        "spark.memory.offHeap.size": "4g",
        "spark.executor.extraClassPath": "/usr/local/lib/python3.8/site-packages/pyspark/jars/*",
        "spark.driver.extraClassPath": "/usr/local/lib/python3.8/site-packages/pyspark/jars/*"
    },
    # 数据集配置
    "dataset": {
        "name": "InterHand",
        "json_dir": "data/InterHand/origin_data/val_joint_3d_shuffled.json",
        "paths_file": "data/InterHand/origin_data/val_image_paths_shuffled.json",
        "scale_enlarge": 1.25
    },
    # 距离矩阵配置
    "distance_matrix_file": "data/InterHand/load_data/hdbscan_spark/distance_matrix_{hand_type}_shuffled.parquet",
    # 样本数量配置
    "max_samples": 10000,
    # HDBSCAN聚类算法配置
    "clustering": {
        "min_cluster_size": 8,
        "min_samples": 5,
        "cluster_selection_epsilon": 0.0
    },
    # 筛选阈值配置
    "threshold": 0.0003,
    # 合并阈值配置
    "merge_threshold": 0.0003,
    # 输出文件配置
    "output_file": "data/InterHand/cluster_data/hdbscan_spark/hand_clusters_hdbscan_{hand_type}_min{min_cluster_size}_th{threshold}_merge{merge_threshold}.json",
    "unfiltered_output_file": "data/InterHand/cluster_data/hdbscan_spark/unfiltered_hdbscan_{hand_type}_clusters.json"
}

def create_spark_session(config):
    """创建优化的Spark会话"""
    builder = SparkSession.builder.appName(config["spark"]["app_name"])
    
    # 应用所有Spark配置
    for key, value in config["spark"].items():
        builder = builder.config(f"spark.{key}", value)
    
    # 创建会话
    spark = builder.getOrCreate()
    
    # 设置日志级别
    spark.sparkContext.setLogLevel("WARN")
    
    return spark

def align_w_scale_vectorized(mtx1, mtx2):
    """向量化的对齐函数"""
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

@udf(returnType=FloatType())
def calculate_distance_udf(data1, data2):
    """计算两个手部姿态之间的距离（UDF版本）"""
    try:
        data1 = np.array(data1, dtype=np.float32)
        data2 = np.array(data2, dtype=np.float32)
        xyz_aligned = align_w_scale_vectorized(data1, data2)
        error = np.mean(np.linalg.norm(data1 - xyz_aligned, axis=1))
        return float(error)
    except Exception as e:
        logger.error(f"计算距离时出错: {str(e)}")
        return None

def load_dataset(spark, json_path, max_samples, hand_type):
    """使用Spark加载数据集"""
    start_time = time.time()
    
    # 读取JSON文件
    with open(json_path, 'r') as f:
        dataset = json.load(f)
    
    if hand_type not in dataset:
        raise ValueError(f"数据文件中没有找到 {hand_type} 手的数据")
    
    hand_data = dataset[hand_type]
    if not isinstance(hand_data, list):
        raise ValueError(f"{hand_type} 手的数据格式错误")
    
    # 处理数据并限制样本数
    processed_data = []
    for i, item in enumerate(hand_data[:max_samples]):
        if isinstance(item, list) and len(item) > 0:
            float_data = [[float(x) for x in point] for point in item]
            processed_data.append((i, float_data))
    
    # 定义schema
    schema = StructType([
        StructField("index", IntegerType(), False),
        StructField("data", ArrayType(ArrayType(FloatType())), False)
    ])
    
    # 创建Spark DataFrame
    df = spark.createDataFrame(processed_data, schema)
    
    # 计算数据读取耗时
    data_reading_time = time.time() - start_time
    logger.info(f"数据读取耗时: {data_reading_time:.2f} 秒")
    logger.info(f"成功加载 {len(processed_data)} 个 {hand_type} 手样本")
    
    return df

def calculate_distance_matrix_spark(df, distance_matrix_file):
    """使用Spark DataFrame API计算距离矩阵"""
    start_time = time.time()
    
    # 创建笛卡尔积
    df1 = df.withColumnRenamed("index", "idx1").withColumnRenamed("data", "data1")
    df2 = df.withColumnRenamed("index", "idx2").withColumnRenamed("data", "data2")
    
    # 使用crossJoin并只保留上三角部分
    cross_df = df1.crossJoin(df2).filter(col("idx1") < col("idx2"))
    
    # 计算距离
    distances_df = cross_df.select(
        col("idx1"),
        col("idx2"),
        calculate_distance_udf(col("data1"), col("data2")).alias("distance")
    ).filter(col("distance").isNotNull())
    
    # 缓存结果
    distances_df = distances_df.cache()
    
    # 保存结果
    distances_df.write.mode("overwrite").parquet(distance_matrix_file)
    
    # 计算耗时
    computation_time = time.time() - start_time
    logger.info(f"距离矩阵计算耗时: {computation_time:.2f} 秒")
    
    return distances_df

def perform_hdbscan_clustering_spark(distances_df, config):
    """使用HDBSCAN进行聚类"""
    start_time = time.time()
    
    # 收集距离矩阵
    distances = distances_df.collect()
    n_samples = int(np.sqrt(len(distances) * 2))  # 计算样本数
    distance_matrix = np.zeros((n_samples, n_samples))
    
    # 填充距离矩阵
    for row in distances:
        i, j, dist = row
        distance_matrix[i, j] = dist
        distance_matrix[j, i] = dist
    
    # 执行HDBSCAN聚类
    clustering = hdbscan.HDBSCAN(
        metric='precomputed',
        min_cluster_size=config["clustering"]["min_cluster_size"],
        min_samples=config["clustering"]["min_samples"],
        cluster_selection_epsilon=config["clustering"]["cluster_selection_epsilon"]
    )
    
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    # 计算聚类耗时
    clustering_time = time.time() - start_time
    logger.info(f"HDBSCAN聚类耗时: {clustering_time:.2f} 秒")
    
    return clustering, cluster_labels, clustering_time

def save_cluster_result(cluster_result, paths_file, output_file, hand_type, config):
    """保存聚类结果"""
    # 读取图像路径
    with open(paths_file, 'r') as f:
        paths = json.load(f)
    
    # 处理聚类结果
    processed_result = {}
    for cluster_id, indices in cluster_result.items():
        if cluster_id == -1:  # 跳过噪声点
            continue
        processed_result[str(cluster_id)] = [paths[i] for i in indices]
    
    # 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(processed_result, f, indent=2)
    
    logger.info(f"聚类结果已保存到: {output_file}")

def process_hand_data_spark(spark, config, hand_type):
    """处理单个手部数据"""
    try:
        # 加载数据集
        df = load_dataset(spark, config["dataset"]["json_dir"], config["max_samples"], hand_type)
        
        # 计算距离矩阵
        distances_df = calculate_distance_matrix_spark(
            df,
            config["distance_matrix_file"].format(hand_type=hand_type)
        )
        
        # 执行聚类
        clustering, cluster_labels, clustering_time = perform_hdbscan_clustering_spark(
            distances_df, config
        )
        
        # 处理聚类结果
        cluster_result = {}
        for i, label in enumerate(cluster_labels):
            if label not in cluster_result:
                cluster_result[label] = []
            cluster_result[label].append(i)
        
        # 保存结果
        save_cluster_result(
            cluster_result,
            config["dataset"]["paths_file"],
            config["output_file"].format(
                hand_type=hand_type,
                min_cluster_size=config["clustering"]["min_cluster_size"],
                threshold=config["threshold"],
                merge_threshold=config["merge_threshold"]
            ),
            hand_type,
            config
        )
        
        # 打印统计信息
        total_samples = len(cluster_labels)
        classified_samples = sum(1 for label in cluster_labels if label != -1)
        unclassified_samples = sum(1 for label in cluster_labels if label == -1)
        
        logger.info(f"\n聚类结果统计:")
        logger.info(f"总样本数: {total_samples}")
        logger.info(f"已分类样本数: {classified_samples}")
        logger.info(f"未分类样本数: {unclassified_samples}")
        logger.info(f"聚类数量: {len(cluster_result)}")
        
    except Exception as e:
        logger.error(f"\n⚠️ 处理过程中出现错误: {str(e)}")
        logger.info("继续处理其他数据...")
        return

def main():
    logger.info("🚀 InterHand HDBSCAN 手势聚类分析 (Spark版本)")
    logger.info("🎯 分布式计算支持大规模数据处理")
    logger.info("=" * 60)
    
    # 创建Spark会话
    spark = create_spark_session(CONFIG)
    
    try:
        # 处理左手数据
        logger.info("\n处理左手数据...")
        process_hand_data_spark(spark, CONFIG, "left")
        
        # 处理右手数据
        logger.info("\n处理右手数据...")
        process_hand_data_spark(spark, CONFIG, "right")
        
    finally:
        # 停止Spark会话
        spark.stop()

if __name__ == "__main__":
    main() 