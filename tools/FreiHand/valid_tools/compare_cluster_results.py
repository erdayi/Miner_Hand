import json
import os
from collections import defaultdict

def load_cluster_results(file_path):
    """加载聚类结果文件
    
    Args:
        file_path: 聚类结果JSON文件路径
        
    Returns:
        dict: 样本到簇标签的映射
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 创建样本到簇标签的映射
        sample_to_cluster = {}
        
        # 处理不同的文件格式
        if 'clusters' in data:  # HDBSCAN格式
            clusters = data['clusters']
            # 处理HDBSCAN格式（列表格式）
            for cluster_idx, samples in enumerate(clusters):
                if not isinstance(samples, list):
                    print(f"警告: 簇 {cluster_idx} 不是列表格式，跳过")
                    continue
                for sample in samples:
                    if isinstance(sample, (int, str)):
                        sample_to_cluster[sample] = cluster_idx
                    else:
                        print(f"警告: 样本 {sample} 格式无效，跳过")
        else:  # OPTICS格式（字典格式）
            # 处理OPTICS格式（字典格式）
            for cluster_label, samples in data.items():
                if not isinstance(samples, list):
                    print(f"警告: 簇 {cluster_label} 不是列表格式，跳过")
                    continue
                for sample in samples:
                    if isinstance(sample, (int, str)):
                        sample_to_cluster[sample] = int(cluster_label)  # 确保标签是整数
                    else:
                        print(f"警告: 样本 {sample} 格式无效，跳过")
                
        if not sample_to_cluster:
            raise ValueError("没有找到有效的聚类结果")
            
        return sample_to_cluster
    except Exception as e:
        print(f"加载文件 {file_path} 失败: {str(e)}")
        return None

def compare_clusters(hdbscan_file, optics_file):
    """比较HDBSCAN和OPTICS的聚类结果
    
    Args:
        hdbscan_file: HDBSCAN聚类结果文件路径
        optics_file: OPTICS聚类结果文件路径
        
    Returns:
        tuple: (是否一致, 不一致的样本列表, 统计信息)
    """
    # 加载两个聚类结果
    print("\n正在加载HDBSCAN结果...")
    hdbscan_clusters = load_cluster_results(hdbscan_file)
    if hdbscan_clusters is None:
        print("❌ HDBSCAN结果加载失败")
        return False, [], {}
        
    print("正在加载OPTICS结果...")
    optics_clusters = load_cluster_results(optics_file)
    if optics_clusters is None:
        print("❌ OPTICS结果加载失败")
        return False, [], {}
    
    # 获取所有样本
    all_samples = set(hdbscan_clusters.keys()) | set(optics_clusters.keys())
    
    # 统计信息
    stats = {
        'total_samples': len(all_samples),
        'hdbscan_clusters': len(set(hdbscan_clusters.values())),
        'optics_clusters': len(set(optics_clusters.values())),
        'inconsistent_samples': 0,
        'consistent_samples': 0
    }
    
    # 检查每个样本的簇分配是否一致
    inconsistent_samples = []
    for sample in all_samples:
        hdbscan_cluster = hdbscan_clusters.get(sample)
        optics_cluster = optics_clusters.get(sample)
        
        if hdbscan_cluster is None or optics_cluster is None:
            # 样本在其中一个结果中缺失
            inconsistent_samples.append((sample, 'missing', hdbscan_cluster, optics_cluster))
            stats['inconsistent_samples'] += 1
        elif hdbscan_cluster != optics_cluster:
            # 样本被分到不同的簇
            inconsistent_samples.append((sample, 'different_clusters', hdbscan_cluster, optics_cluster))
            stats['inconsistent_samples'] += 1
        else:
            stats['consistent_samples'] += 1
    
    # 计算一致性比例
    stats['consistency_ratio'] = stats['consistent_samples'] / stats['total_samples'] if stats['total_samples'] > 0 else 0
    
    # 检查是否完全一致
    is_consistent = len(inconsistent_samples) == 0
    
    return is_consistent, inconsistent_samples, stats

def analyze_inconsistencies(inconsistent_samples, hdbscan_file, optics_file):
    """分析不一致的样本
    
    Args:
        inconsistent_samples: 不一致的样本列表
        hdbscan_file: HDBSCAN聚类结果文件路径
        optics_file: OPTICS聚类结果文件路径
    """
    try:
        # 加载原始聚类结果
        with open(hdbscan_file, 'r', encoding='utf-8') as f:
            hdbscan_data = json.load(f)
        with open(optics_file, 'r', encoding='utf-8') as f:
            optics_data = json.load(f)
        
        # 按类型分组不一致的样本
        missing_samples = []
        different_clusters = defaultdict(list)
        
        for sample, type_, hdbscan_cluster, optics_cluster in inconsistent_samples:
            if type_ == 'missing':
                missing_samples.append(sample)
            else:
                different_clusters[(hdbscan_cluster, optics_cluster)].append(sample)
        
        # 打印分析结果
        print("\n不一致样本分析:")
        print("=" * 50)
        
        if missing_samples:
            print(f"\n缺失样本 ({len(missing_samples)}):")
            for sample in missing_samples:
                print(f"  样本 {sample}")
        
        if different_clusters:
            print(f"\n不同簇分配 ({len(different_clusters)} 种情况):")
            for (hdbscan_cluster, optics_cluster), samples in different_clusters.items():
                print(f"\n  HDBSCAN簇 {hdbscan_cluster} -> OPTICS簇 {optics_cluster} ({len(samples)} 个样本):")
                for sample in samples[:5]:  # 只显示前5个样本
                    print(f"    样本 {sample}")
                if len(samples) > 5:
                    print(f"    ... 还有 {len(samples)-5} 个样本")
    except Exception as e:
        print(f"分析不一致样本时出错: {str(e)}")

def main():
    # 设置文件路径
    hdbscan_file = "data/FreiHand/cluster_data/hdbscan/hand_clusters_hdbscan_min8_th0.001_merge0.001.json"
    optics_file = "data/FreiHand/cluster_data/clusters_optics_32560_0.003_min8_xi0.1.json"
    
    print("🔍 开始比较聚类结果...")
    print(f"HDBSCAN结果: {hdbscan_file}")
    print(f"OPTICS结果: {optics_file}")
    
    try:
        # 比较聚类结果
        is_consistent, inconsistent_samples, stats = compare_clusters(hdbscan_file, optics_file)
        
        if not stats:  # 如果统计信息为空，说明加载失败
            print("❌ 无法比较聚类结果")
            return
        
        # 打印统计信息
        print("\n📊 比较结果统计:")
        print("=" * 50)
        print(f"总样本数: {stats['total_samples']}")
        print(f"HDBSCAN簇数: {stats['hdbscan_clusters']}")
        print(f"OPTICS簇数: {stats['optics_clusters']}")
        print(f"一致样本数: {stats['consistent_samples']}")
        print(f"不一致样本数: {stats['inconsistent_samples']}")
        print(f"一致性比例: {stats['consistency_ratio']*100:.2f}%")
        
        # 分析不一致的样本
        if inconsistent_samples:
            analyze_inconsistencies(inconsistent_samples, hdbscan_file, optics_file)
        
        # 输出最终结论
        print("\n📝 结论:")
        print("=" * 50)
        if is_consistent:
            print("✅ 两个聚类结果完全一致！")
        else:
            print("⚠️ 两个聚类结果存在差异。")
            print(f"   - 共有 {stats['inconsistent_samples']} 个样本的簇分配不一致")
            print(f"   - 一致性比例为 {stats['consistency_ratio']*100:.2f}%")
            
    except Exception as e:
        print(f"\n❌ 程序执行出错: {str(e)}")

if __name__ == "__main__":
    main() 