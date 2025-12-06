'''
InterHand2.6M数据集 HDBSCAN聚类结果验证工具

该工具用于验证HDBSCAN聚类结果与真实多视角聚类结果的一致性。
主要功能：
1. 加载HDBSCAN聚类结果和真实多视角聚类结果
2. 计算两种聚类结果的一致性指标
3. 分析不一致的案例
4. 输出详细的验证报告

验证指标：
1. 聚类数量比较
2. 视角覆盖情况
3. 聚类一致性分析
4. 错误案例分析

使用方法：
python tools/InterHand/verify_tools/verify_hdbscan_cluster_results.py --hdbscan_result "data/InterHand/cluster_data/hdbscan/hand_clusters_hdbscan_left_min8_th0.0003_merge0.0003.json" --true_result "data/InterHand/cluster_data/origin/val_multi_view_true_clusters_sampled_5.json"
'''

import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
import re
import argparse

# 配置文件
CONFIG = {
    'hdbscan_results': {
        'left': "data/InterHand/cluster_data/hdbscan/hand_clusters_hdbscan_left_min8_th0.0003_merge0.0003.json",
        'right': "data/InterHand/cluster_data/hdbscan/hand_clusters_hdbscan_right_min8_th0.0003_merge0.0003.json"
    },
    'true_result': "data/InterHand/cluster_data/origin/val_multi_view_true_clusters_sampled_5.json"
}

def extract_frame_idx(view_path):
    """从视角路径中提取帧号"""
    match = re.search(r'image(\d+)', view_path)
    if match:
        return int(match.group(1))
    return 0

def extract_sequence_name(view_path):
    """从视角路径中提取序列名"""
    return view_path.split('/')[0]

def sort_clusters(clusters):
    """对聚类结果进行排序"""
    cluster_info = []
    for cluster in clusters:
        first_view = cluster[0]
        seq_name = extract_sequence_name(first_view)
        frame_idx = extract_frame_idx(first_view)
        cluster_info.append({
            'cluster': cluster,
            'seq_name': seq_name,
            'frame_idx': frame_idx
        })
    
    cluster_info.sort(key=lambda x: (x['seq_name'], x['frame_idx']))
    return [info['cluster'] for info in cluster_info]

def load_cluster_results(hdbscan_file, true_file):
    """加载聚类结果"""
    print("加载聚类结果...")
    with open(hdbscan_file, 'r') as f:
        hdbscan_result = json.load(f)
    with open(true_file, 'r') as f:
        true_result = json.load(f)
    
    print("对HDBSCAN聚类结果进行排序...")
    hdbscan_result['clusters'] = sort_clusters(hdbscan_result['clusters'])
    
    return hdbscan_result, true_result

def analyze_cluster_sizes(hdbscan_clusters, true_clusters):
    """分析聚类大小分布"""
    hdbscan_sizes = [len(cluster) for cluster in hdbscan_clusters]
    true_sizes = [len(cluster) for cluster in true_clusters]
    
    print("\n聚类大小分析:")
    print(f"HDBSCAN聚类:")
    print(f"  - 总聚类数: {len(hdbscan_sizes)}")
    print(f"  - 最大聚类大小: {max(hdbscan_sizes)}")
    print(f"  - 最小聚类大小: {min(hdbscan_sizes)}")
    print(f"  - 平均聚类大小: {np.mean(hdbscan_sizes):.2f}")
    print(f"  - 标准差: {np.std(hdbscan_sizes):.2f}")
    
    print(f"\n真实聚类:")
    print(f"  - 总聚类数: {len(true_sizes)}")
    print(f"  - 最大聚类大小: {max(true_sizes)}")
    print(f"  - 最小聚类大小: {min(true_sizes)}")
    print(f"  - 平均聚类大小: {np.mean(true_sizes):.2f}")
    print(f"  - 标准差: {np.std(true_sizes):.2f}")

def calculate_coverage(hdbscan_clusters, true_clusters):
    """计算视角覆盖情况"""
    hdbscan_views = set()
    for cluster in hdbscan_clusters:
        hdbscan_views.update(cluster)
    
    true_views = set()
    for cluster in true_clusters:
        true_views.update(cluster)
    
    total_views = len(true_views)
    covered_views = len(hdbscan_views.intersection(true_views))
    coverage_rate = covered_views / total_views if total_views > 0 else 0
    
    hdbscan_cluster_sizes = [len(cluster) for cluster in hdbscan_clusters]
    true_cluster_sizes = [len(cluster) for cluster in true_clusters]
    
    print("\n视角覆盖分析:")
    print(f"真实视角总数: {total_views}")
    print(f"HDBSCAN覆盖视角数: {covered_views}")
    print(f"覆盖率: {coverage_rate:.2%}")
    
    print(f"\n详细统计:")
    print(f"HDBSCAN聚类中的总视角数: {sum(hdbscan_cluster_sizes)}")
    print(f"真实聚类中的总视角数: {sum(true_cluster_sizes)}")
    print(f"HDBSCAN聚类中的唯一视角数: {len(hdbscan_views)}")
    print(f"真实聚类中的唯一视角数: {len(true_views)}")
    
    hdbscan_duplicates = sum(hdbscan_cluster_sizes) - len(hdbscan_views)
    true_duplicates = sum(true_cluster_sizes) - len(true_views)
    print(f"\n重复视角分析:")
    print(f"HDBSCAN聚类中的重复视角数: {hdbscan_duplicates}")
    print(f"真实聚类中的重复视角数: {true_duplicates}")
    
    actual_coverage = sum(hdbscan_cluster_sizes) / sum(true_cluster_sizes)
    print(f"\n实际覆盖率:")
    print(f"实际覆盖率: {actual_coverage:.2%}")

def analyze_cluster_consistency(hdbscan_clusters, true_clusters):
    """分析聚类结果的一致性"""
    # 将每个簇转换为集合，便于比较
    hdbscan_sets = [set(cluster) for cluster in hdbscan_clusters]
    true_sets = [set(cluster) for cluster in true_clusters]
    
    # 记录每个视角在真实聚类中的簇
    true_view_to_cluster = {}
    for i, cluster in enumerate(true_sets):
        for view in cluster:
            true_view_to_cluster[view] = i
    
    # 记录每个视角在HDBSCAN聚类中的簇
    hdbscan_view_to_cluster = {}
    for i, cluster in enumerate(hdbscan_sets):
        for view in cluster:
            hdbscan_view_to_cluster[view] = i
    
    # 找出所有在HDBSCAN中的视角
    hdbscan_views = set(hdbscan_view_to_cluster.keys())
    true_views = set(true_view_to_cluster.keys())
    
    # 找出匹配到不同簇的视角
    mismatched_views = []
    for view in hdbscan_views:
        if view in true_view_to_cluster:
            if true_view_to_cluster[view] != hdbscan_view_to_cluster[view]:
                mismatched_views.append({
                    'view': view,
                    'true_cluster': true_view_to_cluster[view],
                    'hdbscan_cluster': hdbscan_view_to_cluster[view],
                    'true_cluster_size': len(true_sets[true_view_to_cluster[view]]),
                    'hdbscan_cluster_size': len(hdbscan_sets[hdbscan_view_to_cluster[view]])
                })
    
    print("\n匹配到不同簇的视角分析:")
    print(f"总视角数: {len(true_views)}")
    print(f"HDBSCAN中的视角数: {len(hdbscan_views)}")
    print(f"匹配到不同簇的视角数: {len(mismatched_views)}")
    
    if mismatched_views:
        print("\n分配错误的簇样例 (前5个):")
        for i, case in enumerate(mismatched_views[:5]):
            print(f"\n案例 {i+1}:")
            print(f"视角: {case['view']}")
            
            # 获取真实簇和HDBSCAN簇的内容
            true_cluster = true_sets[case['true_cluster']]
            hdbscan_cluster = hdbscan_sets[case['hdbscan_cluster']]
            
            print(f"\n真实簇 {case['true_cluster']} (大小: {case['true_cluster_size']}):")
            print("包含的视角:")
            for view in sorted(true_cluster)[:5]:  # 只显示前5个视角
                print(f"  - {view}")
            if len(true_cluster) > 5:
                print(f"  ... 等 {len(true_cluster)-5} 个视角")
            
            print(f"\nHDBSCAN簇 {case['hdbscan_cluster']} (大小: {case['hdbscan_cluster_size']}):")
            print("包含的视角:")
            for view in sorted(hdbscan_cluster)[:5]:  # 只显示前5个视角
                print(f"  - {view}")
            if len(hdbscan_cluster) > 5:
                print(f"  ... 等 {len(hdbscan_cluster)-5} 个视角")
            
            # 分析两个簇的差异
            common_views = true_cluster.intersection(hdbscan_cluster)
            print(f"\n两个簇的共同视角数: {len(common_views)}")
            print(f"真实簇特有视角数: {len(true_cluster - hdbscan_cluster)}")
            print(f"HDBSCAN簇特有视角数: {len(hdbscan_cluster - true_cluster)}")

def verify_cluster_consistency(hdbscan_clusters, true_clusters):
    """验证聚类结果与真实聚类的一致性"""
    # 1. 验证簇的数量
    if len(hdbscan_clusters) != len(true_clusters):
        print(f"警告：簇的数量不匹配！HDBSCAN: {len(hdbscan_clusters)}, 真实: {len(true_clusters)}")
        return False
    
    # 2. 验证每个簇的内容
    for i, (hdbscan_cluster, true_cluster) in enumerate(zip(hdbscan_clusters, true_clusters)):
        # 转换为集合进行比较
        hdbscan_set = set(hdbscan_cluster)
        true_set = set(true_cluster)
        
        # 检查簇的大小
        if len(hdbscan_set) != len(true_set):
            print(f"警告：簇 {i} 的大小不匹配！HDBSCAN: {len(hdbscan_set)}, 真实: {len(true_set)}")
            return False
        
        # 检查簇的内容是否完全一致
        if hdbscan_set != true_set:
            print(f"警告：簇 {i} 的内容不完全一致！")
            print(f"HDBSCAN特有视角: {hdbscan_set - true_set}")
            print(f"真实簇特有视角: {true_set - hdbscan_set}")
            return False
    
    return True

def analyze_cluster_quality(hdbscan_clusters, true_clusters):
    """分析聚类质量"""
    # 1. 计算每个簇的Jaccard相似度
    jaccard_scores = []
    for hdbscan_cluster in hdbscan_clusters:
        best_match_score = 0
        for true_cluster in true_clusters:
            intersection = len(set(hdbscan_cluster) & set(true_cluster))
            union = len(set(hdbscan_cluster) | set(true_cluster))
            similarity = intersection / union if union > 0 else 0
            best_match_score = max(best_match_score, similarity)
        jaccard_scores.append(best_match_score)
    
    # 2. 计算整体匹配度
    perfect_matches = sum(1 for score in jaccard_scores if score == 1.0)
    match_rate = perfect_matches / len(jaccard_scores) if jaccard_scores else 0
    
    print("\n聚类质量分析:")
    print(f"完美匹配的簇数: {perfect_matches}")
    print(f"匹配率: {match_rate:.2%}")
    print(f"平均Jaccard相似度: {np.mean(jaccard_scores):.4f}")
    
    return match_rate == 1.0

def verify_sequence_continuity(clusters):
    """验证序列的连续性"""
    for i, cluster in enumerate(clusters):
        # 提取所有帧号
        frames = [extract_frame_idx(view) for view in cluster]
        frames.sort()
        
        # 如果只有一个帧，则认为是连续的
        if len(frames) == 1:
            continue
            
        # 检查帧号是否连续
        if max(frames) - min(frames) != len(frames) - 1:
            print(f"警告：簇 {i} 的帧号不连续！")
            print(f"帧号范围: {min(frames)} - {max(frames)}")
            print(f"缺失的帧号: {set(range(min(frames), max(frames)+1)) - set(frames)}")
            return False
    
    return True

def analyze_hand_results(hand_type):
    """分析指定手型的聚类结果"""
    print(f"\n{'='*50}")
    print(f"分析{hand_type}手聚类结果")
    print(f"{'='*50}")
    
    hdbscan_result, true_result = load_cluster_results(
        CONFIG['hdbscan_results'][hand_type],
        CONFIG['true_result']
    )
    
    hdbscan_clusters = hdbscan_result['clusters']
    true_clusters = true_result[f'{hand_type}_clusters']
    
    analyze_cluster_sizes(hdbscan_clusters, true_clusters)
    calculate_coverage(hdbscan_clusters, true_clusters)
    analyze_cluster_consistency(hdbscan_clusters, true_clusters)

def analyze_true_data_distribution(true_clusters):
    """分析真实数据的视角分布"""
    true_view_counts = defaultdict(int)
    for cluster in true_clusters:
        for view in cluster:
            true_view_counts[view] += 1
    
    print("\n真实数据视角分布:")
    for count, num_views in sorted(Counter(true_view_counts.values()).items()):
        print(f"  出现{count}次的视角数: {num_views}")

def analyze_sequence_distribution(clusters):
    """分析聚类结果的序列分布"""
    sequence_counts = defaultdict(int)
    for cluster in clusters:
        for view in cluster:
            seq_name = extract_sequence_name(view)
            sequence_counts[seq_name] += 1
    
    print("\n序列分布:")
    print(f"总序列数: {len(sequence_counts)}")
    print(f"每个序列的平均视角数: {sum(sequence_counts.values())/len(sequence_counts):.2f}")

def analyze_frame_distribution(clusters):
    """分析聚类结果的帧号分布"""
    frame_counts = defaultdict(int)
    for cluster in clusters:
        for view in cluster:
            frame_idx = extract_frame_idx(view)
            frame_counts[frame_idx] += 1
    
    print("\n帧号分布:")
    print(f"总帧数: {len(frame_counts)}")
    print(f"每个帧的平均视角数: {sum(frame_counts.values())/len(frame_counts):.2f}")

def main():
    parser = argparse.ArgumentParser(description='验证HDBSCAN聚类结果')
    parser.add_argument('--hdbscan_result', type=str, help='HDBSCAN聚类结果文件路径')
    parser.add_argument('--true_result', type=str, help='真实聚类结果文件路径')
    args = parser.parse_args()
    
    if args.hdbscan_result and args.true_result:
        CONFIG['hdbscan_results']['left'] = args.hdbscan_result
        CONFIG['true_result'] = args.true_result
    
    for hand_type in ['left', 'right']:
        print(f"\n{'='*50}")
        print(f"验证{hand_type}手聚类结果")
        print(f"{'='*50}")
        
        # 加载结果
        hdbscan_result, true_result = load_cluster_results(
            CONFIG['hdbscan_results'][hand_type],
            CONFIG['true_result']
        )
        
        hdbscan_clusters = hdbscan_result['clusters']
        true_clusters = true_result[f'{hand_type}_clusters']
        
        # 1. 验证簇的一致性
        is_consistent = verify_cluster_consistency(hdbscan_clusters, true_clusters)
        print(f"\n簇一致性验证: {'通过' if is_consistent else '失败'}")
        
        # 2. 分析聚类质量
        is_perfect = analyze_cluster_quality(hdbscan_clusters, true_clusters)
        print(f"聚类质量验证: {'完美匹配' if is_perfect else '存在差异'}")
        
        # 3. 输出详细统计信息
        analyze_cluster_sizes(hdbscan_clusters, true_clusters)
        calculate_coverage(hdbscan_clusters, true_clusters)
        analyze_cluster_consistency(hdbscan_clusters, true_clusters)

if __name__ == "__main__":
    main() 