'''
InterHand2.6M数据集聚类结果验证工具

该工具用于验证OPTICS聚类结果与真实多视角聚类结果的一致性。
主要功能：
1. 加载OPTICS聚类结果和真实多视角聚类结果
2. 计算两种聚类结果的一致性指标
3. 分析不一致的案例
4. 输出详细的验证报告

验证指标：
1. 聚类数量比较
2. 视角覆盖情况
3. 聚类一致性分析
4. 错误案例分析

使用方法：
直接运行脚本即可，参数已在CONFIG中配置：
python tools/InterHand/verify_tools/verify_optics_cluster_results.py
'''

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import re

# 配置文件
CONFIG = {
    'optics_results': {
        'left': "data/InterHand/cluster_data/optics/hand_clusters_optics_left_min5_xi0.05_th0.0003_merge0.0003.json",
        'right': "data/InterHand/cluster_data/optics/hand_clusters_optics_right_min5_xi0.05_th0.0003_merge0.0003.json"
    },
    'true_result': "data/InterHand/cluster_data/origin/val_multi_view_true_clusters_sampled_5.json",
    'output_dir': "data/InterHand/cluster_data/optics/verification_results"
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
    # 为每个聚类找到第一个视角的序列名和帧号
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
    
    # 按序列名和帧号排序
    cluster_info.sort(key=lambda x: (x['seq_name'], x['frame_idx']))
    
    # 返回排序后的聚类
    return [info['cluster'] for info in cluster_info]

def load_cluster_results(optics_file, true_file):
    """加载聚类结果"""
    print("加载聚类结果...")
    with open(optics_file, 'r') as f:
        optics_result = json.load(f)
    with open(true_file, 'r') as f:
        true_result = json.load(f)
    
    # 对OPTICS聚类结果进行排序
    print("对OPTICS聚类结果进行排序...")
    optics_result['clusters'] = sort_clusters(optics_result['clusters'])
    
    return optics_result, true_result

def analyze_cluster_sizes(optics_clusters, true_clusters):
    """分析聚类大小分布"""
    optics_sizes = [len(cluster) for cluster in optics_clusters]
    true_sizes = [len(cluster) for cluster in true_clusters]
    
    print("\n聚类大小分析:")
    print(f"OPTICS聚类:")
    print(f"  - 总聚类数: {len(optics_sizes)}")
    print(f"  - 最大聚类大小: {max(optics_sizes)}")
    print(f"  - 最小聚类大小: {min(optics_sizes)}")
    print(f"  - 平均聚类大小: {np.mean(optics_sizes):.2f}")
    
    print(f"\n真实聚类:")
    print(f"  - 总聚类数: {len(true_sizes)}")
    print(f"  - 最大聚类大小: {max(true_sizes)}")
    print(f"  - 最小聚类大小: {min(true_sizes)}")
    print(f"  - 平均聚类大小: {np.mean(true_sizes):.2f}")

def analyze_cluster_consistency(optics_clusters, true_clusters):
    """分析聚类一致性"""
    # 对OPTICS聚类结果进行排序
    print("对OPTICS聚类结果进行排序...")
    sorted_optics_clusters = sort_clusters(optics_clusters)
    
    # 创建视角到聚类的映射
    optics_view_to_cluster = {}
    for i, cluster in enumerate(sorted_optics_clusters):
        for view in cluster:
            optics_view_to_cluster[view] = i
    
    # 统计信息
    total_clusters = len(true_clusters)
    perfect_matches = 0
    partial_matches = 0
    mismatched_clusters = []
    
    print("\n聚类一致性分析:")
    print(f"真实簇总数: {total_clusters}")
    
    # 分析每个真实簇
    for true_cluster_idx, true_cluster in enumerate(true_clusters):
        # 获取该真实簇中所有视角在OPTICS聚类中的簇标签
        optics_labels = set()
        for view in true_cluster:
            if view in optics_view_to_cluster:
                optics_labels.add(optics_view_to_cluster[view])
        
        # 检查是否所有视角都在同一个OPTICS簇中
        if len(optics_labels) == 1:
            optics_cluster_idx = optics_labels.pop()
            optics_cluster = sorted_optics_clusters[optics_cluster_idx]
            
            # 检查是否有额外的视角
            extra_views = set(optics_cluster) - set(true_cluster)
            if not extra_views:
                perfect_matches += 1
            else:
                partial_matches += 1
                mismatched_clusters.append({
                    'true_cluster_idx': true_cluster_idx,
                    'optics_cluster_idx': optics_cluster_idx,
                    'true_cluster': true_cluster,
                    'optics_cluster': optics_cluster,
                    'extra_views': list(extra_views)
                })
        else:
            mismatched_clusters.append({
                'true_cluster_idx': true_cluster_idx,
                'optics_labels': list(optics_labels),
                'true_cluster': true_cluster,
                'split_into': len(optics_labels)
            })
    
    # 输出统计信息
    print(f"完美匹配的簇数: {perfect_matches} ({perfect_matches/total_clusters*100:.2f}%)")
    print(f"部分匹配的簇数: {partial_matches} ({partial_matches/total_clusters*100:.2f}%)")
    print(f"完全失配的簇数: {len(mismatched_clusters) - partial_matches} ({(len(mismatched_clusters) - partial_matches)/total_clusters*100:.2f}%)")
    
    # 输出错误案例
    if mismatched_clusters:
        print("\n错误案例分析 (前5个):")
        for i, case in enumerate(mismatched_clusters[:5]):
            print(f"\n案例 {i+1}:")
            if 'optics_cluster_idx' in case:
                print(f"真实簇 {case['true_cluster_idx']} 与 OPTICS簇 {case['optics_cluster_idx']} 部分匹配")
                print(f"真实簇大小: {len(case['true_cluster'])}")
                print(f"OPTICS簇大小: {len(case['optics_cluster'])}")
                print(f"额外视角数: {len(case['extra_views'])}")
                print("额外视角示例:")
                for view in case['extra_views'][:3]:
                    print(f"  - {view}")
                if len(case['extra_views']) > 3:
                    print(f"  ... 等 {len(case['extra_views'])-3} 个视角")
            else:
                print(f"真实簇 {case['true_cluster_idx']} 被分割到 {case['split_into']} 个OPTICS簇中")
                print(f"真实簇大小: {len(case['true_cluster'])}")
                print(f"OPTICS簇标签: {case['optics_labels']}")

def calculate_coverage(optics_clusters, true_clusters):
    """计算视角覆盖情况"""
    # 收集所有视角
    optics_views = set()
    for cluster in optics_clusters:
        optics_views.update(cluster)
    
    true_views = set()
    for cluster in true_clusters:
        true_views.update(cluster)
    
    # 计算覆盖情况
    total_views = len(true_views)
    covered_views = len(optics_views.intersection(true_views))
    coverage_rate = covered_views / total_views if total_views > 0 else 0
    
    # 计算每个聚类中的视角数
    optics_cluster_sizes = [len(cluster) for cluster in optics_clusters]
    true_cluster_sizes = [len(cluster) for cluster in true_clusters]
    
    print("\n视角覆盖分析:")
    print(f"真实视角总数: {total_views}")
    print(f"OPTICS覆盖视角数: {covered_views}")
    print(f"覆盖率: {coverage_rate:.2f}%")
    print(f"\n详细统计:")
    print(f"OPTICS聚类中的总视角数: {sum(optics_cluster_sizes)}")
    print(f"真实聚类中的总视角数: {sum(true_cluster_sizes)}")
    print(f"OPTICS聚类中的唯一视角数: {len(optics_views)}")
    print(f"真实聚类中的唯一视角数: {len(true_views)}")
    
    # 检查是否有重复的视角
    optics_duplicates = sum(optics_cluster_sizes) - len(optics_views)
    true_duplicates = sum(true_cluster_sizes) - len(true_views)
    print(f"\n重复视角分析:")
    print(f"OPTICS聚类中的重复视角数: {optics_duplicates}")
    print(f"真实聚类中的重复视角数: {true_duplicates}")
    
    # 计算实际覆盖率
    actual_coverage = sum(optics_cluster_sizes) / sum(true_cluster_sizes)
    print(f"\n实际覆盖率:")
    print(f"实际覆盖率: {actual_coverage:.2f}%")

def analyze_hand_results(hand_type):
    """分析指定手型的聚类结果"""
    print(f"\n{'='*50}")
    print(f"分析{hand_type}手聚类结果")
    print(f"{'='*50}")
    
    # 加载结果
    optics_result, true_result = load_cluster_results(
        CONFIG['optics_results'][hand_type],
        CONFIG['true_result']
    )
    
    optics_clusters = optics_result['clusters']
    true_clusters = true_result[f'{hand_type}_clusters']
    
    # 分析聚类大小
    analyze_cluster_sizes(optics_clusters, true_clusters)
    
    # 计算视角覆盖
    calculate_coverage(optics_clusters, true_clusters)
    
    # 分析聚类一致性
    analyze_cluster_consistency(optics_clusters, true_clusters)
    
    # 保存验证结果
    save_verification_results(hand_type, optics_clusters, true_clusters)

def save_verification_results(hand_type, optics_clusters, true_clusters):
    """保存验证结果到JSON文件"""
    # 创建输出目录
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备结果数据
    results = {
        'hand_type': hand_type,
        'optics_file': CONFIG['optics_results'][hand_type],
        'true_file': CONFIG['true_result'],
        'cluster_sizes': {
            'optics': {
                'total': len(optics_clusters),
                'sizes': [len(cluster) for cluster in optics_clusters]
            },
            'true': {
                'total': len(true_clusters),
                'sizes': [len(cluster) for cluster in true_clusters]
            }
        },
        'coverage': {
            'optics_views': len(set().union(*[set(cluster) for cluster in optics_clusters])),
            'true_views': len(set().union(*[set(cluster) for cluster in true_clusters]))
        }
    }
    
    # 保存结果
    output_file = output_dir / f"verification_{hand_type}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n验证结果已保存到: {output_file}")

def main():
    print("开始验证OPTICS聚类结果...")
    print(f"真实聚类文件: {CONFIG['true_result']}")
    print(f"OPTICS聚类文件:")
    print(f"  - 左手: {CONFIG['optics_results']['left']}")
    print(f"  - 右手: {CONFIG['optics_results']['right']}")
    
    # 分析左手结果
    analyze_hand_results('left')
    
    # 分析右手结果
    analyze_hand_results('right')
    
    print("\n验证完成！")

if __name__ == "__main__":
    main() 