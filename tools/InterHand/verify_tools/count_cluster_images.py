'''
    统计OPTICS聚类结果和真实聚类结果中的图片数量
'''

import json
from pathlib import Path

# 配置文件
CONFIG = {
    'optics_results': {
        'left': "data/InterHand/cluster_data/optics/hand_clusters_optics_left_min5_xi0.05_th0.0003_merge0.0003.json",
        'right': "data/InterHand/cluster_data/optics/hand_clusters_optics_right_min5_xi0.05_th0.0003_merge0.0003.json"
    },
    'true_result': "data/InterHand/cluster_data/origin/val_multi_view_true_clusters_sampled_5.json"
}

def count_images_in_clusters(clusters):
    """统计聚类中的图片数量"""
    total_images = 0
    for cluster in clusters:
        total_images += len(cluster)
    return total_images

def analyze_hand_images(hand_type):
    """分析指定手型的图片数量"""
    print(f"\n{'='*50}")
    print(f"分析{hand_type}手图片数量")
    print(f"{'='*50}")
    
    # 加载OPTICS聚类结果
    with open(CONFIG['optics_results'][hand_type], 'r') as f:
        optics_result = json.load(f)
    
    # 加载真实聚类结果
    with open(CONFIG['true_result'], 'r') as f:
        true_result = json.load(f)
    
    # 统计OPTICS聚类中的图片数量
    optics_images = count_images_in_clusters(optics_result['clusters'])
    
    # 统计真实聚类中的图片数量
    true_images = count_images_in_clusters(true_result[f'{hand_type}_clusters'])
    
    print(f"\nOPTICS聚类结果:")
    print(f"  - 总聚类数: {len(optics_result['clusters'])}")
    print(f"  - 总图片数: {optics_images}")
    
    print(f"\n真实聚类结果:")
    print(f"  - 总聚类数: {len(true_result[f'{hand_type}_clusters'])}")
    print(f"  - 总图片数: {true_images}")
    
    return optics_images, true_images

def main():
    # 分析左手图片
    left_optics, left_true = analyze_hand_images('left')
    
    # 分析右手图片
    right_optics, right_true = analyze_hand_images('right')
    
    # 打印总体统计
    print(f"\n{'='*50}")
    print("总体统计")
    print(f"{'='*50}")
    print(f"\nOPTICS聚类结果:")
    print(f"  - 左手图片数: {left_optics}")
    print(f"  - 右手图片数: {right_optics}")
    print(f"  - 总图片数: {left_optics + right_optics}")
    
    print(f"\n真实聚类结果:")
    print(f"  - 左手图片数: {left_true}")
    print(f"  - 右手图片数: {right_true}")
    print(f"  - 总图片数: {left_true + right_true}")

if __name__ == "__main__":
    main() 