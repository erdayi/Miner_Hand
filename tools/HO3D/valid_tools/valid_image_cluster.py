import json
import os
from collections import Counter

def count_clusters_and_images(json_file_path):
    """统计JSON文件中的簇数量和总图片数量"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 获取簇的数量
        num_clusters = len(data['clusters'])
        
        # 计算总图片数量
        total_images = sum(len(cluster) for cluster in data['clusters'])
        
        print(f"\n文件路径: {json_file_path}")
        print(f"簇的数量: {num_clusters}")
        print(f"总图片数量: {total_images}")
        
        # 统计每种大小的簇的数量
        cluster_sizes = Counter(len(cluster) for cluster in data['clusters'])
        
        print("\n簇的大小分布:")
        for size, count in sorted(cluster_sizes.items()):
            print(f"{size}张图片的簇: {count}个")
            
    except FileNotFoundError:
        print(f"错误：文件 {json_file_path} 未找到")
    except json.JSONDecodeError:
        print(f"错误：文件 {json_file_path} 不是有效的JSON格式")
    except KeyError:
        print(f"错误：文件 {json_file_path} 中没有找到 'clusters' 键")

def analyze_multiple_files(file_paths):
    """分析多个JSON文件"""
    for file_path in file_paths:
        count_clusters_and_images(file_path)

# 使用示例
if __name__ == "__main__":
    # 可以分析多个文件
    json_files = [
        'data/HO3D/cluster_data/origin/order/hand_clusters_0003_interval_1.json',
        'data/HO3D/cluster_data/origin/shuffled/hand_clusters_0003_interval_5.json',
        # 可以添加更多文件路径
    ]
    analyze_multiple_files(json_files)