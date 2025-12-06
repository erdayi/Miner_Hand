import json
from collections import Counter

'''
    本方法旨在判断两个json文件是否相同，并计算聚类正确率
    （辅助验证初始方法和聚类算法结果是否一致）
    比较时会忽略簇内图片的顺序
'''

def sort_json(data):
    if isinstance(data, dict):
        return {key: sort_json(value) for key, value in sorted(data.items())}
    elif isinstance(data, list):
        # 如果列表元素是字符串（图片路径），则将列表转换为集合进行比较
        if all(isinstance(x, str) for x in data):
            return set(data)
        # 对于其他类型的列表，递归处理每个元素
        return [sort_json(item) for item in data]
    return data

def calculate_clustering_accuracy(ground_truth_clusters, predicted_clusters):
    """计算聚类正确率"""
    # 将簇转换为集合形式
    gt_sets = [set(cluster) for cluster in ground_truth_clusters]
    pred_sets = [set(cluster) for cluster in predicted_clusters]
    
    # 统计正确分类的图片数量
    correct_count = 0
    total_images = sum(len(cluster) for cluster in ground_truth_clusters)
    
    # 对于每个真实簇，找到最匹配的预测簇
    for gt_set in gt_sets:
        max_intersection = 0
        for pred_set in pred_sets:
            intersection = len(gt_set.intersection(pred_set))
            max_intersection = max(max_intersection, intersection)
        correct_count += max_intersection
    
    accuracy = correct_count / total_images if total_images > 0 else 0
    return accuracy, correct_count, total_images

def compare_json_files(file1_path, file2_path):
    try:
        with open(file1_path, 'r', encoding='utf-8') as file1:
            data1 = json.load(file1)
        with open(file2_path, 'r', encoding='utf-8') as file2:
            data2 = json.load(file2)

        # 将数据转换为可比较的格式（忽略顺序）
        processed_data1 = sort_json(data1)
        processed_data2 = sort_json(data2)

        # 计算聚类正确率
        accuracy, correct_count, total_images = calculate_clustering_accuracy(
            data1['clusters'], data2['clusters']
        )

        # 打印详细信息
        print(f"\n聚类评估结果:")
        print(f"总图片数量: {total_images}")
        print(f"正确分类数量: {correct_count}")
        print(f"聚类正确率: {accuracy:.2%}")
        print(f"真实簇数量: {len(data1['clusters'])}")
        print(f"预测簇数量: {len(data2['clusters'])}")

        # 比较处理后的数据
        if isinstance(processed_data1, dict) and isinstance(processed_data2, dict):
            # 检查键是否相同
            if processed_data1.keys() != processed_data2.keys():
                return False
            # 比较每个簇
            for key in processed_data1:
                if processed_data1[key] != processed_data2[key]:
                    return False
            return True
        return processed_data1 == processed_data2

    except FileNotFoundError:
        print("错误：文件未找到。")
        return False
    except json.JSONDecodeError:
        print("错误：JSON 解析失败。")
        return False

# 使用示例
if __name__ == "__main__":
    file1_path = 'data/HO3D/cluster_data/ground_truth/hand_clusters_interval_1.json'
    file2_path = 'data/HO3D/cluster_data/optics/shuffled/hand_clusters_optics_interval_1.json'
    result = compare_json_files(file1_path, file2_path)
    print(f"\n两个 JSON 文件是否完全相同: {result}")