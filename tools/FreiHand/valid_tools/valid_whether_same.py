import json
import os

def compare_cluster_files(file_paths):
    """
    比较多个聚类结果JSON文件是否完全一致
    
    参数:
        file_paths: 要比较的JSON文件路径列表
        
    返回:
        bool: 所有文件内容是否完全一致
        dict: 每个文件的聚类统计信息
    """
    cluster_data = []
    stats = {}
    
    # 1. 读取所有文件内容
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            cluster_data.append(data)
            
            # 记录每个文件的统计信息
            stats[file_path] = {
                'cluster_count': len(data),
                'sample_count': sum(len(v) for v in data.values())
            }
    
    # 2. 检查文件数量
    if len(cluster_data) < 2:
        print("需要至少2个文件进行比较")
        return True, stats
    
    # 3. 比较所有文件内容
    base_data = cluster_data[0]
    all_match = True
    
    for i in range(1, len(cluster_data)):
        current_data = cluster_data[i]
        
        # 检查簇数量是否相同
        if len(base_data) != len(current_data):
            print(f"文件 {file_paths[0]} 和 {file_paths[i]} 的簇数量不同")
            all_match = False
            continue
            
        # 检查每个簇的内容是否相同
        for cluster_id in base_data:
            if cluster_id not in current_data:
                print(f"簇ID {cluster_id} 在文件 {file_paths[i]} 中不存在")
                all_match = False
                continue
                
            if set(base_data[cluster_id]) != set(current_data[cluster_id]):
                print(f"簇ID {cluster_id} 的内容在不同文件中不一致")
                all_match = False
    
    return all_match, stats

if __name__ == "__main__":
    # 示例使用 - 请替换为您的实际文件路径
    cluster_dir = "g:/GS/Tencent/Miner/data/FreiHand/mini_data/save_cluster_data/"
    files_to_compare = [
        "clusters_mini_4000_0.0055_min8_xi0.1.json",
        "clusters_mini_4000_0.0055_min8_xi0.02.json",
        "clusters_mini_4000_0.0055_min8_xi0.05.json",
        "clusters_mini_4000_0.0055_min8_xi0.15.json",
        "clusters_mini_4000_0.0055_min6_xi0.1.json",
        "clusters_mini_4000_0.0055_min6_xi0.02.json",
        "clusters_mini_4000_0.0055_min6_xi0.05.json",
        "clusters_mini_4000_0.0055_min6_xi0.15.json",
    ]
    
    full_paths = [os.path.join(cluster_dir, f) for f in files_to_compare]
    are_identical, stats = compare_cluster_files(full_paths)
    
    print("\n聚类结果比较:")
    print(f"所有文件内容完全一致: {'是' if are_identical else '否'}")
    
    print("\n各文件统计信息:")
    for file_path, info in stats.items():
        print(f"{os.path.basename(file_path)}:")
        print(f"  簇数量: {info['cluster_count']}")
        print(f"  样本总数: {info['sample_count']}")