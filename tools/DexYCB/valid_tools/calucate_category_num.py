import json

'''
    本代码用于统计左右手聚类结果数量

    输入:
        hand_clusters_path: 聚类结果JSON路径（包含left_clusters/right_clusters）

    输出:
        左手簇数量, 右手簇数量
'''

def count_clusters(hand_clusters_path):
    print(f"正在加载聚类结果文件: {hand_clusters_path}")
    with open(hand_clusters_path, 'r', encoding='utf-8') as f:
        cluster_data = json.load(f)
    
    left_clusters = cluster_data.get("left_clusters", [])
    right_clusters = cluster_data.get("right_clusters", [])
    
    left_cluster_count = len(left_clusters)
    right_cluster_count = len(right_clusters)
    
    print(f"成功加载聚类数据，左手簇数量: {left_cluster_count}，右手簇数量: {right_cluster_count}")
    return left_cluster_count, right_cluster_count

if __name__ == "__main__":
    # 配置文件路径
    CLUSTER_JSON = "data/minidata/new/hand_clusters_origin_0.0055.json"
    
    # 计算左右手簇数量
    left_cluster_count, right_cluster_count = count_clusters(
        hand_clusters_path=CLUSTER_JSON
    )
    
    # 打印结果
    print("\n================= 最终统计结果 =================")
    print(f"左手簇数量: {left_cluster_count}")
    print(f"右手簇数量: {right_cluster_count}")