import os
import json

def list_cluster_files(base_dir: str, output_file: str):
    """
    列出param_test目录下的所有聚类文件
    
    Args:
        base_dir: param_test目录的路径
        output_file: 输出文件路径
    """
    try:
        # 获取所有文件
        files = [f for f in os.listdir(base_dir) if f.startswith('clusters_') and f.endswith('.json')]
        
        # 提取参数信息
        param_files = []
        for file in files:
            # 去掉'clusters_'前缀和'.json'后缀
            param_str = file[9:-5]
            param_files.append(param_str)
        
        # 按文件名排序
        param_files.sort()
        
        # 保存到文件
        with open(output_file, 'w') as f:
            json.dump(param_files, f, indent=4)
        
        print(f"找到 {len(param_files)} 个聚类文件")
        print(f"文件名已保存到: {output_file}")
        
        # 打印所有文件名
        print("\n所有聚类文件:")
        for i, param in enumerate(param_files, 1):
            print(f"{i}. {param}")
            
    except Exception as e:
        print(f"处理文件时出错：{str(e)}")
        import traceback
        print(traceback.format_exc())

def main():
    # 设置目录和输出文件路径
    base_dir = "data/FreiHand/cluster_data/hdbscan/param_test"
    output_file = "data/FreiHand/cluster_data/hdbscan/param_test/progress.json"
    
    # 执行文件列表
    list_cluster_files(base_dir, output_file)

if __name__ == "__main__":
    main()