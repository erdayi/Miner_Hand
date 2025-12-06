import os
import pandas as pd
from collections import defaultdict


'''
    统计HO3D_v2数据集train目录下各子文件夹中RGB文件夹内的图片数量
    
    参数:
        base_dir (str): HO3D_v2数据集train目录的路径
        
    返回:
        dict: 包含每个子文件夹RGB图片数量的字典
'''
def count_rgb_images(base_dir):
    # 检查基础目录是否存在
    if not os.path.exists(base_dir):
        print(f"错误: 目录 {base_dir} 不存在!")
        return None
    
    # 存储结果的字典
    rgb_counts = defaultdict(int)
    total_count = 0
    
    # 遍历train目录下的所有子文件夹
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        
        # 确保是目录而不是文件
        if os.path.isdir(subdir_path):
            # 构建RGB文件夹路径
            rgb_dir = os.path.join(subdir_path, "rgb")
            
            # 检查RGB文件夹是否存在
            if os.path.exists(rgb_dir) and os.path.isdir(rgb_dir):
                # 计算RGB文件夹中的图片数量
                image_count = len([f for f in os.listdir(rgb_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                rgb_counts[subdir] = image_count
                total_count += image_count
            else:
                print(f"警告: 在 {subdir} 中未找到RGB文件夹")
    
    # 添加总计
    rgb_counts["总计"] = total_count
    
    return rgb_counts

def main():
    # 设置HO3D_v2数据集train目录的路径
    train_dir = r"F:\GS\Dataset\HO3D_v2\train"
    
    # 统计RGB图片数量
    rgb_counts = count_rgb_images(train_dir)
    
    if rgb_counts:
        # 打印结果
        print("\n===== HO3D_v2 训练集RGB图片统计 =====")
        for subdir, count in sorted(rgb_counts.items()):
            if subdir != "总计":
                print(f"{subdir}: {count}张图片")
        print(f"\n总计: {rgb_counts['总计']}张图片")
        
        # 将结果保存为CSV文件
        # df = pd.DataFrame(list(rgb_counts.items()), columns=['子文件夹', 'RGB图片数量'])
        # csv_path = os.path.join(os.path.dirname(train_dir), "ho3d_rgb_counts.csv")
        # df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        # print(f"\n结果已保存至: {csv_path}")

if __name__ == "__main__":
    main()