import os
import cv2
import numpy as np
import json
from pathlib import Path

'''
本方法用于将HDBSCAN算法聚类的结果以整体图片形式保存
'''

def load_image(image_path):
    """加载图像"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法加载图像: {image_path}")
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"加载图像时出错 {image_path}: {str(e)}")
        return None


def combine_images(images, output_path='combined_image.jpg', thumbnail_size=(100, 100)):
    """
    将多个图片拼接成一个整体图片。

    :param images: 包含多个图像的列表
    :param output_path: 拼接后图像的保存路径
    :param thumbnail_size: 每个缩略图的尺寸
    """
    # 获取图像的数量
    num_images = len(images)
    # 如果没有图像，直接返回
    if num_images == 0:
        return

    # 计算网格的大小，确保能够容纳所有图像
    grid_size = int(np.ceil(np.sqrt(num_images)))
    # 计算画布的高度
    canvas_height = grid_size * thumbnail_size[1]
    # 计算画布的宽度
    canvas_width = grid_size * thumbnail_size[0]
    # 创建一个白色背景的画布
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # 遍历所有图像
    for index, img in enumerate(images):
        if img is None:
            continue
        # 将图像调整为指定的缩略图尺寸
        img_resized = cv2.resize(img, thumbnail_size)
        # 计算当前图像在网格中的行和列
        row, col = divmod(index, grid_size)
        # 计算当前图像在画布上的起始纵坐标
        y_start = row * thumbnail_size[1]
        # 计算当前图像在画布上的起始横坐标
        x_start = col * thumbnail_size[0]
        # 将调整后的图像放置到画布上的指定位置
        canvas[y_start:y_start + thumbnail_size[1], x_start:x_start + thumbnail_size[0]] = img_resized

    # 将拼接后的图像保存到指定路径
    cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main():
    # 数据集路径
    base_dir = Path('I:/gs/raw/InterHand2.6M_5fps_batch1')
    image_dir = base_dir / 'images' / 'val' 
    
    # 聚类结果文件路径
    cluster_data_path = "data/InterHand/cluster_data/hdbscan/hand_clusters_hdbscan_right_min8_th0.0003_merge0.0003.json"
    
    # 拼接后图像的输出根目录
    output_root = "data/InterHand/visual_data/hdbscan/right"
    
    # 每个缩略图的尺寸
    thumbnail_size = (100, 100)
    
    # 读取聚类结果的 JSON 文件
    with open(cluster_data_path, "r") as f:
        cluster_data = json.load(f)
    
    # 获取聚类结果
    clusters = cluster_data['clusters']
    
    # 获取聚类参数
    parameters = cluster_data['parameters']
    
    # 统计聚类的数量
    num_clusters = len(clusters)
    # 统计所有聚类中的图像总数
    total_images = sum(len(cluster) for cluster in clusters)
    
    print("\n📊 聚类结果统计：")
    print("=" * 50)
    print(f"总聚类数: {num_clusters}")
    print(f"聚类中的总图像数: {total_images}")
    print("\n聚类参数：")
    print(f"  最小聚类大小: {parameters['min_cluster_size']}")
    print(f"  最小样本数: {parameters['min_samples']}")
    print(f"  阈值: {parameters['threshold']}")
    print(f"  合并阈值: {parameters['merge_threshold']}")
    print(f"  手部类型: {parameters['hand_type']}")
    
    # 确保输出目录存在
    os.makedirs(output_root, exist_ok=True)
    
    # 遍历每个聚类
    for cluster_idx, cluster_paths in enumerate(clusters):
        try:
            # 存储当前聚类中的所有图像
            cluster_images = []
            # 遍历当前聚类中的每个图像路径
            for image_path in cluster_paths:
                # 构建完整的图像路径
                full_image_path = image_dir / image_path
                # 加载图像
                img = load_image(str(full_image_path))
                if img is not None:
                    cluster_images.append(img)
            
            # 如果当前聚类中有图像
            if cluster_images:
                # 拼接后图像的保存路径
                output_path = os.path.join(output_root, f"cluster_{cluster_idx:04d}.jpg")
                # 调用拼接函数将当前聚类中的图像拼接成一个整体图像
                combine_images(cluster_images, output_path, thumbnail_size)
                print(f"聚类 {cluster_idx} 可视化结果已保存到 {output_path}")
        
        except Exception as e:
            # 打印处理当前聚类时出现的错误信息
            print(f"处理聚类 {cluster_idx} 时出错: {str(e)}")
    
    print("\n✅ 所有聚类可视化结果生成完成。")


if __name__ == "__main__":
    main() 