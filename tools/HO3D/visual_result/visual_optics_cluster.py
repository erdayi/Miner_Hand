import os
import cv2
import numpy as np
import json

'''
    本代码用于将HO3D迷你数据集聚类结果可视化,将每个簇的图像拼接成一个整体图片
'''
def combine_images(images, output_path='combined_image.jpg', thumbnail_size=(640, 480)):
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
    cv2.imwrite(output_path, canvas)

if __name__ == "__main__":
    # 数据集基础路径
    dataset_base_path = r"F:\GS\Dataset\HO3D_v2"
    
    # 读取聚类结果文件
    cluster_file = r"data/HO3D/cluster_data/hand_clusters_optics_interval_5.json"
    with open(cluster_file, "r") as f:
        cluster_data = json.load(f)
    
    # 提取聚类结果
    clusters = cluster_data.get("clusters", [])
    
    # 统计聚类的数量和图像总数
    num_clusters = len(clusters)
    total_images = sum(len(samples) for samples in clusters)
    print(f"总聚类数: {num_clusters}")
    print(f"聚类中的图像总数: {total_images}")

    # 拼接后图像的输出根目录
    output_root = r"data/HO3D/visual_data/optics/0003/minpts5_0.05/interval_5"
    # 确保输出目录存在，如果不存在则创建
    os.makedirs(output_root, exist_ok=True)

    # 处理每个聚类
    for cluster_idx, sample_paths in enumerate(clusters):
        try:
            # 存储当前聚类中的所有图像
            cluster_images = []
            # 遍历当前聚类中的每个样本路径
            for image_path in sample_paths:
                # 打印完整路径用于调试
                print(f"尝试读取的完整路径: {image_path}")
                # 检查文件是否存在
                if not os.path.exists(image_path):
                    print(f"文件不存在: {image_path}")
                    continue
                # 读取图像
                img = cv2.imread(image_path)
                if img is not None:
                    # 将图像添加到当前聚类的图像列表中
                    cluster_images.append(img)

            # 如果当前聚类中有图像
            if cluster_images:
                # 拼接后图像的保存路径
                output_path = os.path.join(output_root, f"cluster_{cluster_idx}.jpg")
                # 调用拼接函数将当前聚类中的图像拼接成一个整体图像
                combine_images(cluster_images, output_path)
                print(f"聚类 {cluster_idx} 可视化已保存到 {output_path}")

        except Exception as e:
            # 打印处理当前聚类时出现的错误信息
            print(f"处理聚类 {cluster_idx} 时出错: {str(e)}")

    print("所有聚类可视化生成完成。")