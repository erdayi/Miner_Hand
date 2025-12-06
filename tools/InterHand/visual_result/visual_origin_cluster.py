import os
import cv2
import numpy as np
import json
from pathlib import Path

'''
    本方法用于将InterHand2.6M聚类的结果以整体图片形式保存
'''

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

def main():
    # 配置参数
    params = {
        # 聚类结果文件路径
        "cluster_data_path": "data/InterHand/cluster_data/origin/left/hand_clusters_0003_interval_1.json",
        # 拼接后图像的输出根目录
        "output_root": "data/InterHand/visual_data/origin/0003",
        # 每个缩略图的尺寸
        "thumbnail_size": (100, 100),
    }

    # 读取聚类结果的 JSON 文件
    with open(params["cluster_data_path"], "r") as f:
        cluster_data = json.load(f)

    # 统计聚类的数量
    num_clusters = len(cluster_data["clusters"])
    # 统计所有聚类中的图像总数
    total_images = sum(len(samples) for samples in cluster_data["clusters"])
    print(f"Total number of clusters: {num_clusters}")
    print(f"Total number of images in clusters: {total_images}")

    # 确保输出目录存在
    os.makedirs(params["output_root"], exist_ok=True)

    # 遍历每个聚类
    for cluster_idx, image_paths in enumerate(cluster_data["clusters"]):
        try:
            # 存储当前聚类中的所有图像
            cluster_images = []
            # 遍历当前聚类中的每个图像路径
            for img_path in image_paths:
                # 统一路径格式
                img_path = img_path.replace('\\', '/')
                # 读取图像
                img = cv2.imread(img_path)
                if img is not None:
                    # 将BGR转换为RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    cluster_images.append(img)
                else:
                    print(f"Warning: Could not read image {img_path}")

            # 如果当前聚类中有图像
            if cluster_images:
                # 拼接后图像的保存路径
                output_path = os.path.join(params["output_root"], f"cluster_{cluster_idx}.jpg")
                # 调用拼接函数将当前聚类中的图像拼接成一个整体图像
                combine_images(cluster_images, output_path, params["thumbnail_size"])
                print(f"Cluster {cluster_idx} visualization saved to {output_path}")

        except Exception as e:
            # 打印处理当前聚类时出现的错误信息
            print(f"Error processing cluster {cluster_idx}: {str(e)}")

    print("All cluster visualizations generated successfully.")

if __name__ == "__main__":
    main()
