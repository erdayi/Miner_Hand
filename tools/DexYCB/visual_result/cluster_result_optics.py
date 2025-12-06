import os
import cv2
import numpy as np
import json

'''
    本代码用于将OPTICS迷你数据集OPTICS聚类结果可视化,将每个簇的图像拼接成一个整体图片
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
    dataset_base_path = r"F:\DexYCB\dataset"

    # 读取左手图像路径的JSON文件
    with open("E:\\pose\\MVHandMiner\\data\\left_hand_img_paths.json", "r") as f:
        left_all_image_paths = json.load(f)
    
    # 读取右手图像路径的JSON文件
    with open("E:\\pose\\MVHandMiner\\data\\right_hand_img_paths.json", "r") as f:
        right_all_image_paths = json.load(f)
    
    # 读取新的聚类结果文件
    with open("E:\\pose\\MVHandMiner\\data\\minidata\\hand_clusters_optics_0.0055.json", "r") as f:
        cluster_data = json.load(f)
    
    # 提取左手和右手的聚类结果
    left_cluster_data = cluster_data.get("left_clusters", [])
    right_cluster_data = cluster_data.get("right_clusters", [])
    
    # 统计左手聚类的数量
    left_num_clusters = len(left_cluster_data)
    # 统计左手所有聚类中的图像总数
    left_total_images = sum(len(samples) for samples in left_cluster_data)
    print(f"Total number of left hand clusters: {left_num_clusters}")
    print(f"Total number of left hand images in clusters: {left_total_images}")

    # 统计右手聚类的数量
    right_num_clusters = len(right_cluster_data)
    # 统计右手所有聚类中的图像总数
    right_total_images = sum(len(samples) for samples in right_cluster_data)
    print(f"Total number of right hand clusters: {right_num_clusters}")
    print(f"Total number of right hand images in clusters: {right_total_images}")

    # 拼接后图像的输出根目录
    output_root = r"E:\pose\MVHandMiner\data\cluster_visualization_optics_minitest"
    # 确保输出目录存在，如果不存在则创建
    os.makedirs(output_root, exist_ok=True)

    # 创建左手和右手的子文件夹
    left_output_folder = os.path.join(output_root, "left")
    right_output_folder = os.path.join(output_root, "right")
    os.makedirs(left_output_folder, exist_ok=True)
    os.makedirs(right_output_folder, exist_ok=True)

    # 处理左手聚类
    for cluster_idx, sample_paths in enumerate(left_cluster_data):
        try:
            # 存储当前左手聚类中的所有图像
            cluster_images = []
            # 遍历当前左手聚类中的每个样本路径
            for relative_image_path in sample_paths:
                # 打印完整路径用于调试
                print(f"尝试读取的左手完整路径: {relative_image_path}")
                # 检查文件是否存在
                if not os.path.exists(relative_image_path):
                    print(f"左手文件不存在: {relative_image_path}")
                    continue
                # 读取左手图像
                img = cv2.imread(relative_image_path)
                if img is not None:
                    # 将左手图像添加到当前聚类的图像列表中
                    cluster_images.append(img)

            # 如果当前左手聚类中有图像
            if cluster_images:
                # 拼接后左手图像的保存路径
                output_path = os.path.join(left_output_folder, f"cluster_{cluster_idx}.jpg")
                # 调用拼接函数将当前左手聚类中的图像拼接成一个整体图像
                combine_images(cluster_images, output_path)
                print(f"Left hand Cluster {cluster_idx} visualization saved to {output_path}")

        except Exception as e:
            # 打印处理当前左手聚类时出现的错误信息
            print(f"Error processing left hand cluster {cluster_idx}: {str(e)}")

    # 处理右手聚类
    for cluster_idx, sample_paths in enumerate(right_cluster_data):
        try:
            # 存储当前右手聚类中的所有图像
            cluster_images = []
            # 遍历当前右手聚类中的每个样本路径
            for relative_image_path in sample_paths:
                # 打印完整路径用于调试
                print(f"尝试读取的右手完整路径: {relative_image_path}")
                # 检查文件是否存在
                if not os.path.exists(relative_image_path):
                    print(f"右手文件不存在: {relative_image_path}")
                    continue
                # 读取右手图像
                img = cv2.imread(relative_image_path)
                if img is not None:
                    # 将右手图像添加到当前聚类的图像列表中
                    cluster_images.append(img)

            # 如果当前右手聚类中有图像
            if cluster_images:
                # 拼接后右手图像的保存路径
                output_path = os.path.join(right_output_folder, f"cluster_{cluster_idx}.jpg")
                # 调用拼接函数将当前右手聚类中的图像拼接成一个整体图像
                combine_images(cluster_images, output_path)
                print(f"Right hand Cluster {cluster_idx} visualization saved to {output_path}")

        except Exception as e:
            # 打印处理当前右手聚类时出现的错误信息
            print(f"Error processing right hand cluster {cluster_idx}: {str(e)}")

    print("All cluster visualizations generated successfully.")