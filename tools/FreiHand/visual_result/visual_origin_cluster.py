# import os
# import cv2
# import numpy as np
# import json
#
# from tools.FreiHand.origin_deal.get_data import GetData
#
# '''
#     本方法用于将基础方法聚类的结果以整体图片形式保存
# '''
#
#
# def save_image(image, output_path):
#     """ 保存单张图片 """
#     cv2.imwrite(output_path, image)
#
#
# def combine_images(images, output_path='combined_image.jpg', thumbnail_size=(100, 100)):
#     """ 将多个图片拼接成一个整体图片 """
#     resized_images = [cv2.resize(img, thumbnail_size) for img in images]
#
#     num_images = len(images)
#     grid_size = int(np.ceil(np.sqrt(num_images)))
#     canvas_height = grid_size * thumbnail_size[1]
#     canvas_width = grid_size * thumbnail_size[0]
#
#     canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
#
#     for index, img in enumerate(resized_images):
#         row = index // grid_size
#         col = index % grid_size
#         y_start = row * thumbnail_size[1]
#         y_end = y_start + thumbnail_size[1]
#         x_start = col * thumbnail_size[0]
#         x_end = x_start + thumbnail_size[0]
#         canvas[y_start:y_end, x_start:x_end] = img
#
#     cv2.imwrite(output_path, canvas)
#
#
# def grouping(err_dict, thred=0.005):
#     """ 根据误差筛选相似样本 """
#     grouped_res = []
#     for res in err_dict:
#         if err_dict[res] <= thred:
#             grouped_res.append(int(res))
#     return grouped_res
#
#
# def main():
#     # 配置所有输入参数
#     params = {
#         # 数据集信息
#         "dataset": {
#             "name": "FreiHand",
#             "json_dir": '../../../data/FreiHand/origin_data/train.json',
#             "eval_dir": 'G:/WZY/PalmData/data/FreiHAND/training/rgb',
#             "scale_enlarge": 1.25
#         },
#         # 输入文件路径
#         "aligned_err_path": "../../../data/FreiHand/cluster_data/clusters_origin_32560_0.0055.json",
#         # 输出目录
#         "output_root": "../../../data/FreiHand/visual_data/origin/0055",
#         # 图像处理参数
#         "image_size": (224, 224),
#         "thumbnail_size": (100, 100),
#         # 误差阈值
#         "threshold": 0.005,
#     }
#
#     # 读取数据集
#     dataset = GetData(
#         params["dataset"]["json_dir"],
#         params["image_size"],
#         params["dataset"]["scale_enlarge"]
#     )
#
#     # 读取 JSON 对齐误差文件
#     with open(params["aligned_err_path"], "r") as f:
#         aligned_err = json.load(f)
#
#     # 创建输出目录
#     os.makedirs(params["output_root"], exist_ok=True)
#
#     for anchor_idx, similar_samples in aligned_err.items():
#         try:
#             anchor_idx = int(anchor_idx)  # 确保索引是整数
#
#             # 读取锚点图片
#             anchor_data = dataset.getitem(anchor_idx)
#             anchor_img = np.transpose(anchor_data['img'], (1, 2, 0))
#
#             # 收集锚点和相似样本图片
#             grouped_images = [anchor_img]  # 将锚点图片加入列表
#             for sample_idx in similar_samples.keys():
#                 sample_idx = int(sample_idx)  # 确保索引是整数
#                 sample_data = dataset.getitem(sample_idx)
#                 sample_img = np.transpose(sample_data['img'], (1, 2, 0))
#                 grouped_images.append(sample_img)
#
#             # 将锚点和相似样本拼接成一张图片
#             combined_image_path = os.path.join(params["output_root"], f"anchor_{anchor_idx}_combined.jpg")
#             combine_images(
#                 grouped_images,
#                 combined_image_path,
#                 params["thumbnail_size"]
#             )
#
#             print(f"Processed anchor {anchor_idx}, saved combined image with {len(similar_samples)} samples.")
#
#         except Exception as e:
#             print(f"Error processing anchor {anchor_idx}: {str(e)}")
#
#     print("All combined images saved successfully.")
#
#
# if __name__ == "__main__":
#     main()

# _________________上方代码是借助锚点样本的json进行可视化的展现，下方可视化借助类似OPTICS算法保存的结果进行展示_________________________#

import os
import cv2
import numpy as np
import json

from tools.FreiHand.origin_deal.get_data import GetData

'''
    本方法用于将初始方法聚类的结果（clean 按optics算法保存）以整体图片形式保存
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
    # 数据集信息
    dataset_params = {
        "name": "FreiHand",
        "json_dir": '../../../data/FreiHand/origin_data/train.json',
        "eval_dir": 'G:/WZY/PalmData/data/FreiHAND/training/rgb',
        "scale_enlarge": 1.25
    }

    # 聚类结果文件路径
    cluster_data_path = "../../../data/FreiHand/cluster_data/new_clusters_origin_32560_0.0055_clean.json"

    # 拼接后图像的输出根目录
    output_root = "../../../data/FreiHand/visual_data/origin/0055"

    # 每个缩略图的尺寸
    thumbnail_size = (100, 100)

    # 初始化数据加载器，传入 JSON 文件路径、图像尺寸和缩放比例
    dataset = GetData(dataset_params["json_dir"], (224, 224), dataset_params["scale_enlarge"])

    # 读取聚类结果的 JSON 文件
    with open(cluster_data_path, "r") as f:
        cluster_data = json.load(f)

    # 统计聚类的数量
    num_clusters = len(cluster_data)
    # 统计所有聚类中的图像总数
    total_images = sum(len(samples) for samples in cluster_data.values())
    print(f"Total number of clusters: {num_clusters}")
    print(f"Total number of images in clusters: {total_images}")

    # 确保输出目录存在，如果不存在则创建
    os.makedirs(output_root, exist_ok=True)

    # 遍历每个聚类
    for cluster_label, sample_indices in cluster_data.items():
        try:
            # 存储当前聚类中的所有图像
            cluster_images = []
            # 遍历当前聚类中的每个样本索引
            for sample_idx in sample_indices:
                # 将样本索引转换为整数类型
                sample_idx = int(sample_idx)
                # 根据样本索引获取样本数据
                sample_data = dataset.getitem(sample_idx)
                # 将样本图像的维度进行转换
                sample_img = np.transpose(sample_data['img'], (1, 2, 0))
                # 将转换后的图像添加到当前聚类的图像列表中
                cluster_images.append(sample_img)

            # 如果当前聚类中有图像
            if cluster_images:
                # 拼接后图像的保存路径
                output_path = os.path.join(output_root, f"cluster_{cluster_label}.jpg")
                # 调用拼接函数将当前聚类中的图像拼接成一个整体图像
                combine_images(cluster_images, output_path, thumbnail_size)
                print(f"Cluster {cluster_label} visualization saved to {output_path}")

        except Exception as e:
            # 打印处理当前聚类时出现的错误信息
            print(f"Error processing cluster {cluster_label}: {str(e)}")

    print("All cluster visualizations generated successfully.")


if __name__ == "__main__":
    main()
