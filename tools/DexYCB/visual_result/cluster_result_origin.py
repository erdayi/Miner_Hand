import os
import cv2
import numpy as np
import json
import logging

'''
    本代码用于将OPTICS迷你数据集初始算法聚类结果可视化,将每个簇的图像拼接成一个整体图片
'''

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def combine_images(images, output_path='combined_image.jpg', thumbnail_size=(640, 480)):
    """
    将多个图片拼接成一个整体图片。

    :param images: 包含多个图像的列表
    :param output_path: 拼接后图像的保存路径
    :param thumbnail_size: 每个缩略图的尺寸
    """
    try:
        num_images = len(images)
        if num_images == 0:
            logging.warning(f"No images provided for {output_path}. Skipping.")
            return

        grid_size = int(np.ceil(np.sqrt(num_images)))
        canvas_height = grid_size * thumbnail_size[1]
        canvas_width = grid_size * thumbnail_size[0]
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

        for index, img in enumerate(images):
            img_resized = cv2.resize(img, thumbnail_size)
            row, col = divmod(index, grid_size)
            y_start = row * thumbnail_size[1]
            x_start = col * thumbnail_size[0]
            canvas[y_start:y_start + thumbnail_size[1], x_start:x_start + thumbnail_size[0]] = img_resized

        cv2.imwrite(output_path, canvas)
        logging.info(f"Successfully saved combined image to {output_path}")
    except Exception as e:
        logging.error(f"Error combining images for {output_path}: {e}")


def get_full_image_path(base_path, relative_image_path):
    # 去除相对路径开头可能的多余字符
    relative_image_path = relative_image_path.lstrip('/').lstrip('\\')
    return os.path.join(base_path, relative_image_path)


def visualize_clusters(cluster_json_path, image_paths_json_path, output_folder, label_prefix):
    logging.info(f"Starting visualization for {label_prefix} clusters...")
    try:
        with open(cluster_json_path, 'r', encoding='utf-8') as f:
            cluster_data = json.load(f)
        logging.info(f"Successfully loaded cluster data from {cluster_json_path}")
    except FileNotFoundError:
        logging.error(f"Error: Cluster data file {cluster_json_path} not found.")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {cluster_json_path}: {e}")
        return

    try:
        with open(image_paths_json_path, 'r', encoding='utf-8') as f:
            all_image_paths = json.load(f)
        logging.info(f"Successfully loaded image paths from {image_paths_json_path}")
    except FileNotFoundError:
        logging.error(f"Error: Image paths file {image_paths_json_path} not found.")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {image_paths_json_path}: {e}")
        return

    base_path = r"G:\DexYCB\dataset"
    os.makedirs(output_folder, exist_ok=True)

    for cluster_label, sample_indices in cluster_data.items():
        logging.info(f"Processing {label_prefix} cluster {cluster_label}...")
        cluster_images = []

        anchor_idx = int(cluster_label)
        anchor_relative_image_path = all_image_paths[anchor_idx]
        anchor_image_path = get_full_image_path(base_path, anchor_relative_image_path)
        try:
            if os.path.exists(anchor_image_path):
                anchor_img = cv2.imread(anchor_image_path)
                if anchor_img is not None:
                    cluster_images.append(anchor_img)
                else:
                    logging.warning(f"Could not read anchor image {anchor_image_path} for {label_prefix} cluster {cluster_label}")
            else:
                logging.warning(f"Anchor image {anchor_image_path} for {label_prefix} cluster {cluster_label} does not exist.")
        except Exception as e:
            logging.error(f"Error reading anchor image {anchor_image_path} for {label_prefix} cluster {cluster_label}: {e}")

        for sample_idx_str in sample_indices.keys():
            sample_idx = int(sample_idx_str)
            relative_image_path = all_image_paths[sample_idx]
            full_image_path = get_full_image_path(base_path, relative_image_path)
            try:
                if os.path.exists(full_image_path):
                    img = cv2.imread(full_image_path)
                    if img is not None:
                        cluster_images.append(img)
                    else:
                        logging.warning(f"Could not read sample image {full_image_path} for {label_prefix} cluster {cluster_label}")
                else:
                    logging.warning(f"Sample image {full_image_path} for {label_prefix} cluster {cluster_label} does not exist.")
            except Exception as e:
                logging.error(f"Error reading sample image {full_image_path} for {label_prefix} cluster {cluster_label}: {e}")

        if cluster_images:
            output_path = os.path.join(output_folder, f"{label_prefix}_cluster_{cluster_label}.jpg")
            combine_images(cluster_images, output_path)
        else:
            logging.warning(f"No valid images found for {label_prefix} cluster {cluster_label}. Skipping.")

    logging.info(f"Finished visualization for {label_prefix} clusters.")


if __name__ == "__main__":
    left_cluster_json_path = 'data/minidata/left_clusters_origin_0.0055.json'
    right_cluster_json_path = 'data/minidata/right_clusters_origin_0.0055.json'
    left_image_paths_json_path = 'data/minidata/left_hand_img_paths.json'
    right_image_paths_json_path = 'data/minidata/right_hand_img_paths.json'
    output_root = 'data/cluster_visualization_origin_minitest'
    left_output_folder = os.path.join(output_root, 'left')
    right_output_folder = os.path.join(output_root, 'right')

    visualize_clusters(left_cluster_json_path, left_image_paths_json_path, left_output_folder, 'left')
    visualize_clusters(right_cluster_json_path, right_image_paths_json_path, right_output_folder, 'right')

    logging.info("All cluster visualizations process completed.")
    