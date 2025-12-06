import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image
import os
import argparse
from pathlib import Path

# 定义基础路径
BASE_DIR = Path('I:/gs/raw/InterHand2.6M_5fps_batch1')
IMAGE_DIR = BASE_DIR / 'images' / 'val' / 'Capture0'
CAMERA_FILE = BASE_DIR / 'annotations' / 'val' / 'InterHand2.6M_val_camera.json'

# 定义手指颜色
FINGER_COLORS = {
    'thumb': '#FF0000',    # 红色
    'index': '#00FF00',    # 绿色
    'middle': '#0000FF',   # 蓝色
    'ring': '#FF00FF',     # 紫色
    'pinky': '#00FFFF'     # 青色
}

# 定义手指连接
FINGER_CONNECTIONS = {
    'thumb': [[0,1], [1,2], [2,3], [3,4]],
    'index': [[0,5], [5,6], [6,7], [7,8]],
    'middle': [[0,9], [9,10], [10,11], [11,12]],
    'ring': [[0,13], [13,14], [14,15], [15,16]],
    'pinky': [[0,17], [17,18], [18,19], [19,20]]
}

def load_data(dataset_type):
    """加载图片路径和3D关节点数据"""
    if dataset_type == 'interhand':
        with open('data/InterHand/origin_data/val_image_paths.json', 'r') as f:
            image_paths = json.load(f)
        with open('data/InterHand/origin_data/val_joint_3d.json', 'r') as f:
            joint_3d = json.load(f)
        # 加载相机参数
        with open(CAMERA_FILE, 'r') as f:
            camera_params = json.load(f)
        return image_paths, joint_3d, camera_params
    else:  # freihand
        with open('data/FreiHand/origin_data/eval.json', 'r') as f:
            image_paths = json.load(f)
        with open('data/FreiHand/origin_data/eval_xyz_list.json', 'r') as f:
            joint_3d = json.load(f)
        return image_paths, joint_3d, None

def plot_3d_joints(joints, title, camera_params=None, camera_name=None):
    """绘制3D关节点"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制关节点
    x = [joint[0] for joint in joints]
    y = [joint[1] for joint in joints]
    z = [joint[2] for joint in joints]
    
    # 绘制关节点（使用更小的点）
    ax.scatter(x, y, z, c='black', marker='o', s=20)
    
    # 为每个手指绘制不同颜色的连线
    for finger, connections in FINGER_CONNECTIONS.items():
        color = FINGER_COLORS[finger]
        for connection in connections:
            ax.plot([x[connection[0]], x[connection[1]]],
                   [y[connection[0]], y[connection[1]]],
                   [z[connection[0]], z[connection[1]]], 
                   color=color, linewidth=2)
    
    # 如果有相机参数，绘制相机位置
    if camera_params and camera_name:
        capture_id = camera_name.split('_')[0]  # 假设相机名称格式为 "capture_id_camera_name"
        if capture_id in camera_params:
            cam_pos = camera_params[capture_id]['campos'][camera_name]
            ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], c='red', marker='^', s=100, label='Camera')
            ax.legend()
    
    # 设置坐标轴标签
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(title)
    
    # 设置视角
    ax.view_init(elev=20, azim=45)
    
    # 设置坐标轴范围
    max_range = np.array([max(x)-min(x), max(y)-min(y), max(z)-min(z)]).max() / 2.0
    mid_x = (max(x) + min(x)) * 0.5
    mid_y = (max(y) + min(y)) * 0.5
    mid_z = (max(z) + min(z)) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    return fig

def get_image_path(rel_path):
    """构建并验证图片路径"""
    # 确保路径使用正斜杠
    rel_path = rel_path.replace('\\', '/')
    
    # 如果路径已经包含完整的基础路径，直接返回
    if rel_path.startswith(str(IMAGE_DIR)):
        return rel_path
    
    # 构建完整路径
    abs_path = os.path.join(IMAGE_DIR, rel_path)
    abs_path = abs_path.replace('\\', '/')
    
    # 检查文件是否存在
    if not os.path.exists(abs_path):
        print(f"警告：图片文件不存在: {abs_path}")
        return None
    
    return abs_path

def visualize_sample(dataset_type, hand_type=None, sample_idx=0):
    """可视化单个样本的图片和3D关节点"""
    # 加载数据
    image_paths, joint_3d, camera_params = load_data(dataset_type)
    
    # 获取样本数据
    if dataset_type == 'interhand':
        rel_image_path = image_paths[hand_type][sample_idx]
        abs_image_path = get_image_path(rel_image_path)
        joints = joint_3d[hand_type][sample_idx]
        
        # 从图片路径中提取相机信息
        camera_name = None
        if abs_image_path:
            path_parts = os.path.basename(abs_image_path).split('_')
            if len(path_parts) >= 2:
                camera_name = f"{path_parts[0]}_{path_parts[1]}"
    else:  # freihand
        abs_image_path = image_paths[sample_idx]
        joints = joint_3d[sample_idx]
        camera_name = None
    
    # 创建图形
    plt.figure(figsize=(15, 5))
    
    # 显示图片
    plt.subplot(121)
    if abs_image_path and os.path.exists(abs_image_path):
        try:
            img = Image.open(abs_image_path)
            plt.imshow(img)
            if dataset_type == 'interhand':
                plt.title(f'Image: {os.path.basename(rel_image_path)}\n({hand_type} hand)')
            else:
                plt.title(f'Image: {os.path.basename(abs_image_path)}')
        except Exception as e:
            plt.text(0.5, 0.5, f'Error loading image:\n{str(e)}', 
                    horizontalalignment='center', verticalalignment='center')
            plt.title('Error: Failed to load image')
    else:
        plt.text(0.5, 0.5, f'Image not found:\n{rel_image_path}', 
                horizontalalignment='center', verticalalignment='center')
        plt.title('Error: Image not found')
    plt.axis('off')
    
    # 显示3D关节点
    ax = plt.subplot(122, projection='3d')
    x = [joint[0] for joint in joints]
    y = [joint[1] for joint in joints]
    z = [joint[2] for joint in joints]
    
    # 绘制关节点（使用更小的点）
    ax.scatter(x, y, z, c='black', marker='o', s=20)
    
    # 为每个手指绘制不同颜色的连线
    for finger, connections in FINGER_CONNECTIONS.items():
        color = FINGER_COLORS[finger]
        for connection in connections:
            ax.plot([x[connection[0]], x[connection[1]]],
                   [y[connection[0]], y[connection[1]]],
                   [z[connection[0]], z[connection[1]]], 
                   color=color, linewidth=2)
    
    # 如果有相机参数，绘制相机位置
    if camera_params and camera_name:
        capture_id = camera_name.split('_')[0]
        if capture_id in camera_params:
            cam_pos = camera_params[capture_id]['campos'][camera_name]
            ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], c='red', marker='^', s=100, label='Camera')
            ax.legend()
    
    # 设置标题和视角
    if dataset_type == 'interhand':
        plt.title(f'3D Joints ({hand_type} hand)\nThumb(Red) Index(Green) Middle(Blue) Ring(Purple) Pinky(Cyan)')
    else:
        plt.title(f'3D Joints\nThumb(Red) Index(Green) Middle(Blue) Ring(Purple) Pinky(Cyan)')
    
    # 设置坐标轴范围
    max_range = np.array([max(x)-min(x), max(y)-min(y), max(z)-min(z)]).max() / 2.0
    mid_x = (max(x) + min(x)) * 0.5
    mid_y = (max(y) + min(y)) * 0.5
    mid_z = (max(z) + min(z)) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 设置视角
    ax.view_init(elev=20, azim=45)
    
    # 调整布局
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='可视化手部3D关节点')
    parser.add_argument('--dataset', type=str, choices=['interhand', 'freihand'], 
                      default='interhand', help='选择数据集类型')
    args = parser.parse_args()
    
    if args.dataset == 'interhand':
        # 可视化InterHand数据
        for hand_type in ['left', 'right']:
            for i in range(3):  # 每个手类型显示3个样本
                print(f"\n显示{hand_type}手的第{i+1}个样本")
                visualize_sample('interhand', hand_type, i)
                input("按回车键继续...")
    else:
        # 可视化FreiHand数据
        for i in range(5):  # 显示5个样本
            print(f"\n显示第{i+1}个样本")
            visualize_sample('freihand', sample_idx=i)
            input("按回车键继续...")

if __name__ == "__main__":
    main() 