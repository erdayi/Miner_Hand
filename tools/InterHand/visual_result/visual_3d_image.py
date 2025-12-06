import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具
import numpy as np

def visualize_3d_points(world_coords_3d, image_path=None):
    """
    在3D空间中可视化坐标点。
    如果提供了image_path，则会尝试读取图片，但主要焦点是3D点云可视化。

    参数:
    world_coords_3d (list): 包含3D世界坐标点的列表，每个点是 [x, y, z]。
                               这里我们只使用前21个点。
    image_path (str, optional): 图像文件的路径。如果提供，可以考虑在旁边显示图像，但主要绘图是3D的。
    """
    points_to_plot = np.array(world_coords_3d[:21])
    x_coords = points_to_plot[:, 0]
    y_coords = points_to_plot[:, 1]
    z_coords = points_to_plot[:, 2]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制3D散点图
    ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o', s=50, edgecolors='black')

    # 为每个点添加编号
    for i in range(len(x_coords)):
        ax.text(x_coords[i], y_coords[i], z_coords[i], f'{i}', size=10, zorder=1, color='k')

    ax.set_xlabel('X Coordinate (world units)')
    ax.set_ylabel('Y Coordinate (world units)')
    ax.set_zlabel('Z Coordinate (world units)')
    ax.set_title('3D Keypoints Visualization')

    # 设置坐标轴的比例，使得各个轴的单位长度在视觉上大致相等
    # 这有助于更好地感知3D形状
    max_range = np.array([x_coords.max()-x_coords.min(), 
                        y_coords.max()-y_coords.min(), 
                        z_coords.max()-z_coords.min()]).max() / 2.0
    
    mid_x = (x_coords.max()+x_coords.min()) * 0.5
    mid_y = (y_coords.max()+y_coords.min()) * 0.5
    mid_z = (z_coords.max()+z_coords.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 如果需要，可以在旁边显示2D图像作为参考
    if image_path:
        try:
            image = cv2.imread(image_path)
            if image is not None:
                # 由于主图是3D的，这里可以考虑创建一个新的figure或子图来显示2D图像
                # 为了简单起见，这里只打印图片路径，实际应用中可以更复杂地布局
                print(f"Reference image path: {image_path}")
                # fig_img = plt.figure(figsize=(6,4))
                # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                # plt.title("Reference Image")
                # plt.axis('off')
            else:
                print(f"警告：无法读取参考图片 {image_path}")
        except Exception as e:
            print(f"读取参考图片时发生错误: {e}")

    plt.show()

if __name__ == '__main__':
    # 用户提供的图片路径 (可选，主要用于参考)
    img_path = r"I:\gs\raw\InterHand2.6M_5fps_batch1\images\val\Capture0\ROM07_Rt_Finger_Occlusions\cam400262\image23330.jpg"
    
    # 用户提供的3D世界坐标 (只取前21个点)
    world_coordinates = [
        [-7.815380096435547, -73.01349639892578, 1043.1099853515625],
        [-9.369819641113281, -53.81169891357422, 1067.2900390625],
        [-16.923500061035156, -37.21820068359375, 1097.8599853515625],
        [-12.18179988861084, -2.6924099922180176, 1121.010009765625],
        [6.4014201164245605, -7.8521199226379395, 957.8179931640625],
        [0.16330300271511078, -8.381429672241211, 981.4669799804688],
        [-9.41808032989502, -7.359340190887451, 1006.1199951171875],
        [-17.960500717163086, -5.554870128631592, 1050.9599609375],
        [16.99020004272461, 20.053800582885742, 946.4349975585938],
        [9.250479698181152, 17.909400939941406, 971.0869750976562],
        [-4.9473700523376465, 18.557600021362305, 1000.75],
        [-18.148399353027344, 19.169700622558594, 1049.489990234375],
        [21.126399993896484, 46.15629959106445, 960.0770263671875],
        [13.087800025939941, 43.39139938354492, 983.8920288085938],
        [-1.0669100284576416, 41.85049819946289, 1011.719970703125],
        [-13.011300086975098, 39.053001403808594, 1054.969970703125],
        [15.847599983215332, 85.51750183105469, 994.4450073242188],
        [9.90880012512207, 76.67410278320312, 1014.4099731445312],
        [2.1491100788116455, 68.96040344238281, 1033.6300048828125],
        [-7.561260223388672, 57.0177001953125, 1065.739990234375],
        [-33.93130111694336, 25.24169921875, 1143.239990234375]
    ]

    visualize_3d_points(world_coordinates, img_path)