import os
import pickle
import numpy as np
from scipy.linalg import orthogonal_procrustes

def alignwscale(mtx1, mtx2):
    """
    使用正交对齐方法进行匹配，并返回对齐后的结果。
    :param mtx1: 第一个矩阵，代表手部关键点数据
    :param mtx2: 第二个矩阵，代表手部关键点数据
    :return: 对齐后的第二个矩阵
    """
    # 计算两个矩阵的均值
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    # 减去均值进行中心化
    mtx1t = mtx1 - t1
    mtx2t = mtx2 - t2

    # 计算矩阵的范数，并添加一个小的常数避免除零错误
    s1 = np.linalg.norm(mtx1t) + 1e-8
    # 归一化矩阵
    mtx1t /= s1
    s2 = np.linalg.norm(mtx2t) + 1e-8
    mtx2t /= s2

    # 使用正交普罗克鲁斯特斯分析计算旋转矩阵和缩放因子
    R, s = orthogonal_procrustes(mtx1t, mtx2t)
    # 应用旋转和缩放
    mtx2t = np.dot(mtx2t, R.T) * s
    # 恢复到原始尺度和位置
    mtx2t = mtx2t * s1 + t1

    return mtx2t

def calculatepaerror(mtx1, mtx2):
    """
    计算两个矩阵的对齐误差。
    :param mtx1: 第一个矩阵，代表手部关键点数据
    :param mtx2: 第二个矩阵，代表手部关键点数据
    :return: 每个关键点的对齐误差和整体对齐误差
    """
    mtx2_aligned = alignwscale(mtx1, mtx2)
    error = np.linalg.norm(mtx1 - mtx2_aligned, axis=1)
    overall_error = np.mean(error)

    return error, overall_error

# 指定要比较的 .pkl 文件
pkl_files = [
    r"F:\GS\Dataset\HO3D_v2\train\ABF10\meta\0000.pkl",
    r"F:\GS\Dataset\HO3D_v2\train\ABF11\meta\0000.pkl"
]

# 加载 .pkl 文件并提取 handJoints3D 数据
hand_joints_list = []
for pkl_file in pkl_files:
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
        hand_joints = data['handJoints3D']
        hand_joints_list.append(hand_joints)
        print(f"成功加载 {pkl_file} 的handJoints3D数据,形状: {hand_joints.shape}")

# 计算并打印 PA 对齐误差
for i in range(len(hand_joints_list)):
    for j in range(i + 1, len(hand_joints_list)):
        error, overall_error = calculatepaerror(hand_joints_list[i], hand_joints_list[j])
        print(f"\n图片{i} 对齐到 图片{j} 的每个关键点的对齐误差: {error}")
        print(f"图片{i} 对齐到 图片{j} 的PA对齐误差: {overall_error}")
