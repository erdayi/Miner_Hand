import os
import numpy as np
import json
from functools import lru_cache
import cv2
import pickle
from tqdm import tqdm
from typing import List, Dict

from .kp_preprocess import get_2d3d_perspective_transform

'''
    该 GetData 类是实现手部姿态估计数据预处理的核心模块 (调用kp_preprocess.py的模块)
    
    1.数据加载
        从 json_path 加载所有图像路径列表
        自动关联每张图像的标注文件（.jpg → .json）
        读取标注信息：2D关键点 (uv)、3D关键点 (xyz)、顶点坐标 (vertices)、相机参数 (K)
    
    2.关键点有效性检测
        归一化关键点坐标到 [0,1] 范围
        过滤无效关键点（超出图像边界的点）
        若有效点过少则使用全部关键点
    
    3.几何变换
        根据关键点/边界框计算 中心点 和 缩放比例
        调用 get_2d3d_perspective_transform 生成：
        新相机参数 new_K
        2D图像透视变换矩阵 trans_matrix_2d
        3D坐标变换矩阵 trans_matrix_3d
    
    4.图像预处理
        应用透视变换裁剪并缩放图像至 img_size
        处理灰度图 → BGR 转换
        图像通道格式转换：HWC → CHW (适配PyTorch)
    
    5.坐标系统一变换
        对2D关键点进行齐次坐标变换
        对3D关键点和顶点应用旋转变换
        保持图像与3D空间的几何一致性

'''


class GetData():
    def __init__(self, json_path, img_size=(224, 224), scale_enlarge=1.2, rot_angle=0):
        super().__init__()

        with open(json_path) as f:
            self.all_image_info = json.load(f)
        self.all_info = [{"image_path": image_path} for image_path in self.all_image_info]
        self.img_size = img_size
        self.scale_enlarge = scale_enlarge
        self.rot_angle = rot_angle

    def __len__(self):
        return len(self.all_image_info)

    def read_image(self, img_path):
        img = cv2.imread(img_path)
        return img

    def read_info(self, img_path):
        info_path = img_path.replace('.jpg', '.json')
        with open(info_path) as f:
            info = json.load(f)
        return info

    def getitem(self, index):
        image_path = self.all_image_info[index]
        img = self.read_image(image_path)
        data_dict = self.read_info(image_path)
        h, w = img.shape[:2]
        K = np.array(data_dict['K'])
        if "uv" in data_dict:
            uv = np.array(data_dict['uv'])  # 2D关键点 (N, 2)
            xyz = np.array(data_dict['xyz'])  # 3D关键点 (N, 3)
            vertices = np.array(data_dict['vertices'])  # 顶点坐标 (N, 3)
            uv_norm = uv.copy()
            uv_norm[:, 0] /= w  # 归一化x坐标到 [0, 1]
            uv_norm[:, 1] /= h  # 归一化y坐标到 [0, 1]

            # 计算坐标有效性（是否在图像范围内）
            coord_valid = (uv_norm > 0).astype("float32") * (uv_norm < 1).astype("float32")
            coord_valid = coord_valid[:, 0] * coord_valid[:, 1]  # 关键点(x,y)均有效时为1

            valid_points = [uv[i] for i in range(len(uv)) if coord_valid[i] == 1]
            if len(valid_points) <= 1:  # 若有效点太少，使用全部点
                valid_points = uv
            points = np.array(valid_points)
            min_coord = points.min(axis=0)
            max_coord = points.max(axis=0)
            center = (max_coord + min_coord) / 2  # 关键点中心
            scale = max_coord - min_coord  # 关键点范围
        else:
            # 若没有uv数据，使用边界框计算中心和尺度
            bbox = data_dict['bbox']
            x1, y1, x2, y2 = bbox[:4]
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            scale = np.array([x2 - x1, y2 - y1])
            uv = np.zeros((21, 2), dtype=np.float32)
            xyz = np.zeros((21, 3), dtype=np.float32)

        ori_xyz = xyz.copy()
        ori_vertices = vertices.copy()
        scale = scale * self.scale_enlarge  # 应用缩放因子
        # perspective trans 获取透视变换矩阵
        new_K, trans_matrix_2d, trans_matrix_3d = get_2d3d_perspective_transform(K, center, scale, self.rot_angle,
                                                                                 self.img_size[0])
        # 图像预处理
        img_processed = cv2.warpPerspective(img, trans_matrix_2d, self.img_size)
        if img_processed.ndim == 2:  # 处理灰度图
            img_processed = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2BGR)
        img_processed = np.transpose(img_processed, (2, 0, 1))  # HWC -> CHW
        # 2D关键点变换
        new_uv = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=1)  # 齐次坐标
        new_uv = (trans_matrix_2d @ new_uv.T).T  # 应用变换
        new_uv = new_uv[:, :2] / new_uv[:, 2:]  # 去齐次化
        new_xyz = (trans_matrix_3d @ xyz.T).T  # 3D关键点变换

        vertices = trans_matrix_3d.dot(vertices.T).T  # 顶点坐标变换

        return {
            "img": np.ascontiguousarray(img_processed),  # 处理后的图像（CHW）
            "trans_matrix_2d": trans_matrix_2d,  # 2D变换矩阵
            "trans_matrix_3d": trans_matrix_3d,  # 3D变换矩阵
            "K": new_K,  # 变换后的内参矩阵
            "uv": new_uv,  # 变换后的2D关键点
            "xyz": new_xyz,  # 变换后的3D关键点
            "vertices": vertices,  # 变换后的顶点坐标
            "scale": self.img_size[0],  # 目标图像尺寸（边长）
            "ori_xyz": ori_xyz,  # 原始3D关键点（未变换）
            "ori_vertices": ori_vertices,  # 原始顶点坐标（未变换）
        }


class GetOriginData():
    def __init__(self, dataset_path, set_name, scale_enlarge=1.2, rot_angle=0):
        super().__init__()
        self.base_path = dataset_path
        self.set_name = set_name
        if self.set_name == 'evaluation':
            dataset_name = 'evaluation'
        else:
            dataset_name = 'training'
        self.scale_enlarge = scale_enlarge
        self.rot_angle = rot_angle
        self.img_size = (224, 224)
        self.K_list = self.json_load(os.path.join(self.base_path, '%s_K.json' % dataset_name))
        self.scale_list = self.json_load(os.path.join(self.base_path, '%s_scale.json' % dataset_name))

        if self.set_name == 'training' or self.set_name == 'trainval_train' or self.set_name == 'trainval_val':  # only 32560
            self.mano_list = self.json_load(
                os.path.join(self.base_path, '%s_mano.json' % dataset_name))  # [32560 1 61] float
            self.uv_list = self.json_load(os.path.join(self.base_path, '%s_uv.json' % dataset_name))  # [32560 21 2]
            self.joint_list = self.json_load(os.path.join(self.base_path, '%s_xyz.json' % dataset_name))  # [32560 21 3]
            self.verts_list = []
            # self.verts_list = self.json_load(os.path.join(self.base_path, '%s_verts.json' % self.set_name)) # 谨慎运行此行！此文件大小1.5G！

            # 数据集大小选择
            # ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐
            # FOR 32560
            # mask_idxs = [int(imgname.split(".")[0]) for imgname in sorted(os.listdir(os.path.join(self.base_path, dataset_name, 'mask')))]
            # self.prefix_template = "{:08d}"
            # prefixes = [self.prefix_template.format(idx) for idx in mask_idxs]
            # if self.set_name == 'trainval_train':
            #     prefixes = prefixes[:30000]
            # elif self.set_name == 'trainval_val':
            #     prefixes = prefixes[30000:]
            # del mask_idxs

            # FOR 32560*4
            img_idxs = [int(imgname.split(".")[0]) for imgname in
                        sorted(os.listdir(os.path.join(self.base_path, dataset_name, 'rgb')))]  # len = 130240
            self.K_list = self.K_list + self.K_list + self.K_list + self.K_list
            self.K_list = np.array(self.K_list)
            self.scale_list = self.scale_list + self.scale_list + self.scale_list + self.scale_list
            self.scale_list = np.array(self.scale_list)
            self.mano_list = self.mano_list + self.mano_list + self.mano_list + self.mano_list
            self.mano_list = np.array(self.mano_list)
            self.uv_list = self.uv_list + self.uv_list + self.uv_list + self.uv_list
            self.uv_list = np.array(self.uv_list)
            self.joint_list = self.joint_list + self.joint_list + self.joint_list + self.joint_list
            self.joint_list = np.array(self.joint_list)
            self.verts_list = self.verts_list + self.verts_list + self.verts_list + self.verts_list
            self.verts_list = np.array(self.verts_list)
            self.prefix_template = "{:08d}"
            prefixes = [self.prefix_template.format(idx) for idx in img_idxs]
            del img_idxs
            # ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐

        elif self.set_name == 'evaluation':
            img_idxs = [int(imgname.split(".")[0]) for imgname in
                        sorted(os.listdir(os.path.join(self.base_path, self.set_name, 'rgb')))]
            self.prefix_template = "{:08d}"
            prefixes = [self.prefix_template.format(idx) for idx in img_idxs]
            self.verts_list = self.json_load(os.path.join(self.base_path, '%s_verts.json' % self.set_name))
            self.verts_list = np.array(self.verts_list)
            self.joint_list = self.json_load(os.path.join(self.base_path, '%s_xyz.json' % dataset_name))
            self.joint_list = np.array(self.joint_list)
            self.uv_list = self.json_load(os.path.join(self.base_path, '%s_uv.json' % dataset_name))
            self.uv_list = np.array(self.uv_list)

        image_names = []
        for idx, prefix in enumerate(prefixes):
            image_path = os.path.join(self.base_path, dataset_name, 'rgb', '{}.jpg'.format(prefix))
            image_names.append(image_path)
        self.image_names = image_names
        del image_names
        del prefixes

        # h, w = self.img_size

    def json_load(self, p):
        assert os.path.exists(p), 'File does not exists: %s' % p
        with open(p, 'r') as fi:
            d = json.load(fi)
        return d

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img = cv2.imread(self.image_names[index])
        h, w = img[0].shape[:2]

        K = self.K_list[index]
        uv = self.uv_list[index]
        xyz = self.joint_list[index]
        uv_norm = uv.copy()
        uv_norm[:, 0] /= w
        uv_norm[:, 1] /= h

        coord_valid = (uv_norm > 0).astype("float32") * (uv_norm < 1).astype("float32")  # Nx2x21x2
        coord_valid = coord_valid[:, 0] * coord_valid[:, 1]

        valid_points = [uv[i] for i in range(len(uv)) if coord_valid[i] == 1]
        if len(valid_points) <= 1:
            valid_points = uv

        points = np.array(valid_points)
        min_coord = points.min(axis=0)
        max_coord = points.max(axis=0)
        center = (max_coord + min_coord) / 2
        scale = max_coord - min_coord

        ori_xyz = xyz.copy()
        scale = scale * self.scale_enlarge
        # perspective trans
        new_K, trans_matrix_2d, trans_matrix_3d = get_2d3d_perspective_transform(K, center, scale, self.rot_angle, 224)
        img_processed = cv2.warpPerspective(img, trans_matrix_2d, self.img_size)
        new_uv = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=1)
        new_uv = (trans_matrix_2d @ new_uv.T).T
        new_uv = new_uv[:, :2] / new_uv[:, 2:]
        new_xyz = (trans_matrix_3d @ xyz.T).T

        if img_processed.ndim == 2:
            img_processed = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2BGR)
        img_processed = np.transpose(img_processed, (2, 0, 1))
        return {
            # "img": np.ascontiguousarray(self.image_names[index]),
            "K": self.K_list[index],
            "xyz": self.joint_list[index],
            "scale": self.scale_list[index],
            "ori_xyz": ori_xyz,
        }
