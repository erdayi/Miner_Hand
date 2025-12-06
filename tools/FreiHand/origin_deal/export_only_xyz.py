import json
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool

from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
import sys
import os

from .get_data import GetData




# 移除或注释掉以下两行，它们是尝试动态修改 sys.path，但在包结构下通常不需要
# module_dir = os.path.dirname(os.path.abspath(r"E:\pose\MVHandMiner\miner\get_data.py"))
# sys.path.append(module_dir)



"""
    02-本程序用于导出xyz_list.json和xyz_order.json(依次读取原数据集的每个json文件效率过低，合并按顺序导出所需3d xyz坐标即可)
    步骤：
        1. 加载数据集
        2. 多线程处理数据集
        3. 导出xyz_list.json和xyz_order.json
        4. 训练只需要xyz_list即可，但是查找需要xyz_order即带索引的数据
    调用：
        get_data 用于加载数据
    运行：
        python -m tools.FreiHand.origin_deal.export_only_xyz
"""

def process_chunk(start_idx, end_idx):
    temp_res = {}
    for idx in tqdm(range(start_idx, end_idx), "Mining:"):
        try:
            data = dataset.getitem(idx)
            xyz = data['xyz'].tolist()  # 将NumPy数组转换为Python列表
            temp_res[idx] = xyz  # 转换为Python原生类型
        except Exception as e:
            print(f"Error processing index {idx}: {str(e)}")
    return temp_res


bmk=dict(
    name="FreiHand",
    json_dir = 'data/FreiHand/origin_data/eval.json',
    eval_dir = 'G:/WZY/PalmData/data/FreiHAND/evaluation/rgb',
    scale_enlarge=1.25,
)

dataset = GetData(bmk["json_dir"], (224, 224), bmk["scale_enlarge"])


if __name__ == "__main__":
    xyz_list = []

    # 多线程
    num_processes = 16  # 根据CPU核心数调整
    total_samples = len(dataset)
    # 生成进程参数（动态分配任务范围）
    def generate_chunks(total, num_workers):
        chunk_size = (total + num_workers - 1) // num_workers
        return [(i, min(i+chunk_size, total)) for i in range(0, total, chunk_size)]
    process_args = generate_chunks(total_samples, num_processes)
    # 并行处理（带进度条）
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_chunk, process_args)
    # 合并结果（使用OrderedDict保证顺序）
    from collections import OrderedDict
    xyz_order = OrderedDict()
    for chunk in sorted(results, key=lambda x: next(iter(x.keys()))):
        xyz_order.update(chunk)

    xyz_list = list(xyz_order.values())
    print("Saving xyz_list...")
    with open('data/FreiHand/origin_data/eval_xyz_list.json', 'w', encoding='utf-8') as f:
        json.dump(xyz_list, f, ensure_ascii=False, indent=4)

    print("Saving xyz_order...")
    with open('data/FreiHand/origin_data/eval_xyz_order.json', 'w', encoding='utf-8') as f:
        json.dump(xyz_order, f, ensure_ascii=False, indent=4)