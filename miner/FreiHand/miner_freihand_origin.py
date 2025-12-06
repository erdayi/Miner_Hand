import json
import numpy as np
import time
from tqdm import tqdm
from scipy.linalg import orthogonal_procrustes

'''
    本方法基于初始方法借助锚点和样本进行聚类
    
    输出的两个json文件 
    clusters_origin_32560_0.0055.json 代表初始生成的聚类样式 
    clusters_origin_32560_0.0055_clean.json 代表清洗后参考OPTICS保存的聚类样式保存
'''


def align_w_scale(mtx1, mtx2):
    """
    使用正交对齐方法进行匹配，并返回对齐后的结果
    :param mtx1: 第一个矩阵
    :param mtx2: 第二个矩阵
    :return: 对齐后的第二个矩阵
    """
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1
    return mtx2_t


def read_dataset(json_dir, max_samples):
    """
    读取数据集
    :param json_dir: 数据集的 JSON 文件路径
    :param max_samples: 最大样本数
    :return: 数据集
    """
    dataset = json.load(open(json_dir, 'r'))
    dataset = np.array(dataset[:max_samples])
    return dataset


def mine_anchors(dataset, anchor_idx_list, threshold):
    """
    挖掘锚点
    :param dataset: 数据集
    :param anchor_idx_list: 锚点索引列表
    :param threshold: 误差阈值
    :return: 匹配结果和已处理样本集合
    """
    aligned_err_res = {}
    processed_samples = set()

    for anchor_idx in tqdm(anchor_idx_list, desc="Mining Anchors"):
        anchor_start_time = time.time()

        if anchor_idx in processed_samples:
            continue

        anchor_xyz = dataset[anchor_idx]
        matched_samples = {}

        for idx in range(len(dataset)):
            if idx == anchor_idx or idx in processed_samples:
                continue

            xyz = dataset[idx]
            xyz_aligned = align_w_scale(anchor_xyz, xyz)
            xyz_aligned_err = np.mean(np.linalg.norm(anchor_xyz - xyz_aligned, axis=1))

            if xyz_aligned_err < threshold:
                matched_samples[idx] = float(xyz_aligned_err)
                processed_samples.add(idx)

        if matched_samples:
            aligned_err_res[anchor_idx] = matched_samples

        processed_samples.add(anchor_idx)

        anchor_time_elapsed = time.time() - anchor_start_time
        print(f"Anchor {anchor_idx} processed in {anchor_time_elapsed:.4f} seconds")

    return aligned_err_res, processed_samples


def save_results(aligned_err_res, output_file):
    """
    保存匹配结果到 JSON 文件
    :param aligned_err_res: 匹配结果
    :param output_file: 输出文件路径
    """
    # 对每个聚类的索引进行排序
    for anchor, matches in aligned_err_res.items():
        sorted_matches = dict(sorted(matches.items()))
        aligned_err_res[anchor] = sorted_matches

    # 对聚类结果的键进行排序
    sorted_aligned_err_res = dict(sorted(aligned_err_res.items()))

    # 将键转换为字符串
    aligned_err_res_str_keys = {str(k): {str(i): v for i, v in val.items()} for k, val in
                                sorted_aligned_err_res.items()}

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(aligned_err_res_str_keys, f, indent=4)


def print_statistics(dataset, processed_samples, aligned_err_res):
    """
    打印统计结果
    :param dataset: 数据集
    :param processed_samples: 已处理样本集合
    :param aligned_err_res: 匹配结果
    """
    total_samples = len(dataset)
    classified_samples = len(processed_samples)
    unclassified_samples = total_samples - classified_samples
    num_anchors_with_matches = len(aligned_err_res)

    all_involved_samples = set()
    for anchor, matches in aligned_err_res.items():
        all_involved_samples.add(anchor)
        all_involved_samples.update(matches.keys())
    num_all_involved_samples = len(all_involved_samples)

    print("\n统计结果：")
    print(f"总照片数：{total_samples}")
    print(f"有匹配样本的锚点数量：{num_anchors_with_matches}")
    print(f"一共涉及的样本数量（锚点和匹配样本）：{num_all_involved_samples}")


def read_and_save_clustered_result(input_file, output_file):
    """
    读取聚类好的结果文件，并按照optics样式保存
    :param input_file: 输入的聚类结果文件路径
    :param output_file: 输出文件路径
    """
    # 读取聚类结果文件
    with open(input_file, 'r', encoding='utf-8') as f:
        clustered_result = json.load(f)

    # 初始化新的聚类结果
    new_clusters = {}
    cluster_index = 0

    # 遍历原始聚类结果，每个键是锚点，值是匹配的样本及其误差
    for anchor, samples in clustered_result.items():
        # 提取样本ID（忽略误差值）
        sample_ids = list(samples.keys())
        # 将锚点作为第一个元素，后面跟上所有匹配的样本
        cluster_members = [anchor] + sample_ids
        # 按顺序生成新的簇编号
        new_clusters[cluster_index] = cluster_members
        cluster_index += 1

    # 对聚类结果的键进行排序（虽然已经按顺序生成，但为了保险起见）
    sorted_clusters = dict(sorted(new_clusters.items()))

    # 将键转换为字符串
    final_result = {str(k): v for k, v in sorted_clusters.items()}

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=4)

def main():
    description = """
    本程序按照严格顺序计算手部关键点数据集里每个样本与多个锚点样本之间的对齐误差。
    规则：
    1. 依次从 0 开始选取锚点，后续锚点不再匹配已归类的样本。
    2. 只有误差小于阈值时才存储匹配关系。
    3. 锚点匹配完成后，自己也不能再被后续锚点匹配。
    4. 未匹配到样本的锚点不会出现在最终的 JSON 里。
    """
    print(description)

    # 参数设置
    bmk = dict(
        name="FreiHand",
        json_dir=r'../../data/FreiHand/origin_data/xyz_list.json',
        scale_enlarge=1.25,
    )
    max_samples = 32560
    threshold = 0.0055
    output_file = '../../data/FreiHand/cluster_data/clusters_origin_32560_0.0055.json'

    # 读取数据集
    dataset = read_dataset(bmk["json_dir"], max_samples)
    anchor_idx_list = list(range(max_samples))

    # 开始挖掘
    print("Start mining")
    start_time = time.time()
    aligned_err_res, processed_samples = mine_anchors(dataset, anchor_idx_list, threshold)
    total_time_elapsed = time.time() - start_time
    print(f"Total mining time: {total_time_elapsed:.2f} seconds")

    # 保存结果
    save_results(aligned_err_res, output_file)

    # 打印统计结果
    print_statistics(dataset, processed_samples, aligned_err_res)

    # 将聚类后的结果按照optics存储
    new_file = '../../data/FreiHand/cluster_data/new_clusters_origin_32560_0.0055_clean.json'
    read_and_save_clustered_result(output_file, new_file)

    print("Done")


if __name__ == "__main__":
    main()
