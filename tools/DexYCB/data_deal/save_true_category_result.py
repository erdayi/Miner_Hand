import os
import json
import yaml
import numpy as np

'''
构建保存迷你测试集的真实类别总数 含跨5帧截取逻辑 量化指标的分母

输出
    OUTPUT_JSON = "data/minidata/new/classification_total_result.json"

'''
def collect_image_classes(base_dir, subject_folders, expected_views=8):
    left_clusters = []  # 左手簇：[[视角1路径, 视角2路径, ..., 视角8路径], ...]
    right_clusters = []  # 右手簇：[[视角1路径, 视角2路径, ..., 视角8路径], ...]

    for subject_folder in subject_folders:
        subject_path = os.path.join(base_dir, subject_folder)
        if not os.path.exists(subject_path):
            print(f"主题文件夹 {subject_folder} 不存在，跳过...")
            continue
        print(f"正在处理主题文件夹: {subject_folder}")

        # 获取所有实验子目录并创建索引序列
        experiment_folders = os.listdir(subject_path)
        sequence_indices = list(range(0, 100, 2))  # 每隔一个取一个索引

        # 遍历所有实验子目录
        for idx, experiment_folder_name in enumerate(experiment_folders):
            # 只处理在 sequence_indices 中的索引对应的文件夹
            if idx not in sequence_indices:
                continue
                
            experiment_folder = os.path.join(subject_path, experiment_folder_name)
            if not os.path.isdir(experiment_folder):
                continue
            print(f"正在处理实验文件夹: {experiment_folder}")

            # 解析meta.yml获取关键信息
            meta_path = os.path.join(experiment_folder, "meta.yml")
            if not os.path.exists(meta_path):
                print(f"实验文件夹 {experiment_folder} 缺少meta.yml，跳过...")
                continue
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta_data = yaml.safe_load(f)
                mano_sides = meta_data.get('mano_sides', [])
                serials = meta_data.get('serials', [])
                print(f"解析到mano_sides: {mano_sides}, serials(视角目录): {serials}")
            except Exception as e:
                print(f"解析meta.yml出错: {e}")
                continue

            # 验证视角数量是否符合预期（8个）
            if len(serials) != expected_views:
                print(f"实验 {experiment_folder} 的serials数量不符（预期{expected_views}个，实际{len(serials)}个），跳过...")
                continue

            # 构建所有视角目录路径并检查存在性
            view_dirs = [
                os.path.join(experiment_folder, serial)
                for serial in serials
            ]
            missing_dirs = [vd for vd in view_dirs if not os.path.isdir(vd)]
            if missing_dirs:
                print(f"实验 {experiment_folder} 缺少视角目录: {missing_dirs}，跳过...")
                continue

            # 收集每个视角目录下的{文件名: 完整路径}
            view_image_map = {}  # {视角目录: {文件名: 路径}}
            for view_dir in view_dirs:
                view_image_map[view_dir] = {}
                for filename in os.listdir(view_dir):
                    if filename.endswith('.jpg'):
                        file_path = os.path.join(view_dir, filename)
                        view_image_map[view_dir][filename] = file_path

            # 提取所有视角共有的文件名（必须在所有视角中存在）
            common_filenames = None
            for view_dir, images in view_image_map.items():
                if not images:
                    print(f"视角目录 {view_dir} 中无图片，跳过当前实验...")
                    common_filenames = set()
                    break
                if common_filenames is None:
                    common_filenames = set(images.keys())
                else:
                    common_filenames &= set(images.keys())
            if not common_filenames:
                print(f"实验 {experiment_folder} 中无多视角同名图片，跳过...")
                continue

            # 按文件名分组，生成8视角路径列表（按serials顺序排列）
            valid_frames = []  # 存储所有成功获取的帧
            for filename in common_filenames:
                image_paths = []
                for view_idx in range(expected_views):
                    view_dir = view_dirs[view_idx]
                    file_path = view_image_map[view_dir].get(filename)
                    if file_path:
                        # 提取文件名中的数字部分
                        file_number = int(filename.split('.')[0].split('_')[-1])
                        label_file = f"labels_{file_number:06d}.npz"
                        label_file_path = os.path.join(view_dir, label_file)

                        # 加载标注文件
                        try:
                            data = np.load(label_file_path)
                            joint_3d = data['joint_3d']
                            joint_2d = data['joint_2d']
                            # 检查 joint_3d 和 joint_2d 中是否有 -1.0
                            if -1.0 not in joint_3d and -1.0 not in joint_2d:
                                image_paths.append(file_path)
                                print(f"成功获取图片 {file_path} 及其对应的标注数据")
                        except Exception as e:
                            print(f"加载或检查标注文件 {label_file_path} 出错: {e}")

                # 确保收集到完整8个视角
                if len(image_paths) == expected_views:
                    valid_frames.append((file_number, image_paths, mano_sides))

            # 对成功获取的帧按帧号排序并进行间隔采样
            valid_frames.sort(key=lambda x: x[0])  # 按帧号排序
            interval = 5
            for i, (frame_num, image_paths, sides) in enumerate(valid_frames):
                if i % interval == 0:  # 每隔5帧取一个
                    if 'left' in sides:
                        left_clusters.append(image_paths)
                        print(f"左手簇添加: frame_{frame_num:06d}（{expected_views}视角）")
                    if 'right' in sides:
                        right_clusters.append(image_paths)
                        print(f"右手簇添加: frame_{frame_num:06d}（{expected_views}视角）")

    # 生成统计结果
    result = {
        "left_clusters": left_clusters,
        "right_clusters": right_clusters,
        # "statistics": {
        #     "total_left_clusters": len(left_clusters),
        #     "total_right_clusters": len(right_clusters),
        #     "total_clusters": len(left_clusters) + len(right_clusters)
        # }
    }
    return result


def save_results_to_json(result, output_path):
    """保存结果到JSON，含空数据检查"""
    if not result["left_clusters"] and not result["right_clusters"]:
        print("警告：未找到有效分类数据，JSON文件将为空")
        print("请检查：\n1. serials是否包含8个视角目录\n2. 每个视角目录是否有同名jpg文件\n3. 路径是否正确")
        return

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"分类结果已保存到 {output_path}")
    except Exception as e:
        print(f"保存JSON出错: {e}")


if __name__ == "__main__":
    # 配置参数
    BASE_DIR = "F:\\DexYCB\\dataset"
    SUBJECT_FOLDERS = [  
        "20200709-subject-01",
        "20200813-subject-02",
        "20200820-subject-03",
        "20200903-subject-04",
        "20200908-subject-05",
        "20200918-subject-06",
        "20200928-subject-07",
        "20201002-subject-08",
        "20201015-subject-09",
        "20201022-subject-10"
        ]  # 可添加多个主题文件夹
    OUTPUT_JSON = "data/minidata/new/classification_total_result.json"
    EXPECTED_VIEWS = 8  # 预期的视角数量（根据serials数量确定）

    print("开始执行图片分类...")
    classification_result = collect_image_classes(
        base_dir=BASE_DIR,
        subject_folders=SUBJECT_FOLDERS,
        expected_views=EXPECTED_VIEWS
    )
    print("分类完成")

    # 打印统计信息
    stats = classification_result["statistics"]
    print(f"统计结果：左手簇{stats['total_left_clusters']}个，右手簇{stats['total_right_clusters']}个，总簇{stats['total_clusters']}个")

    # 保存结果
    print("开始保存数据...")
    save_results_to_json(classification_result, OUTPUT_JSON)
    print("保存完成")