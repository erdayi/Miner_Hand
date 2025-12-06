import os
import json
import yaml
import re
'''
本代码旨在构建迷你测试集 10人 / 50次实验 / 8视角 / (原本是72左右帧-不固定 之后每五帧截取一帧 避免出现过于相近手势 14帧?)

输入
    img_paths_json_path = "F:\\DexYCB\\miner\\image_paths.json"
    joint_3d_data_json_path = "F:\\DexYCB\\miner\\joint_3d_data.json"

输出
    left_hand_img_paths_json_path = 'data/minidata/new/left_hand_img_paths.json'
    left_hand_joint_3d_json_path = 'data/minidata/new/left_hand_joint_3d.json'
    right_hand_img_paths_json_path = 'data/minidata/new/right_hand_img_paths.json'
    right_hand_joint_3d_json_path = 'data/minidata/new/right_hand_joint_3d.json'

'''
def collect_subfolder_names(datadir, sequence_indices):
    print(f"开始收集 {datadir} 目录下符合条件的子文件夹信息...")
    # 初始化一个空字典，用于存储筛选后的子文件夹信息
    subfolder_info = {}
    # 定义要处理的具体文件夹名称列表，这些文件夹是我们要重点关注的
    target_folders = [
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
    ]
    # 遍历数据目录下的所有文件和文件夹
    for folder in os.listdir(datadir):
        print(f"正在检查文件夹: {folder}")
        # 拼接当前文件夹的完整路径
        folder_path = os.path.join(datadir, folder)
        # 检查当前文件夹是否在目标文件夹列表中
        if folder in target_folders:
            print(f"文件夹 {folder} 在目标文件夹列表中，继续检查是否为有效目录...")
            # 检查该路径是否为一个有效的目录
            if os.path.isdir(folder_path):
                print(f"文件夹 {folder} 是有效目录，开始获取子文件夹...")
                # 获取该文件夹下的所有子文件夹，使用列表推导式筛选出是目录的项
                child_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
                # 对子文件夹进行排序，默认按字母顺序排序
                child_folders.sort()
                # 确保子文件夹的数量不少于 sequence_indices 中的最大索引值加 1
                if len(child_folders) >= max(sequence_indices) + 1:
                    print(f"文件夹 {folder} 的子文件夹数量足够，开始过滤子文件夹...")
                    # 过滤子文件夹，只保留 sequence_indices 中指定索引的子文件夹
                    filtered_child_folders = [child_folders[index] for index in sequence_indices]
                    # 将子文件夹信息添加到字典中，以当前文件夹名为键，值是包含索引和子文件夹名称的字典列表
                    subfolder_info[folder] = [{"index": index, "subfolder_name": child_folder} for index, child_folder in
                                              zip(sequence_indices, filtered_child_folders)]
                else:
                    # 若子文件夹数量不足，打印警告信息
                    print(f"Warning: The number of subfolders in {folder} is less than the maximum index in sequence_indices.")
            else:
                # 若该路径不是一个目录，打印警告信息
                print(f"Warning: The folder {folder} is not a directory.")
    print(f"完成 {datadir} 目录下符合条件的子文件夹信息收集。")
    # 返回存储子文件夹信息的字典
    return subfolder_info


def filter_and_separate_data(img_paths_json_path, joint_3d_data_json_path, sequence_indices, data_dir,
                             left_hand_img_paths_json_path, left_hand_joint_3d_json_path,
                             right_hand_img_paths_json_path, right_hand_joint_3d_json_path):
    # 加载图片路径和 3D 关节坐标的 JSON 数据
    with open(img_paths_json_path, 'r', encoding='utf-8') as f:
        img_paths = json.load(f)
    with open(joint_3d_data_json_path, 'r', encoding='utf-8') as f:
        joint_3d_data = json.load(f)

    left_hand_img_paths = []
    left_hand_joint_3d = []
    right_hand_img_paths = []
    right_hand_joint_3d = []

    # 获取指定的子文件夹信息
    valid_subfolders = collect_subfolder_names(data_dir, sequence_indices)

    for img_path, joint_3d in zip(img_paths, joint_3d_data):
        parts = img_path.strip('/').split('/')
        subject_folder = parts[0]
        second_dir = parts[1]

        # 检查是否是符合条件的子文件夹
        if subject_folder in valid_subfolders:
            for subfolder_info in valid_subfolders[subject_folder]:
                if subfolder_info["subfolder_name"] == second_dir:
                    meta_file_path = os.path.join(data_dir, subject_folder, second_dir, 'meta.yml')
                    if os.path.exists(meta_file_path):
                        try:
                            with open(meta_file_path, 'r', encoding='utf-8') as f:
                                meta_data = yaml.safe_load(f)
                            mano_sides = meta_data.get('mano_sides', [])
                            if 'left' in mano_sides:
                                left_hand_img_paths.append(img_path)
                                print(f"已筛选出左手图片路径: {img_path}")
                                # 去除多余嵌套
                                if isinstance(joint_3d, list) and len(joint_3d) == 1 and isinstance(joint_3d[0], list):
                                    left_hand_joint_3d.append(joint_3d[0])
                                else:
                                    left_hand_joint_3d.append(joint_3d)
                            elif 'right' in mano_sides:
                                right_hand_img_paths.append(img_path)
                                print(f"已筛选出右手图片路径: {img_path}")
                                # 去除多余嵌套
                                if isinstance(joint_3d, list) and len(joint_3d) == 1 and isinstance(joint_3d[0], list):
                                    right_hand_joint_3d.append(joint_3d[0])
                                else:
                                    right_hand_joint_3d.append(joint_3d)
                        except Exception as e:
                            print(f"读取 {meta_file_path} 时发生错误: {e}")
                    else:
                        print(f"未找到 {meta_file_path} 文件。")

    # 检查并创建保存文件的目录
    for path in [left_hand_img_paths_json_path, left_hand_joint_3d_json_path,
                 right_hand_img_paths_json_path, right_hand_joint_3d_json_path]:
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    # 保存左右手的数据到 JSON 文件
    with open(left_hand_img_paths_json_path, 'w', encoding='utf-8') as f:
        json.dump(left_hand_img_paths, f, indent=4)
    print(f"已完成左手图片路径数据的存储到 {left_hand_img_paths_json_path}")

    with open(left_hand_joint_3d_json_path, 'w', encoding='utf-8') as f:
        json.dump(left_hand_joint_3d, f, indent=4)
    print(f"已完成左手 3D 关节坐标数据的存储到 {left_hand_joint_3d_json_path}")

    with open(right_hand_img_paths_json_path, 'w', encoding='utf-8') as f:
        json.dump(right_hand_img_paths, f, indent=4)
    print(f"已完成右手图片路径数据的存储到 {right_hand_img_paths_json_path}")

    with open(right_hand_joint_3d_json_path, 'w', encoding='utf-8') as f:
        json.dump(right_hand_joint_3d, f, indent=4)
    print(f"已完成右手 3D 关节坐标数据的存储到 {right_hand_joint_3d_json_path}")

    print(f"已完成数据筛选和分离，左手数据保存到 {left_hand_img_paths_json_path} 和 {left_hand_joint_3d_json_path}，右手数据保存到 {right_hand_img_paths_json_path} 和 {right_hand_joint_3d_json_path}。")




def filter_frames_by_interval(left_hand_img_paths_json_path, left_hand_joint_3d_json_path,
                              right_hand_img_paths_json_path, right_hand_joint_3d_json_path,
                              output_left_hand_img_paths_json_path, output_left_hand_joint_3d_json_path,
                              output_right_hand_img_paths_json_path, output_right_hand_joint_3d_json_path,
                              interval=5):
    """
    读取四个输入文件，根据每个视角下每隔 `interval` 帧保留一个的原则，筛选出需要保留的帧，
    并将这些帧的图片路径和 3D 关节坐标保存到新的文件中。

    :param left_hand_img_paths_json_path: 左手图片路径 JSON 文件路径
    :param left_hand_joint_3d_json_path: 左手 3D 关节坐标 JSON 文件路径
    :param right_hand_img_paths_json_path: 右手图片路径 JSON 文件路径
    :param right_hand_joint_3d_json_path: 右手 3D 关节坐标 JSON 文件路径
    :param output_left_hand_img_paths_json_path: 输出左手图片路径 JSON 文件路径
    :param output_left_hand_joint_3d_json_path: 输出左手 3D 关节坐标 JSON 文件路径
    :param output_right_hand_img_paths_json_path: 输出右手图片路径 JSON 文件路径
    :param output_right_hand_joint_3d_json_path: 输出右手 3D 关节坐标 JSON 文件路径
    :param interval: 每个视角下保留帧的间隔，默认为 5
    """
    # 加载输入文件
    with open(left_hand_img_paths_json_path, 'r', encoding='utf-8') as f:
        left_hand_img_paths = json.load(f)
    with open(left_hand_joint_3d_json_path, 'r', encoding='utf-8') as f:
        left_hand_joint_3d = json.load(f)
    with open(right_hand_img_paths_json_path, 'r', encoding='utf-8') as f:
        right_hand_img_paths = json.load(f)
    with open(right_hand_joint_3d_json_path, 'r', encoding='utf-8') as f:
        right_hand_joint_3d = json.load(f)

    # 初始化输出列表
    output_left_hand_img_paths = []
    output_left_hand_joint_3d = []
    output_right_hand_img_paths = []
    output_right_hand_joint_3d = []

    # 定义一个函数来处理每个手的数据
    def process_hand_data(img_paths, joint_3d, output_img_paths, output_joint_3d):
        # 使用字典来存储每个视角的帧索引
        view_frames = {}
        for img_path, joint_3d_data in zip(img_paths, joint_3d):
            parts = img_path.strip('/').split('/')
            subject_folder = parts[0]
            second_dir = parts[1]
            frame_name = parts[-1]
            frame_number = int(frame_name.split('_')[-1].split('.')[0])
            
            if (subject_folder, second_dir) not in view_frames:
                view_frames[(subject_folder, second_dir)] = []
            view_frames[(subject_folder, second_dir)].append((frame_number, img_path, joint_3d_data))
        
        # 对每个视角的帧进行排序并筛选
        for view, frames in view_frames.items():
            frames.sort(key=lambda x: x[0])  # 按帧号排序
            for i, (frame_number, img_path, joint_3d_data) in enumerate(frames):
                if i % interval == 0:
                    output_img_paths.append(img_path)
                    output_joint_3d.append(joint_3d_data)

    # 处理左手数据
    process_hand_data(left_hand_img_paths, left_hand_joint_3d, output_left_hand_img_paths, output_left_hand_joint_3d)
    # 处理右手数据
    process_hand_data(right_hand_img_paths, right_hand_joint_3d, output_right_hand_img_paths, output_right_hand_joint_3d)

    # 检查并创建保存文件的目录
    for path in [output_left_hand_img_paths_json_path, output_left_hand_joint_3d_json_path,
                 output_right_hand_img_paths_json_path, output_right_hand_joint_3d_json_path]:
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    # 保存筛选后的数据到新的文件
    with open(output_left_hand_img_paths_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_left_hand_img_paths, f, indent=4)
    print(f"已完成筛选后的左手图片路径数据的存储到 {output_left_hand_img_paths_json_path}")

    with open(output_left_hand_joint_3d_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_left_hand_joint_3d, f, indent=4)
    print(f"已完成筛选后的左手 3D 关节坐标数据的存储到 {output_left_hand_joint_3d_json_path}")

    with open(output_right_hand_img_paths_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_right_hand_img_paths, f, indent=4)
    print(f"已完成筛选后的右手图片路径数据的存储到 {output_right_hand_img_paths_json_path}")

    with open(output_right_hand_joint_3d_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_right_hand_joint_3d, f, indent=4)
    print(f"已完成筛选后的右手 3D 关节坐标数据的存储到 {output_right_hand_joint_3d_json_path}")

    print(f"已完成数据筛选，筛选后的数据保存到 {output_left_hand_img_paths_json_path}, {output_left_hand_joint_3d_json_path}, {output_right_hand_img_paths_json_path}, {output_right_hand_joint_3d_json_path}。")



def filter_frames_by_interval(left_hand_img_paths_json_path, left_hand_joint_3d_json_path,
                              right_hand_img_paths_json_path, right_hand_joint_3d_json_path,
                              output_left_hand_img_paths_json_path, output_left_hand_joint_3d_json_path,
                              output_right_hand_img_paths_json_path, output_right_hand_joint_3d_json_path,
                              interval=5):
    """
    读取四个输入文件，根据每个实验+视角组合下每隔 `interval` 帧保留一个的原则，筛选出需要保留的帧，
    并将这些帧的图片路径和 3D 关节坐标保存到新的文件中。

    参数说明：
    :param left_hand_img_paths_json_path:  左手图片路径 JSON 文件路径（输入）
    :param left_hand_joint_3d_json_path:   左手 3D 关节坐标 JSON 文件路径（输入）
    :param right_hand_img_paths_json_path: 右手图片路径 JSON 文件路径（输入）
    :param right_hand_joint_3d_json_path:  右手 3D 关节坐标 JSON 文件路径（输入）
    :param output_left_hand_img_paths_json_path:  输出左手图片路径 JSON 文件路径
    :param output_left_hand_joint_3d_json_path:   输出左手 3D 关节坐标 JSON 文件路径
    :param output_right_hand_img_paths_json_path: 输出右手图片路径 JSON 文件路径
    :param output_right_hand_joint_3d_json_path:  输出右手 3D 关节坐标 JSON 文件路径
    :param interval: 每个实验视角组合内保留帧的间隔，默认为 5
    """
    # 加载输入文件
    with open(left_hand_img_paths_json_path, 'r') as f:
        left_img_paths = json.load(f)
    with open(left_hand_joint_3d_json_path, 'r') as f:
        left_joint_data = json.load(f)
    with open(right_hand_img_paths_json_path, 'r') as f:
        right_img_paths = json.load(f)
    with open(right_hand_joint_3d_json_path, 'r') as f:
        right_joint_data = json.load(f)

    # 初始化输出容器
    output_left_img, output_left_joint = [], []
    output_right_img, output_right_joint = [], []

    def process_hand_data(img_paths, joint_data, out_img, out_joint):
        experiment_view_dict = {}
        for img_path, joint in zip(img_paths, joint_data):
            # 解析路径结构：subject/experiment/camera/frame
            path_parts = img_path.strip('/').split('/')
            subject = path_parts[0]
            experiment = path_parts[1]
            camera = path_parts[2]
            
            # 提取帧号（假设文件名格式为 color_000001.jpg）
            frame_num = int(path_parts[-1].split('_')[-1].split('.')[0])
            
            # 使用组合键：人员+实验+视角
            key = (subject, experiment, camera)
            if key not in experiment_view_dict:
                experiment_view_dict[key] = []
            experiment_view_dict[key].append( (frame_num, img_path, joint) )

        # 按组合键排序后处理
        for key in sorted(experiment_view_dict.keys()):
            frames = sorted(experiment_view_dict[key], key=lambda x: x[0])  # 按帧号排序
            for idx, (fnum, path, data) in enumerate(frames):
                if idx % interval == 0:  # 从第0帧开始每隔interval帧取一个
                    out_img.append(path)
                    out_joint.append(data)

    # 处理双手数据
    process_hand_data(left_img_paths, left_joint_data, output_left_img, output_left_joint)
    process_hand_data(right_img_paths, right_joint_data, output_right_img, output_right_joint)

    # 保存输出文件
    def save_data(data, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} items to {path}")

    save_data(output_left_img, output_left_hand_img_paths_json_path)
    save_data(output_left_joint, output_left_hand_joint_3d_json_path)
    save_data(output_right_img, output_right_hand_img_paths_json_path)
    save_data(output_right_joint, output_right_hand_joint_3d_json_path)
    
def main(img_paths_json_path, joint_3d_data_json_path, sequence_indices, data_dir):
    left_hand_img_paths_json_path = 'data/minidata/new/disorder/left_hand_img_paths.json'
    left_hand_joint_3d_json_path = 'data/minidata/new/disorder/left_hand_joint_3d.json'
    right_hand_img_paths_json_path = 'data/minidata/new/disorder/right_hand_img_paths.json'
    right_hand_joint_3d_json_path = 'data/minidata/new/disorder/right_hand_joint_3d.json'

    # 获取数据
    filter_and_separate_data(img_paths_json_path, joint_3d_data_json_path, sequence_indices, data_dir,
                             left_hand_img_paths_json_path, left_hand_joint_3d_json_path,
                             right_hand_img_paths_json_path, right_hand_joint_3d_json_path)


    output_left_hand_img_paths_json_path = 'data/minidata/new/disorder/left_hand_img_paths_clean.json'
    output_left_hand_joint_3d_json_path = 'data/minidata/new/disorder/left_hand_joint_3d_clean.json'
    output_right_hand_img_paths_json_path = 'data/minidata/new/disorder/right_hand_img_paths_clean.json'
    output_right_hand_joint_3d_json_path = 'data/minidata/new/disorder/right_hand_joint_3d_clean.json'

    
    # 截帧筛选数据
    filter_frames_by_interval(
        left_hand_img_paths_json_path,left_hand_joint_3d_json_path,right_hand_img_paths_json_path
        ,right_hand_joint_3d_json_path,output_left_hand_img_paths_json_path,output_left_hand_joint_3d_json_path,
        output_right_hand_img_paths_json_path,output_right_hand_joint_3d_json_path)

if __name__ == "__main__":
    img_paths_json_path = "F:\\DexYCB\\miner\\image_paths.json"
    joint_3d_data_json_path = "F:\\DexYCB\\miner\\joint_3d_data.json"
    # sequence_indices = [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99]  # 要保留的子文件夹索引
    sequence_indices = list(range(0, 20, 2))  # 修改为每隔一个取一个索引，从0到98
    data_dir = "F:\\DexYCB\\dataset"

    main(img_paths_json_path, joint_3d_data_json_path, sequence_indices, data_dir)
    