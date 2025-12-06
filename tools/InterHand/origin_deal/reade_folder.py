import os
from pathlib import Path

'''
   读取eval图片目录下的所有动作类型
'''

def get_folders_with_os(path):
    """使用os模块获取指定路径下的所有文件夹名称"""
    folders = []
    # 检查路径是否存在
    if not os.path.exists(path):
        print(f"路径 {path} 不存在")
        return folders
    
    # 遍历路径下的所有项目
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        # 判断是否为文件夹
        if os.path.isdir(item_path):
            folders.append(item)
    
    return folders

def get_folders_with_pathlib(path):
    """使用pathlib模块获取指定路径下的所有文件夹名称"""
    folders = []
    p = Path(path)
    # 检查路径是否存在
    if not p.exists():
        print(f"路径 {path} 不存在")
        return folders
    
    # 遍历路径下的所有项目
    for item in p.iterdir():
        # 判断是否为文件夹
        if item.is_dir():
            folders.append(item.name)
    
    return folders

def get_folders_recursive(path):
    """递归获取指定路径下的所有文件夹名称（包括子文件夹）"""
    folders = []
    p = Path(path)
    if not p.exists():
        print(f"路径 {path} 不存在")
        return folders
    
    # 递归遍历所有子目录
    for item in p.glob('**/'):
        # 排除当前路径本身
        if item != p:
            # 计算相对于原始路径的文件夹名称
            relative_path = item.relative_to(p)
            folders.append(str(relative_path))
    
    return folders

# 指定要读取的目录路径
directory_path = r"I:\gs\raw\InterHand2.6M_5fps_batch1\images\val\Capture0"

# 使用os模块获取文件夹名称
folders_os = get_folders_with_os(directory_path)
print("使用os模块获取的文件夹名称:")
for folder in folders_os:
    print(folder)

print("\n------------------------\n")

# 使用pathlib模块获取文件夹名称
folders_pathlib = get_folders_with_pathlib(directory_path)
print("使用pathlib模块获取的文件夹名称:")
for folder in folders_pathlib:
    print(folder)

print("\n------------------------\n")

# 递归获取所有子文件夹名称
recursive_folders = get_folders_recursive(directory_path)
print("递归获取的所有文件夹名称:")
for folder in recursive_folders:
    print(folder)