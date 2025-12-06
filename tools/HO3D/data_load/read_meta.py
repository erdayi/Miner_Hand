import pickle
import os
import pprint

# 元数据文件路径
meta_path = r'F:\GS\Dataset\HO3D_v2\train\MC1\meta\0001.pkl'  # 请替换为实际文件名

# 读取.pkl文件
def load_meta_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# 格式化输出字典内容
def print_meta_data(meta_data):
    pp = pprint.PrettyPrinter(indent=4, width=100, depth=None)
    print("\n完整元数据内容:")
    pp.pprint(meta_data)
    
    # 打印所有键值对
    print("\n所有字段:")
    for key, value in meta_data.items():
        print(f"{key}: ", end="")
        if isinstance(value, (list, dict, tuple)):
            print(f"{type(value)} 长度/大小: {len(value)}")
        elif hasattr(value, 'shape'):
            print(f"numpy.ndarray 形状: {value.shape}")
        else:
            print(value)

# 示例使用
if __name__ == '__main__':
    try:
        meta_data = load_meta_data(meta_path)
        print("成功加载元数据")
        
        # 打印完整数据
        print_meta_data(meta_data)
        
    except FileNotFoundError:
        print(f"错误: 文件 {meta_path} 不存在")
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")