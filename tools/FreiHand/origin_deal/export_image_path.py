import json
import os

'''
    01-构建FreiHand训练集和验证集的图片路径对应的JSON
'''
dataset_dir = "G:/WZY/PalmData/data/FreiHAND"
train_dir = "training/rgb/"
eval_dir = "evaluation/rgb/"

train_num = 130240
eval_num = 3960

# 生成训练集路径清单（带缩进）
train_json = []
for i in range(train_num):
    image_path = os.path.join(dataset_dir, train_dir, f"{i:08d}.jpg")
    train_json.append(image_path)
with open("../../../data/FreiHand/origin_data/train.json", "w") as f:
    json.dump(train_json, f, indent=4)  # 添加缩进参数

# 生成验证集路径清单（带缩进）
eval_json = []
for i in range(eval_num):
    image_path = os.path.join(dataset_dir, eval_dir, f"{i:08d}.jpg")
    eval_json.append(image_path)
with open("../../../data/FreiHand/origin_data/eval.json", "w") as f:
    json.dump(eval_json, f, indent=4)  # 添加缩进参数