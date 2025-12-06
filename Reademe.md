# 相似人手姿挖掘算法

注：本项目将详细展示不同算法对于相似人手挖掘代码，含可视化，借助重构目录展示更为清晰的目录结构


OPTICS聚类评估结果:
轮廓系数(Silhouette Score): -0.2784 (范围[-1,1], 越接近1越好)
Davies-Bouldin指数: 22.5610 (越小越好, 0表示最佳聚类)
Calinski-Harabasz指数: 1.7682 (越大表示聚类效果越好)
检测到的聚类数量: 4069


原始数据聚类评估结果:
轮廓系数(Silhouette Score): -0.3547 (范围[-1,1], 越接近1越好)
Davies-Bouldin指数: 21.7652 (越小越好, 0表示最佳聚类)
Calinski-Harabasz指数: 0.6546 (越大表示聚类效果越好)
检测到的聚类数量: 1739


# 安装依赖
pip install cupy-cuda11x  # 或 cupy-cuda12x
# 或者
pip install torch torchvision

# 运行GPU加速版本
python miner_freihand_optics_pro_gpu.py --gpu

pip install faiss-gpu  # GPU版本
# 或
pip install faiss-cpu  # CPU版本