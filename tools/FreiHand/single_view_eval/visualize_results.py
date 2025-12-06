import os
import json
import numpy as np
from typing import Dict, List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def create_visualization(evaluation_dir: str, output_dir: str) -> None:
    """
    创建可视化结果
    
    Args:
        evaluation_dir: 包含评估结果的目录
        output_dir: 输出可视化结果的目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载评估结果
    with open(os.path.join(evaluation_dir, "evaluation_results.json"), "r") as f:
        results = json.load(f)
    
    # 准备数据
    data = []
    for result in results:
        params = result["parameters"]
        metrics = result["metrics"]
        data.append({
            "min_cluster_size": params["min_cluster_size"],
            "min_samples": params["min_samples"],
            "alpha": params["alpha"],
            "distance_threshold": params["distance_threshold"],
            "silhouette_score": metrics["silhouette_score"],
            "davies_bouldin": metrics["davies_bouldin"],
            "calinski_harabasz": metrics["calinski_harabasz"],
            "intra_cluster_distance": metrics["intra_cluster_distance"],
            "inter_cluster_distance": metrics["inter_cluster_distance"],
            "cluster_entropy": metrics["cluster_entropy"],
            "unclassified_ratio": metrics["unclassified_ratio"],
            "total_clusters": metrics["total_clusters"],
            "classified_samples": metrics["classified_samples"],
            "unclassified_samples": metrics["unclassified_samples"]
        })
    
    # 创建HTML文件
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>HDBSCAN聚类评估指标可视化</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            .chart-container {
                transition: all 0.3s ease;
            }
            .chart-container:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
            }
        </style>
    </head>
    <body class="bg-gray-50 font-inter">
        <div class="container mx-auto px-4 py-8">
            <h1 class="text-3xl font-bold mb-8">HDBSCAN聚类评估指标可视化</h1>
            
            <!-- 概览部分 -->
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-lg font-semibold mb-2">Silhouette Score</h3>
                    <p class="text-2xl font-bold text-blue-600" id="avg-silhouette"></p>
                </div>
                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-lg font-semibold mb-2">Davies-Bouldin Index</h3>
                    <p class="text-2xl font-bold text-green-600" id="avg-davies"></p>
                </div>
                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-lg font-semibold mb-2">Calinski-Harabasz Index</h3>
                    <p class="text-2xl font-bold text-purple-600" id="avg-calinski"></p>
                </div>
                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-lg font-semibold mb-2">总聚类数量</h3>
                    <p class="text-2xl font-bold text-gray-800" id="total-clusters"></p>
                </div>
            </div>
            
            <!-- 图表部分 -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-lg font-semibold mb-4">聚类质量指标对比</h3>
                    <div id="metrics-comparison" class="h-80"></div>
                </div>
                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-lg font-semibold mb-4">参数与聚类数量关系</h3>
                    <div id="cluster-count" class="h-80"></div>
                </div>
                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-lg font-semibold mb-4">参数与Silhouette Score关系</h3>
                    <div id="silhouette-score" class="h-80"></div>
                </div>
                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-lg font-semibold mb-4">参数与Davies-Bouldin Index关系</h3>
                    <div id="davies-bouldin" class="h-80"></div>
                </div>
            </div>
        </div>
        
        <script>
            // 数据
            const data = """ + json.dumps(data) + """;
            
            // 更新概览卡片
            function updateOverviewCards() {
                const avgSilhouette = data.reduce((sum, d) => sum + d.silhouette_score, 0) / data.length;
                const avgDavies = data.reduce((sum, d) => sum + d.davies_bouldin, 0) / data.length;
                const avgCalinski = data.reduce((sum, d) => sum + d.calinski_harabasz, 0) / data.length;
                const totalClusters = data.reduce((sum, d) => sum + d.total_clusters, 0) / data.length;
                
                document.getElementById('avg-silhouette').textContent = avgSilhouette.toFixed(4);
                document.getElementById('avg-davies').textContent = avgDavies.toFixed(4);
                document.getElementById('avg-calinski').textContent = avgCalinski.toFixed(4);
                document.getElementById('total-clusters').textContent = Math.round(totalClusters);
            }
            
            // 创建图表
            function createCharts() {
                // 聚类质量指标对比
                const metricsComparison = document.getElementById('metrics-comparison');
                Plotly.newPlot(metricsComparison, [{
                    type: 'scatterpolar',
                    r: [
                        data[0].silhouette_score,
                        data[0].calinski_harabasz / 1000,
                        1 / (1 + data[0].davies_bouldin),
                        data[0].total_clusters / 1000,
                        data[0].classified_samples / 32560
                    ],
                    theta: ['Silhouette Score', 'Calinski-Harabasz', 'Davies-Bouldin', '聚类数量', '分类样本数'],
                    fill: 'toself',
                    name: '当前参数'
                }], {
                    polar: {
                        radialaxis: {
                            visible: true,
                            range: [0, 1]
                        }
                    }
                });
                
                // 参数与聚类数量关系
                const clusterCount = document.getElementById('cluster-count');
                Plotly.newPlot(clusterCount, [{
                    type: 'scatter',
                    mode: 'lines+markers',
                    x: data.map(d => d.min_cluster_size),
                    y: data.map(d => d.total_clusters),
                    name: '聚类数量'
                }], {
                    title: 'min_cluster_size vs 聚类数量',
                    xaxis: { title: 'min_cluster_size' },
                    yaxis: { title: '聚类数量' }
                });
                
                // 参数与Silhouette Score关系
                const silhouetteScore = document.getElementById('silhouette-score');
                Plotly.newPlot(silhouetteScore, [{
                    type: 'scatter',
                    mode: 'lines+markers',
                    x: data.map(d => d.min_cluster_size),
                    y: data.map(d => d.silhouette_score),
                    name: 'Silhouette Score'
                }], {
                    title: 'min_cluster_size vs Silhouette Score',
                    xaxis: { title: 'min_cluster_size' },
                    yaxis: { title: 'Silhouette Score' }
                });
                
                // 参数与Davies-Bouldin Index关系
                const daviesBouldin = document.getElementById('davies-bouldin');
                Plotly.newPlot(daviesBouldin, [{
                    type: 'scatter',
                    mode: 'lines+markers',
                    x: data.map(d => d.min_cluster_size),
                    y: data.map(d => d.davies_bouldin),
                    name: 'Davies-Bouldin Index'
                }], {
                    title: 'min_cluster_size vs Davies-Bouldin Index',
                    xaxis: { title: 'min_cluster_size' },
                    yaxis: { title: 'Davies-Bouldin Index' }
                });
            }
            
            // 初始化
            updateOverviewCards();
            createCharts();
        </script>
    </body>
    </html>
    """
    
    # 保存HTML文件
    with open(os.path.join(output_dir, "index_compare.html"), "w", encoding="utf-8") as f:
        f.write(html_content)

if __name__ == "__main__":
    evaluation_dir = "data/FreiHand/cluster_data/hdbscan/evaluation"
    output_dir = "data/FreiHand/cluster_data/hdbscan/visualization"
    create_visualization(evaluation_dir, output_dir) 