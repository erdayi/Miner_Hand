# import json
# from pyecharts import options as opts
# from pyecharts.charts import Bar
#
# '''
#     本方法借助PyEcharts库统计基础方法聚类簇内的图片数量分布，将可视化结果以cluster_image_distribution.html网页形式保存
# '''
#
#
# def count_images_from_json(json_file_path):
#     """
#     根据 JSON 文件中的样本索引统计对应图片的数量和聚类数量
#     :param json_file_path: JSON 文件的路径
#     :return: 对应图片的总数、聚类的数量、簇大小统计字典
#     """
#     try:
#         with open(json_file_path, 'r') as f:
#             cluster_data = json.load(f)
#
#         total_image_count = 0
#         cluster_size_stats = {}
#         for cluster_id, sample_indices in cluster_data.items():
#             # 簇大小 = 锚点(1) + 相似样本数量
#             cluster_size = 1 + len(sample_indices)
#             total_image_count += cluster_size
#             if cluster_size in cluster_size_stats:
#                 cluster_size_stats[cluster_size] += 1
#             else:
#                 cluster_size_stats[cluster_size] = 1
#
#         num_clusters = len(cluster_data)
#         return total_image_count, num_clusters, cluster_size_stats
#
#     except FileNotFoundError:
#         print(f"错误：未找到文件 {json_file_path}。")
#         return 0, 0, {}
#     except json.JSONDecodeError:
#         print(f"错误：无法解析 {json_file_path} 为有效的 JSON 数据。")
#         return 0, 0, {}
#
#
# def plot_cluster_size_distribution(cluster_size_stats, title="聚类簇大小分布", sort_by_count=True):
#     """
#     使用 pyecharts 绘制簇大小分布的柱状图
#
#     Args:
#         cluster_size_stats: 簇大小统计字典（键：簇大小，值：出现次数）
#         title: 图表标题
#         sort_by_count: 是否按簇数量降序排序
#     """
#     if not cluster_size_stats:
#         print("没有有效的统计数据，无法生成图表。")
#         return
#
#     # 按要求排序
#     if sort_by_count:
#         # 按簇数量降序排序
#         sorted_stats = sorted(cluster_size_stats.items(), key=lambda x: x[1], reverse=True)
#     else:
#         # 按簇大小升序排序
#         sorted_stats = sorted(cluster_size_stats.items())
#
#     sizes = [item[0] for item in sorted_stats]
#     counts = [item[1] for item in sorted_stats]
#
#     # 创建柱状图
#     bar = (
#         Bar(init_opts=opts.InitOpts(width="1200px", height="600px"))
#         .add_xaxis(sizes)
#         .add_yaxis(
#             "簇的数量",
#             counts,
#             color="#4a6fe3",  # 统一使用蓝色
#             label_opts=opts.LabelOpts(
#                 position="top",
#                 color="black",
#                 font_size=12
#             ),
#         )
#         .set_global_opts(
#             title_opts=opts.TitleOpts(is_show=False),  # 不显示标题
#             xaxis_opts=opts.AxisOpts(
#                 name="簇内图片数量",
#                 name_textstyle_opts=opts.TextStyleOpts(font_size=16),
#                 axislabel_opts=opts.LabelOpts(
#                     font_size=12,
#                     # rotate=30,  # 横轴标签倾斜30度
#                     interval=0,  # 强制显示所有标签
#                     overflow="breakAll"  # 标签过长时换行显示
#                 ),
#                 axisline_opts=opts.AxisLineOpts(
#                     linestyle_opts=opts.LineStyleOpts(color="#333")
#                 )
#             ),
#             yaxis_opts=opts.AxisOpts(
#                 name="簇的数量",
#                 name_textstyle_opts=opts.TextStyleOpts(font_size=16),
#                 axislabel_opts=opts.LabelOpts(font_size=12),
#                 axisline_opts=opts.AxisLineOpts(
#                     linestyle_opts=opts.LineStyleOpts(color="#333")
#                 ),
#                 splitline_opts=opts.SplitLineOpts(
#                     linestyle_opts=opts.LineStyleOpts(color="#eee")
#                 )
#             ),
#             toolbox_opts=opts.ToolboxOpts(
#                 is_show=True,
#                 feature=opts.ToolBoxFeatureOpts(
#                     save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
#                         type_="png",
#                         pixel_ratio=1,
#                         name="簇大小分布"
#                     ),
#                     data_view=opts.ToolBoxFeatureDataViewOpts(is_show=True),
#                     data_zoom=opts.ToolBoxFeatureDataZoomOpts(is_show=True)
#                 )
#             ),
#             legend_opts=opts.LegendOpts(is_show=False)
#
#         )
#     )
#
#     # 渲染图表
#     bar.render("cluster_image_distribution.html")
#     return bar
#
#
# def main():
#     # 配置参数
#     params = {
#         "json_file_path": "../../../data/FreiHand/cluster_data/clusters_origin_32560_0.0055.json",
#         "chart_title": "FreiHand数据集聚类簇大小分布"
#     }
#
#     # 分析聚类数据
#     total_images, num_clusters, size_stats = count_images_from_json(params["json_file_path"])
#
#     # 计算统计信息
#     if size_stats:
#         max_size = max(size_stats.keys())
#         min_size = min(size_stats.keys())
#         avg_size = sum(k * v for k, v in size_stats.items()) / num_clusters if num_clusters > 0 else 0
#
#         print(f"总簇数: {num_clusters}")
#         print(f"总图片数: {total_images}")
#         print(f"最大簇大小: {max_size}")
#         print(f"最小簇大小: {min_size}")
#         print(f"平均簇大小: {avg_size:.2f}")
#
#     # 输出详细统计
#     print("\n详细统计:")
#     print("簇大小\t簇数量")
#     for size in sorted(size_stats.keys()):
#         print(f"{size}\t{size_stats[size]}")
#
#     # 绘制并保存图表
#     plot_cluster_size_distribution(size_stats, params["chart_title"])
#     print("\n图表已保存为 cluster_image_distribution.html")
#     print("请在浏览器中打开该文件查看交互式图表")
#
#
# if __name__ == "__main__":
#     main()

# ————————下方方法借助PyEcharts库统计基础方法聚类（clean按optics结果保存）簇内的图片数量分布，将可视化结果以cluster_image_distribution.html网页形式保存——————#
import json
from pyecharts import options as opts
from pyecharts.charts import Bar

'''
    本方法借助PyEcharts库统计基础方法聚类（clean 按optics结果保存）簇内的图片数量分布，将可视化结果以cluster_image_distribution.html网页形式保存
'''


def count_images_from_json(json_file_path):
    """
    根据 JSON 文件中的样本索引统计对应图片的数量和聚类数量
    :param json_file_path: JSON 文件的路径
    :return: 对应图片的总数、聚类的数量、簇大小统计字典
    """
    try:
        with open(json_file_path, 'r') as f:
            cluster_data = json.load(f)

        total_image_count = 0
        cluster_size_stats = {}

        for cluster_id, sample_indices in cluster_data.items():
            # 直接使用样本数量（不额外加1）
            cluster_size = len(sample_indices)
            total_image_count += cluster_size
            if cluster_size in cluster_size_stats:
                cluster_size_stats[cluster_size] += 1
            else:
                cluster_size_stats[cluster_size] = 1

        num_clusters = len(cluster_data)
        return total_image_count, num_clusters, cluster_size_stats

    except FileNotFoundError:
        print(f"错误：未找到文件 {json_file_path}。")
        return 0, 0, {}
    except json.JSONDecodeError:
        print(f"错误：无法解析 {json_file_path} 为有效的 JSON 数据。")
        return 0, 0, {}


def plot_cluster_size_distribution(cluster_size_stats, title="聚类簇大小分布", sort_by_count=True):
    """
    使用 pyecharts 绘制簇大小分布的柱状图

    Args:
        cluster_size_stats: 簇大小统计字典（键：簇大小，值：出现次数）
        title: 图表标题
        sort_by_count: 是否按簇数量降序排序
    """
    if not cluster_size_stats:
        print("没有有效的统计数据，无法生成图表。")
        return

    # 按要求排序
    if sort_by_count:
        # 按簇数量降序排序
        sorted_stats = sorted(cluster_size_stats.items(), key=lambda x: x[1], reverse=True)
    else:
        # 按簇大小升序排序
        sorted_stats = sorted(cluster_size_stats.items())

    sizes = [item[0] for item in sorted_stats]
    counts = [item[1] for item in sorted_stats]

    # 创建柱状图
    bar = (
        Bar(init_opts=opts.InitOpts(width="1200px", height="600px"))
        .add_xaxis(sizes)
        .add_yaxis(
            "簇的数量",
            counts,
            color="#4a6fe3",  # 统一使用蓝色
            label_opts=opts.LabelOpts(
                position="top",
                color="black",
                font_size=12
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(is_show=False),  # 不显示标题
            xaxis_opts=opts.AxisOpts(
                name="簇内图片数量",
                name_textstyle_opts=opts.TextStyleOpts(font_size=16),
                axislabel_opts=opts.LabelOpts(
                    font_size=12,
                    # rotate=30,  # 横轴标签倾斜30度
                    interval=0,  # 强制显示所有标签
                    overflow="breakAll"  # 标签过长时换行显示
                ),
                axisline_opts=opts.AxisLineOpts(
                    linestyle_opts=opts.LineStyleOpts(color="#333")
                )
            ),
            yaxis_opts=opts.AxisOpts(
                name="簇的数量",
                name_textstyle_opts=opts.TextStyleOpts(font_size=16),
                axislabel_opts=opts.LabelOpts(font_size=12),
                axisline_opts=opts.AxisLineOpts(
                    linestyle_opts=opts.LineStyleOpts(color="#333")
                ),
                splitline_opts=opts.SplitLineOpts(
                    linestyle_opts=opts.LineStyleOpts(color="#eee")
                )
            ),
            toolbox_opts=opts.ToolboxOpts(
                is_show=True,
                feature=opts.ToolBoxFeatureOpts(
                    save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                        type_="png",
                        pixel_ratio=1,
                        name="簇大小分布"
                    ),
                    data_view=opts.ToolBoxFeatureDataViewOpts(is_show=True),
                    data_zoom=opts.ToolBoxFeatureDataZoomOpts(is_show=True)
                )
            ),
            legend_opts=opts.LegendOpts(is_show=False)

        )
    )

    # 渲染图表
    bar.render("cluster_image_distribution.html")
    return bar


def main():
    # 配置参数
    params = {
        "json_file_path": "../../../data/FreiHand/cluster_data/new_clusters_origin_32560_0.0055_clean.json",
        "chart_title": "FreiHand数据集聚类簇大小分布"
    }

    # 分析聚类数据
    total_images, num_clusters, size_stats = count_images_from_json(params["json_file_path"])

    # 计算统计信息
    if size_stats:
        max_size = max(size_stats.keys())
        min_size = min(size_stats.keys())
        avg_size = sum(k * v for k, v in size_stats.items()) / num_clusters if num_clusters > 0 else 0

        print(f"总簇数: {num_clusters}")
        print(f"总图片数: {total_images}")
        print(f"最大簇大小: {max_size}")
        print(f"最小簇大小: {min_size}")
        print(f"平均簇大小: {avg_size:.2f}")

    # 输出详细统计
    print("\n详细统计:")
    print("簇大小\t簇数量")
    for size in sorted(size_stats.keys()):
        print(f"{size}\t{size_stats[size]}")

    # 绘制并保存图表
    plot_cluster_size_distribution(size_stats, params["chart_title"])
    print("\n图表已保存为 cluster_image_distribution.html")
    print("请在浏览器中打开该文件查看交互式图表")


if __name__ == "__main__":
    main()