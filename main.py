import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio as rio
from matplotlib import colors
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.patheffects import withStroke
from matplotlib.image import imread
from matplotlib.colors import ListedColormap
import os
from osgeo import gdal


def raster_pixel_proportions(raster_path, bins, output_path):
    # 打开栅格数据文件
    dataset = gdal.Open(raster_path)

    # 获取栅格数据的宽度和高度
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize

    # 获取栅格数据的地理参考信息
    transform = dataset.GetGeoTransform()

    # 获取栅格数据的第一个波段
    band = dataset.GetRasterBand(1)

    # 将栅格数据读取为numpy数组
    data = band.ReadAsArray(0, 0, cols, rows)

    # 关闭栅格数据文件
    dataset = None
    data_no_zero = np.where(data == 0, np.nan, data)
    data_filtered = data_no_zero[~np.isnan(data_no_zero)]
    # 使用numpy的digitize方法将栅格数据分到不同的区间
    digitized = np.digitize(data_filtered, bins, right=False)

    # 统计每个区间中的像素数量
    pixel_counts = np.bincount(digitized.flatten())[1:]

    # 计算每个区间中的像素比例
    total_pixels = np.sum(pixel_counts)
    pixel_proportions = pixel_counts / total_pixels

    # 将结果保存到csv文件
    bin_labels = ["High negative correlation", "moderate negative correlation", "weak negative correlation",
                  "weak positive correlation", "moderate positive correlation", "high positive correlation"]
    df = pd.DataFrame({raster_path: pixel_proportions}).T
    df.to_csv(output_path, mode='a', header=None)


def combine_images_vertically(images, output_path):
    """
    将多张PNG图像纵向组合为一张图像，并保存为PNG格式。

    参数：
    images: 包含多张PNG图像的列表。
    output_path: 组合后图像的保存路径。

    返回值：
    无返回值。
    """
    # 读取所有PNG图像
    pil_images = [Image.open(image) for image in images]

    # 获取每张图像的尺寸
    width, height = pil_images[0].size

    # 创建一个新的空白图像，用于存储组合后的图像
    combined_image = Image.new('RGB', (width, height * len(pil_images)))

    # 将所有图像按顺序拼接到新图像上
    for i, pil_image in enumerate(pil_images):
        combined_image.paste(pil_image, (0, i * height))

    # 保存组合后的图像
    combined_image.save(output_path)


def combine_images_vertically_limit(images, output_path, limit):
    """
    将多张PNG图像纵向组合为一张图像，并保存为PNG格式。

    参数：
    images: 包含多张PNG图像的列表。
    output_path: 组合后图像的保存路径。

    返回值：
    无返回值。
    """
    # 读取所有PNG图像
    pil_images = [Image.open(image) for image in images]

    # 获取每张图像的尺寸
    width, height = pil_images[0].size

    # 计算每列最多可以容纳几张图像
    max_images_per_column = limit

    # 计算组合后图像的总行数和总列数
    num_images = len(pil_images)
    num_rows = (num_images + max_images_per_column - 1) // max_images_per_column
    num_columns = min(num_images, max_images_per_column)

    # 创建一个新的空白图像，用于存储组合后的图像
    combined_image = Image.new('RGB', (num_columns * width, num_rows * height))

    # 将所有图像按顺序拼接到新图像上
    for i in range(num_rows):
        for j in range(num_columns):
            index = i * num_columns + j
            if index < num_images:
                pil_image = pil_images[index]

            else:
                pil_image = Image.new('RGB', (width, height), color='white')

            combined_image.paste(pil_image, (j * width, i * height))

    # 保存组合后的图像
    combined_image.save(output_path)


def plot_raster_images_continuous(filenames, titles, area, topics, cliprange, fontsize=20):
    """
    绘制多张栅格图像，图像按照给定的名称列表横向排列，共用一个统一的colorbar，并且
    色带的范围和图像渲染的范围均为图像像素值范围的1%~99%。绘制的图像包含经纬度边框、
    海岸线和国界线等地图特征，并且图像显示的范围从栅格数据的元数据中读取。

    :param filenames: 包含栅格数据文件名的列表
    :type filenames: list
    :param titles: 包含每张图像名称的列表
    :type titles: list
    """
    # 读取栅格数据和元数据
    data = []
    bounds = []
    tif_path = './0_cartopy_data/NE1_50M_SR_W.tif'
    for filename in filenames:
        with rio.open(filename) as src:
            data.append(src.read(1, masked=True))
            transform = src.transform
            height, width = src.height, src.width
            left, bottom = transform * (0, height)
            right, top = transform * (width, 0)
            bounds.append((left, right, bottom, top))

    vmin_max_data = np.where(np.array(data).flatten() == 0, np.nan, np.array(data).flatten())
    # 获取像素值范围的1%~99%
    vmin = np.nanpercentile(vmin_max_data, cliprange[0])
    vmax = np.nanpercentile(vmin_max_data, cliprange[1])

    # 创建图例
    fig, ax = plt.subplots(nrows=1, ncols=len(data), figsize=(30, 5),
                           subplot_kw=dict(projection=ccrs.PlateCarree()))
    for i, d in enumerate(data):
        colors_list = ['#2b83ba', "#abdda4", "#ffffbf", "#fdae61", "#d7191c"]
        self_defined_cmap = colors.LinearSegmentedColormap.from_list('custom_cmap', colors_list)

        ax[i].imshow(
            imread(tif_path),
            origin='upper',
            transform=ccrs.PlateCarree(),
            extent=[-180, 180, -90, 90]
        )
        im = ax[i].imshow(d, vmin=vmin, vmax=vmax, cmap=self_defined_cmap, extent=bounds[i],
                          transform=ccrs.PlateCarree())
        ax[i].set_title(titles[i], fontsize=fontsize)
        ax[i].coastlines()
        ax[i].add_feature(cfeature.LAND.with_scale('110m'), edgecolor='black')
        rivers = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '110m',
                                              edgecolor='blue',
                                              facecolor='none')

        # ax[i].add_feature(rivers)
        ax[i].set_extent(bounds[i])

        # 添加经纬度边框
        xlocs = np.linspace(bounds[i][0], bounds[i][1], 2)
        ylocs = np.linspace(bounds[i][2], bounds[i][3], 3)
        ax[i].set_xticks(xlocs, crs=ccrs.PlateCarree())
        ax[i].set_yticks(ylocs, crs=ccrs.PlateCarree())
        ax[i].xaxis.set_ticklabels(['{:.1f}°E'.format(x) for x in ax[i].get_xticks()], fontsize=fontsize)
        ax[i].yaxis.set_ticklabels(['{:.1f}°N'.format(y) for y in ax[i].get_yticks()], fontsize=fontsize)
        ax[i].gridlines(linestyle='--')
        ax[i].tick_params(axis='x', which='both', pad=10)



    # 添加统一的colorbar

    plt.subplots_adjust(wspace=1, hspace=0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, location='right')
    cbar.ax.set_ylabel(topics, fontsize=fontsize)
    plt.savefig('./result/' + topics + '_fig_' + area + '.png')
    # 显示图像

    plt.show()


def plot_raster_images_segmentation(filenames, titles, area, topics, cliprange, colorbounds, colorstypes, fontsize=20):
    """
    绘制多张栅格图像，图像按照给定的名称列表横向排列，共用一个统一的colorbar，并且
    色带的范围和图像渲染的范围均为图像像素值范围的1%~99%。绘制的图像包含经纬度边框、
    海岸线和国界线等地图特征，并且图像显示的范围从栅格数据的元数据中读取。

    :param filenames: 包含栅格数据文件名的列表
    :type filenames: list
    :param titles: 包含每张图像名称的列表
    :type titles: list
    """
    # 读取栅格数据和元数据
    data = []
    bounds = []
    tif_path = './0_cartopy_data/NE1_50M_SR_W.tif'
    for filename in filenames:
        with rio.open(filename) as src:
            data.append(src.read(1, masked=True))
            transform = src.transform
            height, width = src.height, src.width
            left, bottom = transform * (0, height)
            right, top = transform * (width, 0)
            bounds.append((left, right, bottom, top))

    vmin_max_data = np.where(np.array(data).flatten() == 0, np.nan, np.array(data).flatten())
    # 获取像素值范围的1%~99%
    vmin = np.nanpercentile(vmin_max_data, cliprange[0])
    vmax = np.nanpercentile(vmin_max_data, cliprange[1])

    # 创建图例
    fig, ax = plt.subplots(nrows=1, ncols=len(data), figsize=(30, 4),
                           subplot_kw=dict(projection=ccrs.PlateCarree()))
    for i, d in enumerate(data):
        cmap = ListedColormap(colorstypes)
        norm = plt.Normalize(colorbounds[0], colorbounds[-1])

        ax[i].imshow(
            imread(tif_path),
            origin='upper',
            transform=ccrs.PlateCarree(),
            extent=[-180, 180, -90, 90]
        )
        im = ax[i].imshow(d, norm=norm, cmap=cmap, extent=bounds[i],
                          transform=ccrs.PlateCarree())
        ax[i].set_title(titles[i], fontsize=fontsize)
        ax[i].coastlines()
        ax[i].add_feature(cfeature.LAND.with_scale('110m'), edgecolor='black')
        rivers = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '110m',
                                              edgecolor='blue',
                                              facecolor='none')

        # ax[i].add_feature(rivers)
        ax[i].set_extent(bounds[i])

        # 添加经纬度边框
        xlocs = np.linspace(bounds[i][0], bounds[i][1], 2)
        ylocs = np.linspace(bounds[i][2], bounds[i][3], 3)
        ax[i].set_xticks(xlocs, crs=ccrs.PlateCarree())
        ax[i].set_yticks(ylocs, crs=ccrs.PlateCarree())
        ax[i].xaxis.set_ticklabels(['{:.1f}°E'.format(x) for x in ax[i].get_xticks()], fontsize=fontsize)
        ax[i].yaxis.set_ticklabels(['{:.1f}°N'.format(y) for y in ax[i].get_yticks()], fontsize=fontsize)
        ax[i].gridlines(linestyle='--')
        ax[i].tick_params(axis='x', which='both', pad=10)

    # 添加统一的colorbar

    plt.subplots_adjust(wspace=0.25, hspace=0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, location='right', ticks=colorbounds, boundaries=colorbounds)
    cbar.ax.set_ylabel(topics, fontsize=fontsize)
    plt.savefig('./result/' + topics + '_fig_' + area + '.png')
    # 显示图像

    plt.show()


def chl_year(file_dir, area):
    path1 = file_dir + 'area_spring_chl_avg.tif'
    path2 = file_dir + 'area_summer_chl_avg.tif'
    path3 = file_dir + 'area_fall_chl_avg.tif'
    path4 = file_dir + 'area_winter_chl_avg.tif'
    path5 = file_dir + 'area_year_chl_avg.tif'
    plot_raster_images_continuous([path1, path2, path3, path4, path5],
                                  ['spring', 'summer', 'fall', 'winter', 'year'], area, 'chl', [1, 99])


def chl_slope_year(file_dir, area):
    path1 = file_dir + 'area_spring_slope.tif'
    path2 = file_dir + 'area_summer_slope.tif'
    path3 = file_dir + 'area_fall_slope.tif'
    path4 = file_dir + 'area_winter_slope.tif'
    path5 = file_dir + 'area_yearly_slope.tif'

    for file in [path1, path2, path3, path4, path5]:
        bins = [-1, 0, 1]
        output_path = './result/slope_output.csv'
        raster_pixel_proportions(file, bins, output_path)

    plot_raster_images_continuous([path1, path2, path3, path4, path5],
                                  ['spring', 'summer', 'fall', 'winter', 'year'], area, 'slope', [1, 99])


def chl_cv_year(file_dir, area):
    path1 = file_dir + 'area_spring_cv.tif'
    path2 = file_dir + 'area_summer_cv.tif'
    path3 = file_dir + 'area_fall_cv.tif'
    path4 = file_dir + 'area_winter_cv.tif'
    path5 = file_dir + 'area_yearly_cv.tif'
    colorbounds = [0,0.05,0.10,0.15,0.2]
    colortypes = ['#2b83ba', "#abdda4", "#ffffbf", "#fdae61", "#d7191c"]

    for file in [path1, path2, path3, path4, path5]:
        bins = [0,0.05,0.10,0.15,0.2]
        output_path = './result/cv_output.csv'
        raster_pixel_proportions(file, bins, output_path)

    # plot_raster_images_segmentation([path1, path2, path3, path4, path5],
    #                                 ['spring', 'summer', 'fall', 'winter', 'year'], area, 'cv', [2, 98], colorbounds,
    #                                 colortypes)

    plot_raster_images_continuous([path1, path2, path3, path4, path5],
                                  ['spring', 'summer', 'fall', 'winter', 'year'], area, 'cv', [2, 98])


def chl_sst_corr(file_dir, area):
    files = os.listdir(file_dir)
    tif_files = [file_dir + f for f in files if "Temperature" in f and f.endswith(".tif")]

    for file in tif_files:
        bins = [-1.0, -0.7, -0.4, 0, 0.4, 0.7, 1.0]
        output_path = './result/corr_output.csv'
        raster_pixel_proportions(file, bins, output_path)

    colorbounds = [-1.0, -0.7, -0.4, -0.1, 0, 0.1, 0.4, 0.7, 1.0]
    colortypes = ['#2b83ba', "#91cba8", "#ddf1b4", "#fedf99", "#f59053", "#d7191c"]

    plot_raster_images_segmentation(tif_files,['Temperature rising ','Temperature stabilisation'] , area, 'corr', [1, 99], colorbounds, colortypes)


def plot_correlation_matrix(csv_file, area, p_values=True, alpha=0.01):
    """
    绘制相关性矩阵

    :param alpha:
    :param p_values:
    :param area:
    :param csv_file: 数据文件的地址
    """
    # 读取数据
    data_t = pd.read_csv(csv_file, index_col=0)
    data = data_t.T

    # 归一化处理
    data_norm = (data - data.mean()) / data.std()

    # 计算相关系数矩阵
    corr = np.corrcoef(data_norm, rowvar=False)

    # 计算p值矩阵
    p_values = np.zeros_like(corr)
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            index1 = data_norm.columns[i]
            index2 = data_norm.columns[j]
            _, p = pearsonr(data_norm[index1], data_norm[index2])
            p_values[i, j] = p
            p_values[j, i] = p

    # 绘制相关矩阵
    sns.set(style="white")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(7, 6))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                xticklabels=data.columns, yticklabels=data.columns)

    # 在方格中添加文本标签
    for i in range(len(corr)):
        for j in range(len(corr)):
            if mask[i, j]:
                continue
            if p_values[i, j] < alpha:
                text = ax.text(j + 0.5, i + 0.5, "**", ha="center", va="center", color="w", fontsize=16,
                               path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='black')])

                text = ax.text(j + 0.5, i + 0.5, round(p_values[i, j], 3), ha="center", va="center", color="w",
                               fontsize=16,
                               path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='black')])
            else:
                if p_values[i, j] < 5 * alpha:
                    text = ax.text(j + 0.5, i + 0.5, "*", ha="center", va="center", color="w", fontsize=16,
                                   path_effects=[
                                       plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='black')])
                    text = ax.text(j + 0.5, i + 0.5, round(p_values[i, j], 3), ha="center", va="center", color="w",
                                   fontsize=16,
                                   path_effects=[
                                       plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='black')])
                else:
                    text = ax.text(j + 0.5, i + 0.5, "", ha="center", va="center", color="w")
                    text = ax.text(j + 0.5, i + 0.5, round(p_values[i, j], 3), ha="center", va="center", color="w",
                                   fontsize=16,
                                   path_effects=[
                                       plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='black')])

    plt.title(area + ' Correlation Matrix')
    plt.savefig('./result/' + area + '_Corr_Matrix.png')
    plt.show()
def plot_correlation_Scatter_plot(csv_file, area, p_values=True, alpha=0.01):
    """
    绘制相关性散点图

    :param alpha:
    :param p_values:
    :param area:
    :param csv_file: 数据文件的地址
    """
    # 读取数据
    data_t = pd.read_csv(csv_file, index_col=0)
    data = data_t.T

    # 归一化处理
    data_norm = (data - data.mean()) / data.std()

    # 计算相关系数矩阵
    corr = data_norm.corr()

    # 绘制相关散点图
    g = sns.pairplot(data_norm, kind="reg", diag_kind="kde")
    # 设置图表标题作为全图的标题
    g.fig.suptitle(area + ' Correlation Scatter Plot', y=1)
    # # 在散点图中添加文本标签
    # if p_values:
    #     p_values = corr.values
    #     for i in range(len(corr)):
    #         for j in range(i + 1, len(corr)):
    #             if p_values[i, j] < alpha:
    #                 plt.text(0.5, 0.5, "*", ha="center", va="center", color="w", fontsize=16,
    #                          path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='black')])


    plt.savefig('./result/' + area + '_Corr_Scatter_Plot.png')
    plt.show()

    return True

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    area_list = ['amazon', 'bohai', 'donghai', 'mexico', 'USEastCoast']
    # area_list = ['Iberian']

    for area in area_list:
        print(area)

        # chl_year('./06_' + area + '_result_file/', area)
        # chl_slope_year('./06_' + area + '_result_file/', area)
        # chl_cv_year('./06_' + area + '_result_file/', area)
        chl_sst_corr('./06_' + area + '_result_file/', area)

        # plot_correlation_matrix('./08_' + area + '_result_file/monthly_environmental_factors.csv', area)
        # plot_correlation_Scatter_plot('./08_' + area + '_result_file/monthly_environmental_factors.csv', area)

    # for area in area_list:
    #     print(area)
    #
    #     chl_year('./06_' + area + '_result_file/', area)


# combine_images_vertically(['./result/chl_fig_bohai.png', './result/chl_fig_mexico.png',
#                            './result/chl_fig_donghai.png', './result/chl_fig_USEastCoast.png',
#                            './result/chl_fig_amazon.png'], './chl.png')
# combine_images_vertically(['./result/slope_fig_bohai.png', './result/slope_fig_mexico.png',
#                            './result/slope_fig_donghai.png', './result/slope_fig_USEastCoast.png',
#                            './result/slope_fig_amazon.png'], './slope.png')
# combine_images_vertically(['./result/cv_fig_bohai.png', './result/cv_fig_mexico.png',
#                            './result/cv_fig_donghai.png', './result/cv_fig_USEastCoast.png',
#                            './result/cv_fig_amazon.png'], './cv.png')
#
# combine_images_vertically(['./result/corr_fig_bohai.png', './result/corr_fig_mexico.png',
#                            './result/corr_fig_donghai.png', './result/corr_fig_USEastCoast.png',
#                            './result/corr_fig_amazon.png'], './corr_fig.png')
#
# combine_images_vertically_limit(['./result/donghai_chl.png', './result/USEastCoast_chl.png',
#                            './result/amazon_chl.png','./result/bohai_chl.png', './result/mexico_chl.png'], './chl_line.png',3)

# combine_images_vertically_limit([
#                            './result/MK_test_chl_donghai.png', './result/MK_test_chl_USEastCoast.png',
#                            './result/MK_test_chl_amazon.png','./result/MK_test_chl_bohai.png', './result/MK_test_chl_mexico.png'], './chl_mk.png',3)
#
# combine_images_vertically_limit([
#                            './result/MK_test_sst_donghai.png', './result/MK_test_sst_USEastCoast.png',
#                            './result/MK_test_sst_amazon.png','./result/MK_test_sst_bohai.png', './result/MK_test_sst_mexico.png'], './sst_mk.png',3)
#
# combine_images_vertically_limit([
#                            './result/donghai_Corr_Matrix.png', './result/USEastCoast_Corr_Matrix.png',
#                            './result/amazon_Corr_Matrix.png','./result/bohai_Corr_Matrix.png', './result/mexico_Corr_Matrix.png',], './Corr_Matrix.png',3)
# combine_images_vertically_limit([
#                            './result/donghai_Corr_Scatter_Plot.png', './result/USEastCoast_Corr_Scatter_Plot.png',
#                            './result/amazon_Corr_Scatter_Plot.png','./result/bohai_Corr_Scatter_Plot.png', './result/mexico_Corr_Scatter_Plot.png',], './Corr_Scatter_Plot.png',3)

# combine_images_vertically_limit(['./result/donghai_sst.png', './result/USEastCoast_sst.png',
#                            './result/amazon_sst.png', './result/bohai_sst.png',
#                            './result/mexico_sst.png'], './sst.png',3)