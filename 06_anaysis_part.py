import datetime
import os
import numpy as np
from osgeo import gdal, osr
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn import preprocessing
import scipy.stats as stats
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from numba import jit
import pymannkendall as mk
import re
import threading

from tqdm import tqdm


def log_func(func):
    def wrapper(*args, **kwargs):
        print(f"Function Name: {func.__name__}")
        print(f"Arguments: {args}")
        print(f"Keyword Arguments: {kwargs}")
        return func(*args, **kwargs)

    return wrapper


def verify_the_existence_of_folder(target_data_dir):
    """
    校验文件夹是否存在如果不存在则创建一个
    :param target_data_dir: 目标文件夹名
    :return:
    """

    if not os.path.exists(target_data_dir):
        os.makedirs(target_data_dir)


def date_range(begin_date, end_date):
    dates = []
    dt = datetime.datetime.strptime(begin_date, "%Y%m%d")
    date = begin_date[:]
    while date <= end_date:
        dates.append(date)
        dt = dt + datetime.timedelta(1)
        date = dt.strftime("%Y%m%d")
    return dates


def month_range(begin_date, end_date):
    month_set = set()
    for date in date_range(begin_date, end_date):
        month_set.add(date[0:6])
    month_list = []
    for month in month_set:
        month_list.append(month)
    return sorted(month_list)


def year_range(begin_date, end_date):
    year_set = set()
    for date in date_range(begin_date, end_date):
        year_set.add(date[0:4])
    year_list = []
    for year in year_set:
        year_list.append(year)
    return sorted(year_list)


def gdal_analysis(in_path):
    '''
    获取内容为一维数组
    :param in_path:
    :return:
    '''
    tif_data = gdal.Open(in_path)
    band = tif_data.GetRasterBand(1)
    ndv = band.GetNoDataValue()  # nodata value
    values = np.array(band.ReadAsArray()).ravel()
    values = values[values != ndv]

    return values


def get_tif_band_value(in_path):
    '''
    获取内容为栅格数值数组
    :param in_path:
    :return:
    '''
    tif_data = gdal.Open(in_path)
    band = tif_data.GetRasterBand(1)
    ndv = band.GetNoDataValue()  # nodata value
    values = band.ReadAsArray()

    return values


def get_yearly_mean_data(input_data_dir, begin_date, end_date):
    years = year_range(begin_date, end_date)

    files = glob.glob(input_data_dir + '*.tif')
    yearly_value = []
    for year in years:
        tif_value_mean_list = []
        for file in files:
            numbers = re.findall('\d{6}', file)
            if year == numbers[0][0:4]:
                tif_value_mean = np.nanmean(gdal_analysis(file))
                tif_value_mean_list.append(tif_value_mean)
        year_mean = np.nanmean(tif_value_mean_list)
        yearly_value.append(year_mean)

    return years, yearly_value


def get_monthly_mean_data(input_data_dir):
    months = [str(m + 1).rjust(2, '0') for m in range(12)]

    files = glob.glob(input_data_dir + '*.tif')
    chl_monthly_value = []
    for month in months:
        tif_value_mean_list = []
        for file in files:
            numbers = re.findall('\d{6}', file)
            if month == numbers[0][4:6]:
                tif_value_mean = np.nanmean(gdal_analysis(file))
                tif_value_mean_list.append(tif_value_mean)
        year_mean = np.nanmean(tif_value_mean_list)
        chl_monthly_value.append(year_mean)
    return months, chl_monthly_value


def linear_regression(x_data, y_data):
    """
    线性回归，获取函数及R2
    """
    x_train = np.array(x_data).reshape(-1, 1)
    y_train = np.array(y_data).reshape(-1, 1)
    lm = linear_model.LinearRegression()
    # 拟合模型
    lm.fit(x_train, y_train)
    y_pred = lm.predict(x_train)
    # 查看回归方程系数
    print('Coefficients:', lm.coef_)
    # 查看回归方程截距
    print('intercept:', lm.intercept_)

    r2_score_number = r2_score(y_train, y_pred)
    print('r2_score:', r2_score_number)
    return x_train, y_pred, lm.coef_, lm.intercept_, r2_score_number


def print_yearly_mean_fig(input_data_dir, data_type, result_path, begin_date, end_date):
    years, yearly_value = get_yearly_mean_data(input_data_dir, begin_date, end_date)
    x_train, y_pred, linear_regression_x, linear_regression_y, r2_score_number = linear_regression(years,
                                                                                                   yearly_value)
    #
    # plt.figure(figsize=(20, 10), dpi=300)
    # plt.plot(years, yearly_value)
    # regression_data_x = [str(x[0]) for x in x_train]
    # regression_data_y = [float(y) for y in y_pred]
    # print(regression_data_y, regression_data_x)
    # plt.plot(regression_data_x, regression_data_y)
    # plt.xlabel("years")
    # plt.ylabel(data_type)
    # plt.grid(axis='y')
    # plt.savefig(result_path + 'yearly_' + data_type + '_mean.png')
    # plt.show()

    return yearly_value


def create_tif(band_data, rows, cols, geotransform, projection, new_tif_name):
    # 创建tif文件
    driver = gdal.GetDriverByName('GTiff')
    out_tif_name = new_tif_name
    out_tif = driver.Create(out_tif_name, rows, cols, 1, gdal.GDT_Float32)

    # 设置影像的显示范围
    # Lat_re前需要添加负号
    out_tif.SetGeoTransform(geotransform)

    # 定义投影
    prj = osr.SpatialReference()
    prj.ImportFromEPSG(4326)
    out_tif.SetProjection(prj.ExportToWkt())
    data_array = band_data

    # 数据导出

    out_tif.GetRasterBand(1).WriteArray(data_array)  # 将数据写入内存，此时没有写入到硬盘
    out_tif.GetRasterBand(1).SetNoDataValue(0)
    out_tif.FlushCache()  # 将数据写入到硬盘
    out_tif = None  # 关闭tif文件


@log_func
def print_year_chl_mean_tif(input_data_dir, result_path, chl_value_array=None):
    files = glob.glob(input_data_dir + '*.tif')
    for i in tqdm(range(len(files))):
        if i == 0:
            chl_value_array = get_tif_band_value(files[i])
        else:
            chl_value_array += get_tif_band_value(files[i])

    tif_dataset = gdal.Open(files[0])

    img_width = tif_dataset.RasterXSize
    img_height = tif_dataset.RasterYSize
    geotransform = tif_dataset.GetGeoTransform()
    proj = tif_dataset.GetProjection()
    new_tif_name = result_path + 'area_year_chl_avg.tif'
    create_tif(chl_value_array / len(files), img_width, img_height, geotransform, proj, new_tif_name)


def create_year_mean_tif(input_data_dir, result_path, begin_date, end_date, type, chl_value_array=None, ):
    files = glob.glob(input_data_dir + '*.tif')
    years = year_range(begin_date, end_date)
    for year in years:
        for file in files:
            numbers = re.findall('\d{6}', file)
            index = 0
            if year == numbers[0][0:4]:
                if index == 0:
                    chl_value_array = get_tif_band_value(file)
                    index += 1
                else:
                    chl_value_array += get_tif_band_value(file)
                    index += 1

        tif_dataset = gdal.Open(files[0])

        img_width = tif_dataset.RasterXSize
        img_height = tif_dataset.RasterYSize
        geotransform = tif_dataset.GetGeoTransform()
        proj = tif_dataset.GetProjection()
        new_tif_name = result_path + year + '_' + type + '_avg.tif'
        create_tif(chl_value_array / len(files), img_width, img_height, geotransform, proj, new_tif_name)


@log_func
def print_monthly_mean_tif(input_data_dir, result_path, type):
    months = [str(m + 1).rjust(2, '0') for m in range(12)]
    files = glob.glob(input_data_dir + '*.tif')

    tif_dataset = gdal.Open(files[0])

    img_width = tif_dataset.RasterXSize
    img_height = tif_dataset.RasterYSize

    for m in tqdm(range(len(months))):

        month_file_list = []
        chl_value_array = np.zeros((img_height, img_width), dtype=np.float)

        for file in files:
            numbers = re.findall('\d{6}', file)
            if numbers[0][4:6] == months[m]:
                month_file_list.append(file)

        for i in range(len(month_file_list)):
            # print(month_file_list[i], np.nanmean(gdal_analysis(month_file_list[i])))
            chl_value_array += get_tif_band_value(month_file_list[i])

        tif_dataset = gdal.Open(month_file_list[0])

        img_width = tif_dataset.RasterXSize
        img_height = tif_dataset.RasterYSize
        geotransform = tif_dataset.GetGeoTransform()
        proj = tif_dataset.GetProjection()
        new_tif_name = result_path + 'area_' + months[m] + '_' + type + '_avg.tif'
        create_tif(chl_value_array / len(month_file_list), img_width, img_height, geotransform, proj, new_tif_name)


def get_yearly_tif_data_array(input_data_dir, begin_date, end_date, tif_value=None):
    tif_value_tif_array = []
    years = year_range(begin_date, end_date)
    files = glob.glob(input_data_dir + '*.tif')
    for i in tqdm(range(len(years))):
        file_list = []
        for file in files:
            numbers = re.findall('\d{6}', file)
            if years[i] == numbers[0][0:4]:
                file_list.append(file)

        for i in range(len(file_list)):
            if i == 0:
                tif_value = get_tif_band_value(file_list[i])
            else:
                tif_value += get_tif_band_value(file_list[i])
        tif_value_tif_array.append(tif_value / len(file_list))
    return tif_value_tif_array


def get_slope_and_cv_trend_band_value_by_pixel(yearly_tif_data_array):
    def calculate_slope_trend(series_list):

        # 计算公式中需要用到的参数
        n = len(series_list)
        xi = np.array(range(1, n + 1))
        data_array = np.array(series_list)

        # 按照公式计算坡度
        slope_result = (n * np.sum(xi * data_array) - np.sum(xi) * np.sum(data_array)) / (
                n * np.sum(xi ** 2) - np.sum(xi) ** 2)

        return slope_result

    def calculate_coefficient_value(series_list):
        mean = np.mean(series_list)  # 计算平均值
        std = np.std(series_list, ddof=0)  # 样本标准差
        if mean == 0:
            cv = 0
        else:
            cv = std / mean
        return cv

    def remove_outliers(data):
        mean = np.mean(data)
        std = np.std(data)
        threshold_upper = mean + (3 * std)
        threshold_lower = mean - (3 * std)
        filtered_data = [value for value in data if threshold_lower <= value <= threshold_upper]
        return filtered_data
    # def rs_analysis(data):
    #     """
    #     R/S 分析检测时间序列数据自相关性
    #     :param data: 时间序列数据
    #     :return: R/S 比值
    #     """
    #
    #     # 计算时间序列数据的平均值和标准差
    #     mean = np.mean(data)
    #     std = np.std(data)
    #
    #     # 计算每个时间尺度下的 R/S 值
    #     rs = []
    #     for i in range(1, len(data)):
    #         # 将数据按照时间尺度 i 分割为 num_seg 个子序列
    #         num_seg = int(np.floor(len(data) / i))
    #         data_seg = np.array_split(data[:i * num_seg], num_seg)
    #
    #         # 计算每个子序列的均值和标准差
    #         means = [np.mean(seg) for seg in data_seg]
    #         stds = [np.std(seg) for seg in data_seg]
    #
    #         # 计算每个子序列的 R/S 值
    #         rs_i = [(np.max(seg) - np.min(seg)) / std for seg, std in zip(data_seg, stds)]
    #         rs.append(np.mean(rs_i))
    #
    #     # 计算 R/S 比值
    #     rs_ratio = np.cumsum(rs) / np.arange(1, len(rs) + 1)
    #
    #     return rs_ratio[-1]

    width = len(yearly_tif_data_array[0])
    height = len(yearly_tif_data_array[0][0])

    slope_tif_array = np.zeros((width, height))
    coefficient_tif_array = np.zeros((width, height))
    for w in tqdm(range(width)):
        for h in range(height):
            yearly_pixel_values = []
            for i in range(len(yearly_tif_data_array)):
                yearly_pixel_values.append(yearly_tif_data_array[i][w][h])

            handled_list=remove_outliers(yearly_pixel_values)

            pixel_slope_value = calculate_slope_trend(handled_list)
            pixel_cv_value = calculate_coefficient_value(handled_list)
            # rs_analysis_value = rs_analysis(yearly_pixel_values)

            slope_tif_array[w][h] = pixel_slope_value
            coefficient_tif_array[w][h] = pixel_cv_value

    return slope_tif_array, coefficient_tif_array


@log_func
def print_year_chl_slop_tif(input_data_dir, result_path, begin_date, end_date, tif_value=None):
    yearly_tif_data_array = get_yearly_tif_data_array(input_data_dir, begin_date, end_date)
    slope_tif_year_array, cv_tif_year_array = get_slope_and_cv_trend_band_value_by_pixel(yearly_tif_data_array)

    files = glob.glob(input_data_dir + '*.tif')
    tif_dataset = gdal.Open(files[0])

    img_width = tif_dataset.RasterXSize
    img_height = tif_dataset.RasterYSize
    geotransform = tif_dataset.GetGeoTransform()
    proj = tif_dataset.GetProjection()
    new_slope_tif_name = result_path + 'area_yearly_slope.tif'
    new_cv_tif_name = result_path + 'area_yearly_cv.tif'
    create_tif(slope_tif_year_array, img_width, img_height, geotransform, proj, new_slope_tif_name)
    create_tif(cv_tif_year_array, img_width, img_height, geotransform, proj, new_cv_tif_name)

    return True


@log_func
def print_seasonal_chl_slop_tif(input_data_dir, result_path, begin_date, end_date, tif_value=None):
    def get_seasonal_tif_data_array(input_data_dir, begin_date, end_date, season_month, tif_value=None):
        tif_value_tif_array = []
        years = year_range(begin_date, end_date)
        files = glob.glob(input_data_dir + '*.tif')
        for i in tqdm(range(len(years))):
            file_list = []
            for file in files:
                numbers = re.findall('\d{6}', file)
                if years[i] == numbers[0][0:4] and numbers[0][4:6] in season_month:
                    file_list.append(file)

            for i in range(len(file_list)):
                if i == 0:
                    tif_value = get_tif_band_value(file_list[i])
                else:
                    tif_value += get_tif_band_value(file_list[i])
            tif_value_tif_array.append(tif_value / len(file_list))
        return tif_value_tif_array

    season_list = ['spring', 'summer', 'fall', 'winter']
    season_month_list = [['03', '04', '05'], ['06', '07', '08'], ['09', '10', '11'], ['12', '01', '02']]

    files = glob.glob(input_data_dir + '*.tif')
    tif_dataset = gdal.Open(files[0])

    img_width = tif_dataset.RasterXSize
    img_height = tif_dataset.RasterYSize
    geotransform = tif_dataset.GetGeoTransform()
    proj = tif_dataset.GetProjection()

    for i in range(len(season_list)):
        seasonal_tif_data_array = get_seasonal_tif_data_array(input_data_dir, begin_date, end_date,
                                                              season_month_list[i])
        slope_tif_season_array, cv_tif_season_array = get_slope_and_cv_trend_band_value_by_pixel(
            seasonal_tif_data_array)

        new_slope_tif_name = result_path + 'area_' + season_list[i] + '_slope.tif'
        new_cv_tif_name = result_path + 'area_' + season_list[i] + '_cv.tif'
        create_tif(slope_tif_season_array, img_width, img_height, geotransform, proj, new_slope_tif_name)
        create_tif(cv_tif_season_array, img_width, img_height, geotransform, proj, new_cv_tif_name)


@log_func
def get_the_correlation_matrix(statistics_file_path):
    def get_pearsonr_value(list_a, list_b, datatype):
        r, p = stats.pearsonr(list_a, list_b)  # 相关系数和P值
        print(datatype + '的相关系数r为 = %6.3f，p值为 = %6.3f' % (r, p))

    data = pd.read_csv(statistics_file_path, sep=',', header='infer', index_col=0)
    chl_list = data.loc['chl']
    sst_list = data.loc['sst']
    par_list = data.loc['par']
    wind_list = data.loc['wind']

    # 相关性分析
    get_pearsonr_value(chl_list, par_list, 'par')
    get_pearsonr_value(chl_list, sst_list, 'sst')
    get_pearsonr_value(chl_list, wind_list, 'wind')

    # 主成分分析
    factors_list = [chl_list, par_list, sst_list, wind_list]
    column_lst = ['chl', 'par', 'sst', 'wind']

    # 计算列表两两间的相关系数
    data_dict = {}  # 创建数据字典，为生成Dataframe做准备
    for col, gf_lst in zip(column_lst, factors_list):
        data_dict[col] = gf_lst

    unstrtf_df = pd.DataFrame(data_dict)
    cor1 = unstrtf_df.corr()  # 计算相关系数，得到一个矩阵
    print(cor1)
    print(unstrtf_df.columns.tolist())


@log_func
def get_environmental_factors_line_fig(chl_data_path, par_data_path, sst_data_path, ccmp_data_path, result_path,
                                       begin_date, end_date):
    chl_list = print_yearly_mean_fig(chl_data_path, 'chl', result_path, begin_date=begin_date, end_date=end_date)
    par_list = print_yearly_mean_fig(par_data_path, 'par', result_path, begin_date=begin_date, end_date=end_date)
    sst_list = print_yearly_mean_fig(sst_data_path, 'sst', result_path, begin_date=begin_date, end_date=end_date)
    wind_list = print_yearly_mean_fig(ccmp_data_path, 'wind', result_path, begin_date=begin_date, end_date=end_date)

    data_list = [chl_list, par_list, sst_list, wind_list]
    column_lst = ['chl', 'par', 'sst', 'wind']

    year_list = year_range(begin_date='19980101', end_date='20201231')
    year_mean_data = pd.DataFrame(data=data_list, columns=year_list,
                                  index=column_lst)  # 数据有四列，列名分别为'chl', 'par', 'sst', 'wind'
    year_mean_data.to_csv(result_path + 'environmental_factors.csv', encoding='gbk')


@log_func
def get_monthly_environmental_factors_line_fig(chl_data_path, par_data_path, sst_data_path, ccmp_data_path, result_path,
                                               begin_date, end_date):
    # 绘制叶绿素浓度在每个月之间的浓度变化趋势图

    months, chl_yearly_value = get_monthly_mean_data(chl_data_path)
    months, par_yearly_value = get_monthly_mean_data(par_data_path)
    months, sst_yearly_value = get_monthly_mean_data(sst_data_path)
    months, ccmp_yearly_value = get_monthly_mean_data(ccmp_data_path)

    data_list = [chl_yearly_value, par_yearly_value, sst_yearly_value, ccmp_yearly_value]
    column_lst = ['chl', 'par', 'sst', 'wind']

    month_list = months
    month_mean_data = pd.DataFrame(data=data_list, columns=month_list,
                                   index=column_lst)  # 数据有四列，列名分别为'chl', 'par', 'sst', 'wind'
    month_mean_data.to_csv(result_path + 'monthly_environmental_factors.csv', encoding='gbk')


# @log_func
def calculate_factors_pca(statistics_file_path, area):
    def get_bartlett_sphericity_value(data):
        chi_square_value, p_value = calculate_bartlett_sphericity(data)
        print('bartlett', chi_square_value, p_value)
        return chi_square_value, p_value

    def get_kmo_value(data):
        kmo_all, kmo_model = calculate_kmo(data)
        print('kmo', kmo_model)
        return kmo_all

    data = pd.read_csv(statistics_file_path, sep=',', header='infer', index_col=0)

    analyse_data = data[1:].T

    chi_square_value, p_value = get_bartlett_sphericity_value(analyse_data)
    kmo_all = get_kmo_value(analyse_data)

    pca = PCA(n_components=2)
    reduce = pca.fit_transform(analyse_data)  # 进行降维

    # print('各维度方差：', pca.explained_variance_)  # 方差贡献绝对值
    print('各成分贡献率：', pca.explained_variance_ratio_)  # 各成分方差贡献占比
    print('降维后维度数量：', pca.n_components_)

    # print(analyse_data.columns, analyse_data.index)

    pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    np.set_printoptions(precision=3)
    print('初始载荷矩阵：\n', pca_loadings)
    print('-------------------' + area + '---------------------')


def get_yearly_tif_value(files, years, value_array=None):
    yearly_tif_value_list = []

    for year in years:
        years_file = []
        for file in files:
            numbers = re.findall('\d{6}', file)
            if year == numbers[0][0:4]:
                years_file.append(file)

        for i in range(len(years_file)):
            if i == 0:
                value_array = None
                value_array = get_tif_band_value(years_file[i])
            else:
                value_array += get_tif_band_value(years_file[i])

        yearly_tif_value_list.append(value_array)

    return yearly_tif_value_list

def compute_correlation(row, col, arr1, arr2, src_nodata=None):
    if src_nodata is None or arr1[0, row, col] != src_nodata:
        x1 = np.array(arr1)[:, row, col]
        x2 = np.array(arr2)[:, row, col]
        corr = np.corrcoef(x1, x2)[0, 1]
        return row, col, corr
    else:
        return row, col, np.nan

def compute_tif_correlations(arr1, arr2, src_nodata=None):
    """
    计算相关系数，c为通道数，h为行数，w为列数
    :param arr1: 影像1的数据，np数组，shape为[c,h,w]
    :param arr2: 影像2的数据，np数组，shape为[c,h,w]
    :param src_nodta: 忽略值，数字
    :return: 相关系数图像，np数组，shape为[h,w]
    """
    band1 = arr1[0]
    out = band1 * 0
    rows, cols = out.shape
    for row in tqdm(range(rows)):
        for col in range(cols):
            if src_nodata is None:
                x1 = np.array(arr1)[:, row, col]
                x2 = np.array(arr2)[:, row, col]
                corr = np.corrcoef(x1, x2)[0, 1]
                out[row, col] = corr
            else:
                if band1[row, col] != src_nodata:
                    x1 = np.array(arr1)[:, row, col]
                    x2 = np.array(arr2)[:, row, col]
                    corr = np.corrcoef(x1, x2)[0, 1]
                    out[row, col] = corr
    return out


def get_correlations_between_sst_and_chlorophyll(sst_files_dir, chl_files_dir, result_path, data_type,period_name,
                                                 begin_date='19980101', end_date='20201231'):
    years = year_range(begin_date, end_date)

    sst_files = glob.glob(sst_files_dir + '*.tif')
    chl_files = glob.glob(chl_files_dir + '*.tif')

    sst_array = get_yearly_tif_value(sst_files, years)
    chl_array = get_yearly_tif_value(chl_files, years)

    correlations_tif_value_array = compute_tif_correlations(sst_array, chl_array, 0)

    tif_dataset = gdal.Open(sst_files[0])

    img_width = tif_dataset.RasterXSize
    img_height = tif_dataset.RasterYSize
    geotransform = tif_dataset.GetGeoTransform()
    proj = tif_dataset.GetProjection()
    new_tif_name = result_path + period_name +'.tif'
    create_tif(correlations_tif_value_array, img_width, img_height, geotransform, proj, new_tif_name)


def mk_analyse(file_path, result_path, area, type, fontsize=16):
    df = pd.read_csv(file_path, sep=',', header='infer', index_col=0).T

    # 获取数据
    x = df.index
    y = df[type]
    n = len(y)

    # 正序计算
    # 定义累计量序列Sk，长度n，初始值为0
    Sk = np.zeros(n)
    UFk = np.zeros(n)

    # 定义Sk序列元素s
    s = 0
    for i in range(1, n):
        for j in range(0, i):
            if y.iloc[i] > y.iloc[j]:
                s += 1

        Sk[i] = s
        E = (i + 1) * (i / 4)
        Var = (i + 1) * i * (2 * (i + 1) + 5) / 72
        UFk[i] = (Sk[i] - E) / np.sqrt(Var)

    # 逆序计算
    # 定义逆累计量序列Sk2
    y2 = np.zeros(n)
    Sk2 = np.zeros(n)
    UBk = np.zeros(n)

    s = 0
    y2 = y[::-1]

    for i in range(1, n):
        for j in range(0, i):
            if y2.iloc[i] > y2.iloc[j]:
                s += 1
        Sk2[i] = s
        E = (i + 1) * (i / 4)
        Var = (i + 1) * i * (2 * (i + 1) + 5) / 72
        UBk[i] = -(Sk2[i] - E) / np.sqrt(Var)
    UBk2 = UBk[::-1]

    # 字体设置
    plt.rcParams['font.family'] = ['MicroSoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 画图
    plt.figure(figsize=(20, 10), dpi=300)
    plt.plot(range(n, ), UFk, label='UF', color='red', marker='s')
    plt.plot(range(n, ), UBk2, label='UB', color='blue', linestyle='--', marker='o')
    plt.ylabel('Mann-Kendall test value', fontsize=fontsize)
    plt.xlabel('Year', fontsize=fontsize)

    # 添加辅助线
    x_lim = plt.xlim()

    # 添加显著水平线和y=0
    plt.plot(x_lim, [-1.96, -1.96], ':', color='black', label='5% significant level')
    plt.plot(x_lim, [0, 0], '--', color='black')
    plt.plot(x_lim, [1.96, 1.96], ':', color='black')
    plt.xticks(range(n), x.tolist(), fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # plt.legend(loc='upper right', bbox_to_anchor=(0.9,0.95),ncol=3,fancybox=True)

    # 设置图例位置，第一个参数调整左右位置，第二个参数调整上下位置
    plt.legend(bbox_to_anchor=(0.8, 0.8), facecolor='w', frameon=False, fontsize=fontsize)

    # 添加文本注释
    plt.text(1.7, 5, 'mutagenicity testing', fontsize=20)
    plt.savefig("./result/Mk_test_" + type + '_' + area + ".svg",dpi=300,format="svg")

    result = mk.original_test(y, alpha=0.05)  # alpha默认为0.05
    print(result)


def print_seasonal_chl_mean_tif(input_data_dir, result_path, begin_date, end_date, tif_value=None):
    def get_seasonal_tif_data_array(input_data_dir, begin_date, end_date, season_month, tif_value=None):
        tif_value_tif_array = []
        years = year_range(begin_date, end_date)
        files = glob.glob(input_data_dir + '*.tif')
        for i in tqdm(range(len(years))):
            file_list = []
            for file in files:
                numbers = re.findall('\d{6}', file)
                if years[i] == numbers[0][0:4] and numbers[0][4:6] in season_month:
                    file_list.append(file)

            for i in range(len(file_list)):
                if i == 0:
                    tif_value = get_tif_band_value(file_list[i])
                else:
                    tif_value += get_tif_band_value(file_list[i])
            tif_value_tif_array.append(tif_value / len(file_list))
        return tif_value_tif_array

    season_list = ['spring', 'summer', 'fall', 'winter']
    season_month_list = [['03', '04', '05'], ['06', '07', '08'], ['09', '10', '11'], ['12', '01', '02']]

    files = glob.glob(input_data_dir + '*.tif')
    tif_dataset = gdal.Open(files[0])

    img_width = tif_dataset.RasterXSize
    img_height = tif_dataset.RasterYSize
    geotransform = tif_dataset.GetGeoTransform()
    proj = tif_dataset.GetProjection()

    year_list = year_range(begin_date, end_date, )

    four_season_chl_list = []
    for i in range(len(season_list)):
        seasonal_tif_data_array = get_seasonal_tif_data_array(input_data_dir, begin_date, end_date,
                                                              season_month_list[i])
        season_yearly_chl_list = []
        season_tif_data = seasonal_tif_data_array[0]

        for year_chl_index in range(len(seasonal_tif_data_array)):
            values = np.array(seasonal_tif_data_array[year_chl_index]).ravel()
            tif_value_mean = np.nanmean(values)
            season_yearly_chl_list.append(tif_value_mean)
            if year_chl_index == 0:
                tif_value = seasonal_tif_data_array[year_chl_index]
            else:
                tif_value += seasonal_tif_data_array[year_chl_index]

        new_tif_name = result_path + 'area_' + season_list[i] + '_chl_avg.tif'
        create_tif(tif_value / len(seasonal_tif_data_array), img_width, img_height, geotransform, proj, new_tif_name)

        four_season_chl_list.append(season_yearly_chl_list)

    four_season_chl_df = pd.DataFrame(data=four_season_chl_list, columns=year_list, index=season_list)
    four_season_chl_df.to_csv(result_path + 'seasonly_month_chl.csv', encoding='gbk')


def analysis_data(area, time_list):
    chl_data_path = './05_' + area + '_data/chl/preprocess_result/'
    par_data_path = './05_' + area + '_data/par/preprocess_result/'
    sst_data_path = './05_' + area + '_data/sst/preprocess_result/'
    ccmp_data_path = './05_' + area + '_data/ccmp/preprocess_result/'
    result_path = './06_' + area + '_result_file/'
    result_path2 = './08_' + area + '_result_file/'
    #
    verify_the_existence_of_folder(result_path)
    verify_the_existence_of_folder(result_path2)

    # create_year_mean_tif(chl_data_path, result_path, begin_date='19980101', end_date='20201231', type='chl')
    # create_year_mean_tif(par_data_path, result_path, begin_date='19980101', end_date='20201231', type='par')
    # create_year_mean_tif(sst_data_path, result_path, begin_date='19980101', end_date='20201231', type='sst')
    # create_year_mean_tif(ccmp_data_path, result_path, begin_date='19980101', end_date='20201231', type='ccmp')
    # # # 绘制年度的叶绿素浓度变化趋势折线图
    # get_environmental_factors_line_fig(chl_data_path, par_data_path, sst_data_path, ccmp_data_path, result_path,
    #                                    begin_date='19980101', end_date='20201231')
    # get_monthly_environmental_factors_line_fig(chl_data_path, par_data_path, sst_data_path, ccmp_data_path, result_path,
    #                                            begin_date='19980101', end_date='20201231')
    # # # 绘制年度平均的叶绿素浓度图
    # print_year_chl_mean_tif(chl_data_path, result_path)
    # #
    # # # # 绘制逐月的叶绿素浓度图
    # print_monthly_mean_tif(chl_data_path, result_path, 'chl')
    # print_monthly_mean_tif(par_data_path, result_path, 'par')
    # print_monthly_mean_tif(sst_data_path, result_path, 'sst')
    # print_monthly_mean_tif(ccmp_data_path, result_path, 'ccmp')
    # #
    # # # 绘制逐季度的叶绿素浓度图
    # print_seasonal_chl_mean_tif(chl_data_path, result_path, begin_date='19980101', end_date='20201231')
    # #
    # # 绘制逐像元的年度的slope&cv变化趋势
    # print_year_chl_slop_tif(chl_data_path, result_path, begin_date='19980101', end_date='20201231')
    # #
    # # 分季度绘制逐像元的slope&cv变化趋势
    # print_seasonal_chl_slop_tif(chl_data_path, result_path, begin_date='19980101', end_date='20201231')

    # # 获取各环境要素的相关性矩阵和r2
    # get_the_correlation_matrix(result_path + 'monthly_environmental_factors.csv')
    # #
    # # 各要素的主成分分析及参数检验
    # calculate_factors_pca(result_path + 'monthly_environmental_factors.csv', area)
    # # 获取各环境要素的相关性矩阵和r2
    # get_the_correlation_matrix(result_path2 + area+'_monthly_environmental_factors.csv')
    # # #
    # # # 各要素的主成分分析及参数检验
    # calculate_factors_pca(result_path2 + area+'_monthly_environmental_factors.csv', area)

    # # # 根据时间间隔获取海洋温度和叶绿素浓度的关系
    get_correlations_between_sst_and_chlorophyll(sst_data_path, chl_data_path, result_path, 'sst',area+' Temperature stabilisation',
                                                 begin_date=time_list[0],
                                                 end_date=time_list[1])
    get_correlations_between_sst_and_chlorophyll(sst_data_path, chl_data_path, result_path, 'sst',area+' Temperature rising ',
                                                 begin_date=time_list[2],
                                                 end_date=time_list[3])
    # get_correlations_between_sst_and_chlorophyll(ccmp_data_path, chl_data_path, result_path, 'ccmp')
    # get_correlations_between_sst_and_chlorophyll(par_data_path, chl_data_path, result_path, 'par')
    # get_correlations_between_sst_and_chlorophyll(sst_data_path, chl_data_path, result_path, 'sst')

    #
    # M-K分析
    mk_analyse(result_path + 'environmental_factors.csv', result_path, area, 'sst')
    # mk_analyse(result_path + 'environmental_factors.csv', result_path, area, 'chl')


if __name__ == '__main__':
    # area_list = ['donghai','USEastCoast','amazon', 'bohai', 'mexico']
    # time_list = [['20040101', '20071231', '20140101', '20171231'],
    #              ['20040101', '20071231', '20130101', '20161231'],
    #              ['20060101', '20091231', '20020101', '20051231'],
    #              ['19990101', '20021231', '20100101', '20131231'],
    #              ['20170101', '20201231', '20140101', '20171231']]
    area_list = ['USEastCoast']
    time_list = [['20040101', '20071231', '20130101', '20161231']]

    threads = []
    for i in range(len(area_list)):
        thread = threading.Thread(target=analysis_data, args=(area_list[i], time_list[i]))
        thread.start()
        threads.append(thread)

    # 等待所有线程执行完毕
    for thread in threads:
        thread.join()

    print("All tasks completed.")

