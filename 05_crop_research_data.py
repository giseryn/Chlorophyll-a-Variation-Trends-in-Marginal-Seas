import os
import shutil
from osgeo import gdal, osr, ogr, gdalconst
import datetime
import numpy as np
import pandas as pd
from sklearn import linear_model
from tqdm import tqdm
import numba


def verify_the_existence_of_folder(target_data_dir):
    """
    校验文件夹是否存在如果不存在则创建一个
    :param target_data_dir: 目标文件夹名
    :return:
    """

    if not os.path.exists(target_data_dir):
        os.makedirs(target_data_dir)


def log_func(func):
    def wrapper(*args, **kwargs):
        print(f"Function Name: {func.__name__}")
        print(f"Arguments: {args}")
        print(f"Keyword Arguments: {kwargs}")
        return func(*args, **kwargs)

    return wrapper


@log_func
def fill_missing_values(source_data_dir, target_data_dir):
    def copy_tif_file(srcfile, dstpath):
        """
        复制文件到指定文件夹下
        :param srcfile: 原始文件的路径
        :param dstpath: 目标文件夹路径
        :return:
        """
        if not os.path.isfile(srcfile):
            print("%s not exist!" % srcfile)
        else:
            if not os.path.exists(dstpath):
                os.makedirs(dstpath)
            shutil.copy(srcfile, dstpath)

    def filenodata(source_data):
        input_raster = gdal.Open(source_data, 1)
        dataset = input_raster.GetRasterBand(1)

        gdal.FillNodata(
            targetBand=dataset,
            maskBand=None,
            maxSearchDist=100,
            smoothingIterations=0
        )

        return dataset

    verify_the_existence_of_folder(target_data_dir)

    files = os.listdir(source_data_dir)
    data_list = []

    for file in files:
        if '.tif' in file:
            data_list.append(file)
            copy_tif_file(source_data_dir + file, target_data_dir)

    for i in tqdm(range(len(data_list))):
        data = data_list[i]
        filenodata(target_data_dir + '/' + data)


@log_func
def resampling_tif(source_data_dir, target_data_dir, stand_resample_file_path):
    def ReprojectImages(inputfilePath, referencefilefilePath, outputfilePath):
        # 若采用gdal.Warp()方法进行重采样
        # 获取输出影像信息
        inputrasfile = gdal.Open(inputfilePath, gdal.GA_ReadOnly)
        inputProj = inputrasfile.GetProjection()
        # 获取参考影像信息
        referencefile = gdal.Open(referencefilefilePath, gdal.GA_ReadOnly)
        referencefileProj = referencefile.GetProjection()
        referencefileTrans = referencefile.GetGeoTransform()
        bandreferencefile = referencefile.GetRasterBand(1)
        x = referencefile.RasterXSize
        y = referencefile.RasterYSize
        nbands = referencefile.RasterCount
        # 创建重采样输出文件（设置投影及六参数）
        driver = gdal.GetDriverByName('GTiff')
        output = driver.Create(outputfilePath, x, y, nbands, bandreferencefile.DataType)
        output.SetGeoTransform(referencefileTrans)
        output.SetProjection(referencefileProj)
        options = gdal.WarpOptions(srcSRS=inputProj, dstSRS=referencefileProj, resampleAlg=gdalconst.GRA_Bilinear)
        gdal.Warp(output, inputfilePath, options=options)

    if not os.path.exists(target_data_dir):
        os.makedirs(target_data_dir)

    files = os.listdir(source_data_dir)

    i = 0
    for i in tqdm(range(len(files))):
        if files[i].split('.')[-1] == 'tif':
            ReprojectImages(source_data_dir + files[i], stand_resample_file_path, target_data_dir + files[i])
        i = i + 1


@log_func
def crop_region(tif_dir_path, mask_path, result_path):
    """
    根据掩膜裁剪栅格数据
    :param tif_dir_path:
    :param mask_path:
    :param result_path:
    :return:
    """

    # @numba.jit()
    def crop_raster(input_shape_path, input_raster_path, result_path, tif_dir_path):
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        output_raster = result_path + 'crop_' + input_raster_path
        # 矢量文件路径，打开矢量文件
        input_raster = gdal.Open(tif_dir_path + input_raster_path)
        # 开始裁剪，一行代码，爽的飞起
        ds = gdal.Warp(output_raster,
                       input_raster,
                       format='GTiff',
                       cropToCutline=True,
                       cutlineDSName=input_shape_path,
                       dstNodata=None)

    files = os.listdir(tif_dir_path)
    data_list = []
    for file in files:
        if 'tif' in file:
            data_list.append(file)

    for i in tqdm(range(len(data_list))):
        data = data_list[i]
        crop_raster(mask_path, data, result_path, tif_dir_path)


@log_func
def regression_chl_tif_data(source_data_dir, target_data_dir, target_dir):
    def dateRange(beginDate, endDate):
        dates = []
        dt = datetime.datetime.strptime(beginDate, "%Y%m%d")
        date = beginDate[:]
        while date <= endDate:
            dates.append(date)
            dt = dt + datetime.timedelta(1)
            date = dt.strftime("%Y%m%d")
        return dates

    def monthRange(beginDate, endDate):
        monthSet = set()
        for date in dateRange(beginDate, endDate):
            monthSet.add(date[0:6])
        monthList = []
        for month in monthSet:
            monthList.append(month)
        return sorted(monthList)

    def gdal_analysis(in_path):
        """
        读取栅格文件，批量统计栅格的文件名、计数、最小值、最大值、总和、平均值、中位数、标准差。并将统计结果保存为表格（.csv）文件。
        in_path:待统计的栅格所在的文件夹
        out_csv:生成的表格.csv文件所存放的位置
        """
        tifs = [os.path.join(in_path, i) for i in os.listdir(in_path) if i.endswith(".tif")]

        aqua_data_dict = {}
        seawifs_data_dict = {}
        aqua_data_path_dict = {}
        seawifs_data_path_dict = {}
        for in_tif in tifs:
            bname = os.path.basename(in_tif)
            fname = os.path.splitext(bname)[0]
            ftimerange = fname.split('.')[1]

            rds = gdal.Open(in_tif)  # type:gdal.Dataset
            if rds.RasterCount != 1:
                print("Warning, RasterCount > 1")
            band = rds.GetRasterBand(1)  # type:gdal.Band
            ndv = band.GetNoDataValue()  # nodata value

            # 读取栅格至大小为n*1的数组中
            values = np.array(band.ReadAsArray()).ravel()
            # 排除空值区
            values = values[values != ndv]

            if 'AQUA' in fname:
                aqua_data_dict[ftimerange] = np.mean(values)
                aqua_data_path_dict[ftimerange] = bname
            else:
                seawifs_data_dict[ftimerange] = np.mean(values)
                seawifs_data_path_dict[ftimerange] = bname

        aqua_mean_df = pd.DataFrame.from_dict(aqua_data_dict, orient='index', columns=['aqua'])
        seawifs_mean_df = pd.DataFrame.from_dict(seawifs_data_dict, orient='index', columns=['seawifs'])
        aqua_path_df = pd.DataFrame.from_dict(aqua_data_path_dict, orient='index', columns=['aqua_path'])
        seawifs_path_df = pd.DataFrame.from_dict(seawifs_data_path_dict, orient='index', columns=['seawifs_path'])
        aqua_df = pd.concat([aqua_mean_df, aqua_path_df], axis=1)
        seawifs_df = pd.concat([seawifs_mean_df, seawifs_path_df], axis=1)
        res = pd.concat([aqua_df, seawifs_df], axis=1)

        return res

    def generate_regression_data(data_df, out_csv):
        """
        计算得到每个月拟合时使用的数据集
        """
        res = []
        for i in tqdm(range(12)):
            month = i + 1
            month_aqua_list = []
            month_seawifs_list = []
            for index, row in data_df.iterrows():
                if month == int(index[4:6]) and row["aqua"] >= 0 and row["seawifs"] >= 0:
                    month_aqua_list.append(row["aqua"])
                    month_seawifs_list.append(row["seawifs"])
            # print(month_aqua_list)
            # print(month_seawifs_list)
            regression_a, regression_b, r2_score = get_generate_code(month_aqua_list, month_seawifs_list)
            temp = [month, regression_a, regression_b, r2_score]

            res.append(temp)
        res = pd.DataFrame(res)
        res.columns = ["month", "regression_a", "regression_b", "r2_score"]
        res.to_csv(out_csv)

        return res

    def get_generate_code(month_aqua_list, month_seawifs_list):
        """
        根据前置的计算结果确定需要纠正的月份
        """
        x_train = np.array(month_seawifs_list).reshape(-1, 1)
        y_train = np.array(month_aqua_list).reshape(-1, 1)
        lm = linear_model.LinearRegression()
        # 拟合模型
        lm.fit(x_train, y_train)
        y_pred = lm.predict(x_train)
        # 查看回归方程系数
        # print('Coefficients:', lm.coef_)
        # 查看回归方程截距
        # print('intercept:', lm.intercept_)
        from sklearn.metrics import r2_score

        r2_score_number = r2_score(y_train, y_pred)
        # print('r2_score:', r2_score_number)
        return lm.coef_, lm.intercept_, r2_score_number

    def make_monthly_data(source_data_dir, target_data_dir, data_df, regression_result_df):

        def read_seawifs_img(source_data_dir, img_path, regression_result_df, month):
            regression_metedata = regression_result_df.loc[month]
            """读取遥感数据信息"""
            dataset = gdal.Open(source_data_dir + img_path, 0)

            img_width = dataset.RasterXSize
            img_height = dataset.RasterYSize
            adf_GeoTransform = dataset.GetGeoTransform()
            im_Proj = dataset.GetProjection()

            img_data = np.array(dataset.ReadAsArray(0, 0, img_width, img_height), dtype=float)  # 将数据写成数组，对应栅格矩阵
            if (regression_metedata['r2_score'] > 0.5):
                img_data_ = img_data * regression_metedata['regression_a'] + regression_metedata['regression_b']
            else:
                img_data_ = img_data
            del dataset
            return img_width, img_height, adf_GeoTransform, im_Proj, img_data_

        def read_img(source_data_dir, img_path):
            """读取遥感数据信息"""
            dataset = gdal.Open(source_data_dir + img_path, 0)

            img_width = dataset.RasterXSize
            img_height = dataset.RasterYSize
            adf_GeoTransform = dataset.GetGeoTransform()
            im_Proj = dataset.GetProjection()

            img_data = np.array(dataset.ReadAsArray(0, 0, img_width, img_height), dtype=float)  # 将数据写成数组，对应栅格矩阵

            del dataset
            return img_width, img_height, adf_GeoTransform, im_Proj, img_data

        def read_both_img(source_data_dir, aqua_path, seawifs_path, regression_result_df, month):
            """读取遥感数据信息"""

            regression_metedata = regression_result_df.loc[month]
            """读取遥感数据信息"""
            aqua_dataset = gdal.Open(source_data_dir + aqua_path, 0)
            seawifs_dataset = gdal.Open(source_data_dir + seawifs_path, 0)

            img_width = seawifs_dataset.RasterXSize
            img_height = seawifs_dataset.RasterYSize
            adf_GeoTransform = seawifs_dataset.GetGeoTransform()
            im_Proj = seawifs_dataset.GetProjection()

            aqua_data = np.array(aqua_dataset.ReadAsArray(0, 0, img_width, img_height), dtype=float)
            seawifs_data = np.array(seawifs_dataset.ReadAsArray(0, 0, img_width, img_height), dtype=float)

            if (regression_metedata['r2_score'] > 0.5):
                img_data = ((seawifs_data * regression_metedata['regression_a'] + regression_metedata[
                    'regression_b']) + aqua_data) / 2
            else:
                img_data = seawifs_data
            del aqua_dataset, seawifs_dataset

            return img_width, img_height, adf_GeoTransform, im_Proj, img_data

        def arr2img(metedata, save_path):
            # 保存为jpg格式
            # plt.imsave(save_path, arr)
            # metedata  img_width, img_height, adf_GeoTransform, im_Proj, img_data

            # 保存为TIF格式
            driver = gdal.GetDriverByName("GTiff")
            datasetnew = driver.Create(save_path, metedata[0], metedata[1], 1, gdal.GDT_Float32)
            datasetnew.SetGeoTransform(metedata[2])
            datasetnew.SetProjection(metedata[3])
            band = datasetnew.GetRasterBand(1)
            band.WriteArray(metedata[4])
            datasetnew.FlushCache()  # Write to disk.必须有清除缓存

        res = []
        for i in tqdm(range(12)):
            month = i + 1
            for index, row in data_df.iterrows():
                if month == int(index[4:6]):
                    result_filename = target_data_dir + 'area_chl_' + index[0:6] + '.tif'
                    if index[0:6] == 200801:
                        print(1)
                    if row["aqua"] > 0:
                        if row["seawifs"] > 0:
                            metedata = read_both_img(source_data_dir, row["aqua_path"], row["seawifs_path"],
                                                     regression_result_df, i)
                            arr2img(metedata, result_filename)
                        else:
                            metedata = read_img(source_data_dir, row["aqua_path"])
                            arr2img(metedata, result_filename)
                    else:
                        metedata = read_seawifs_img(source_data_dir, row["seawifs_path"], regression_result_df, i)
                        arr2img(metedata, result_filename)

        return True

    if not os.path.exists(target_data_dir):
        os.makedirs(target_data_dir)

    month_list = monthRange(beginDate='19980101', endDate='20201231')

    files = os.listdir(source_data_dir)

    data_list = []
    for file in files:
        if file.split('.')[-1] == 'tif':
            data_list.append(file)

    tif_mean_df = gdal_analysis(source_data_dir)

    regression_result_df = generate_regression_data(tif_mean_df, out_csv=target_dir + '/regression_chl_result.csv')

    make_monthly_data(source_data_dir, target_data_dir, tif_mean_df, regression_result_df)
    return True


@log_func
def regression_par_tif_data(source_data_dir, target_data_dir, target_dir):
    def dateRange(beginDate, endDate):
        dates = []
        dt = datetime.datetime.strptime(beginDate, "%Y%m%d")
        date = beginDate[:]
        while date <= endDate:
            dates.append(date)
            dt = dt + datetime.timedelta(1)
            date = dt.strftime("%Y%m%d")
        return dates

    def monthRange(beginDate, endDate):
        monthSet = set()
        for date in dateRange(beginDate, endDate):
            monthSet.add(date[0:6])
        monthList = []
        for month in monthSet:
            monthList.append(month)
        return sorted(monthList)

    def gdal_analysis(in_path):
        """
        读取栅格文件，批量统计栅格的文件名、计数、最小值、最大值、总和、平均值、中位数、标准差。并将统计结果保存为表格（.csv）文件。
        in_path:待统计的栅格所在的文件夹
        out_csv:生成的表格.csv文件所存放的位置
        """
        tifs = [os.path.join(in_path, i) for i in os.listdir(in_path) if i.endswith(".tif")]

        aqua_data_dict = {}
        seawifs_data_dict = {}
        aqua_data_path_dict = {}
        seawifs_data_path_dict = {}
        for in_tif in tifs:
            bname = os.path.basename(in_tif)
            fname = os.path.splitext(bname)[0]
            ftimerange = fname.split('.')[1]

            rds = gdal.Open(in_tif)  # type:gdal.Dataset
            if rds.RasterCount != 1:
                print("Warning, RasterCount > 1")
            band = rds.GetRasterBand(1)  # type:gdal.Band
            ndv = band.GetNoDataValue()  # nodata value

            # 读取栅格至大小为n*1的数组中
            values = np.array(band.ReadAsArray()).ravel()
            # 排除空值区
            values = values[values != ndv]

            if 'AQUA' in fname:
                aqua_data_dict[ftimerange] = np.mean(values)
                aqua_data_path_dict[ftimerange] = bname
            else:
                seawifs_data_dict[ftimerange] = np.mean(values)
                seawifs_data_path_dict[ftimerange] = bname

        aqua_mean_df = pd.DataFrame.from_dict(aqua_data_dict, orient='index', columns=['aqua'])
        seawifs_mean_df = pd.DataFrame.from_dict(seawifs_data_dict, orient='index', columns=['seawifs'])
        aqua_path_df = pd.DataFrame.from_dict(aqua_data_path_dict, orient='index', columns=['aqua_path'])
        seawifs_path_df = pd.DataFrame.from_dict(seawifs_data_path_dict, orient='index', columns=['seawifs_path'])
        aqua_df = pd.concat([aqua_mean_df, aqua_path_df], axis=1)
        seawifs_df = pd.concat([seawifs_mean_df, seawifs_path_df], axis=1)
        res = pd.concat([aqua_df, seawifs_df], axis=1)

        return res

    def generate_regression_data(data_df, out_csv):
        """
        计算得到每个月拟合时使用的数据集
        """
        res = []
        for i in tqdm(range(12)):
            month = i + 1
            month_aqua_list = []
            month_seawifs_list = []
            for index, row in data_df.iterrows():
                if month == int(index[4:6]) and row["aqua"] >= 0 and row["seawifs"] >= 0:
                    month_aqua_list.append(row["aqua"])
                    month_seawifs_list.append(row["seawifs"])
            # print(month_aqua_list)
            # print(month_seawifs_list)
            regression_a, regression_b, r2_score = get_generate_code(month_aqua_list, month_seawifs_list)
            temp = [month, regression_a, regression_b, r2_score]

            res.append(temp)
        res = pd.DataFrame(res)
        res.columns = ["month", "regression_a", "regression_b", "r2_score"]
        res.to_csv(out_csv)

        return res

    def get_generate_code(month_aqua_list, month_seawifs_list):
        """
        根据前置的计算结果确定需要纠正的月份
        """
        x_train = np.array(month_seawifs_list).reshape(-1, 1)
        y_train = np.array(month_aqua_list).reshape(-1, 1)
        lm = linear_model.LinearRegression()
        # 拟合模型
        lm.fit(x_train, y_train)
        y_pred = lm.predict(x_train)
        # 查看回归方程系数
        # print('Coefficients:', lm.coef_)
        # 查看回归方程截距
        # print('intercept:', lm.intercept_)
        from sklearn.metrics import r2_score

        r2_score_number = r2_score(y_train, y_pred)
        # print('r2_score:', r2_score_number)
        return lm.coef_, lm.intercept_, r2_score_number

    def make_monthly_data(source_data_dir, target_data_dir, data_df, regression_result_df):

        def read_seawifs_img(source_data_dir, img_path, regression_result_df, month):
            regression_metedata = regression_result_df.loc[month]
            """读取遥感数据信息"""
            dataset = gdal.Open(source_data_dir + img_path, 0)

            img_width = dataset.RasterXSize
            img_height = dataset.RasterYSize
            adf_GeoTransform = dataset.GetGeoTransform()
            im_Proj = dataset.GetProjection()

            img_data = np.array(dataset.ReadAsArray(0, 0, img_width, img_height), dtype=float)  # 将数据写成数组，对应栅格矩阵
            if (regression_metedata['r2_score'] > 0.5):
                img_data_ = img_data * regression_metedata['regression_a'] + regression_metedata['regression_b']
            else:
                img_data_ = img_data
            del dataset
            return img_width, img_height, adf_GeoTransform, im_Proj, img_data_

        def read_img(source_data_dir, img_path):
            """读取遥感数据信息"""
            dataset = gdal.Open(source_data_dir + img_path, 0)

            img_width = dataset.RasterXSize
            img_height = dataset.RasterYSize
            adf_GeoTransform = dataset.GetGeoTransform()
            im_Proj = dataset.GetProjection()

            img_data = np.array(dataset.ReadAsArray(0, 0, img_width, img_height), dtype=float)  # 将数据写成数组，对应栅格矩阵

            del dataset
            return img_width, img_height, adf_GeoTransform, im_Proj, img_data

        def read_both_img(source_data_dir, aqua_path, seawifs_path, regression_result_df, month):
            """读取遥感数据信息"""

            regression_metedata = regression_result_df.loc[month]
            """读取遥感数据信息"""
            aqua_dataset = gdal.Open(source_data_dir + aqua_path, 0)
            seawifs_dataset = gdal.Open(source_data_dir + seawifs_path, 0)

            img_width = seawifs_dataset.RasterXSize
            img_height = seawifs_dataset.RasterYSize
            adf_GeoTransform = seawifs_dataset.GetGeoTransform()
            im_Proj = seawifs_dataset.GetProjection()

            aqua_data = np.array(aqua_dataset.ReadAsArray(0, 0, img_width, img_height), dtype=float)
            seawifs_data = np.array(seawifs_dataset.ReadAsArray(0, 0, img_width, img_height), dtype=float)

            if (regression_metedata['r2_score'] > 0.5):
                img_data = ((seawifs_data * regression_metedata['regression_a'] + regression_metedata[
                    'regression_b']) + aqua_data) / 2
            else:
                img_data = seawifs_data
            del aqua_dataset, seawifs_dataset

            return img_width, img_height, adf_GeoTransform, im_Proj, img_data

        def arr2img(metedata, save_path):
            # 保存为jpg格式
            # plt.imsave(save_path, arr)
            # metedata  img_width, img_height, adf_GeoTransform, im_Proj, img_data

            # 保存为TIF格式
            driver = gdal.GetDriverByName("GTiff")
            datasetnew = driver.Create(save_path, metedata[0], metedata[1], 1, gdal.GDT_Float32)
            datasetnew.SetGeoTransform(metedata[2])
            datasetnew.SetProjection(metedata[3])
            band = datasetnew.GetRasterBand(1)
            band.WriteArray(0.002 * metedata[4] + 65.5)
            datasetnew.FlushCache()  # Write to disk.必须有清除缓存

        res = []
        for i in tqdm(range(12)):
            month = i + 1
            for index, row in data_df.iterrows():
                if month == int(index[4:6]):
                    result_filename = target_data_dir + 'area_par_' + index[0:6] + '.tif'
                    if row["aqua"] > 0:
                        if row["seawifs"] > 0:
                            metedata = read_both_img(source_data_dir, row["aqua_path"], row["seawifs_path"],
                                                     regression_result_df, i)
                            arr2img(metedata, result_filename)
                        else:
                            metedata = read_img(source_data_dir, row["aqua_path"])
                            arr2img(metedata, result_filename)
                    else:
                        metedata = read_seawifs_img(source_data_dir, row["seawifs_path"], regression_result_df, i)
                        arr2img(metedata, result_filename)

        return True

    if not os.path.exists(target_data_dir):
        os.makedirs(target_data_dir)

    month_list = monthRange(beginDate='19980101', endDate='20201231')

    files = os.listdir(source_data_dir)

    data_list = []
    for file in files:
        if file.split('.')[-1] == 'tif':
            data_list.append(file)

    tif_mean_df = gdal_analysis(source_data_dir)

    regression_result_df = generate_regression_data(tif_mean_df, out_csv=target_dir + '/regression_par_result.csv')

    make_monthly_data(source_data_dir, target_data_dir, tif_mean_df, regression_result_df)
    return True


@log_func
def del_files(dir_path):
    # os.walk会得到dir_path下各个后代文件夹和其中的文件的三元组列表，顺序自内而外排列，
    # 如 log下有111文件夹，111下有222文件夹：[('D:\\log\\111\\222', [], ['22.py']), ('D:\\log\\111', ['222'], ['11.py']), ('D:\\log', ['111'], ['00.py'])]
    for root, dirs, files in os.walk(dir_path, topdown=False):
        print(root)  # 各级文件夹绝对路径
        # 第一步：删除文件
        for name in files:
            os.remove(os.path.join(root, name))  # 删除文件
        # 第二步：删除空文件夹
        for name in dirs:
            os.rmdir(os.path.join(root, name))  # 删除一个空目录


def prepar_data(area):
    mask_path = './04_' + area + '/mask/mask.shp'
    chl_source = './05_' + area + '_data/chl'
    par_source = './05_' + area + '_data/par'
    sst_source = './05_' + area + '_data/sst'
    ccmp_source = './05_' + area + '_data/ccmp'

    # chl part
    chl_source = './05_' + area + '_data/chl'
    fill_missing_values('./04_' + area + '/chl/', chl_source + '/filled_tif_data/')
    resampling_tif(chl_source + '/filled_tif_data/', chl_source + '/resampleing_tif/', './04_basic_file/sample.tif')
    crop_region(chl_source + '/resampleing_tif/', mask_path, chl_source + '/crop_no_regression_data/')
    regression_chl_tif_data(chl_source + '/crop_no_regression_data/', chl_source + '/mod_result_tif/', chl_source)
    crop_region(chl_source + '/mod_result_tif/', mask_path, chl_source + '/mod_crop_result_tif/')
    crop_region(chl_source + '/mod_crop_result_tif/', mask_path, chl_source + '/preprocess_result/')

    del_files(chl_source + '/mod_crop_result_tif/')
    del_files(chl_source + '/mod_result_tif/')
    del_files(chl_source + '/crop_no_regression_data/')
    del_files(chl_source + '/resampleing_tif/')
    del_files(chl_source + '/filled_tif_data/')

    # par part
    par_source = './05_' + area + '_data/par'
    fill_missing_values('./04_' + area + '/par/', par_source + '/filled_tif_data/')
    resampling_tif(par_source + '/filled_tif_data/', par_source + '/resampleing_tif/', './04_basic_file/sample.tif')
    crop_region(par_source + '/resampleing_tif/', mask_path, par_source + '/crop_no_regression_data/')
    regression_par_tif_data(par_source + '/crop_no_regression_data/', par_source + '/mod_result_tif/', par_source)
    crop_region(par_source + '/mod_result_tif/', mask_path, par_source + '/mod_crop_result_tif/')

    del_files(par_source + '/mod_result_tif/')
    del_files(par_source + '/crop_no_regression_data/')
    del_files(par_source + '/resampleing_tif/')
    del_files(par_source + '/filled_tif_data/')

    # # sst part
    # sst_source = './05_' + area + '_data/sst'
    # resampling_tif('./04_' + area + '/sst/', sst_source + '/resampleing_tif/', './04_basic_file/sample.tif')
    # crop_region(sst_source + '/resampleing_tif/', mask_path, sst_source + '/crop_no_regression_data/')
    #
    # del_files(sst_source + '/resampleing_tif/')
    #
    # # ccmp part
    # ccmp_source = './05_' + area + '_data/ccmp'
    # resampling_tif('./04_' + area + '/ccmp/', ccmp_source + '/resampleing_tif/', './04_basic_file/sample.tif')
    # crop_region(ccmp_source + '/resampleing_tif/', mask_path, ccmp_source + '/crop_no_regression_data/')
    #
    # del_files(ccmp_source + '/resampleing_tif/')
    #
    # crop_region(chl_source + '/mod_crop_result_tif/', mask_path, chl_source + '/preprocess_result/')
    #
    # crop_region(par_source + '/mod_crop_result_tif/', mask_path, par_source + '/preprocess_result/')
    #
    # crop_region(sst_source + '/crop_no_regression_data/', mask_path, sst_source + '/preprocess_result/')
    #
    # crop_region(ccmp_source + '/crop_no_regression_data/', mask_path, ccmp_source + '/preprocess_result/')
    #
    # del_files(chl_source + '/mod_crop_result_tif/')
    # del_files(par_source + '/mod_crop_result_tif/')
    # del_files(sst_source + '/crop_no_regression_data/')
    # del_files(ccmp_source + '/crop_no_regression_data/')


if __name__ == '__main__':
    # area_list = ['amazon', 'bohai', 'donghai', 'mexico', 'USEastCoast']
    area_list = ['Iberian']
    for area in area_list:
        prepar_data(area)
