import netCDF4 as nc
import glob
import xarray as xr
import os
from osgeo import gdal, osr
import numpy as np


def par_nc2tif(data, target_data_dir):
    tmp_data = nc.Dataset(data)  # 利用.Dataset()方法读取nc数据
    # print(tmp_data)

    Lat_data = tmp_data.variables['lat'][:]
    Lon_data = tmp_data.variables['lon'][:]
    # print(Lat_data)
    # [58.63129069 58.62295736 58.61462403 ... 15.77295736 15.76462403
    #  15.75629069]
    # print(Lon_data)
    # [ 71.29005534  71.29838867  71.30672201 ... 136.67338867 136.68172201
    #  136.69005534]

    tmp_arr = np.asarray(tmp_data.variables['par'])

    # 影像的左上角&右下角坐标
    Lonmin, Latmax, Lonmax, Latmin = [Lon_data.min(), Lat_data.max(), Lon_data.max(), Lat_data.min()]
    # Lonmin, Latmax, Lonmax, Latmin
    # (71.29005533854166, 58.63129069182766, 136.6900553385789, 15.756290691830095)

    # 分辨率计算
    Num_lat = len(Lat_data)  # 5146
    Num_lon = len(Lon_data)  # 7849
    Lat_res = (Latmax - Latmin) / (float(Num_lat) - 1)
    Lon_res = (Lonmax - Lonmin) / (float(Num_lon) - 1)
    # print(Num_lat, Num_lon)
    # print(Lat_res, Lon_res)
    # 5146 7849
    # 0.00833333333333286 0.008333333333338078

    # i=0,1,2,3,4,5,6,7,8,9,...
    # 创建tif文件
    driver = gdal.GetDriverByName('GTiff')
    out_tif_name = data.split('\\')[-1].replace('.nc', '.tif')
    out_tif = driver.Create(target_data_dir + out_tif_name, Num_lon, Num_lat, 1, gdal.GDT_Float32)

    # 设置影像的显示范围
    # Lat_re前需要添加负号
    geotransform = (Lonmin, Lon_res, 0.0, Latmax, 0.0, -Lat_res)
    out_tif.SetGeoTransform(geotransform)

    # 定义投影
    prj = osr.SpatialReference()
    prj.ImportFromEPSG(4326)
    out_tif.SetProjection(prj.ExportToWkt())
    data_array = tmp_arr

    # 数据导出

    out_tif.GetRasterBand(1).WriteArray(data_array)  # 将数据写入内存，此时没有写入到硬盘
    out_tif.GetRasterBand(1).SetNoDataValue(-32767)
    out_tif.FlushCache()  # 将数据写入到硬盘
    out_tif = None  # 关闭tif文件


def par_convert_nc_to_tif(source_data_dir, target_data_dir):
    if not os.path.exists(target_data_dir):
        os.makedirs(target_data_dir)

    data_list = glob.glob(source_data_dir + "/" + '*par.nc')

    for i in range(len(data_list)):
        data = data_list[i]
        par_nc2tif(data, target_data_dir)
        print('par nc to tif :' + str(i + 1) + '/' + str(len(data_list)))


def chl_nc2tif(data, target_data_dir):
    tmp_data = nc.Dataset(data)  # 利用.Dataset()方法读取nc数据
    # print(tmp_data)

    Lat_data = tmp_data.variables['lat'][:]
    Lon_data = tmp_data.variables['lon'][:]
    # print(Lat_data)
    # [58.63129069 58.62295736 58.61462403 ... 15.77295736 15.76462403
    #  15.75629069]
    # print(Lon_data)
    # [ 71.29005534  71.29838867  71.30672201 ... 136.67338867 136.68172201
    #  136.69005534]

    tmp_arr = np.asarray(tmp_data.variables['chlor_a'])

    # 影像的左上角&右下角坐标
    Lonmin, Latmax, Lonmax, Latmin = [Lon_data.min(), Lat_data.max(), Lon_data.max(), Lat_data.min()]
    # Lonmin, Latmax, Lonmax, Latmin
    # (71.29005533854166, 58.63129069182766, 136.6900553385789, 15.756290691830095)

    # 分辨率计算
    Num_lat = len(Lat_data)  # 5146
    Num_lon = len(Lon_data)  # 7849
    Lat_res = (Latmax - Latmin) / (float(Num_lat) - 1)
    Lon_res = (Lonmax - Lonmin) / (float(Num_lon) - 1)
    # print(Num_lat, Num_lon)
    # print(Lat_res, Lon_res)
    # 5146 7849
    # 0.00833333333333286 0.008333333333338078

    # i=0,1,2,3,4,5,6,7,8,9,...
    # 创建tif文件
    driver = gdal.GetDriverByName('GTiff')
    out_tif_name = data.split('\\')[-1].replace('.nc', '.tif')
    out_tif = driver.Create(target_data_dir + out_tif_name, Num_lon, Num_lat, 1, gdal.GDT_Float32)

    # 设置影像的显示范围
    # Lat_re前需要添加负号
    geotransform = (Lonmin, Lon_res, 0.0, Latmax, 0.0, -Lat_res)
    out_tif.SetGeoTransform(geotransform)

    # 定义投影
    prj = osr.SpatialReference()
    prj.ImportFromEPSG(4326)
    out_tif.SetProjection(prj.ExportToWkt())
    data_array = tmp_arr

    # 数据导出

    out_tif.GetRasterBand(1).WriteArray(data_array)  # 将数据写入内存，此时没有写入到硬盘
    out_tif.GetRasterBand(1).SetNoDataValue(-32767)
    out_tif.FlushCache()  # 将数据写入到硬盘
    out_tif = None  # 关闭tif文件


def chl_convert_nc_to_tif(source_data_dir, target_data_dir):
    if not os.path.exists(target_data_dir):
        os.makedirs(target_data_dir)

    data_list = glob.glob(source_data_dir + "/" + '*chlor_a.nc')

    for i in range(len(data_list)):
        data = data_list[i]
        chl_nc2tif(data, target_data_dir)
        print('chl nc to tif :' + str(i + 1) + '/' + str(len(data_list)))


def ccmp_convert_nc_to_tif(source_data_dir, target_data_dir):
    if not os.path.exists(target_data_dir):
        os.makedirs(target_data_dir)

    data_list = glob.glob(source_data_dir + "/" + '*.nc')

    for i in range(len(data_list)):
        data = data_list[i]
        ccmp_nc2tif(data, target_data_dir)
        print('CCMP nc to tif :' + str(i + 1) + '/' + str(len(data_list)))


def ccmp_nc2tif(data, Output_folder):
    if not os.path.exists(Output_folder):
        os.makedirs(Output_folder)
    tmp_data = nc.Dataset(data)  # 利用.Dataset()方法读取nc数据

    Lat_data = tmp_data.variables['latitude'][:]
    Lon_data = tmp_data.variables['longitude'][:]
    if max(Lon_data) > 180:
        tmp_data = ccmp_convert_360_to_180(data)

    Lat_data = tmp_data.variables['latitude'][:]
    Lon_data = tmp_data.variables['longitude'][:]

    # print(Lat_data)
    # [58.63129069 58.62295736 58.61462403 ... 15.77295736 15.76462403
    #  15.75629069]
    # print(Lon_data)
    # [ 71.29005534  71.29838867  71.30672201 ... 136.67338867 136.68172201
    #  136.69005534]

    tmp_arr = np.sqrt(
        np.square(np.asarray(tmp_data.variables['uwnd'])) + np.square(np.asarray(tmp_data.variables['vwnd'])))

    # 影像的左上角&右下角坐标
    Lonmin, Latmax, Lonmax, Latmin = [Lon_data.min(), Lat_data.max(), Lon_data.max(), Lat_data.min()]
    # Lonmin, Latmax, Lonmax, Latmin
    # (71.29005533854166, 58.63129069182766, 136.6900553385789, 15.756290691830095)

    # 分辨率计算
    Num_lat = len(Lat_data)  # 5146
    Num_lon = len(Lon_data)  # 7849
    Lat_res = (Latmax - Latmin) / (float(Num_lat) - 1)
    Lon_res = (Lonmax - Lonmin) / (float(Num_lon) - 1)
    # print(Num_lat, Num_lon)
    # print(Lat_res, Lon_res)
    # 5146 7849
    # 0.00833333333333286 0.008333333333338078

    # i=0,1,2,3,4,5,6,7,8,9,...
    # 创建tif文件
    driver = gdal.GetDriverByName('GTiff')
    out_tif_name = data.split('\\')[-1].replace('.nc', '.tif')
    out_tif = driver.Create(Output_folder + out_tif_name, Num_lon, Num_lat, 1, gdal.GDT_Float32)

    # 设置影像的显示范围
    # Lat_re前需要添加负号
    geotransform = (Lonmin, Lon_res, 0.0, Latmin, 0.0, Lat_res)
    out_tif.SetGeoTransform(geotransform)

    # 定义投影
    prj = osr.SpatialReference()
    prj.ImportFromEPSG(4326)
    out_tif.SetProjection(prj.ExportToWkt())
    data_array = tmp_arr[0]

    # 数据导出

    out_tif.GetRasterBand(1).WriteArray(data_array)  # 将数据写入内存，此时没有写入到硬盘
    out_tif.GetRasterBand(1).SetNoDataValue(-999)
    out_tif.FlushCache()  # 将数据写入到硬盘
    out_tif = None  # 关闭tif文件


def ccmp_convert_360_to_180(data):
    lon_name = 'longitude'  # whatever name is in the data

    # Adjust lon values to make sure they are within (-180, 180)
    ds = xr.open_dataset(data)
    ds['_longitude_adjusted'] = xr.where(
        ds[lon_name] > 180,
        ds[lon_name] - 360,
        ds[lon_name])

    # reassign the new coords to as the main lon coords
    # and sort DataArray using new coordinate values
    ds = (
        ds
        .swap_dims({lon_name: '_longitude_adjusted'})
        .sel(**{'_longitude_adjusted': sorted(ds._longitude_adjusted)})
        .drop(lon_name))

    ds = ds.rename({'_longitude_adjusted': lon_name})
    return ds


def sst_nc2tif(data, Output_folder):
    if not os.path.exists(Output_folder):
        os.makedirs(Output_folder)
    tmp_data = nc.Dataset(data)  # 利用.Dataset()方法读取nc数据

    Lat_data = tmp_data.variables['lat'][:]
    Lon_data = tmp_data.variables['lon'][:]
    if max(Lon_data) > 180:
        tmp_data = sst_convert_360_to_180(data)

    Lat_data = tmp_data.variables['lat'][:]
    Lon_data = tmp_data.variables['lon'][:]

    # print(Lat_data)
    # [58.63129069 58.62295736 58.61462403 ... 15.77295736 15.76462403
    #  15.75629069]
    # print(Lon_data)
    # [ 71.29005534  71.29838867  71.30672201 ... 136.67338867 136.68172201
    #  136.69005534]

    tmp_arr = np.asarray(tmp_data.variables['sst'])

    # 影像的左上角&右下角坐标
    Lonmin, Latmax, Lonmax, Latmin = [Lon_data.min(), Lat_data.max(), Lon_data.max(), Lat_data.min()]
    # Lonmin, Latmax, Lonmax, Latmin
    # (71.29005533854166, 58.63129069182766, 136.6900553385789, 15.756290691830095)

    # 分辨率计算
    Num_lat = len(Lat_data)  # 5146
    Num_lon = len(Lon_data)  # 7849
    Lat_res = (Latmax - Latmin) / (float(Num_lat) - 1)
    Lon_res = (Lonmax - Lonmin) / (float(Num_lon) - 1)
    # print(Num_lat, Num_lon)
    # print(Lat_res, Lon_res)
    # 5146 7849
    # 0.00833333333333286 0.008333333333338078

    # i=0,1,2,3,4,5,6,7,8,9,...
    # 创建tif文件
    driver = gdal.GetDriverByName('GTiff')
    out_tif_name = data.split('\\')[-1].replace('.nc', '.tif')
    out_tif = driver.Create(Output_folder + out_tif_name, Num_lon, Num_lat, 1, gdal.GDT_Float32)

    # 设置影像的显示范围
    # Lat_re前需要添加负号
    geotransform = (Lonmin, Lon_res, 0.0, Latmin, 0.0, Lat_res)
    out_tif.SetGeoTransform(geotransform)

    # 定义投影
    prj = osr.SpatialReference()
    prj.ImportFromEPSG(4326)
    out_tif.SetProjection(prj.ExportToWkt())
    data_array = tmp_arr[0][0]

    # 数据导出

    out_tif.GetRasterBand(1).WriteArray(data_array)  # 将数据写入内存，此时没有写入到硬盘
    out_tif.GetRasterBand(1).SetNoDataValue(-999)
    out_tif.FlushCache()  # 将数据写入到硬盘
    out_tif = None  # 关闭tif文件


def sst_covert_nc_to_tif(source_data_dir, target_data_path):
    if not os.path.exists(target_data_path):
        os.makedirs(target_data_path)

    data_list = glob.glob(source_data_dir + "/" + '*.nc')

    for i in range(len(data_list)):
        data = data_list[i]
        sst_nc2tif(data, target_data_path)
        print('sst nc to tif :' + str(i + 1) + '/' + str(len(data_list)))


def sst_convert_360_to_180(data):
    lon_name = 'lon'  # whatever name is in the data

    # Adjust lon values to make sure they are within (-180, 180)
    ds = xr.open_dataset(data)
    ds['_longitude_adjusted'] = xr.where(
        ds[lon_name] > 180,
        ds[lon_name] - 360,
        ds[lon_name])

    # reassign the new coords to as the main lon coords
    # and sort DataArray using new coordinate values
    ds = (
        ds
        .swap_dims({lon_name: '_longitude_adjusted'})
        .sel(**{'_longitude_adjusted': sorted(ds._longitude_adjusted)})
        .drop(lon_name))

    ds = ds.rename({'_longitude_adjusted': lon_name})
    return ds


if __name__ == '__main__':
    par_convert_nc_to_tif('./02_par_nc_data/requested_files', './03_par_tif/')
    chl_convert_nc_to_tif('./02_chl_nc_data/requested_files', './03_chl_tif/')
    # ccmp_convert_nc_to_tif('./02_CCMP_nc_data', './03_CCMP_tif/')
    # sst_covert_nc_to_tif('./02_sst_nc_data', './03_tif_data/')
