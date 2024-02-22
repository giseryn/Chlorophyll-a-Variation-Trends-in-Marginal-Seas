import time
import os
from osgeo import gdal, osr, ogr
from tqdm import tqdm

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box


def verify_the_existence_of_folder(target_data_dir):
    """
    校验文件夹是否存在如果不存在则创建一个
    :param target_data_dir: 目标文件夹名
    :return:
    """

    if not os.path.exists(target_data_dir):
        os.makedirs(target_data_dir)


def fn_timer(func):
    def wrapper(*args, **kwargs):
        t_start = time.time()
        result = func(*args, **kwargs)
        t_end = time.time()
        print("函数 %s 运行时间：%.6f 秒" % (func.__name__, t_end - t_start))
        return result

    return wrapper


@fn_timer
def intersect_shp(shp1_path, shp2_path, out_shp_path):
    # 读取矢量数据
    shp1 = gpd.read_file(shp1_path)
    shp2 = gpd.read_file(shp2_path)

    # 计算与shp2相交的要素id
    intersect_ids = list(shp1[shp1.intersects(shp2.geometry.unary_union)].index)

    # 根据id提取要素，保存为新的矢量数据集
    intersect_shp_data = shp1.loc[intersect_ids]
    intersect_shp_data.to_file(out_shp_path)


@fn_timer
def create_vector_grid(space_range, raster_path, vector_path):
    # 获取栅格数据的分辨率
    from osgeo import gdal
    ds = gdal.Open(raster_path)
    geo_transform = ds.GetGeoTransform()
    cell_size = geo_transform[1]

    # 获取空间范围
    xmin, ymin, xmax, ymax = space_range

    # 计算需要创建的网格数量
    x_count = int((xmax - xmin) / cell_size)
    y_count = int((ymax - ymin) / cell_size)

    # 创建矢量网格数据
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(vector_path):
        driver.DeleteDataSource(vector_path)
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(4326)
    ds = driver.CreateDataSource(vector_path)
    layer = ds.CreateLayer('grid', spatial_ref, ogr.wkbPolygon)
    idField = ogr.FieldDefn('id', ogr.OFTInteger)
    layer.CreateField(idField)

    # 以左下角为起点，一个一个网格创建
    for i in range(x_count):
        for j in range(y_count):
            # 计算矩形四个点坐标
            x_min = xmin + i * cell_size
            y_min = ymin + j * cell_size
            x_max = x_min + cell_size
            y_max = y_min + cell_size

            # 创建矩形
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(x_min, y_min)
            ring.AddPoint(x_max, y_min)
            ring.AddPoint(x_max, y_max)
            ring.AddPoint(x_min, y_max)
            ring.AddPoint(x_min, y_min)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            # 添加到图层中
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetGeometry(poly)
            feature.SetField('id', i * y_count + j)
            layer.CreateFeature(feature)
            feature = None

    ds = None


def create_musk(coords, place_folders, basic_raster, basic_shp):
    """

    :param coords: 空间范围，示例：[-60, -10, -40, 10]
    :param place_folders: 研究区域名称 示例：'./04_amazon/'
    :param basic_raster: 示例的栅格数据地址
    :param basic_shp: 导出的矢量网格数据地址
    :return:
    """

    verify_the_existence_of_folder(place_folders)

    temp_vector_address = place_folders + 'temp_shp.shp'  # 矢量网格数据地址
    result_vector_address = place_folders + 'musk_shp.shp'  # 矢量网格数据地址

    create_vector_grid(coords, basic_raster, temp_vector_address)

    intersect_shp(temp_vector_address, basic_shp, result_vector_address)


def crop_raster(src_path, output_path, bbox):
    """
    Crop a raster file based on a given bounding box.

    Parameters:
    src_path (str): Path to the input raster file (.tif).
    output_path (str): Path to the output raster file (.tif).
    bbox (tuple): A tuple of four elements representing the bounding box
        in the format of (xmin, ymin, xmax, ymax).

    Returns:
    None
    """
    # Convert the bounding box into a shapely geometry object.
    geometry = box(*bbox)

    # Read the raster file with rasterio.
    with rasterio.open(src_path) as src:
        # Crop the raster file using the bounding box geometry.
        out_image, out_transform = mask(src, [geometry], crop=True)

        # Copy the metadata from the input file.
        out_meta = src.meta.copy()

    # Update the metadata for the output file.
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    # Write the cropped raster file to disk.
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)


def crop_region(tif_dir_path, bbox, output_path_dir):
    """

    :param tif_dir_path:
    :param bbox:
    :param output_path:
    :return:
    """

    print(output_path_dir)

    verify_the_existence_of_folder(output_path_dir)

    files = os.listdir(tif_dir_path)
    data_list = []
    for file in files:
        if 'tif' in file:
            data_list.append(file)

    for i in tqdm(range(len(data_list))):
        data = data_list[i]
        input_path = tif_dir_path + data
        output_path = output_path_dir + data
        crop_raster(input_path, output_path, bbox)
        # print('crop region :' + str(i) + '/' + str(len(data_list)))


def crop_data(area_dir, bbox):
    """

    :param area_dir:
    :param bbox:
    :return:
    """

    print(area_dir)

    basic_raster_path = './04_basic_file/sample.tif'
    basic_shp_path = './04_basic_file/ne_110m_ocean.shp'
    musk_dir = area_dir + '/mask/'
    chl_dir = area_dir + '/chl/'
    par_dir = area_dir + '/par/'
    ccmp_dir = area_dir + '/ccmp/'
    sst_dir = area_dir + '/sst/'

    # create_musk(bbox, musk_dir, basic_raster_path, basic_shp_path)
    crop_region('./03_chl_tif/', bbox, chl_dir)
    crop_region('./03_par_tif/', bbox, par_dir)
    crop_region('./03_CCMP_tif/', bbox, ccmp_dir)
    crop_region('./03_sst_tif/', bbox, sst_dir)


if __name__ == '__main__':
    # create_musk([-61, -6, -37, 16], './04_amazon/mask/', basic_raster_path, basic_shp_path)
    # create_musk([117, 37, 123, 41], './04_bohai/mask/', basic_raster_path, basic_shp_path)
    # create_musk([-98.86, 17.09, -81.28, 31.16], './04_mexico/mask/', basic_raster_path, basic_shp_path)
    # create_musk([117, 23, 131, 33], './04_donghai/mask/', basic_raster_path, basic_shp_path)
    # create_musk([-81.6, 27.5, -57.66, 43.3], './04_USEastCoast/mask/', basic_raster_path, basic_shp_path)

    crop_data('./04_amazon/', [-61, -6, -37, 16])
    crop_data('./04_bohai/', [117, 37, 123, 41])
    crop_data('./04_mexico/', [-98.86, 17.09, -81.28, 31.16])
    crop_data('./04_donghai/', [117, 23, 131, 33])
    crop_data('./04_USEastCoast/', [-81.6, 27.5, -57.66, 43.3])
    # crop_data('./04_newfoundland/', [-71.3, 42.7,-43.5, 60.5])
    # crop_data('./04_SouthUS/', [-81.53, 25.23, -74.94, 35.21])

