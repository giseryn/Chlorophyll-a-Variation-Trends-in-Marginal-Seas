import os
import tarfile
import time


def verify_the_existence_of_folder(target_data_dir):
    """
    校验文件夹是否存在如果不存在则创建一个
    :param target_data_dir: 目标文件夹名
    :return:
    """

    if not os.path.exists(target_data_dir):
        os.makedirs(target_data_dir)


def with_progress_bar(func):
    def wrapper(*args, **kwargs):
        items = func(*args, **kwargs)
        func_name = func.__name__
        for i, item in enumerate(items):
            print(f"{func_name}：{i + 1}/{len(items)}")
            # 你的操作
            yield item

    return wrapper


def extract_tar_file(path, file_name, extract_path):
    """
    解压缩指定文件到指定目录
    :param path: 压缩文件所在目录
    :param file_name: 压缩文件名
    :param extract_path: 解压缩路径
    :return: 无返回
    """
    file_full_path = os.path.join(path, file_name)  # 获取待解压文件的完整路径
    tar_file = tarfile.open(file_full_path)  # 打开tar文件
    tar_file.extractall(extract_path)  # 解压所有文件到指定目录
    tar_file.close()  # 关闭tar文件


@with_progress_bar
def find_tar_files(path, target):
    """
    查找指定目录下包含指定字符串的tar文件
    :param path: 目标目录
    :param target: 查找字符串
    :return: 返回包含指定字符串的tar文件列表
    """
    tar_files = []
    for file in os.listdir(path):
        if file.endswith('.tar') and target in file:
            tar_files.append(file)
    return tar_files


def handle_nasa_source_data(input_path, output_path, keywords):
    """
    从指定路径下将符合要求的压缩包解压到指定文件夹内
    :param input_path: 压缩包地址
    :param output_path: 解压缩地址
    :param keywords: 匹配关键词
    :return:
    """

    verify_the_existence_of_folder(output_path)

    # 查找所有包含'keywords'的tar文件
    target_files = find_tar_files(input_path, keywords)

    # 解压缩每个tar文件到指定目录
    for file in target_files:
        extract_tar_file(input_path, file, output_path)


if __name__ == '__main__':
    # 指定原始压缩文件目录和目标解压目录

    handle_nasa_source_data('./01_download_nasa_data', './02_chl_nc_data', 'chl')
    # handle_nasa_source_data('./01_download_nasa_data', './02_par_nc_data', 'par')
