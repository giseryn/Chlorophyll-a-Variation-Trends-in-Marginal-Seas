import os
import datetime
import glob
import re
import numpy as np
from osgeo import gdal, osr
import pandas as pd
import matplotlib.pyplot as plt


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


def year_range(begin_date, end_date):
    year_set = set()
    for date in date_range(begin_date, end_date):
        year_set.add(date[0:4])
    year_list = []
    for year in year_set:
        year_list.append(year)
    return sorted(year_list)


def get_yearly_mean_data(input_data_dir, begin_date, end_date):
    years = year_range(begin_date, end_date)

    files = glob.glob(input_data_dir + '*.tif')
    yearly_value = []
    monthly_value=[]
    for year in years:
        tif_value_mean_list = []
        for file in files:
            numbers = re.findall('\d{6}', file)
            if year == numbers[0][0:4]:
                tif_value_mean = np.nanmean(gdal_analysis(file))
                tif_value_mean_list.append(tif_value_mean)
                monthly_value.append(tif_value_mean)

        spring_avg = np.mean(tif_value_mean_list[3:6])
        summer_avg = np.mean(tif_value_mean_list[6:9])
        fall_avg = np.mean(tif_value_mean_list[9:12])
        winter_avg = np.mean(tif_value_mean_list[0:2] + tif_value_mean_list[11:12])
        year_avg = np.mean(tif_value_mean_list)

        tif_value_mean_list.append(spring_avg)
        tif_value_mean_list.append(summer_avg)
        tif_value_mean_list.append(fall_avg)
        tif_value_mean_list.append(winter_avg)
        tif_value_mean_list.append(year_avg)

        yearly_value.append(tif_value_mean_list)

    return years, yearly_value,monthly_value


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


def print_yearly_mean_fig(input_data_dir, data_type, result_path, begin_date, end_date):
    years, yearly_value,monthly_value = get_yearly_mean_data(input_data_dir, begin_date, end_date)

    return yearly_value,monthly_value


def month_range(begin_date, end_date):
    month_set = set()
    for date in date_range(begin_date, end_date):
        month_set.add(date[0:6])
    month_list = []
    for month in month_set:
        month_list.append(month)
    return sorted(month_list)


def get_environmental_factors_line_fig(chl_data_path, par_data_path, sst_data_path, ccmp_data_path, result_path,
                                       begin_date, end_date):
    def normalized_data(lst):
        normalized_lst = (lst - np.min(lst)) / (np.max(lst) - np.min(lst))
        return [normalized_lst.tolist()]
    verify_the_existence_of_folder(result_path)
    chl_list,chl_month_list = print_yearly_mean_fig(chl_data_path, 'chl', result_path, begin_date=begin_date, end_date=end_date)
    par_list,par_month_list = print_yearly_mean_fig(par_data_path, 'par', result_path, begin_date=begin_date, end_date=end_date)
    sst_list,sst_month_list = print_yearly_mean_fig(sst_data_path, 'sst', result_path, begin_date=begin_date, end_date=end_date)
    ccmp_list,ccmp_month_list = print_yearly_mean_fig(ccmp_data_path, 'ccmp', result_path, begin_date=begin_date, end_date=end_date)
    print(chl_list)

    month_list = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                  "November", "December", "spring_avg", "summer_avg", "fall_avg", "winter_avg", "year_avg"]

    year_list = year_range(begin_date='19980101', end_date='20201231')
    monthly_list = month_range(begin_date='19980101', end_date='20201231')

    year_mean_data = pd.DataFrame(data=chl_list + par_list + sst_list + ccmp_list, columns=month_list,
                                  index=year_list + year_list + year_list + year_list)

    month_mean_data = pd.DataFrame(data=normalized_data(chl_month_list) + normalized_data(par_month_list)+
                                        normalized_data(sst_month_list) + normalized_data(ccmp_month_list)
                                   , columns=monthly_list,index=['Chl-a','PAR','SST','CCMP'])
    year_mean_data.to_csv(result_path + area + '_environmental_factors.csv', encoding='gbk')
    month_mean_data.T.to_csv(area + '_monthly_environmental_factors.csv', encoding='gbk')
    return result_path + area + '_environmental_factors.csv'


def create_correlation_charts(csv_file, result_path, area):
    print(csv_file)

    def normalized_data(lst):
        normalized_lst = (lst - np.min(lst)) / (np.max(lst) - np.min(lst))
        return normalized_lst

    month_list = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                  "November", "December"]
    chl_data = pd.read_csv(csv_file, index_col=0, skiprows=range(0, 0), nrows=23).mean(axis=0).tolist()[0:12]
    par_data = pd.read_csv(csv_file, index_col=0, skiprows=range(0, 23), nrows=23).mean(axis=0).tolist()[0:12]
    sst_data = pd.read_csv(csv_file, index_col=0, skiprows=range(0, 46), nrows=23).mean(axis=0).tolist()[0:12]
    wind_data = pd.read_csv(csv_file, index_col=0, skiprows=range(0, 69), nrows=23).mean(axis=0).tolist()[0:12]

    monthly_data = [normalized_data(chl_data), normalized_data(par_data),
                    normalized_data(sst_data), normalized_data(wind_data)]

    monthly_data_pd = pd.DataFrame(data=monthly_data, columns=month_list, index=['chl', 'par', 'sst', 'wind'])
    monthly_data_pd.to_csv(result_path + area + '_monthly_environmental_factors.csv')


def analysis_data(area):
    chl_data_path = './05_' + area + '_data/chl/preprocess_result/'
    par_data_path = './05_' + area + '_data/par/preprocess_result/'
    sst_data_path = './05_' + area + '_data/sst/preprocess_result/'
    ccmp_data_path = './05_' + area + '_data/ccmp/preprocess_result/'
    result_path = './result/'
    result_path = './08_'+area+'_result_file/'
    csv_path = get_environmental_factors_line_fig(chl_data_path, par_data_path, sst_data_path, ccmp_data_path,
                                                  result_path,
                                                  begin_date='19980101', end_date='20201231')

    # csv_path = result_path + area + '_environmental_factors.csv'

    # create_correlation_charts(csv_path, result_path, area)
    #
    # plot_chlorophyll(csv_path, result_path, area)
    # yearly_chl_plot_and_bar(csv_path, result_path, area)


def plot_chlorophyll(csv_file, result_path, area):
    """
    从CSV文件中读取最后五列数据，绘制成折线图。
    第一列为年份，最后五列为四个季节和全年叶绿素平均值。
    """

    # 读取数据
    data = pd.read_csv(csv_file, index_col=0, nrows=92)

    # 绘制折线图
    data.iloc[47:69, -1:].plot(figsize=(15, 5))

    # 添加标题和标签
    plt.title(area + ' SST Average by Season and Year', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Chlorophyll Average', fontsize=16)

    # 添加图例
    plt.legend(data.iloc[:, -1:].columns, loc='best', fontsize=16)
    plt.savefig('./result/' + area + '_sst.png')
    # 显示图形
    plt.show()


def yearly_chl_plot_and_bar(csv_path, result_path, area):
    df_source = pd.read_csv(csv_path, index_col=0, nrows=23)
    df = df_source.iloc[:, -5:]
    columns = list(df.columns)
    index = list(df.index)
    y = []
    y1 = list(df[columns[0]])
    y2 = list(df[columns[1]])
    y3 = list(df[columns[2]])
    y4 = list(df[columns[3]])
    y5 = list(df[columns[4]])
    x = []
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    for year in index:
        for season in columns:
            if season == columns[4]:
                x5.append(str(year))
            elif season == columns[0]:
                x1.append(str(year) + '_' + str(season))
                x.append(str(year) + '_' + str(season))
            elif season == columns[1]:
                x2.append(str(year) + '_' + str(season))
                x.append(str(year) + '_' + str(season))
            elif season == columns[2]:
                x3.append(str(year) + '_' + str(season))
                x.append(str(year) + '_' + str(season))
            else:
                x4.append(str(year) + '_' + str(season))
                x.append(str(year) + '_' + str(season))

    for year in range(len(index)):
        y.append(y1[year])
        y.append(y2[year])
        y.append(y3[year])
        y.append(y4[year])

    # 创建图表对象
    fig, ax = plt.subplots(figsize=(15, 5))
    # 创建第二个y轴
    ax2 = ax.twiny()
    # 绘制折线图
    colors = ['#abdda4', '#ffffbf', '#fdae61', '#2b83ba','#d7191c']
    ax.plot(x5, y5, linewidth=2.0, color=colors[4])

    # for i, group in enumerate(y):
    #     ax.bar(np.arange(len(y)) + i * 0.2, group, width=0.2, color=colors[i % 4])
    ax2.bar(x, y, color='black', alpha=0.5, width=1.0, align='center')
    ax2.plot(x1, y1, linewidth=2.0, color=colors[0])

    ax2.plot(x2, y2, linewidth=2.0, color=colors[1])

    ax2.plot(x3, y3, linewidth=2.0, color=colors[2])

    ax2.plot(x4, y4, linewidth=2.0, color=colors[3])

    ax2.bar(x1, y1, color=colors[0], alpha=0.5, width=1.0, align='center')
    ax2.bar(x2, y2, color=colors[1], alpha=0.5, width=1.0, align='center')
    ax2.bar(x3, y3, color=colors[2], alpha=0.5, width=1.0, align='center')
    ax2.bar(x4, y4, color=colors[3], alpha=0.5, width=1.0, align='center')

    # 设置轴标签
    ax.set_ylabel('chl-a')
    ax.set_xlabel('years')
    plt.ylim(min(y) - 0.05, max(y) + 0.05)
    plt.xticks([], [])

    # 设置标题
    plt.title(area + ' Year-to-year curve of chlorophyll a concentration')
    leg = ax.legend(['year'], loc='upper right')
    leg.set_bbox_to_anchor((0.9, 1))
    ax2.legend(['spring', 'summer', 'fall', 'winter'], loc='upper right')
    # 显示图表
    plt.savefig(result_path + area + '_chl.png')
    plt.show()


if __name__ == '__main__':
    area_list = ['amazon', 'bohai', 'donghai', 'mexico', 'USEastCoast']
    # area_list = ['Iberian']

    for area in area_list:
        analysis_data(area)
