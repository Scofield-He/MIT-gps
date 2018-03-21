#! python3
# -*- coding: utf-8 -*-

import os
# import gc
# import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator


def df_aggregation(glon_c):
    """
    :param glon_c: 传入唯一参数，地理经度
    :return: df，包含doy等限定条件，及tec极小值位置，以及Kp，AE指数等
    """
    filepath = "C:\\tmp\\sorted tec_profile_gdlat.csv"
    bigTable = pd.read_csv(filepath, encoding='gb2312')
    filepath_kp = "C:\\tmp\\sorted kp_index.csv"
    bigTable_kp = pd.read_csv(filepath_kp, encoding='gb2312')
    filepath_AE = "C:\\tmp\\sorted AE_index.csv"
    bigTable_AE = pd.read_csv(filepath_AE, encoding='gb2312')

    df_tec = bigTable.query("glon == {}".format(glon_c))
    trough_mini = df_tec["trough_min_lat"]
    df_kp = bigTable_kp.query("glon == {}".format(glon_c))
    df_ae = bigTable_AE.query(("glon == {}".format(glon_c))).drop(['date'], axis=1)
    # print(len(df_ae))            # AE指数截止时间2017-6-30日，因此df_ae数据少于df_kp
    # print("length of tec trough min dataframe: ", df_tec_prof.__len__())
    # print("length of kp index dataframe: ", df_kp.__len__())
    # print("length of ae index dataframe: ", df_ae.__len__())

    df_kp.insert(5, "trough_min_lat", trough_mini)                # 插入槽极小的纬度，键值顺序相同，故直接插入即可
    df_new = pd.merge(df_kp, df_ae, on=['year', 'doy', 'glon', 'lt'], how='outer')      # 按照键值合并
    # print("length of df_new: {}\n".format(len(df_new)), df_new.columns)
    if len(df_new) != len(df_kp):
        raise Exception("df merge Error")
    return df_new


def df_query(df, season, y1, m1, d1, y2, m2, d2, idx, idx_range):
    """
    :param df: 待query的df
    :param season: 季节，限定春秋、夏、冬、全年四种情况
    :param y1: 起始年
    :param m1: 起始月
    :param d1: 起始日
    :param y2: 结束年
    :param m2: 结束月
    :param d2: 结束日
    :param idx: 某磁场活动指数
    :param idx_range: 磁场活动指数范围
    :return: 经过query后的dataFrame, 即给定时间段，给定季节，改定此活动强度范围的数据
    """
    print('length of pre_query df', len(df))
    df = df.dropna()
    print('length of not null trough min: ', len(df))
    # df = df.query("35 < trough_min_lat <= 70 ")

    doy1 = datetime.date(y1, m1, d1).timetuple().tm_yday
    doy2 = datetime.date(y2, m2, d2).timetuple().tm_yday
    df = df.query("({} < year < {}) | (year == {} & doy >= {}) | (year == {} & doy <= {}) "
                  .format(y1, y2, y1, doy1, y2, doy2))

    if season == 'summer':
        df = df.query("121 <= doy <= 243")
    elif season == 'winter':
        df = df.query("(doy >= 305) | (doy <= 59)")
    elif season == 'equinox':
        df = df.query("(60 <= doy <= 120) | (244 <= doy <= 304)")
    elif season != 'year':
        raise Exception('arg Error: season illegal')

    df = df.query("{} <= {} < {}".format(idx_range[0], idx, idx_range[1]))
    print('length of df after query: ', len(df))
    return df


def xtick_lt_formatter(x, _):
    return '{}'.format(int(x % 24))


def plot_LatLt(df, ax, ssn):
    df_dusk = df.query('18 <= lt <=23')
    # corrcoef = np.corrcoef(df_dusk['lt'], df['trough_min_lat'])  # 计算相关系数
    corrcoef = df_dusk['lt'].corr(df_dusk['trough_min_lat'])
    print('cc between lt & lat before midnight: {}'.format(corrcoef))

    lt, lat = df['lt'], df["trough_min_lat"]
    lt = [_ + 0.5 for _ in lt]

    lt = np.array([_ + 24 if _ < 12 else _ for _ in lt])            #
    # print('length of lt in season{} ;  {}'.format(season, len(lt)))
    lat = np.array(lat)
    # print('length of lat in season {} ;  {}'.format(season, len(lat)))

    # 统计每个点出现的次数
    lat_lt = list(zip(lt, lat))
    count_of_point = []
    for _ in lat_lt:
        count_of_point.append(lat_lt.count(_))                       # 对每个点出现的次数进行统计

    # figure = plt.figure(figsize=(6, 4))
    # ax = figure.add_subplot(111)
    plt.sca(ax)
    plt.scatter(lt, lat, color='b', s=[_ / 4 for _ in count_of_point])
    plt.text(0.70, 0.80, '{}\ncount:{}'.format(ssn, len(lat)), transform=ax.transAxes, color='black')
    ax.xaxis.set_major_formatter(FuncFormatter(xtick_lt_formatter))
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    plt.xlabel('lt (h)')
    plt.ylabel('Geographic-Latitude (degree)')
    plt.ylim(35, 70)

    # plt.savefig(figure_path + '{}'.format(season))
    # plt.close()
    # plt.show()
    return True


def plot_lat_lt(ssn_list, idx_range_list):
    for index_range in idx_range_list:
        folder_name = 'glon[{}°]_[{}.{}-{}.{}]_Kp9[{}-{})'. \
            format(glon, year1, month1, year2, month2, index_range[0], index_range[1])
        figure_path = lt_path + "{}\\".format(folder_name)
        if not os.path.exists(figure_path):
            os.mkdir(figure_path)

        figure = plt.figure(figsize=(10, 8))
        for index, ssn in enumerate(ssn_list):
            print(ssn)
            DF1 = df_query(DF, ssn, year1, month1, day1, year2, month2, day2, index_name, index_range)
            ax1 = figure.add_subplot(221 + index)
            plt.sca(ax1)
            plot_LatLt(DF1, ax1, ssn)

        figure.suptitle('2014.9-2017.9 glon_{}° kp9[{}_{}]'.format(glon, index_range[0], index_range[1]),
                        fontsize=16, x=0.5, y=0.95)
        figure.savefig(lt_path + '01 kp9_{}-{}'.format(index_range[0], index_range[1]))


def get_median_percentiles(ssn_list, df):
    dic_median_values = {}
    dic_25percentile_values = {}
    dic_75percentile_values = {}

    for ssn in ssn_list:
        dic_median_values[ssn] = {}
        dic_25percentile_values[ssn], dic_75percentile_values[ssn] = {}, {}
        for index_range in index_range_list:
            kp9_str = 'kp9_{}-{}'.format(index_range[0], index_range[1])
            dic_median_values[ssn][kp9_str] = []
            dic_25percentile_values[ssn][kp9_str] = []
            dic_75percentile_values[ssn][kp9_str] = []

            DF1 = df_query(df, ssn, year1, month1, day1, year2, month2, day2, index_name, index_range)
            data = DF1[['lt', 'trough_min_lat']]
            for localtime in lt_list:
                data_lt = data.query("lt == {}".format(localtime))
                dic_median_values[ssn]['kp9_{}-{}'.format(index_range[0], index_range[1])].append(
                    data_lt['trough_min_lat'].median())
                dic_25percentile_values[ssn]['kp9_{}-{}'.format(index_range[0], index_range[1])].append(
                    data_lt['trough_min_lat'].quantile(0.25))
                dic_75percentile_values[ssn]['kp9_{}-{}'.format(index_range[0], index_range[1])].append(
                    data_lt['trough_min_lat'].quantile(0.75))
    return dic_median_values, dic_25percentile_values, dic_75percentile_values


def plot_fig(ssn_list, lt_lst, dic_median, dic_25, dic_75):
    lt_lst = [_ + 24 if _ < 12 else _ for _ in lt_lst]
    fig = plt.figure(figsize=(10, 8))

    for index, ssn in enumerate(ssn_list):
        ax2 = fig.add_subplot(221 + index)
        plt.sca(ax2)
        colors = 'bgr'
        for idx, index_range in enumerate(index_range_list):
            kp9_str = 'kp9_{}-{}'.format(index_range[0], index_range[1])
            y_error = [[i - j for i, j in zip(dic_median[ssn][kp9_str], dic_25[ssn][kp9_str])],
                       [i - j for i, j in zip(dic_75[ssn][kp9_str], dic_median[ssn][kp9_str])]]
            plt.plot(lt_lst, dic_median[ssn][kp9_str], color=colors[idx],
                     label='kp9 in range[{}, {}]'.format(index_range[0], index_range[1]))
            plt.errorbar(lt_lst, dic_median[ssn][kp9_str], yerr=y_error, fmt='-o', color=colors[idx])

        plt.text(0.3, 0.85, '{}'.format(ssn), transform=ax2.transAxes, color='black')
        # plt.text(0.80, 0.90, 'count:{}'.format(len(lat)), transform=ax.transAxes, color='black')
        plt.legend()
        plt.xlabel('Local time (h)')
        plt.xlim(17, 30)
        plt.xticks(lt_lst, [_ - 24 if _ >= 24 else _ for _ in lt_lst])
        plt.ylabel('Geographic-Lat (degree)')
        plt.ylim(40, 65)

    fig.suptitle("{}.{}-{}.{} glon_{}° trough_mini_lat-lt".format(year1, month1, year2, month2, glon),
                 fontsize=16, x=0.5, y=0.95)
    fig.savefig(lt_path + '01 trough_mini_lat -- lt _ with error bar')
    # plt.show()
    fig.clear()

glon = -90
year1, month1, day1 = 2014, 9, 1
year2, month2, day2 = 2017, 9, 1
index_name = 'kp9'
season_list = ['equinox', 'summer', 'winter', 'year']
kp9_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
index_range_list = [[0, 2], [2, 4], [4, 9]]
lt_list = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
lt_path = "C:\\DATA\\GPS_MIT\\millstone\\summary graph\\scatter plot\\lat-lt\\"
# 作槽极小纬度lat与地方时lt关系图，分季；

DF = df_aggregation(glon)  # 从3个csv文件中得到聚合df
plot_lat_lt(season_list, index_range_list)

# 取lat-lt图中同一lt下lat的均值或中值，在同一子图中作出mean(median) value - lt 线图，同一季节不同kp范围在一张图中；
dict_median, dict_25, dict_75 = get_median_percentiles(season_list, DF)
plot_fig(season_list, lt_list, dict_median, dict_25, dict_75)

print("work done!")
