#! python3
# -*- coding: utf-8 -*-

import os
# import gc
# import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def df_aggregation(glon_c):
    """
    :param glon_c: 传入唯一参数，地理经度
    :return: df，包含doy等，及tec极小值位置，以及Kp，AE指数等
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
    print(len(df_ae))            # AE指数截止时间2017-6-30日，因此df_ae数据少于df_kp
    # print("length of tec trough min dataframe: ", df_tec_prof.__len__())
    # print("length of kp index dataframe: ", df_kp.__len__())
    # print("length of ae index dataframe: ", df_ae.__len__())

    df_kp.insert(5, "trough_min_lat", trough_mini)                # 插入槽极小的纬度，键值顺序相同，故直接插入即可
    df_new = pd.merge(df_kp, df_ae, on=['year', 'doy', 'glon', 'lt'], how='outer')      # 按照键值合并
    print("length of df_new: {}\n".format(len(df_new)), df_new.columns)
    if len(df_new) != len(df_kp):
        raise Exception("df merge Error")
    return df_new


def df_query(df, season, y1, m1, d1, y2, m2, d2):
    """
    :param df: 待query的df
    :param season: 季节，限定春秋、夏、冬、全年四种情况
    :param y1: 起始年
    :param m1: 起始月
    :param d1: 起始日
    :param y2: 结束年
    :param m2: 结束月
    :param d2: 结束日
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

    print('length of df after query: ', len(df))
    return df


def get_median_percentiles(ssn_list, df):
    dic_median_values = {}
    dic_25percentile_values = {}
    dic_75percentile_values = {}

    for ssn in ssn_list:
        dic_median_values[ssn] = {}
        dic_25percentile_values[ssn], dic_75percentile_values[ssn] = {}, {}
        DF1 = df_query(df, ssn, year1, month1, day1, year2, month2, day2)
        for lt in lt_list:
            lt_str = 'lt_{}'.format(lt)
            dic_median_values[ssn][lt_str] = []
            dic_25percentile_values[ssn][lt_str] = []
            dic_75percentile_values[ssn][lt_str] = []

            DF2 = DF1.query("lt == {}".format(lt))
            data = DF2[['kp9', 'trough_min_lat']]
            # data.loc[:, 'kp9'] = [np.floor(_) for _ in data['kp9']]          # 向下取整，range(0, 9, 1)
            data.loc[:, 'kp9'] = [0.5 * (_ // 0.5) for _ in data['kp9']]

            for kp9_index in kp9_list:
                data_kp9 = data.query('kp9 == {}'.format(kp9_index))
                if len(data_kp9['trough_min_lat']) > 0:
                    dic_median_values[ssn][lt_str].append(data_kp9['trough_min_lat'].median())
                    dic_25percentile_values[ssn][lt_str].append(data_kp9['trough_min_lat'].quantile(0.25))
                    dic_75percentile_values[ssn][lt_str].append(data_kp9['trough_min_lat'].quantile(0.75))
                else:
                    dic_median_values[ssn][lt_str].append(np.nan)
                    dic_25percentile_values[ssn][lt_str].append(np.nan)
                    dic_75percentile_values[ssn][lt_str].append(np.nan)

    return dic_median_values, dic_25percentile_values, dic_75percentile_values


def get_fitted_line(ssn_list, df):
    dic_fitted_values = {}
    for ssn in ssn_list:
        dic_fitted_values[ssn] = {}
        DF1 = df_query(df, ssn, year1, month1, day1, year2, month2, day2)
        for lt in lt_list:
            lt_str = 'lt_{}'.format(lt)
            dic_fitted_values[ssn][lt_str] = []

            DF2 = DF1.query("lt == {}".format(lt))
            data = DF2[['kp9', 'trough_min_lat']]
            # data['kp9'] = [np.floor(_) for _ in data['kp9']]  # 向下取整，range(0, 9, 1)

            z = list(np.polyfit(list(data['kp9']), list(data['trough_min_lat']), 1))  # 1阶多项式拟合
            x = list(set(data['kp9']))
            y = [z[0] * _ + z[1] for _ in x]
            dic_fitted_values[ssn][lt_str].extend([x, y])
    return dic_fitted_values


def plot_fig(ssn_list, kp9_lst, dic_median, dic_25, dic_75, dic_fitted, plot_med=False):
    """
    :param ssn_list: 4个季节
    :param kp9_lst: 分辨率1
    :param dic_median: 每一kp9的槽极小纬度中值
    :param dic_25: 槽极小纬度1/4分位数
    :param dic_75: 槽极小纬度3/4分位数
    :param dic_fitted: 线性拟合线段的x, y值
    :param plot_med: False as default, which means plot fitted segment;True for median & percentiles
    :return: none
    """
    fig = plt.figure(figsize=(9, 7))

    for index, ssn in enumerate(ssn_list):
        ax2 = fig.add_subplot(141 + index)                   # 四季，四张图
        plt.sca(ax2)
        colors = 'rgbyck'
        for idx, lt in enumerate(lt_list):                  # 每张图6个地方时，6条线
            lt_str = 'lt_{}'.format(lt)
            if not plot_med:
                plt.plot(dic_fitted[ssn][lt_str][0], dic_fitted[ssn][lt_str][1], color=colors[idx],
                         label='lt={}'.format(lt))
            else:
                y_err = [[i - j for i, j in zip(dic_median[ssn][lt_str], dic_25[ssn][lt_str])],
                         [i - j for i, j in zip(dic_75[ssn][lt_str], dic_median[ssn][lt_str])]]
                plt.errorbar(kp9_lst, dic_median[ssn][lt_str], yerr=y_err, color=colors[idx], fmt='-o',
                             label='lt={}'.format(lt))
        plt.text(0.2, 0.10, '{}'.format(ssn), transform=ax2.transAxes, color='black')
        plt.legend()
        plt.xlabel('Kp9 index')
        plt.xlim(-1, 8)
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
        if index == 0:
            plt.ylabel('Geographic-Lat (degree)')
        plt.ylim(35, 65)
        plt.yticks([_ for _ in range(35, 66)])
        ax2.yaxis.set_major_locator(MultipleLocator(5))
        ax2.yaxis.set_minor_locator(MultipleLocator(1))
    fig.suptitle("{}.{}-{}.{} glon_{}° trough_mini_lat-kp9".format(year1, month1, year2, month2, glon),
                 fontsize=16, x=0.5, y=0.95)
    # fig.savefig(kp_path + '00 _ median & percentiles')
    plt.show()
    fig.clear()


if __name__ == '__main__':
    glon = -90
    year1, month1, day1 = 2014, 9, 1
    year2, month2, day2 = 2017, 8, 31
    index_name = 'kp9'
    season_list = ['equinox', 'summer', 'winter', 'year']
    kp9_list = np.linspace(0, 8, 9)
    lt_list = [19, 21, 0, 3]
    kp_path = "C:\\DATA\\GPS_MIT\\millstone\\summary graph\\scatter plot\\lat-index\\"
    folder_name = 'glon_{}°{}.{}-{}.{}'.format(glon, year1, month1, year2, month2)
    figure_path = kp_path + "{}\\".format(folder_name)
    if not os.path.exists(figure_path):
        os.mkdir(figure_path)

    DF = df_aggregation(glon)  # 从3个csv文件中得到聚合df

    dict_median, dict_25, dict_75 = get_median_percentiles(season_list, DF)
    dict_fitted = get_fitted_line(season_list, DF)  # 各季节，各地方时，得到槽极小与磁场指数的线性拟合x，y值；
    plot_fig(season_list, kp9_list, dict_median, dict_25, dict_75, dict_fitted, True)
    # True作中值及上下四分位数、False作拟合线
    print("work done!")
