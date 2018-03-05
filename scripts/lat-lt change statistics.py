#! python3
# -*- coding: utf-8 -*-

import os
# import gc
# import time
import numpy as np
# from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt


def data_gen(glon_c, lt1):
    filepath = "C:\\tmp\\sorted tec_profile_gdlat.csv"
    bigTable = pd.read_csv(filepath, encoding='gb2312')
    filepath_kp = "C:\\tmp\\sorted kp_index.csv"
    bigTable_kp = pd.read_csv(filepath_kp, encoding='gb2312')
    filepath_AE = "C:\\tmp\\sorted AE_index.csv"
    bigTable_AE = pd.read_csv(filepath_AE, encoding='gb2312')

    df_tec_prof = bigTable.query("glon == {} & lt == {}".format(glon_c, lt1))
    print("length of tec trough min dataframe: ", df_tec_prof.__len__())
    # print(df_tec_prof[:5])
    trough_mini = df_tec_prof["trough_min_lat"]

    df_kp = bigTable_kp.query("glon == {} & lt == {}".format(glon_c, lt1))
    print("length of kp index dataframe: ", df_kp.__len__())
    print(df_kp[:5])

    df_ae = bigTable_AE.query(("glon == {} & lt == {}".format(glon_c, lt1))).drop(['date'], axis=1)
    print("length of ae index dataframe: ", df_ae.__len__())
    print(df_ae[:5])
    # ae_index, ae6_index = df_ae["AE"], df_ae["AE6"]

    # df_total = pd.merge(df_tec_prof, df_kp, how='inner')
    # print(len(df_total), df_total.columns)
    # print(df_total["kp"].describe())

    # df_new = df_kp.copy()
    df_kp.insert(5, "trough_min_lat", trough_mini)                # 插入槽极小的纬度，键值顺序相同，故直接插入即可
    df_new = pd.merge(df_kp, df_ae, on=['year', 'doy', 'glon', 'lt'], how='outer')      # 按照键值合并
    print(len(df_new), df_new.columns)
    print(df_new[:10])
    return df_new


def data_process(df, season, y1, y2):
    print('count of df from function data_gen', len(df))
    df = df.dropna()
    print('count of not null trough min: ', len(df))
    df = df.query("35 < trough_min_lat <= 65 ")
    df = df.query("{} <= year <= {}".format(y1, y2))

    if season == 'summer':
        ret = df.query("121 <= doy <= 243")
    elif season == 'winter':
        ret = df.query("(doy >= 305) | (doy <= 59)")
    elif season == 'equinox':
        ret = df.query("(60 <= doy <= 120) | (244 <= doy <= 304)")
    else:
        ret = df
    return ret


def data_plotIndex(df, Index, season):
    data = df[['kp', 'kp9', 'C9', 'Cp', 'sum_8kp', 'mean_8ap', 'F10.7', 'trough_min_lat']]
    print("data corrcoef matrix: \n", data.corr())

    trough_min_lat = df["trough_min_lat"]
    mag_index = df[Index]

    corrcoef = round(mag_index.corr(trough_min_lat), 2)
    print('corrcoef between kp and trough min lat: ', corrcoef)

    loc_and_index = list(zip(mag_index, trough_min_lat))          # 统计每个点的个数
    count_of_point = []
    for _ in loc_and_index:
        count_of_point.append(loc_and_index.count(_))

    figure1 = plt.figure(figsize=(8, 6))                         # 所有数据做线性拟合，重合的数据用点的大小表示数量
    ax1 = figure1.add_subplot(111)
    plt.sca(ax1)
    plt.scatter(mag_index, trough_min_lat, color='b', s=[_ for _ in count_of_point])

    z = list(np.polyfit(list(mag_index), list(trough_min_lat), 1))           # 线性拟合,得到ax + b中 a&b的值
    x = list(set(mag_index))
    y = [z[0] * _ + z[1] for _ in x]
    plt.plot(x, y, 'r--', label='{:.2f} * x + {:.2f}'.format(round(z[0], 2), round(z[1], 2)))
    plt.text(0.72, 0.82, 'cc = {} count: {}'.format(corrcoef, len(trough_min_lat)),
             transform=ax1.transAxes, color='black')
    plt.xlabel(Index)
    plt.ylabel('gdlat')
    plt.ylim(35, 65)
    plt.legend()
    title = '{}-{} {} glon_{}°lt_{}\n gdlat-{} linear fit'.format(year1, year2, season, glon, lt, Index)
    plt.title(title)
    figure_path = "C:\\DATA\\GPS_MIT\\millstone\\summary graph\\scatter plot\\lat-index\\{}\\".format(folder_name)
    if not os.path.exists(figure_path):
        os.mkdir(figure_path)
    figure1.savefig(figure_path + '{} {}-{} glon_{}°lt_{} gdlat-{} linear_fit'.
                    format(season, year1, year2, glon, lt, Index))
    plt.close()
    # figure2 = plt.figure(figsize=(8, 6))

    return True


# index_list = ['AE', 'AE6', 'kp', 'kp9', 'C9', 'Cp', 'sum_8kp', 'mean_8ap']
index_list = ['kp9']
season_list = ["year"]
glon = -90
year1, year2 = 2015, 2016
for lt in [22, 23, 0, 1, 2, 3, 4, 5, 18, 19, 20, 21]:
    folder_name = '{}-{}_{}_lt'.format(year1, year2, glon)
    DF = data_gen(glon, lt)
    for ssn in season_list:
        DF1 = data_process(DF, ssn, year1, year2)
        for _ in index_list:
            data_plotIndex(DF1, _, ssn)
