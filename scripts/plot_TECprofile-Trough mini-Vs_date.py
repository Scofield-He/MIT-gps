#! python3
# -*- coding: utf-8 -*-

# import os
# import gc
# import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator
import matplotlib


matplotlib.rcParams.update({'font.size': 12})        # 修改所有label和tick的大小


def plot_TECProfile_date(df_TEC, lt, dates, out_path):
    df_TEC = df_TEC['{}-{}-{}'.format(dates[0], dates[1], dates[2]): '{}-{}-{}'.format(dates[3], dates[4], dates[5])]\
        .query('lt == {}'.format(lt))

    figure = plt.figure(figsize=(12, 5))
    x, y = [], []                              # date and gdlat
    TEC_profiles = df_TEC.ix[:, 5:]            # 取TEC纬度剖面数据
    print(len(TEC_profiles))
    # print(TEC_profiles.head())
    color = []                                 # 将纬度剖面数据变成1维，每日的值做归一化？
    for row in range(len(TEC_profiles)):
        TEC_profile = TEC_profiles.ix[row, ]
        date = TEC_profiles.index[row].date()
        tec_profile = list(np.array(TEC_profile))
        min_tec, max_tec = min(tec_profile), max(tec_profile)
        # tec_profile = [15 * (_ - min_tec) / (max_tec - min_tec) for _ in tec_profile]
        color += list(tec_profile)
        # x += [(date - min(df_TEC.index)).days] * len(tec_profile)
        x += [date] * len(tec_profile)
        y += [_ for _ in range(30, 81)]
    print(len(x), len(y), len(color))
    plt.scatter(x, y, s=35, c=color, cmap=plt.cm.jet, marker='s', vmin=0, vmax=10)
    plt.xlim(min(df_TEC.index) - datetime.timedelta(1),
             max(df_TEC.index) + datetime.timedelta(1))
    plt.ylim(29, 81)
    plt.plot(df_TEC.index, df_TEC['trough_min_lat'], 'wx', linewidth=3, ms=12)

    plt.ylabel('Geo. Latitude(°)')
    plt.yticks([30, 40, 50, 60, 70, 80])
    plt.xlabel('date', labelpad=0)          # 调整xlabel的位置
    title = 'TEC Profiles and Corresponding Latitude of Mid-Latitude Trough at {} LT ' \
            'from {}-{}-{} to {}-{}-{}'.format(lt, dates[0], dates[1], dates[2], dates[3], dates[4], dates[5])
    plt.title(title)

    for label in plt.gca().xaxis.get_ticklabels():
        label.set_rotation(0)

    plt.subplots_adjust(left=0.06, bottom=0.08, right=0.93, top=0.92)
    cax = plt.axes([0.94, 0.08, 0.01, 0.84])
    c = plt.colorbar(cax=cax, orientation='vertical', pad=0.05, ticks=[0, 2, 4, 6, 8, 10])
    c.set_label('TEC(TECU)')
    # normalized or not
    # figure.savefig(out_path + 'lt_{}\\LS\\'.format(lt) + 'normalized tec profiles at {} LT from {}-{}-{} to {}-{}-{}'.
    #                format(lt, dates[0], dates[1], dates[2], dates[3], dates[4], dates[5]))
    figure.savefig(out_path + 'tec profiles at lt_{} from {}-{}-{} to {}-{}-{}'.
                   format(lt, dates[0], dates[1], dates[2], dates[3], dates[4], dates[5]))
    plt.show()
    return True


# TEC Profile and corrected trough_min_gdlat data
path_TEC_profile_csv = 'C:\\tmp\\sorted tec_profile_gdlat.csv'

# Solar Wind and corrected trough_min_gdlat data
path_SolarWind_csv = 'C:\\tmp\\data.csv'

# Local time and date range
localtime_list = [[19, 20], [21, 22], [23, 0], [1, 2], [3, 4]]
gdlon = -90
Kp_range = [0, 9]
date_range_list = [[2014, 10, 20, 2015, 7, 1],
                   [2015, 3, 1, 2015, 7, 1],
                   [2015, 6, 17, 2015, 7, 25],
                   [2015, 9, 1, 2015, 12, 15],
                   [2016, 1, 10, 2016, 2, 22],
                   [2016, 7, 1, 2017, 6, 1],
                   [2016, 8, 1, 2016, 10, 31],
                   [2017, 2, 1, 2017, 5, 31],
                   [2016, 10, 1, 2016, 12, 31]]

# figure outpath
outpath = 'C:\\tmp\\figure\\solar-wind\\'

data_TEC = pd.read_csv(path_TEC_profile_csv, encoding='gb2312').query('glon == {}'.format(gdlon))
data_TEC['date'] = pd.to_datetime(data_TEC['date'])
data_TEC = data_TEC.set_index('date')
data_Vs = pd.read_csv(path_SolarWind_csv, encoding='gb2312').query('glon == {}'.format(gdlon))
data_Vs['date'] = pd.to_datetime(data_Vs['date'])
data_Vs = data_Vs.set_index('date')

LocalTime = [20, 22, 0, 2, 4]
for localtime in LocalTime:
    date_range = [2015, 9, 1, 2015, 12, 15]
    print('localtime: {} ; date_range: {} --------------------'
          .format(localtime, date_range))
    plot_TECProfile_date(data_TEC, localtime, date_range, outpath)
