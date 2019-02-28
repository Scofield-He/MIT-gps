#! python3
# -*- coding: utf-8 -*-

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd


def data_aggregation(path_profile, path_trough, glon, lt=None):
    df1 = pd.read_csv(path_profile, encoding='gb2312').\
        query('glon == {}'.format(glon)).drop(columns=['year', 'doy', 'glon'])
    var_names = ['date', 'lt', 'mit_auto'] + ['%d' % _ for _ in range(30, 81)]
    df1.columns = var_names               # 重命名列
    df1['date'] = pd.to_datetime(df1['date'])
    df1 = df1.set_index('date')['2014-9-1': '2017-8-31'].reset_index('date')

    df2 = pd.read_csv(path_trough, usecols=[0, 1, 2, 3, 4, 5, 7, 9], encoding='gb2312').\
        query('glon == {}'.format(glon)).drop(columns=['year', 'doy', 'glon'])
    df2['date'] = pd.to_datetime(df2['date'])
    df2 = df2.set_index('date')['2014-9-1': '2017-8-31'].reset_index('date')

    # 按照纬度剖面合并，删除不用的列
    df = pd.merge(df2, df1, on=['date', 'lt'], how='outer').drop(columns=[])

    if lt is not None:
        df = df.query('lt == {}'.format(lt))

    print(df.head(10))
    print(len(df))
    return df


def plot_figure(df, lt, t1, t2, outpath):
    df = df.query('lt == {}'.format(lt)).set_index('date').sort_index()[t1: t2]

    df = df.drop(columns=['lt', 'mit_auto', 'AE6', 'Kp9'])
    print(df.head())

    figure = plt.figure(figsize=(10, 4))
    df_profiles = df.drop(columns=['trough_min_lat']).T
    print('df.T:\n', df_profiles)
    ax1 = figure.add_subplot(111)
    im = ax1.imshow(df_profiles, vmax=12, vmin=1, cmap=plt.cm.jet, origin='lower')

    ax1.set_ylabel('gdlat (degree)')
    ax1.set_xlabel('date')
    plt.sca(ax1)
    auto_xticks = [_ for _ in ax1.get_xticks() if 0 <= _ < len(df)]
    print(auto_xticks)
    date_ticks = list(map(lambda x: x.strftime(format='%Y-%m-%d'),
                      [(df.index[0] + datetime.timedelta(_)).date() for _ in auto_xticks]))
    print(date_ticks)
    plt.xticks(auto_xticks, date_ticks)
    auto_yticks = [int(_) for _ in ax1.get_yticks() if 0 <= _ <= 50]
    print(auto_yticks)
    plt.yticks(auto_yticks, list(map(lambda x: x+30, auto_yticks)))

    trough_min_lat = np.array(df['trough_min_lat'])
    ax1.plot(trough_min_lat - 30, 'r.', alpha=0.6)

    cax = plt.axes([0.93, 0.12, 0.01, 0.8])
    plt.colorbar(im, cax=cax, label='TEC (TECU)')

    plt.subplots_adjust(top=0.92, bottom=0.12, left=0.06, right=0.92,)
    plt.show()
    figure.savefig(outpath + 'profiles_{}-{}'.format(t1.strftime('%Y%m%d'), t2.strftime('%Y%m%d')))

    return None


if __name__ == '__main__':
    data_path = 'C:\\DATA\\GPS_MIT\\millstone\\'
    path_tec_profiles = data_path + 'tec_profile_gdlat_A.csv'
    path_MIT = data_path + 'bigtable.csv'
    geo_lon = -90
    DF = data_aggregation(path_tec_profiles, path_MIT, glon=geo_lon)

    t_from, t_to = datetime.date(2016, 10, 10), datetime.date(2017, 2, 20)
    localtime = 0
    path_figure = 'C:\\Users\\user\\Desktop\\mid-Lat-trough_solar-Wind\\figures\\new\\'
    plot_figure(DF, localtime, t_from, t_to, path_figure)
