#! python3
# -*- coding: utf-8 -*-

import os
import gc
import time
import numpy as np
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt


def plot_tec_prof(year, glon, lt):
    out_path = figure_path + '{}\\glon_{}\\lt_{}-{}\\'.format(year, glon, lt, lt+1)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # 从bigTable中得到当前glon与lt，年份year中的所有行，以doy作为index
    df = bigTable_tec.query("year=={} & glon=={} & lt=={}".format(year, glon, lt)).set_index(['date', 'doy'])
    trough_min_lats = df["trough_min_lat"].fillna(0.0).astype(int)               # 用0填充nan
    df = df.drop(["year", "glon", "lt", "trough_min_lat"], axis=1)

    index_cur_year = bigTable_kp.query("year=={} & glon=={} & lt=={}".format(year, glon, lt)).set_index(['date', 'doy'])
    kp_index_cur_year = index_cur_year.drop(["year", "glon", "lt"], axis=1).kp

    # 开始作每一doy的tec 纬度剖面图
    lats = [_ for _ in range(30, 81)]
    for date, doy in df.index:
        # if doy not in [353]:
        #    continue
        if year == 2017 and doy > 273:
            break
        time0 = time.time()
        print(date, doy, glon, lt, end='  ')
        mean_tec = df.loc[date, doy]
        kp = kp_index_cur_year.loc[date, doy]
        # print(mean_tec)
        # print(kp)

        # fig = plt.figure(figsize=(8, 4))
        plt.scatter(lats, mean_tec, c='b', marker='o', s=20, label="TEC")

        w = np.isnan(mean_tec)                                  # 插值，处理nan情况
        mean_tec[w] = 0.
        x = np.linspace(30, 80, 100)
        y = interpolate.UnivariateSpline(lats, mean_tec, w=~w, s=5)(x)
        plt.plot(x, y, 'k--')
        plt.xlim(30, 80)
        diy_xticks = list(np.linspace(30, 80, 26))
        plt.xticks(diy_xticks)
        plt.xlabel("Geographic Latitude(degree)")
        plt.ylim(0, min(2 * mean_tec.mean(), 20))
        # plt.ylim(0, 8)
        plt.ylabel("TEC(TECU)")

        cur_min = trough_min_lats.loc[date, doy]
        if cur_min:                                             # 非0的数据为判定槽极小的位置
            plt.scatter(cur_min, mean_tec.loc["gdlat-{}°".format(cur_min)], c='r', marker='^', s=40, label='trough min')

        figure_title = '{} mean TEC lat profile  \n lt:{}-{}  glon:{}  year:{}  doy:{:3d} kp:{}'.\
            format(date, lt, lt+1, glon, year, doy, kp)
        # figure_title = '{} lt:{}-{}  glon:{}'.format(date, lt, lt + 1, glon)
        plt.legend()
        plt.title(figure_title)
        plt.savefig(out_path + "{:03d}".format(doy))
        # plt.savefig('C:\\Users\\user\\Desktop\\to liujing\\' + '{:03d}'.format(doy))
        plt.close()
        print("time cost : {}s".format(round(time.time()-time0, 2)))
        # break

    del df
    del index_cur_year, kp_index_cur_year
    gc.collect()

    return True


tec_datapath = "C:\\DATA\\GPS_MIT\\millstone\\tec_profile_gdlat.csv"
bigTable_tec = pd.read_csv(tec_datapath, encoding='gb2312')
kp_datapath = "C:\\DATA\\GPS_MIT\\millstone\\kp_index.csv"
bigTable_kp = pd.read_csv(kp_datapath, encoding='gb2312')

figure_path = "C:\\DATA\\GPS_MIT\\millstone\\daily tec profiles\\"

years = [2017, 2016, 2015, 2014]
glon_list = [0]
# lt_list = [22, 23, 0, 1, 2, 3, 4, 5, 18, 19, 20, 21]
lt_list = [22, 23, 0]
for Year in years:
    for Glon in glon_list:
        for Lt in lt_list:
            plot_tec_prof(year=Year, glon=Glon, lt=Lt)

print("work done")
