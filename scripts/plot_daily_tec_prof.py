#! python3
# -*- coding: utf-8 -*-

import os
import gc
import time
import datetime
import numpy as np
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt


def plot_tec_prof(year, glon, lt):
    out_path = figure_path + '{}\\glon_{}\\lt_{}-{}\\'.format(year, glon, lt, lt+1)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # 从bigTable中得到当前glon与lt，年份year中的所有行，以doy作为index
    df = bigTable.query("year=={} & glon=={} & lt=={}".format(year, glon, lt)).set_index(['doy'])
    trough_min_lats = df["trough_min_lat"].fillna(0.0).astype(int)               # 用0填充nan
    df = df.drop(["year", "glon", "lt", "trough_min_lat"], axis=1)

    # 开始做每一doy的tec 纬度剖面图
    lats = [_ for _ in range(30, 81)]
    for doy in df.index:
        print(doy)
        mean_tec = df.loc[doy]

        plt.scatter(lats, mean_tec, c='b', marker='o', s=20, label="TEC")

        w = np.isnan(mean_tec)                                  # 插值，处理nan情况
        mean_tec[w] = 0.
        x = np.linspace(30, 80, 100)
        y = interpolate.UnivariateSpline(lats, mean_tec, w=~w, s=5)(x)
        plt.plot(x, y, 'k--')
        plt.xlim(30, 80)
        diy_xticks = list(np.linspace(30, 80, 26))
        plt.xticks(diy_xticks)
        plt.ylim(0, 1.5 * mean_tec.max())

        cur_min = trough_min_lats.loc[doy]
        if cur_min:                                             # 非0的数据为判定槽极小的位置
            plt.scatter(cur_min, mean_tec.loc["gdlat-{}°".format(cur_min)], c='r', marker='^', s=40, label='trough min')

        figure_title = 'mean TEC lat profile  \n lt:{}-{}  glon:{}  year:{}  doy:{:3d}'.\
            format(lt, lt+1, glon, year, doy)
        plt.legend()
        plt.title(figure_title)
        plt.savefig(out_path + "{:03d}".format(doy))
        plt.close()

    del df
    gc.collect()

    return True


datapath = "C:\\DATA\\GPS_MIT\\millstone\\millstone_tec_profile.csv"
bigTable = pd.read_csv(datapath, encoding='gb2312')
figure_path = "C:\\DATA\\GPS_MIT\\millstone\\daily tec profiles\\"

years = [2017, 2016, 2015, 2014, 2013, 2012]
glon_list = [-120, -90, 0, 30]
lt_list = [22, 23, 0, 1, 2, 3, 4, 5, 18, 19, 20, 21]


for Year in years:
    for Glon in glon_list:
        for Lt in lt_list:
            plot_tec_prof(year=Year, glon=Glon, lt=Lt)

print("work done")
