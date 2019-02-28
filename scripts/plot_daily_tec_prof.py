#! python3
# -*- coding: utf-8 -*-

import os
import gc
import time
import numpy as np
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


def plot_tec_prof(year, glon, lt, data_range):
    # out_path = figure_path + '{}\\figure\\profiles_{}\\lt_{}-{}\\'.format(year, glon, lt, lt + 1)
    out_path = figure_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # 从bigTable中得到当前glon与lt，年份year中的所有行，以doy作为index
    df = bigTable_tec.query("year=={} & glon=={} & lt=={}".format(year, glon, lt)).set_index(['date', 'doy'])
    trough_min_lats = df["trough_min_lat"].fillna(0.0).astype(int)               # 用0填充nan
    df = df.drop(["year", "glon", "lt", "trough_min_lat"], axis=1)

    index_cur_year = bigTable_kp.query("year=={} & glon=={} & lt=={}".format(year, glon, lt)).set_index(['date', 'doy'])
    kp_index_cur_year = index_cur_year.drop(["year", "glon", "lt"], axis=1).kp

    # 开始作每一doy的tec 纬度剖面图
    matplotlib.rcParams.update({'font.size': 15})  # 修改所有label和tick的大小
    lats = [_ for _ in range(30, 81)]
    for date, doy in df.index:
        if doy not in [353]:
            continue
        # if year == 2017 and doy > 273:
        #     break
        time0 = time.time()
        print(date, doy, glon, lt, end='  ')
        mean_tec = df.loc[date, doy]
        kp = kp_index_cur_year.loc[date, doy]
        # print(mean_tec)
        # print(kp)

        fig = plt.figure(figsize=(10, 5.5))
        plt.scatter(lats, mean_tec, c='k', marker='o', s=40, alpha=0.5, label="TEC")

        w = np.isnan(mean_tec)                                  # 插值曲线，处理nan情况
        mean_tec[w] = 0.
        x = np.linspace(30, 80, 100)
        y = interpolate.UnivariateSpline(lats, mean_tec, w=~w, s=5)(x)      # 能外推的样条插值, 默认k=3,即三阶样条曲线
        plt.plot(x, y, 'k--', linewidth=2)
        plt.xlim(30, 80)
        diy_xticks = list(np.linspace(30, 80, 26))
        plt.xticks(diy_xticks)
        plt.xlabel("Geographic Latitude(degree)")
        y_min, y_max = 0, min(2 * mean_tec.mean(), 20)
        plt.ylim(y_min, y_max)
        plt.ylabel("TEC(TECU)")
        plt.subplots_adjust(top=0.92, left=0.08, right=0.95, bottom=0.12)

        if data_range == 'B':
            x1, x2 = 36, 61
            mean_y = mean_tec[6:32].mean()
        elif data_range == 'A':
            x1, x2 = 45, 70
            mean_y = mean_tec[15:41].mean()
        else:
            raise ValueError('tec profiles data source is invalid!')
        plt.plot([x1, x1], [y_min, y_max], 'k:', linewidth=1.5)                # 作出程序自动识别时槽极小取均值范围
        plt.plot([x2, x2], [y_min, y_max], 'k:', linewidth=1.5)
        plt.plot([x1, x2], [mean_y, mean_y], 'k-.', linewidth=1.5)

        cur_min = trough_min_lats.loc[date, doy]
        if cur_min:                                             # 非0的数据为判定槽极小的位置
            plt.scatter(cur_min, mean_tec.loc["gdlat-{}°".format(cur_min)], c='k', marker='^', s=80, label='trough min')

        figure_title = 'date:{}  doy:{:3d}  lt:{}h  glon:{}  kp:{}'.format(date, doy, lt, glon, kp)
        # figure_title = '{} lt:{}-{}  glon:{}'.format(date, lt, lt + 1, glon)
        plt.legend()
        # plt.title(figure_title)
        # fig.savefig(out_path + "{} {:03d}_{} lt_{} glon_{}".format(year, doy, data_range, lt, glon))
        fig.savefig(out_path + "figure2_lt-{}.eps".format(lt), fmt='eps')
        fig.savefig(out_path + "figure2_lt-{}.jpg".format(lt), fmt='jpg')
        # plt.show()
        plt.close()
        print("time cost : {}s".format(round(time.time()-time0, 2)))
        # break

    del df
    del index_cur_year, kp_index_cur_year
    gc.collect()
    return


# tec_datapath = "C:\\DATA\\GPS_MIT\\millstone\\tec_profile_gdlat.csv"
# tec_datapath = "C:\\DATA\\GPS_MIT\\millstone\\tec_profile_gdlat_B.csv"
tec_datapath = "C:\\DATA\\GPS_MIT\\millstone\\tec_profile_gdlat_A.csv"
bigTable_tec = pd.read_csv(tec_datapath, encoding='gb2312')
kp_datapath = "C:\\DATA\\GPS_MIT\\millstone\\kp_index.csv"
bigTable_kp = pd.read_csv(kp_datapath, encoding='gb2312')

# figure_path = "C:\\DATA\\GPS_MIT\\millstone\\"
figure_path = "C:\\tmp\\figure\\eps\\tec_prof\\"

# years = [2017, 2016, 2015, 2014]
# glon_list = [-120, -90, 0, 30]
# lt_list = [22, 23, 0, 1, 2, 3, 4, 5, 18, 19, 20, 21]
years = [2016]
glon_list = [-90]
lt_list = [22, 23, 0, 1]
for Year in years:
    for Glon in glon_list:
        for Lt in lt_list:
            plot_tec_prof(year=Year, glon=Glon, lt=Lt, data_range='B')

print("work done")
