#! python3
# -*- coding: utf-8 -*-

import os
# import gc
# import time
import datetime
import numpy as np
# from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


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
    # print(df_kp[:5])

    df_ae = bigTable_AE.query(("glon == {} & lt == {}".format(glon_c, lt1))).drop(['date'], axis=1)
    print("length of ae index dataframe: ", df_ae.__len__())
    # print(df_ae[:5])
    # ae_index, ae6_index = df_ae["AE"], df_ae["AE6"]

    # df_total = pd.merge(df_tec_prof, df_kp, how='inner')
    # print(len(df_total), df_total.columns)
    # print(df_total["kp"].describe())

    # df_new = df_kp.copy()
    df_kp.insert(5, "trough_min_lat", trough_mini)                # 插入槽极小的纬度，键值顺序相同，故直接插入即可
    df_new = pd.merge(df_kp, df_ae, on=['year', 'doy', 'glon', 'lt'], how='outer')      # 按照键值合并
    print(len(df_new), df_new.columns)
    # print(df_new[:10])
    return df_new


def data_process(df, season, y1, m1, d1, y2, m2, d2):
    print('count of df from function data_gen', len(df))
    df = df.dropna()
    print('count of not null trough min: ', len(df))
    # df = df.query("35 < trough_min_lat <= 70 ")
    doy1 = datetime.date(y1, m1, d1).timetuple().tm_yday
    doy2 = datetime.date(y2, m2, d2).timetuple().tm_yday
    df = df.query("({} < year < {}) | (year == {} & doy >= {}) | (year == {} & doy <= {}) "
                  .format(y1, y2, y1, doy1, y2, doy2))

    if season == 'summer':
        ret = df.query("121 <= doy <= 243")
    elif season == 'winter':
        ret = df.query("(doy >= 305) | (doy <= 59)")
    elif season == 'equinox':
        ret = df.query("(60 <= doy <= 120) | (244 <= doy <= 304)")
    else:
        ret = df
    return ret


def record_coef(df, Index):
    # data = df[['AE', 'AE6', 'kp', 'kp9', 'F10.7', 'trough_min_lat']]
    # print("data corrcoef matrix: \n", data.corr())

    trough_min_lat = df["trough_min_lat"]
    mag_index = df[Index]

    corrcoef = round(mag_index.corr(trough_min_lat), 2)
    print('corrcoef between kp and trough min lat: ', corrcoef)

    z = list(np.polyfit(list(mag_index), list(trough_min_lat), 1))         # 线性拟合,得到ax + b中 a&b的值
    lt_seq = lt if lt > 12 else lt + 24                                    # 使lt值易于按照时间排序
    parameters.append([lt_seq, round(z[0], 2), round(z[1], 2), corrcoef])  # 记录斜率，截距及相关系数

    return True


def xtick_lt_formatter(x, _):
    return '{}'.format(int(x % 24))


def param_lt(param):
    print(param)
    Lt, slope, intercept, cc = param[0], param[1], param[2], param[3]
    """
    Lt = [_ + 24 if _ < 12 else _ for _ in Lt]
    Lt.sort()
    """
    print(['{}'.format(_ % 24) for _ in Lt])

    figure2 = plt.figure(figsize=(9, 5))
    ax1 = figure2.add_subplot(221)
    plt.sca(ax1)
    plt.scatter(Lt, slope)
    plt.xlabel('lt')
    # plt.xticks([Lt], ['{}'.format(_ % 24) for _ in Lt])
    ax1.xaxis.set_major_formatter(FuncFormatter(xtick_lt_formatter))
    plt.ylabel('slope')

    ax2 = figure2.add_subplot(222)
    plt.sca(ax2)
    plt.scatter(Lt, intercept)
    plt.xlabel('lt')
    # plt.xticks([Lt], [r'${}$'.format(_ % 24) for _ in Lt])
    ax2.xaxis.set_major_formatter(FuncFormatter(xtick_lt_formatter))
    plt.ylabel('intercept')

    ax3 = figure2.add_subplot(223)
    plt.sca(ax3)
    plt.scatter(Lt, cc)
    plt.xlabel('lt')
    ax3.xaxis.set_major_formatter(FuncFormatter(xtick_lt_formatter))
    plt.ylabel('corrcoef')

    ax4 = figure2.add_subplot(224)
    plt.sca(ax4)
    plt.scatter(slope, intercept)
    z = list(np.polyfit(slope, intercept, 1))         # 线性拟合,得到ax + b中 a&b的值
    x = np.linspace(-3, -1.4, 9)
    y = np.array([z[0] * _ + z[1] for _ in x])
    plt.plot(x, y, 'r--', label='{:.2f} * x + {:.2f}'.format(round(z[0], 2), round(z[1], 2)))
    plt.text(0.80, 0.80, 'cc = {}'.format(round(np.corrcoef(slope, intercept)[0][1], 2)),
             transform=ax4.transAxes, color='black')
    plt.xlabel('slope')
    plt.ylabel('intercept')

    figure2.suptitle("{}-{} glon_{}°Kp9".format(year1, year2, glon), fontsize=16, x=0.5, y=0.95)
    # figure2.savefig(figure_path + 'fitted coef - lt {}_{} glon_{}° Kp9'.format(year1, year2, glon))
    plt.show()
    return True


if __name__ == '__main__':
    # index_list = ['AE', 'AE6', 'kp', 'kp9']
    index_list = ['kp9']
    season_list = ["year"]
    kp9_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    glon = -90
    year1, month1, day1 = 2014, 9, 1
    year2, month2, day2 = 2017, 9, 1
    parameters = []

    for lt in [22, 23, 0, 1, 2, 3, 4, 5, 18, 19, 20, 21]:
        DF = data_gen(glon, lt)
        for ssn in season_list:
            DF1 = data_process(DF, ssn, year1, month1, day1, year2, month2, day2)
            folder_name = '{}.{}.{}-{}.{}.{}_{}_lt'.format(year1, month1, day1, year2, month2, day2, glon)
            figure_path = "C:\\DATA\\GPS_MIT\\millstone\\summary graph\\scatter plot\\lat-lt\\{}\\".format(folder_name)
            if not os.path.exists(figure_path):
                os.mkdir(figure_path)

            for _ in index_list:
                record_coef(DF1, _)  # 记录槽极小位置与对应
    parameters.sort()  # 按照第一列，即lt排序
    parameters = np.array(parameters).T  # 转置，以方便取各行
    print(parameters)
    param_lt(parameters)
    print("work done!")
