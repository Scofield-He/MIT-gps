#! python3
# -*- coding: utf-8 -*-

# import os
# import gc
# import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator


if __name__ == '__main__':
    glon_sector = -90
    lt1, lt2 = 18, 6
    col_names = ['year', 'date', 'doy', 'glon', 'lt', 'trough_min_lat']
    csv_path = 'C:\\DATA\\GPS_MIT\\millstone\\'

    df_corrected = pd.read_csv('C:\\tmp\\bigtable.csv', encoding='gb2312'). \
        query('glon == {}'.format(glon_sector))[col_names]
    df_corrected['date'] = pd.to_datetime(df_corrected['date'])
    df_corrected = df_corrected.set_index('date')
    df_corrected = df_corrected['2014-9-1': '2017-8-31'].reset_index()
    df_corrected = df_corrected.query('lt >= {} or lt < {}'.format(lt1, lt2))
    print('df_corrected, length: ', len(df_corrected))
    print(df_corrected.head())

    dfA = pd.read_csv(csv_path + 'tec_profile_gdlat_A.csv', encoding='gb2312').\
        query('glon == {}'.format(glon_sector))[col_names]
    dfA['date'] = pd.to_datetime(dfA['date'])
    dfA = dfA.set_index('date')
    dfA = dfA['2014-9-1': '2017-8-31'].reset_index()
    dfA = dfA.query('lt >= {} or lt < {}'.format(lt1, lt2))
    print('dfA, length: ', len(dfA))
    print(dfA.head())

    dfB = pd.read_csv(csv_path + 'tec_profile_gdlat_B.csv', encoding='gb2312').\
        query('glon == {}'.format(glon_sector))[col_names]
    dfB['date'] = pd.to_datetime(dfB['date'])
    dfB = dfB.set_index('date')
    dfB = dfB['2014-9-1': '2017-8-31'].reset_index()
    dfB = dfB.query('lt >= {} or lt < {}'.format(lt1, lt2))
    print('dfB, length: ', len(dfB))
    print(dfB.head())

    # figure_path = "C:\\Users\\user\\Desktop\\submission version1 reply\\figures\\"
    figure_path = "C:\\tmp\\figure\\eps\\"
    lats = list(range(30, 81))

    def f(_df, source):
        trough_corrected = list(_df['trough_min_lat'])  # 人工识别后的槽极小纬度分布
        trough_count = [0] * len(lats)
        for lat in lats:
            trough_count[lat - 30] = trough_corrected.count(lat)
        # print(len(trough_count), trough_count)
        # print(len(lats), lats)
        d_title = {0: 'manual identification', 1: 'old automatic progress identification',
                   2: 'new automatic progress identification'}
        d_color = {0: 'black', 1: 'green', 2: 'black'}
        figure_TroughCorrected = plt.figure(figsize=(6, 4))  # 作图
        plt.bar(lats, trough_count, color=d_color[source], label='trough number')
        plt.legend(loc='upper right')
        plt.xlabel('gdlat(°)')
        plt.ylabel('count')
        plt.xlim(30, 80)

        # plt.title('latitudinal distribution of troughs identified manually at -90°at {}:00-{}:00 LT'.format(lt1, lt2))
        if source == 0:
            figure_TroughCorrected.savefig(figure_path + 'figure3.eps', fmt='eps')
            figure_TroughCorrected.savefig(figure_path + 'figure3.jpg', fmt='jpg')
            # figure_TroughCorrected.savefig(figure_path + 'figure3.eps'.format(d_title[source], lt1, lt2))
        # plt.show()
        plt.close()

        count_nan = 0  # 人工识别后nan的数目
        for _ in trough_corrected:
            if np.isnan(_):
                count_nan += 1
        print(d_title[source], count_nan)
        return trough_count

    # 统计人工校正后的槽极小结果分布情况
    trough_count_manual = f(df_corrected, 0)

    # 原程序识别方法，需要人工修正的数目统计，区分范围与类别；dfA
    trough_count_A = f(dfA, 1)

    # 改进为磁纬后，需要人工修正的数目统计，区分范围与类别；dfB
    trough_count_B = f(dfB, 2)

    # 同一张图，对比三个结果的纬度分布
    def f1(trough_manual=None, trough_1=None, trough_2=None):
        figure_comparison = plt.figure(figsize=(8, 4))
        width = 0.4 if (not trough_2 or not trough_1) else 0.3
        if trough_manual:
            plt.bar(lats, trough_count_manual, width=width, facecolor='blue', label='manual')
        if trough_1:
            plt.bar(np.array(lats) - width, trough_count_A, width=width, facecolor='green', label='old')
        if trough_2:
            plt.bar(np.array(lats) + width, trough_count_B, width=width, facecolor='black', label='new')
        plt.legend(loc='upper right')
        plt.xlim([30, 80])
        plt.xlabel('gdlat(°)')
        plt.ylabel('count')
        plt.title('Comparison of Latitudinal-Distributions of Mid-Lat-Trough at {}:00-{}:00 LT'.format(lt1, lt2))
        figure_comparison.savefig(figure_path + 'comparison between {}-{}-{} at {}-{}'.format(
            trough_manual, trough_1, trough_2, lt1, lt2))
        plt.show()
        return True

    # f1(trough_manual='manual', trough_1='A', trough_2='B')
    # f1(trough_manual='manual', trough_1='A')
    # f1(trough_manual='manual', trough_2='B')

    # 给出具体每个区域，两种程序识别与人工的差异
    def f2(df1, df2):           # df2为人工识别结果
        # 作纬度分布图对比两个结果
        df = pd.merge(df1, df2, on=col_names[:-1])
        # print('df, length: ', len(df))
        # print(df.head())
        x, y = np.array(df['trough_min_lat_x']), np.array(df['trough_min_lat_y'])
        count_equal = 0           # 与人工识别结果相同
        count_LatError = 0           # 非nan，槽位置错误
        count_nanError = 0           # 有槽，未识别出
        count_isError = 0            # 无槽，识别出槽

        for _ in range(len(x)):
            if np.isnan(x[_]) and np.isnan(y[_]):
                count_equal += 1
            elif not np.isnan(x[_]) and not np.isnan(y[_]):
                if x[_] != y[_]:
                    count_LatError += 1
                    # print(x[_], y[_])
                else:
                    count_equal += 1
            elif np.isnan(x[_]):
                count_nanError += 1
            else:
                count_isError += 1
        print('total', 'equal', ' valueError', 'isButNan', 'notButVal')
        print('{:5d} {:5d} {:10d} {:8d} {:9d}'.format(
            len(df), count_equal, count_LatError, count_nanError, count_isError))
        return True

    # 对比旧方法结果与人工识别结果；
    # f2(dfA, df_corrected)

    # 对比新方法结果与人工识别结果：
    # f2(dfB, df_corrected)

    print('Work Done')
