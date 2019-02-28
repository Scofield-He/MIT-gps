#! python3
# -*- coding: utf-8 -*-

# import os
# import gc
# import time
# import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def df_aggregation(lt, y1, m1, d1, y2, m2, d2):
    filepath_trough = "C:\\tmp\\sorted tec_profile_gdlat.csv"
    table_trough = pd.read_csv(filepath_trough, encoding='gb2312')
    table_trough['date'] = pd.to_datetime(table_trough['date'])
    table_trough = table_trough[['year', 'doy', 'date', 'glon', 'lt', 'trough_min_lat']]
    # print(table_trough.head(3))

    filepath_kp = "C:\\tmp\\sorted kp_index.csv"
    table_kp = pd.read_csv(filepath_kp, encoding='gb2312')
    table_kp['date'] = pd.to_datetime(table_kp['date'])
    table_kp = table_kp[['year', 'doy', 'date', 'glon', 'lt', 'kp', 'kp9', 'C9']]  # 'F10.7'
    # print(table_kp.head(3))

    if len(table_trough) != len(table_kp):
        print("length of trough: {}".format(len(table_trough)))
        print("length of kp: {}".format(len(table_kp)))
        raise Exception("table length not equal")

    print('\nlength of table_trough or table_kp : {}\n'.format(len(table_kp)))

    table_new = pd.merge(table_trough, table_kp, on=['year', 'date', 'doy', 'glon', 'lt'], how='outer')
    table_new = table_new.set_index('date')
    # print(table_new.head())
    table_new = table_new['{}-{}-{}'.format(y1, m1, d1): '{}-{}-{}'.format(y2, m2, d2)]
    table_new = table_new.query("lt == {} | lt == {} | lt == {}".format(lt[0], lt[1], lt[2]))    # 只需要午夜前1h的数据
    return table_new


def plot_longitude(df, lt):
    Kp9_range1, Kp9_range2 = [2, 9], [0, 2]
    if lt[0] <= lt[1]:
        df = df.query("{} <= lt <= {}".format(lt[0], lt[1]))
    else:
        df = df.query("lt >= {} | lt <= {}".format(lt[0], lt[1]))
    df1 = df.query("{} < kp9 <= {} ".format(Kp9_range1[0], Kp9_range1[1]))     # Kp9 (2, 9]
    df2 = df.query("{} <= kp9 <= {} ".format(Kp9_range2[0], Kp9_range2[1]))     # Kp9 [0, 2]

    longitudes = ['-120', '-90', '0', '30']

    data_longitude1 = [df1.query('glon == {}'.format(_)) for _ in longitudes]
    prof_count1 = [len(_) for _ in data_longitude1]
    data_longitude1 = [_.dropna() for _ in data_longitude1]
    trough_count1 = [len(_) for _ in data_longitude1]
    print('high magnetic condition:\n', prof_count1, trough_count1)

    data_longitude2 = [df2.query('glon == {}'.format(_)) for _ in longitudes]
    prof_count2 = np.array([len(_) for _ in data_longitude2])
    data_longitude2 = [_.dropna() for _ in data_longitude2]
    trough_count2 = np.array([len(_) for _ in data_longitude2])
    print('low magnetic condition:\n', prof_count2, trough_count2)

    # 箱型图，槽极小位置的经度分布，统一为磁纬
    box_figure = plt.figure()
    ax = box_figure.add_subplot(111)
    bias = [6, 9, -2, -3]     # bias = {-120: 6, -90: 9, 0: -2, 30: -3}
    data_corrected1 = [np.array(data_longitude1[_]['trough_min_lat']) + bias[_] for _ in range(4)]      # 4个经度带
    data_corrected2 = [np.array(data_longitude2[_]['trough_min_lat']) + bias[_] for _ in range(4)]
    boxprops1, boxprops2 = dict(color='black'), dict(color='slategrey')
    medianprops1, medianprops2 = dict(color='black', linestyle='--'), dict(color='slategrey', linestyle='--')
    whiskerprops1, whiskerprops2 = dict(color='black'), dict(color='slategrey')
    capprops1, capprops2 = dict(color='black'), dict(color='slategrey')
    plt.boxplot(data_corrected1, notch=False, sym='', vert=True, whis=[5, 95], showfliers=False, showbox=True,
                showcaps=True, positions=[0.85, 1.85, 2.85, 3.85], widths=0.2, boxprops=boxprops1,
                whiskerprops=whiskerprops1, capprops=capprops1, medianprops=medianprops1)
    plt.boxplot(data_corrected2, notch=False, sym='', vert=True, whis=[5, 95], showfliers=False, showbox=True,
                showcaps=True, positions=[1.15, 2.15, 3.15, 4.15], widths=0.2, boxprops=boxprops2,
                whiskerprops=whiskerprops2, capprops=capprops2, medianprops=medianprops2)
    plt.xlabel('Geo. Longitude(degree)', fontsize=12)
    plt.xticks([1, 2, 3, 4], [-120, -90, 0, 30], fontsize=12)
    plt.xlim([0.2, 5])
    plt.yticks(fontsize=12)
    plt.ylabel('Gm. Latitude(degree)', fontsize=12)
    # plt.ylim([52, 66])
    # plt.title('boxplot of lat_longitude lt_{}-{}'.format(lt[0], lt[1] + 1), fontsize=12)
    ax.text(0.8, 0.1, 'Kp9∈(2, 9]', transform=ax.transAxes, color='k')
    ax.text(0.8, 0.2, 'Kp9∈[0, 2]', transform=ax.transAxes, color='slategrey')
    plt.subplots_adjust(left=0.1, right=0.95, top=0.93)
    plt.savefig(outpath + 'figure4.eps', fmt='eps')
    plt.savefig(outpath + 'figure4.jpg', fmt='jpg')
    plt.show()
    plt.close('all')


# outpath = 'C:\\DATA\\GPS_MIT\\millstone\\summary graph\\'
# outpath = 'C:\\tmp\\figure\\lat-longitude\\'
outpath = 'C:\\tmp\\figure\\eps\\'
if __name__ == '__main__':
    year1, month1, day1, year2, month2, day2 = 2014, 9, 1, 2017, 8, 31
    DF = df_aggregation([22, 23, 0], year1, month1, day1, year2, month2, day2)    # lt: 22, 23, 0
    print(DF.columns)
    print('length of data Df: ', len(DF))

    plot_longitude(DF, [23, 0])         # lt: 23-1, 槽的数目直方图，槽极小位置箱线图
