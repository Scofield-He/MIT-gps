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


def plot_longitude(df, Kp9, lt):
    df = df.query("{} <= kp9 <= {} ".format(Kp9[0], Kp9[1]))
    if lt[0] <= lt[1]:
        df = df.query("{} <= lt <= {}".format(lt[0], lt[1]))
    else:
        df = df.query("lt >= {} | lt <= {}".format(lt[0], lt[1]))

    longitudes = ['-120', '-90', '0', '30']
    data_longitude = [df.query('glon == {}'.format(_)) for _ in longitudes]
    prof_count = [len(_) for _ in data_longitude]
    data_longitude = [_.dropna() for _ in data_longitude]
    trough_count = [len(_) for _ in data_longitude]
    print(prof_count, trough_count)

    plt.bar(longitudes, prof_count, fc='lightskyblue', label='lat-prof count')
    plt.bar(longitudes, trough_count, fc='blue', label='trough number')
    plt.ylim(0, max(prof_count) * 1.3)
    plt.ylabel('count')
    # plt.xticks(['-120', '-90', '0', '30'])
    plt.xlabel('Geo. Longitude (degree)')
    plt.legend()
    plt.title('2014/9/1-2017/8/31  lt_{}-{}  Kp9_{}-{}'.format(lt[0], lt[1] + 1, Kp9[0], Kp9[1]))
    plt.savefig(outpath + 'counts at longitudes lt_{}-{} Kp9_{}-{}'.format(lt[0], lt[1] + 1, Kp9[0], Kp9[1]))
    plt.show()
    plt.close('all')

    # 箱型图，槽极小位置的经度分布，统一为磁纬
    bias = [6, 9, -2, -3]     # bias = {-120: 6, -90: 9, 0: -2, 30: -3}
    data_corrected = [np.array(data_longitude[_]['trough_min_lat']) + bias[_] for _ in range(4)]      # 4个经度带
    plt.boxplot(data_corrected)
    plt.xlabel('Geo. Longitude(degree)')
    plt.xticks([1, 2, 3, 4], [-120, -90, 0, 30])
    plt.ylabel('Gm. Latitude(degree)')
    plt.title('boxplot of lat_longitude lt_{}-{} Kp9_{}-{}'.format(lt[0], lt[1] + 1, Kp9[0], Kp9[1]))
    plt.savefig(outpath + 'boxplot of lat_month lt_{}-{} Kp9_{}-{}'.format(lt[0], lt[1] + 1, Kp9[0], Kp9[1]))
    plt.show()
    plt.close('all')


# outpath = 'C:\\DATA\\GPS_MIT\\millstone\\summary graph\\'
outpath = 'C:\\tmp\\figure\\lat-longitude\\'
if __name__ == '__main__':
    year1, month1, day1, year2, month2, day2 = 2014, 9, 1, 2017, 8, 31
    DF = df_aggregation([22, 23, 0], year1, month1, day1, year2, month2, day2)    # lt: 22, 23, 0
    print(DF.columns)
    print('length of data Df: ', len(DF))

    plot_longitude(DF, [0, 9], [22, 0])         # lt: 23-1, 槽的数目直方图，槽极小位置箱线图
