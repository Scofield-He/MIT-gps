#! python3
# -*- coding: utf-8 -*-

# import os
# import gc
# import time
# import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def dateparse1(dates):
    return pd.datetime.strptime(dates, '%Y-%m-%d')


def dateparse2(dates):
    return pd.datetime.strptime(dates, '%Y/%m/%d')


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


def plot_pie_isnan(df):
    # df中槽极小存在的占比, 饼图
    print(df.columns)
    print(df.head())
    total_count = len(df)
    not_nan_count = df.trough_min_lat.count()
    nan_count = total_count - not_nan_count
    print(total_count, not_nan_count)
    # 饼图
    plt.rcParams['font.sans-serif'] = ['SimHei']                # 解决中文乱码问题
    plt.figure(figsize=(6, 5))
    labels = [u'槽结构可分辨', u'槽结构不明显']
    sizes = [not_nan_count, nan_count]
    colors = ['lightskyblue', 'yellowgreen']
    explode = (0, 0)
    patches, text1, text2 = plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                                    autopct='%2.1f%%', shadow=False, startangle=90, pctdistance=0.55)
    for t in text2:
        t.set_size = 50
    plt.axis('equal')
    plt.legend()
    plt.savefig(outpath + 'pie plot lt_22-0 longitude_4all')
    plt.show()
    return True


def plot_month(df, Kp9, lt):
    df = df.query("{} <= kp9 <= {} ".format(Kp9[0], Kp9[1]))
    if lt[0] <= lt[1]:
        df = df.query("{} <= lt <= {}".format(lt[0], lt[1]))
    else:
        df = df.query("lt >= {} | lt <= {}".format(lt[0], lt[1]))
    print(df.head())
    # print(pd.concat([df['2014-10'], df['2015-10'], df['2016-10'], df['2017-10']])) # 同月堆叠
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    data_month = []
    for month in months:
        data_month.append(pd.concat([df['{}-{}'.format(_, month)] for _ in [2014, 2015, 2016, 2017]]))

    counts_month = [len(_) for _ in data_month]
    if sum(counts_month) != len(df):
        print(counts_month)
        print(sum(counts_month), len(df))
        raise Exception('sum of data in months not equal total count!')

    # 柱状图，各月槽的数目与总数据
    if (lt[1] + 25 - lt[0]) % 24 == 3:
        total_data_months = [1116, 1020, 1116, 1080, 1116, 1080, 1116, 1116, 1080, 1116, 1080, 1116]
    elif (lt[1] + 25 - lt[0]) % 24 == 2:
        total_data_months = [744, 680, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
    else:
        raise Exception('input lt Error: lt[0] != 22 or 23')

    plt.bar(months, total_data_months, fc='lightskyblue', label='lat-prof count')         # 总数据
    plt.bar(months, counts_month, fc='blue', label='trough number')      # 槽的数目
    if (lt[1] + 25 - lt[0]) % 24 == 3:
        plt.ylim(0, 1300)
    if (lt[1] + 25 - lt[0]) % 24 == 2:
        plt.ylim(0, 900)
    plt.ylabel('count')
    plt.xticks([_ for _ in range(1, 13)])
    plt.xlabel('month')
    plt.legend()
    plt.title('2014/9/1-2017/8/31  lt_{}-{}  Kp9_{}-{}'.format(lt[0], lt[1] + 1, Kp9[0], Kp9[1]))
    # plt.savefig(outpath + 'bar plot\\counts of months in 3years data')
    plt.savefig(outpath + 'counts of months lt_{}-{} Kp9_{}-{}'.format(lt[0], lt[1] + 1, Kp9[0], Kp9[1]))
    # plt.show()
    plt.close('all')

    # 箱型图，槽极小位置，不同经度统一为磁纬
    glons = [-120, -90, 0, 30]
    bias = {-120: 6, -90: 9, 0: -2, 30: -3}
    data_month_corrected = [[] for _ in range(12)]
    for glon in glons:
        data_glon_month = [data_month[_].query('glon == {}'.format(glon)) for _ in range(12)]
        lat_glon_month = [np.array(data_glon_month[_]['trough_min_lat']) + bias[glon] for _ in range(12)]
        data_month_corrected = [np.append(data_month_corrected[_], lat_glon_month[_]) for _ in range(12)]
        plt.boxplot(lat_glon_month)
        plt.xlabel('month')
        plt.ylabel('Gm Latitude(degree)')
        plt.title('boxplot of lat-month glon_{}° lt_{}-{} Kp9_{}-{}'.format(glon, lt[0], lt[1] + 1, Kp9[0], Kp9[1]))
        # plt.savefig(outpath + 'bar plot\\glon_{}° in 3 years data at low kp9'.format(glon))
        plt.savefig(outpath + 'glon_{}° lt_{}-{} Kp9_{}-{}'.format(glon, lt[0], lt[1] + 1, Kp9[0], Kp9[1]))
        # plt.show()
        plt.close('all')
    # 所有4个经度的数据，箱线图
    plt.boxplot(data_month_corrected)
    plt.xlabel('month')
    plt.ylabel('Gm Latitude(degree)')
    plt.title('boxplot of lat_month lt_{}-{} Kp9_{}-{}'.format(lt[0], lt[1] + 1, Kp9[0], Kp9[1]))
    # plt.savefig(outpath + 'bar plot\\boxplot of lats-month of 3 years data at low kp9')
    plt.savefig(outpath + 'boxplot of lat_month lt_{}-{} Kp9_{}-{}'.format(lt[0], lt[1] + 1, Kp9[0], Kp9[1]))
    plt.show()
    plt.close('all')
    return True


# outpath = 'C:\\DATA\\GPS_MIT\\millstone\\summary graph\\'
outpath = 'C:\\tmp\\figure\\lat-month\\'
if __name__ == '__main__':
    year1, month1, day1, year2, month2, day2 = 2014, 9, 1, 2017, 8, 31
    DF = df_aggregation([22, 23, 0], year1, month1, day1, year2, month2, day2)    # lt: 22, 23, 0
    print(DF.columns)
    print('length of data Df: ', len(DF))
    plot_pie_isnan(DF)                      # lt: 22, 23, 0

    # 丢失缺失值，只保留有槽极小的数据
    DF = DF.dropna()
    trough_count = len(DF)
    print(trough_count)

    plot_month(DF, [0, 9], [23, 0])         # lt: 23-1, 槽的数目直方图，槽极小位置箱线图

    # step2, df中槽极小的经度分布
    longitudes = ['-120', '-90', '0', '30']
    counts_glon = [len(DF.query("glon == {} & (lt == 23 | lt == 0)".format(_))) for _ in longitudes]
    print(counts_glon)
