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


def df_aggregation(y1, m1, d1, y2, m2, d2):
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
    return table_new


def plot_pie_isnan(df, lt, glon):
    # df中槽极小存在的占比, 饼图
    print(df.columns)
    print(df.head())
    if lt[0] <= lt[1]:
        df = df.query("{} <= lt <= {}".format(lt[0], lt[1]))
    else:
        df = df.query("lt >= {} | lt <= {}".format(lt[0], lt[1]))

    if len(glon) == 1:
        df = df.query('glon == {}'.format(glon[0]))
    elif len(glon) == 4:
        pass
    else:
        print(glon)
        raise Exception("glon param error: ")    # 取单一经度带或不作限制

    total_count = len(df)
    not_nan_count = df.trough_min_lat.count()
    nan_count = total_count - not_nan_count
    print(total_count, not_nan_count)
    # 饼图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码问题

    fig = plt.figure(figsize=(5, 4))
    plt.gcf()
    ax = plt.axes([0.05, 0.05, 0.85, 0.85])
    labels = [u'中纬槽', u'无中纬槽']
    data = [not_nan_count, nan_count]
    colors = ['lightskyblue', 'yellowgreen']
    explode = (0, 0)

    patches, text1, text2 = ax.pie(x=data, explode=explode, colors=colors, labels=labels,
                                   autopct='%2.1f%%', shadow=False, startangle=60, pctdistance=0.5,
                                   wedgeprops=dict(ec='w'), textprops=dict(color='k'))
    for t1, t2 in zip(text1, text2):
        t1.set_size(12)
        t2.set_size(12)
    plt.axis('equal')
    # ax.legend(patches, labels, loc='upper right')
    fig.suptitle('LT range:{}h-{}h\ntotal number of TEC profiles: {}'.format(lt[0], lt[1]+1, total_count), fontsize=15)
    fig.tight_layout()
    plt.savefig(outpath + 'pie plot lt_{}-{} for glon_{}'.format(lt[0], lt[1] + 1, glon))
    plt.show()
    plt.close('all')
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

    counts_month_prof = [len(_) for _ in data_month]
    if sum(counts_month_prof) != len(df):
        print(counts_month_prof)
        print(sum(counts_month_prof), len(df))
        raise Exception('sum of data in months not equal total count!')
    data_month = [_.dropna() for _ in data_month]
    counts_month_trough = [len(_) for _ in data_month]

    # 柱状图，各月槽的数目与总数据
    bar_width = 0.3
    if Kp9[1] == 9:
        color = 'blue'
        color1 = 'lightskyblue'
    elif Kp9[1] == 2:
        color = 'green'
        color1 = 'lightGreen'
    else:
        raise Exception('error Kp9 range')
    plt.bar(np.array(months) - bar_width, counts_month_prof, bar_width, align='edge', alpha=1, edgecolor='white',
            fc=color1, label='lat-prof counts')
    plt.bar(months, counts_month_trough, bar_width, align='edge', fc=color, edgecolor='white', label='trough number')
    if (lt[1] + 25 - lt[0]) % 24 == 3 and Kp9[1] == 9:
        plt.ylim(0, 1300)
    if (lt[1] + 25 - lt[0]) % 24 == 2 and Kp9[1] == 9:
        plt.ylim(0, 900)
    plt.ylabel('counts', fontsize=12)
    plt.xticks([_ for _ in range(1, 13)], fontsize=12)
    plt.xlabel('month', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.title('2014/9/1-2017/8/31  lt_{}-{}  Kp9_{}-{}'.format(lt[0], lt[1] + 1, Kp9[0], Kp9[1]), fontsize=12)
    # plt.savefig(outpath + 'counts of months lt_{}-{} Kp9∈[{}, {}]'.format(lt[0], lt[1] + 1, Kp9[0], Kp9[1]))
    plt.show()
    plt.close('all')

    # 箱型图，槽极小位置，不同经度统一为磁纬
    glons = [-120, -90, 0, 30]
    bias = {-120: 6, -90: 9, 0: -2, 30: -3}
    data_month_corrected = [[] for _ in range(12)]
    boxprops, whiskerprops, capprops = dict(color=color), dict(color=color), dict(color=color)
    for glon in glons:
        data_glon_month = [data_month[_].query('glon == {}'.format(glon)) for _ in range(12)]
        lat_glon_month = [np.array(data_glon_month[_]['trough_min_lat']) + bias[glon] for _ in range(12)]
        data_month_corrected = [np.append(data_month_corrected[_], lat_glon_month[_]) for _ in range(12)]
        plt.boxplot(lat_glon_month, notch=False, sym='', vert=True, whis=[5, 95], showfliers=False,
                    showcaps=True, boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops)
        plt.xlabel('month', fontsize=12)
        plt.ylabel('Gm Latitude(degree)', fontsize=12)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.title('boxplot of lat-month glon_{}° lt_{}-{} Kp9_{}-{}'.format(glon, lt[0], lt[1] + 1, Kp9[0], Kp9[1]),
                  fontsize=12)
        # plt.savefig(outpath + 'glon_{}° lt_{}-{} Kp9_{}-{}'.format(glon, lt[0], lt[1] + 1, Kp9[0], Kp9[1]))
        # plt.show()
        plt.close('all')
    # 所有4个经度的数据，箱线图
    plt.boxplot(data_month_corrected, notch=False, sym='', vert=True, whis=[5, 95], showfliers=False,
                showcaps=True, boxprops=boxprops, whiskerprops=whiskerprops,
                capprops=capprops)
    plt.xlabel('month', fontsize=12)
    plt.ylabel('Gm Latitude(degree)', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('boxplot of lat_month lt_{}-{} Kp9_{}-{}'.format(lt[0], lt[1] + 1, Kp9[0], Kp9[1]), fontsize=12)
    # plt.savefig(outpath + 'boxplot of lat_month lt_{}-{} Kp9_{}-{}'.format(lt[0], lt[1] + 1, Kp9[0], Kp9[1]))
    plt.show()
    plt.close('all')
    return True


# outpath = 'C:\\DATA\\GPS_MIT\\millstone\\summary graph\\'
outpath = 'C:\\tmp\\figure\\lat-month\\'
if __name__ == '__main__':
    year1, month1, day1, year2, month2, day2 = 2014, 9, 1, 2017, 8, 31
    DF = df_aggregation(year1, month1, day1, year2, month2, day2)  # lt: 22, 23, 0
    print(DF.columns)
    print('length of data Df: ', len(DF))
    plot_pie_isnan(DF, [22, 0], [-90])

    # plot_month(DF, [0, 2], [23, 0])  # lt: 23-1, 槽的数目直方图，槽极小位置箱线图

    # 丢失缺失值，只保留有槽极小的数据
    DF = DF.dropna()
    trough_count = len(DF)
    print(trough_count)
