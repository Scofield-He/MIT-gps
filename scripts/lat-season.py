#! python3
# -*- coding: utf-8 -*-

# import os
# import gc
# import time
# import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams.update({'font.size': 12})        # 修改所有label和tick的大小


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
    # plt.savefig(outpath + 'pie plot lt_{}-{} for glon_{}'.format(lt[0], lt[1] + 1, glon))
    plt.show()
    plt.close('all')
    return True


def plot_month(df, Kp9, lt):
    if Kp9[0] == 0:
        df = df.query("{} <= kp9 <= {} ".format(Kp9[0], Kp9[1]))
    else:
        df = df.query("{} < kp9 <= {} ".format(Kp9[0], Kp9[1]))
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
    fig_bar = plt.figure(figsize=(6, 4))
    plt.gcf()
    bar_width = 0.3
    if Kp9[1] == 9:
        color = 'blue'
        color1 = 'lightskyblue'
    elif Kp9[1] == 2:
        color = 'green'
        color1 = 'lightGreen'
    else:
        raise Exception('error Kp9 range')
    color, color1 = 'black', 'slategrey'
    ax1 = fig_bar.add_subplot(111)
    plt.sca(ax1)
    plt.bar(np.array(months) - bar_width, counts_month_prof, bar_width, align='edge', alpha=1, edgecolor='white',
            fc=color1, label='lat-prof counts')
    plt.bar(months, counts_month_trough, bar_width, align='edge', fc=color, edgecolor='white', label='trough number')
    plt.ylim(0, 1.3 * max(counts_month_prof))
    plt.ylabel('counts', fontsize=12)
    plt.xticks([_ for _ in range(1, 13)], fontsize=12)
    plt.xlabel('month', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()

    """
    ax2 = plt.twinx(ax1)
    occur_rate = list(map(lambda x, y: x / y, counts_month_trough, counts_month_prof))
    ax2.plot(months, occur_rate, 'o-', ms=2, color='slategrey', alpha=0.6)
    ax2.plot(months, [0.8] * len(months), 'r--', linewidth=0.5)
    ax2.set_ylabel('occurrence rate')
    ax2.set_ylim(0.3, 1)
    """

    plt.gcf()
    plt.subplots_adjust(left=0.12, right=0.9, top=0.93)
    # plt.title('2014/9/1-2017/8/31  lt_{}-{}  Kp9_{}-{}'.format(lt[0], lt[1] + 1, Kp9[0], Kp9[1]), fontsize=12)
    # fig_bar.savefig(outpath + 'occur_rate lt_{}-{} Kp9∈[{}, {}]'.format(lt[0], lt[1] + 1, Kp9[0], Kp9[1]))
    if Kp9 == [0, 2]:
        fig_bar.savefig(outpath + 'figure6.eps', fmt='eps')
        fig_bar.savefig(outpath + 'figure6.jpg', fmt='jpg')
    elif Kp9 == [0, 9]:
        fig_bar.savefig(outpath + 'figure5.eps', fmt='eps')
        fig_bar.savefig(outpath + 'figure5.jpg', fmt='jpg')
    else:
        fig_bar.savefig(outpath + 'occurrence rate_month_Kp9_{}-{}'.format(Kp9[0], Kp9[1]))
    plt.show()
    plt.close('all')

    # 箱型图，槽极小位置，不同经度统一为磁纬
    glons = [-120, -90, 0, 30]
    bias = {-120: 6, -90: 9, 0: -2, 30: -3}
    data_month_corrected = [[] for _ in range(12)]
    boxprops, whiskerprops, capprops = dict(color=color1), dict(color=color1), dict(color=color1)
    medianprops = dict(color=color, linestyle='--')
    fig_box1 = plt.figure(figsize=(9, 7))
    plt.gcf()
    for idx, glon in enumerate(glons):
        ax = fig_box1.add_subplot(221 + idx)
        plt.sca(ax)
        data_glon_month = [data_month[_].query('glon == {}'.format(glon)) for _ in range(12)]
        lat_glon_month = [np.array(data_glon_month[_]['trough_min_lat']) + bias[glon] for _ in range(12)]
        data_month_corrected = [np.append(data_month_corrected[_], lat_glon_month[_]) for _ in range(12)]
        plt.boxplot(lat_glon_month, notch=False, sym='', vert=True, whis=[5, 95], showfliers=False,
                    showcaps=True, boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops,
                    medianprops=medianprops)
        plt.xlabel('month', fontsize=12)
        plt.ylabel('Gm Latitude(degree)', fontsize=12)
        if Kp9[0] == 0:
            ax.set_ylim(54, 67)
        elif Kp9[0] == 2:
            ax.set_ylim(50, 64)
        else:
            ax.set_ylim(48, 62)
        ax.text(0.06, 0.88, '{}) {}°'.format(chr(ord('a') + idx), glon), transform=ax.transAxes, color='k')
        # plt.title('Glon_{}° LT_{}-{} Kp9_{}-{}'.format(glon, lt[0], lt[1] + 1, Kp9[0], Kp9[1]))
        # plt.savefig(outpath + 'glon_{}° lt_{}-{} Kp9_{}-{}'.format(glon, lt[0], lt[1] + 1, Kp9[0], Kp9[1]))
        # plt.show()
        # plt.close('all')
    # fig_box1.suptitle('2014/9/1-2017/8/31  23:00-01:00 LT  {}<Kp9≤9'.format(Kp9[0]))
    fig_box1.subplots_adjust(top=0.93, left=0.1, right=0.95)
    # fig_box1.savefig(outpath + '4 longitudes box_plot Kp9_{}-{}'.format(Kp9[0], Kp9[1]))
    if Kp9 == [0, 2]:
        fig_box1.savefig(outpath + 'figure7.eps', fmt='eps')
        fig_box1.savefig(outpath + 'figure7.jpg', fmt='jpg')
    elif Kp9 == [2, 9]:
        fig_box1.savefig(outpath + 'figure8.eps', fmt='eps')
        fig_box1.savefig(outpath + 'figure8.jpg', fmt='jpg')
    plt.show()

    # 所有4个经度的数据，箱线图
    plt.boxplot(data_month_corrected, notch=False, sym='', vert=True, whis=[5, 95], showfliers=False,
                showcaps=True, boxprops=boxprops, whiskerprops=whiskerprops,
                capprops=capprops)
    plt.xlabel('month', fontsize=12)
    plt.ylabel('Gm Latitude(degree)', fontsize=12)
    plt.subplots_adjust(left=0.11, right=0.95, top=0.93)
    plt.title('boxplot of lat_month lt_{}-{} Kp9_{}-{}'.format(lt[0], lt[1] + 1, Kp9[0], Kp9[1]), fontsize=12)
    # plt.savefig(outpath + 'boxplot of lat_month lt_{}-{} Kp9_{}-{}'.format(lt[0], lt[1] + 1, Kp9[0], Kp9[1]))
    plt.show()
    plt.close('all')
    return True


# outpath = 'C:\\DATA\\GPS_MIT\\millstone\\summary graph\\'
# outpath = 'C:\\tmp\\figure\\lat-month\\'
outpath = 'C:\\tmp\\figure\\eps\\'
if __name__ == '__main__':
    year1, month1, day1, year2, month2, day2 = 2014, 9, 1, 2017, 8, 31
    DF = df_aggregation(year1, month1, day1, year2, month2, day2)  # lt: 22, 23, 0
    print(DF.columns)
    print('length of data Df: ', len(DF))
    # plot_pie_isnan(DF, [23, 0], [-90])

    plot_month(DF, Kp9=[2, 9], lt=[23, 0])  # lt: 23-1, 槽的数目直方图，槽极小位置箱线图

    # 丢失缺失值，只保留有槽极小的数据
    DF = DF.dropna()
    trough_count = len(DF)
    # print(trough_count)
