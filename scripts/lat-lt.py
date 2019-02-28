#! python3
# -*- coding: utf-8 -*-

# import os
# import gc
# import time
# import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator
import matplotlib


matplotlib.rcParams.update({'font.size': 12})        # 修改所有label和tick的大小


def season_classifier(DF, season):
    if season == 'whole year':
        df = DF
    elif season == 'equinox':
        df = pd.concat([DF['2014-9-1': '2014-10-31'], DF['2015-3-1': '2015-4-30'],
                        DF['2015-9-1': '2015-10-31'], DF['2016-3-1': '2016-4-30'],
                        DF['2016-9-1': '2016-10-31'], DF['2017-3-1': '2017-4-30']])
    elif season == 'summer':
        df = pd.concat([DF['2015-5-1': '2015-8-31'],
                        DF['2016-5-1': '2016-8-31'],
                        DF['2017-5-1': '2017-8-31']])
    elif season == 'winter':
        df = pd.concat([DF['2014-11-1': '2015-2-28'], DF['2015-11-1': '2016-2-29'], DF['2016-11-1': '2017-2-28']])
    else:
        raise Exception("parameter 'season' Error!")
    return df


def plot_ltCount(DF, Kp9, seasons):
    lt_list = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
    lt_list_tmp = [_ + 24 if _ < 12 else _ for _ in lt_list]
    DF = DF.query('glon == {} & {} < Kp9 < {}'.format(glon, Kp9[0], Kp9[1]))
    data_lt = []
    for lt in lt_list_tmp:
        data_lt.append(DF.query("lt == {}".format(lt % 24)))

    fig = plt.figure(figsize=(8, 6))
    for i, season in enumerate(seasons):
        data_lt_season = [season_classifier(_, season) for _ in data_lt]
        countsOfProf_season = [len(_) for _ in data_lt_season]
        data_lt_season = [_.dropna() for _ in data_lt_season]
        countsOfTrough_season = [len(_) for _ in data_lt_season]
        ax = fig.add_subplot(221 + i)
        plt.sca(ax)
        bar_width = 0.34
        if Kp9 == 0 and Kp9[1] == 9:
            color = 'blue'
            color1 = 'lightskyblue'
            figure_name = 'counts_lt_all Kp'
        elif Kp9[0] == 0 and Kp9[1] == 2:
            color = 'green'
            color1 = 'lightGreen'
            figure_name = 'figure12'
        elif Kp9[0] == 2:
            color = 'k'
            color1 = 'grey'
            figure_name = 'figure13'
        else:
            color = 'r'
            color1 = 'orange'
            figure_name = 'figure14'
            # raise Exception('error Kp9 range')
        if Kp9:
            color, color1 = 'black', 'slategrey'
        ax.bar(np.array(lt_list_tmp), countsOfProf_season, bar_width, align='edge',
               alpha=1, edgecolor='white', fc=color1, label='lat-prof counts')
        ax.bar(np.array(lt_list_tmp) + bar_width, countsOfTrough_season, bar_width, align='edge',
               fc=color, edgecolor='white', label='trough counts')
        # ax.set_title(season, loc='left')
        ax.text(0.2, 0.85, '{}'.format(season), transform=ax.transAxes, color='k')
        ax.set_xlabel('LT(h)')
        ax.set_ylabel('counts')
        ax.set_ylim(0, max(countsOfProf_season) * 1.3)
        ax.legend(loc='upper right', fontsize=8)
        plt.xticks(lt_list_tmp, lt_list)

    fig.tight_layout()
    fig.subplots_adjust(top=0.93, left=0.12, right=0.95)
    fig.suptitle('2014/9/1-2-17/8/31  glon:-90  Kp9:{}'.format(Kp9))
    # fig.savefig(figure_path + 'trough & prof counts Kp9_{}'.format(Kp9))

    fig.savefig(figure_path + figure_name + '.eps', fmt='eps')
    fig.savefig(figure_path + figure_name + '.jpg', fmt='jpg')
    plt.show()
    return True


def plot_LatLtScatter(DF, Kp9, seasons):
    figure = plt.figure(figsize=(9, 7))
    DF = DF.query('{} <= Kp9 < {}'.format(Kp9[0], Kp9[1]))
    for idx, season in enumerate(seasons):
        print(season)
        df = season_classifier(DF, season)

        # lt = [_ + 0.5 for _ in df['lt']]
        lt = np.array([_ + 24 if _ < 12 else _ for _ in df['lt']])
        lat = np.array(df['trough_min_lat'])
        # 统计每个点出现的个数
        lat_lt, counts_of_points = list(zip(lt, lat)), []
        for _ in lat_lt:
            counts_of_points.append(lat_lt.count(_))

        ax = figure.add_subplot(221 + idx)
        plt.sca(ax)
        ax.scatter(lt, lat, color='b', s=[_/4 for _ in counts_of_points])
        ax.text(0.65, 0.85, '{}'.format(season), transform=ax.transAxes, color='black')

        ax.set_xlim(17, 30)
        print(ax.get_xticks())
        x = [_ for _ in range(18, 30)]
        plt.xticks(x, [_ % 24 for _ in x])
        # ax.xaxis.set_major_locator(MultipleLocator(2))
        # ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_xlabel('LT(h)')
        ax.set_ylabel('MIT_min_gdlat(°)')
        if Kp9[0] == 0 and Kp9[1] == 2:
            ax.set_ylim(40, 70)
        elif Kp9[0] == 2:
            ax.set_ylim(35, 65)
        elif Kp9[0] == 4:
            ax.set_ylim(30, 60)
        else:
            ax.set_ylim(30, 70)
    figure.subplots_adjust(top=0.93, left=0.1, right=0.95)
    # figure.tight_layout()
    plt.gcf()
    plt.suptitle('2014/9/1-2017/8/31  glon:-90° Kp9:{}'.format(Kp9))
    figure.savefig(figure_path + 'lat_lt scatter Kp9_{}'.format(Kp9))
    plt.show()
    return True


def get_MediansAndPercentiles(DF, Kp9_RangList, seasons, LocalTimes):
    dict_medians, dict_25percentiles, dict_75percentiles = {}, {}, {}
    for season in seasons:
        dict_medians[season] = {}
        dict_25percentiles[season], dict_75percentiles[season] = {}, {}
        DF_season = season_classifier(DF, season)
        for Kp9_range in Kp9_RangList:
            Kp9_range_str = 'Kp9_{}'.format(Kp9_range)
            dict_medians[season][Kp9_range_str] = []
            dict_25percentiles[season][Kp9_range_str] = []
            dict_75percentiles[season][Kp9_range_str] = []

            df = DF_season.query('{} <= Kp9 < {}'.format(Kp9_range[0], Kp9_range[1]))[['lt', 'trough_min_lat']]
            for localtime in LocalTimes:
                df_lt = df.query('lt == {}'.format(localtime))
                dict_medians[season][Kp9_range_str].append(df_lt['trough_min_lat'].median())
                dict_25percentiles[season][Kp9_range_str].append(df_lt['trough_min_lat'].quantile(0.25))
                dict_75percentiles[season][Kp9_range_str].append(df_lt['trough_min_lat'].quantile(0.75))

    return dict_medians, dict_25percentiles, dict_75percentiles


def plot_LatLtPercentile(DF, seasons, Kp9, LocalTimes):
    dic_median, dic_25percentile, dic_75percentile = get_MediansAndPercentiles(DF, Kp9, seasons, LocalTimes)
    lt = [_ + 24 if _ < 12 else _ for _ in LocalTimes]
    figure = plt.figure(figsize=(9, 7))
    # colors = 'bgr'
    colors = 'kkk'
    fmts = ['-o', '-d', '-s']
    for idx, season in enumerate(seasons):
        ax = figure.add_subplot(221 + idx)
        plt.sca(ax)
        for index, _ in enumerate(Kp9):
            Kp9_str = 'Kp9_{}'.format(_)
            y_err = [[i - j for i, j in zip(dic_median[season][Kp9_str], dic_25percentile[season][Kp9_str])],
                     [i - j for i, j in zip(dic_75percentile[season][Kp9_str], dic_median[season][Kp9_str])]]
            if season == 'summer' and _ == [0, 2]:
                ax.errorbar(lt[1:], dic_median[season][Kp9_str][1:], yerr=[y_err[0][1:], y_err[1][1:]], fmt=fmts[index],
                            color=colors[index], ms=5, capsize=5, label='{}'.format(Kp9_str))
            else:
                ax.errorbar(lt, dic_median[season][Kp9_str], yerr=y_err, fmt=fmts[index], color=colors[index],
                            ms=5, capsize=5, label='{}'.format(Kp9_str))
        ax.text(0.28, 0.85, '{}'.format(season), transform=ax.transAxes, color='k')
        ax.legend()
        ax.set_xlabel('Localtime(h)')
        ax.set_ylabel('MIT_min_gdlat(°)')
        ax.set_xlim(17, 30)
        plt.xticks(lt, [_ - 24 if _ > 24 else _ for _ in lt])
        # ax.xaxis.set_major_locator(MultipleLocator(2))
        # ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_ylim(39, 65)
    # figure.suptitle('2014/9/1-2017/8/31  glon: -90°')
    figure.subplots_adjust(top=0.93, left=0.1, right=0.95)
    # figure.savefig(figure_path + 'lat_lt ErrorBarPlot')
    figure.savefig(figure_path + 'figure15.eps', fmt='eps')
    figure.savefig(figure_path + 'figure15.jpg', fmt='jpg')
    plt.show()
    figure.clear()
    return True


if __name__ == '__main__':
    glon = -90
    year1, month1, day1 = 2014, 9, 1
    year2, month2, day2 = 2017, 8, 31
    index_item = 'Kp9'

    # figure_path = 'C:\\tmp\\figure\\lat-lt\\'
    figure_path = "C:\\tmp\\figure\\eps\\"
    table_path = "C:\\tmp\\bigtable.csv"
    bigtable = pd.read_csv(table_path, encoding='gb2312')
    bigtable['date'] = pd.to_datetime(bigtable["date"])  # date格式转换为datetime.date
    bigtable = bigtable.set_index('date').sort_index()
    bigtable = bigtable["{}-{}-{}".format(year1, month1, day1): "{}-{}-{}".format(year2, month2, day2)]

    Seasons = ['whole year', 'equinox', 'summer', 'winter']
    Kp9_index = [[0, 2], [2, 4], [4, 9]]
    LTs = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
    for Kp9_idx in Kp9_index:
        plot_ltCount(bigtable, Kp9_idx, Seasons)

    bigtable = bigtable.query('glon == {}'.format(glon)).dropna()

    # for _ in [[0, 2], [2, 4], [4, 9], [0, 9]]:
    #     plot_LatLtScatter(bigtable, [0, 2], Seasons)

    plot_LatLtPercentile(bigtable, Seasons, Kp9_index, LTs)
