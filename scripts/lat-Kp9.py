#! python3
# -*- coding: utf-8 -*-

# import os
# import gc
# import time
# import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib
from mpl_toolkits.mplot3d import Axes3D


matplotlib.rcParams.update({'font.size': 12})        # 修改所有label和tick的大小


def season_classifier(DF, season):
    if season == 'whole year':
        df = DF
    elif season == 'equinox':
        df = pd.concat([DF['2014-9-1': '2014-10-30'], DF['2015-3-1': '2015-4-30'],
                        DF['2015-9-1': '2015-10-30'], DF['2016-3-1': '2016-4-30'],
                        DF['2016-9-1': '2016-10-30'], DF['2017-3-1': '2017-4-30']])
    elif season == 'summer':
        df = pd.concat([DF['2015-5-1': '2015-8-31'],
                        DF['2016-5-1': '2016-8-31'],
                        DF['2017-5-1': '2017-8-31']])
    elif season == 'winter':
        df = pd.concat([DF['2014-11-1': '2015-2-28'], DF['2015-11-1': '2016-2-29'], DF['2016-11-1': '2017-2-28']])
    else:
        raise Exception("parameter 'season' Error!")
    return df


def get_percentiles(DF):
    dict_Median, dict_25thPercentile, dict_75thPercentile = {}, {}, {}
    for season in season_list:
        dict_Median[season], dict_25thPercentile[season], dict_75thPercentile[season] = {}, {}, {}
        df_season = season_classifier(DF, season)
        for lt in lt_list:
            lt_str = 'lt_{}'.format(lt)
            dict_Median[season][lt_str] = []
            dict_25thPercentile[season][lt_str] = []
            dict_75thPercentile[season][lt_str] = []

            df = df_season.query("lt == {}".format(lt))[['Kp9', 'trough_min_lat']]

            for Kp9 in Kp9_list:
                data_Kp9 = df.query("{} <= Kp9 < {}".format(Kp9-0.5, Kp9+0.5))
                if len(data_Kp9) > 3:
                    dict_Median[season][lt_str].append(data_Kp9['trough_min_lat'].median())
                    dict_25thPercentile[season][lt_str].append(data_Kp9['trough_min_lat'].quantile(0.25))
                    dict_75thPercentile[season][lt_str].append(data_Kp9['trough_min_lat'].quantile(0.75))
                else:
                    dict_Median[season][lt_str].append(np.nan)
                    dict_25thPercentile[season][lt_str].append(np.nan)
                    dict_75thPercentile[season][lt_str].append(np.nan)
                    # raise Exception('no data at Kp9: {}'.format(Kp9))
    return dict_Median, dict_25thPercentile, dict_75thPercentile


def get_fitted_line(DF):
    dict_fitted = {}
    for season in season_list:
        dict_fitted[season] = {}
        df_season = season_classifier(DF, season)
        for lt in lt_list:
            lt_str = 'lt_{}'.format(lt)
            df = df_season.query("lt == {}".format(lt))[['Kp9', 'trough_min_lat']]

            z = list(np.polyfit(list(df['Kp9']), list(df['trough_min_lat']), 1))
            x = list(set(df['Kp9']))
            y = [z[0] * _ + z[1] for _ in x]

            dict_fitted[season][lt_str] = [x, y]
    return dict_fitted


def plot_lat_Kp9(DF, mode=False):
    fig = plt.figure(figsize=(10, 8))
    # colors = 'rgbyck'
    colors = 'kkkkkk'
    fmts = ['-o', '-d', '-s', '-^', '-v', '-p']
    for idx, season in enumerate(season_list):
        ax = fig.add_subplot(141 + idx)
        plt.sca(ax)
        for i, lt in enumerate(lt_list):
            lt_str = 'lt_{}'.format(lt)
            if mode:
                dic_median, dic_25th, dic_75th = get_percentiles(DF)
                y_err = [[i - j for i, j in zip(dic_median[season][lt_str], dic_25th[season][lt_str])],
                         [i - j for i, j in zip(dic_75th[season][lt_str], dic_median[season][lt_str])]]
                ax.errorbar(Kp9_list, dic_median[season][lt_str], yerr=y_err, fmt=fmts[i], ms=5, c=colors[i],
                            capsize=5, alpha=0.8, label='lt={}'.format(lt))
            else:
                dic_fitted = get_fitted_line(DF)
                ax.plot(dic_fitted[season][lt_str][0], dic_fitted[season][lt_str][1], color=colors[i],
                        label='lt={}'.format(lt))
        ax.text(0.2, 0.05, '{}'.format(season), transform=ax.transAxes, color='k')
        ax.legend()
        ax.set_xlabel('Kp9 index')
        ax.set_xticks([_ for _ in range(9)])
        ax.set_xlim(-0.5, 9)
        ax.set_ylim(35, 65)
        if idx == 0:
            ax.set_ylabel('trough_min_gdlat(°)')
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
    title = '{}/{}/{}-{}/{}/{} glon:{} MIT_min_gdlat-Kp9'.format(year1, month1, day1, year2, month2, day2, glon)
    # fig.suptitle(title)
    fig.subplots_adjust(top=0.93, left=0.09, right=0.95)
    if mode:
        # fig.savefig(figure_path + 'MIT_min_gdlat-Kp9 lt-{}'.format(lt_list))
        fig.savefig(figure_path + 'figure11_non-colored.eps', fmt='eps')
        fig.savefig(figure_path + 'figure11_non-colored.jpg', fmt='jpg')
    else:
        fig.savefig(figure_path + 'MIT_min_gdlat-Kp9 Linear fit lt-{}'.format(lt_list))
    plt.show()
    return True


def plot_3dScatter_lat_Kp9lt(DF):
    fig = plt.figure()
    ax = Axes3D(fig)
    x, y, z = np.array(DF['lt']), np.array(DF['Kp9']), np.array(DF['trough_min_lat'])
    f = x < 12
    x = x + 24 * f           # 地方时连续
    ax.scatter(x, y, z, s=0.5)
    plt.xticks([_ for _ in range(18, 30)], [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5])
    ax.view_init(elev=20, azim=60)
    ax.set_xlabel('LT(h)')
    ax.set_ylabel('Kp9 index')
    ax.set_zlabel('trough_min_gdlat(°)')
    fig.savefig(figure_path + '3dScatter')
    plt.show()
    return True


if __name__ == '__main__':
    glon = -90
    year1, month1, day1 = 2014, 9, 1
    year2, month2, day2 = 2017, 8, 31
    index_item = 'Kp9'

    # figure_path = 'C:\\tmp\\figure\\lat-MagneticField_index\\'
    figure_path = 'C:\\tmp\\figure\\eps\\'
    table_path = "C:\\tmp\\bigtable.csv"
    bigtable = pd.read_csv(table_path, encoding='gb2312')
    bigtable['date'] = pd.to_datetime(bigtable["date"])  # date格式转换为datetime.date
    bigtable = bigtable.set_index('date').sort_index()
    bigtable = bigtable["{}-{}-{}".format(year1, month1, day1): "{}-{}-{}".format(year2, month2, day2)]
    bigtable = bigtable.query('glon == {}'.format(glon)).dropna()

    Kp9_list = np.linspace(0.5, 8.5, 9)           # 取8个值，代表Kp9∈[0, 1)等区间；
    season_list = ['whole year', 'equinox', 'summer', 'winter']
    lt_list = [18, 20, 22, 0, 2, 4]

    plot_lat_Kp9(bigtable, mode=True)
    # plot_3dScatter_lat_Kp9lt(bigtable)
