#! python3
# -*- coding: utf-8 -*-

import os
# import gc
# import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def MIT_F107(df, glon, Kp9, lt):
    df = df.query('glon == {} & {} <= kp9 <= {}'.format(glon, Kp9[0], Kp9[1]))
    if lt[0] <= lt[1]:
        df = df.query('{} <= lt <= {}'.format(lt[0], lt[1]))
    else:
        df = df.query('lt >= {} | lt <= {}'.format(lt[0], lt[1]))
    print(df.columns)

    y_f107, y_mit = df['F10.7'], df['trough_min_lat']

    y_f107_rolmean = y_f107.rolling(window=360, center=True, min_periods=1).median()
    y_new = y_f107 - y_f107_rolmean
    for i in range(len(y_f107)):
        if y_f107.iloc[i] == np.nan:
            print(y_f107.iloc[i])
        if y_f107.iloc[i] > 230:
            y_f107.iloc[i] = np.nan
            continue
    for i in range(len(y_new)):
        if y_new.iloc[i] > 100:
            y_new.iloc[i] = np.nan
            continue
    print(y_f107.max())

    fig = plt.figure(figsize=(8, 6))
    plt.gcf()
    ax1 = plt.axes([0.1, 0.1, 0.5, 0.23])
    ax1.plot(y_f107, linewidth=1)
    ax1.plot(y_f107_rolmean, 'r--', linewidth=1)
    ax1.set_xlabel('date')
    ax1.set_title('a) F10.7 index', loc='left')
    xticks = ax1.get_xticks()
    print(len(xticks), xticks)
    ax1.set_xticks(xticks[::2])
    ax1.set_ylabel('F10.7 index')
    ax1.text(0.7, 0.85, 'count:{}'.format(y_f107.count()), transform=ax1.transAxes, color='black')

    ax2 = plt.axes([0.1, 0.38, 0.5, 0.23])
    ax2.plot(y_new, linewidth=1)
    ax2.set_ylim(-50, )
    ax2.set_xticks([])
    ax2.set_ylabel('detrended F10.7 index')
    ax2.set_title('b) Detrended F10.7 index', loc='left')

    ax3 = plt.axes([0.1, 0.66, 0.5, 0.25])
    ax3.plot(y_mit, linewidth=0.8)
    ax3.set_xticks([])
    ax3.set_ylabel('Trough_min lat(°)')
    ax3.set_title('c) Trough_min_gdlat', loc='left')    # lt=0 & glon=-90°
    ax3.text(0.7, 0.85, 'count:{}'.format(y_mit.count()), transform=ax3.transAxes, color='black')

    ax4 = plt.axes([0.7, 0.1, 0.25, 0.38])
    cc = y_f107.corr(y_mit)
    ax4.scatter(y_mit, y_f107, s=3)
    ax4.text(0.7, 0.85, 'cc:{}'.format(round(cc, 2)), transform=ax4.transAxes, color='black')
    ax4.text(0.1, 0.85, 'd)', transform=ax4.transAxes, color='black')
    ax4.set_ylabel('F10.7 index')
    ax4.set_yticks([50, 100, 150, 200])
    ax4.set_xlabel(' Trough_min_gdlat(°)')
    ax4.set_xlim(35, 65)
    # ax4.set_title('e)', loc='left')
    # plt.legend()

    ax5 = plt.axes([0.7, 0.53, 0.25, 0.38])
    cc = y_new.corr(y_mit)
    ax5.scatter(y_mit[:], y_new, s=3)
    ax5.text(0.7, 0.85, 'cc:{}'.format(round(cc, 2)), transform=ax5.transAxes, color='black')
    ax5.set_ylabel(' Detrended F10.7 index')
    ax5.text(0.1, 0.85, 'e)', transform=ax5.transAxes, color='black')
    ax5.set_xlim(35, 65)
    # ax5.set_title('d)', loc='left')

    # plt.suptitle('MIT lat & F10.7 index')
    fig.savefig(figure_path + 'lat-F107\\f107-mit glon_-90° lt_{}-{}  kp9_{}-{}'.format(lt[0], lt[1]+1, Kp9[0], Kp9[1]))
    # plt.show()
    plt.close('all')

    figure = plt.figure(figsize=(9, 5))
    ssn_dict = {'equinox': [3, 4, 9, 10], 'summer': [5, 6, 7, 8], 'winter': [11, 12, 1, 2]}
    for idx, ssn in enumerate(ssn_dict):
        months = ssn_dict[ssn]
        print(ssn, months)
        if ssn == 'equinox':
            df_ssn = df['2014-9': '2014-10']
            # print(df_ssn)
            for year in [2015, 2016, 2017]:
                df_ssn = df_ssn.append(df['{}-3'.format(year): '{}-4'.format(year)])
                if year != 2017:
                    df_ssn = df_ssn.append(df['{}-9'.format(year): '{}-10'.format(year)])
        elif ssn == 'summer':
            df_ssn = df['2015-5': '2015-8']
            for year in [2016, 2017]:
                df_ssn = df_ssn.append(df['{}-5'.format(year): '{}-8'.format(year)])
        else:
            df_ssn = df['2014-11': '2014-12']
            for year in [2015, 2016, 2017]:
                df_ssn = df_ssn.append(df['{}-1'.format(year): '{}-2'.format(year)])
                if year != 2017:
                    df_ssn = df_ssn.append(df['{}-11'.format(year): '{}-12'.format(year)])

        ax = figure.add_subplot(131 + idx)
        plt.sca(ax)
        print(df_ssn.columns)
        # print(len(df_ssn))
        plt.plot(df_ssn['F10.7'], df_ssn['trough_min_lat'], 'b.')
        cc = df_ssn['F10.7'].corr(df_ssn['trough_min_lat'])
        # count = len(df_ssn)
        count = df_ssn['trough_min_lat'].count()
        plt.xlabel('F10.7')
        # plt.xlim(0, 250)
        plt.xticks([50, 100, 150, 200])
        plt.ylim(36, 62)
        plt.yticks([_ for _ in range(40, 62, 5)])
        ax.set_title('{}) {}'.format(chr(ord('a') + idx), ssn), loc='left')
        plt.text(0.6, 0.88, '  cc: {}\n  count:{}'.format(round(cc, 2), count), transform=ax.transAxes, color='black')
        if idx == 0:
            plt.ylabel('Geographic Latitude(°)')
    title = 'date: 2014/9/1-2017/8/31   longitude-sector: {}°  lt: {}-{}   kp9: {}-{}'.format(
        glon, lt[0], lt[1]+1, Kp9[0], Kp9[1])
    plt.suptitle(title)
    # figure.tight_layout()
    plt.show()
    figure.savefig(figure_path + 'lat-F107\\2014-9-1_2017-8-31 longitude sector{}° lt range{}-{} kp range{}-{}'.format(
        glon, lt[0], lt[1] + 1, Kp9[0], Kp9[1]))
    return True


table_path = "C:\\tmp\\bigtable.csv"
bigtable = pd.read_csv(table_path, encoding='gb2312')
bigtable['date'] = pd.to_datetime(bigtable["date"])
print(bigtable.head(3))
bigtable = bigtable.set_index('date').sort_index()


if __name__ == '__main__':
    glon_c, lt_range, kp9_range = -90, [23, 0], [0, 9]          # F107分辨率1天，故只需要1个数据每天
    figure_path = "C:\\tmp\\figure\\"
    MIT_F107(bigtable, glon=glon_c, Kp9=kp9_range, lt=lt_range)
