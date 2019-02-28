#! python3
# -*- coding: utf-8 -*-

import os
# import gc
# import time
import datetime
import numpy as np
import pandas as pd
from scipy import interpolate
# from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from astropy.stats import LombScargle
import matplotlib


matplotlib.rcParams.update({'font.size': 12})        # 修改所有label和tick的大小


def bigtable_SolarWind(txt_path, csv_path, glon):
    """
    get .csv file which save Solar Wind data
    :param txt_path: origin .txt datafile
    :param csv_path: the path that .csv file saved 
    :param glon: for calculating lt for glon
    :return: none, get saved .csv
    """
    col_widths = [4, 4, 3, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 9]        # 定宽数据宽度
    col_names = ['year', 'doy', 'Hour', 'Lat_of_Earth', 'Lon_of_Earth', 'BR', 'BT', 'BN',
                 '|B', 'Solar_Wind_Speed', 'THETA_elevation_angle_of_v', 'PHI_azimuth_angle_of_v',
                 'Ion_Density', 'Temperature']                     # 自定义列名

    table = pd.read_fwf(txt_path, widths=col_widths, header=None, names=col_names)
    lt_0 = np.array(table['Hour'])
    lt_glon = (lt_0 + glon // 15) % 24                             # 计算地方时
    # doy_bias = (lt_0 + glon // 15) // 24
    year = np.array(table['year'])
    doy = np.array(table['doy'])
    date = np.array(list(map(lambda y, d: datetime.date(y, 1, 1) + datetime.timedelta(int(d) - 1),
                             year, doy)))          # date与世界时保持一致, 便于求平均
    # print(date[:5])
    table.insert(0, 'date', date)
    table.insert(4, 'lt'.format(fixed_glon), lt_glon)

    table['date'] = pd.to_datetime(table['date'])
    table = table.set_index('date')
    print(table.head())
    table = table['2014-9-1': '2017-8-31']

    table.to_csv(csv_path, index=True)


def MIT_doy(df, lt, y1, m1, d1, y2, m2, d2):
    print(df.columns)
    df = df.sort_index()                                                     # 按照date排序
    df = df['{}-{}-{}'.format(y1, m1, d1): '{}-{}-{}'.format(y2, m2, d2)]    # 取给定时间段数据

    figure = plt.figure(figsize=(10, 6))
    ax1 = figure.add_subplot(411)
    plt.sca(ax1)
    df1 = df.query('lt >= {} | lt <= {}'.format(lt[0], lt[-1]))  # 取午夜地方时，23-1
    ax1.plot(df1['trough_min_lat'], 'bx', ms=5, alpha=0.5)
    plt.ylim(38, 56)
    plt.yticks([40, 45, 50, 55])
    # ax1.spines['left'].set_color('blue')                                  # 设置子图左轴颜色
    plt.ylabel('MIT gdlat (°)', color='b')                                 # 设置子图label颜色
    # ax1.yaxis.label.set_color('blue')                                     # 设置子图label颜色
    ax1.tick_params(axis='y', colors='blue')                                # 设置子图y刻度颜色
    ax1.set_xticks([])

    df1 = df1.dropna().reset_index(['date'])
    for i in range(len(df1) - 1):
        if df1.date[i+1] == df1.date[i]:
            pass

    df1 = df1.drop_duplicates(['date'])
    df1 = df1.set_index(['date'])
    x = np.array([(_ - df1.index[0]).days for _ in df1.index])
    y = np.array(df1['trough_min_lat'])
    print(x)
    print(y)
    # print(len(x) == len(y))
    f = interpolate.splrep(x, y, s=50)
    y_new = interpolate.splev(x, f, der=0)
    print(y_new)
    plt.plot(df1.index, y_new, 'r')

    plt.title('longitude-sector: {}  {}-{}-{}: {}-{}-{}  lt: {}-{}h'.format(
        fixed_glon, y1, m1, d1, y2, m2, d2, lt[0], lt[-1] + 1))

    ax2 = figure.add_subplot(412)
    plt.sca(ax2)
    plt.plot(df['Solar_Wind_Speed'], 'r+', ms=3, alpha=0.5)
    plt.ylabel('Vs (km/s)', color='r')
    plt.yticks([300, 400, 500, 600, 700])
    ax2.set_xticks([])
    # ax2.spines['left'].set_color('red')
    # ax2.yaxis.label.set_color('red')
    ax2.tick_params(axis='y', colors='red')
    # ax2.grid()

    ax3 = figure.add_subplot(413)
    plt.sca(ax3)
    plt.plot(df.index, df['BN'], 'k.', ms=3)
    plt.ylim(-15, 15)
    plt.ylabel('Bz(nT)')
    # ax3.spines['top'].set_color('none')
    # ax3.spines['right'].set_color('none')
    ax3.xaxis.set_ticks_position('bottom')
    ax3.spines['bottom'].set_position(('data', 0))
    plt.yticks([-10, -5, 0, 5, 10])
    # ax3.set_xticks([])
    plt.xticks([])
    # ax3.grid()

    ax4 = figure.add_subplot(414)
    plt.sca(ax4)
    plt.bar(df1.index, df1['kp'], color='black')
    plt.ylim(0, 9)
    plt.yticks([0, 3, 6], ['{}'.format(_) for _ in [0, 3, 6]])
    ax4.tick_params(axis='x', rotation=15)
    plt.ylabel('Kp index\n')
    # ax4.grid()

    figure.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()
    figure.savefig(figure_path + '\\change with time\\1year {}-{}-{}_{}-{}-{} ls-90° lt_{}-{}'.format(
        y1, m1, d1, y2, m2, d2, lt[0], lt[1]+1))


def MIT_F107(df, Kp, lt, y0, m0, d0, y1, m1, d1):
    df = df['{}-{}-{}'.format(y0, m0, d0):'{}-{}-{}'.format(y1, m1, d1)]
    df = df.query('{} <= Kp <= {}'.format(Kp[0], Kp[1]))
    df = df.mask(df['F10.7'] > 230)             # 设f10.7 > 230的值为NaN
    if lt[0] <= lt[1]:
        df = df.query('{} <= lt <= {}'.format(lt[0], lt[1]))
    else:
        df = df.query('lt >= {} | lt <= {}'.format(lt[0], lt[1]))
    print(df.columns)

    y_f107, y_mit = df['F10.7'], df['trough_min_lat']
    fig = plt.figure(figsize=(12, 6))
    plt.gcf()
    ax1 = plt.axes([0.08, 0.1, 0.57, 0.38])
    ax1.plot(y_f107, 'x', ms=3, alpha=0.8)
    ax1.set_xlabel('date')
    ax1.set_ylabel('F10.7 (sfu)')
    ax1.text(0.1, 0.9, 'a)', transform=ax1.transAxes)

    ax2 = plt.axes([0.08, 0.53, 0.57, 0.38])
    ax2.plot(y_mit, 'x', ms=3, alpha=0.8)
    # ax2.set_xlabel('date')
    ax2.set_ylabel('trough_min_gdlat (°)')
    ax2.text(0.1, 0.9, 'b)', transform=ax2.transAxes)

    ax3 = plt.axes([0.72, 0.1, 0.25, 0.8])
    cc = y_f107.corr(y_mit)
    ax3.text(0.8, 0.9, 'c)', transform=ax3.transAxes)
    ax3.text(0.6, 0.05, 'n={}\nc.c={}'.format(len(y_f107), round(cc, 2)), transform=ax3.transAxes, color='black')
    ax3.scatter(y_f107, y_mit, s=3)
    ax3.set_xlabel('F10.7 (sfu)')
    ax3.set_ylabel('trough_min_gdlat (°)')
    # plt.legend()

    fig.subplots_adjust(left=0.08, bottom=0.1, right=0.95, top=0.93)
    outpath = figure_path + 'lt_{}\\mit-f107\\'.format(lt[1])
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    fig.savefig(outpath + 'f107-mit_undetrend date {}_{}_{}-{}_{}_{} LT_{}-{} Kp_{}-{}'.format(
        y0, m0, d0, y1, m1, d1, lt[0], lt[1] + 1, Kp[0], Kp[1]))
    # plt.show()

    y_f107_rolmean = y_f107.rolling(window=27, center=True).mean()
    y_new = y_f107 - y_f107_rolmean
    fig = plt.figure(figsize=(12, 6))
    plt.gcf()
    ax1 = plt.axes([0.08, 0.1, 0.57, 0.38])
    ax1.plot(y_new, 'x', ms=3, alpha=0.8)
    ax1.set_xlabel('date')
    ax1.set_ylabel('detrend F10.7 (sfu)')
    ax1.text(0.1, 0.9, 'a)', transform=ax1.transAxes)

    ax2 = plt.axes([0.08, 0.53, 0.57, 0.38])
    ax2.plot(y_mit, 'x', ms=3, alpha=0.8)
    ax2.set_ylabel('trough_min_gdlat (°)')
    ax2.text(0.1, 0.9, 'b)', transform=ax2.transAxes)

    ax3 = plt.axes([0.72, 0.1, 0.25, 0.8])
    print(len(y_new), len(y_mit), y_new.index[0])
    cc = y_new.corr(y_mit)
    ax3.scatter(y_new, y_mit[:], s=3)
    ax3.text(0.8, 0.9, 'c)', transform=ax3.transAxes)
    ax3.text(0.6, 0.05, 'n={}\nc.c={}'.format(len(y_new), round(cc, 2)), transform=ax3.transAxes, color='black')
    ax3.set_xlabel('detrend F10.7 (sfu)')
    ax3.set_ylabel('trough_min_gdlat (°)')
    # plt.legend()
    fig.savefig(outpath + 'f107-mit_detrend date {}_{}_{}-{}_{}_{} LT_{}-{} Kp_{}-{}'.format(
        y0, m0, d0, y1, m1, d1, lt[0], lt[1] + 1, Kp[0], Kp[1]))
    # plt.show()

    """
    figure = plt.figure(figsize=(9, 5))
    ssn_dict = {'equinox': [3, 4, 9, 10], 'summer': [5, 6, 7, 8], 'winter': [11, 12, 1, 2]}
    max_trough_gdlat, min_trough_gdlat = max(df['trough_min_lat']), min(df['trough_min_lat'])
    max_f107, min_f107 = max(df['F10.7']), min(df['F10.7'])
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
        df_ssn_f, df_ssn_trough = df_ssn['F10.7'], df_ssn['trough_min_lat']
        plt.plot(df_ssn_f, df_ssn_trough, 'bx', alpha=0.8)
        cc = df_ssn_f.corr(df_ssn_trough)
        count = len(df_ssn)
        ax.set_xlabel('F10.7 (sfu)')
        ax.set_xlim(min_f107 - 30, max_f107 + 30)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(25))

        ax.set_ylim(min_trough_gdlat - 3, max_trough_gdlat + 3)
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(1))

        ax.text(0.5, 0.9, '{}) {}'.format(chr(ord('a') + idx), ssn), transform=ax.transAxes)
        plt.text(0.55, 0.05, 'n={}\nc.c={}'.format(count, round(cc, 2)), transform=ax.transAxes)
        if idx == 0:
            plt.ylabel('trough_min_gdlat (°)')
    title = 'date: {}/{}/{}-{}/{}/{}   longitude-sector: {}°  lt: {}-{}   kp: {}-{}'.format(
        y0, m0, d0, y1, m1, d1, fixed_glon, lt[0], lt[1]+1, Kp[0], Kp[1])
    plt.suptitle(title)
    plt.subplots_adjust(left=0.07, bottom=0.1, right=0.95, top=0.9)
    plt.show()
    figure.savefig(outpath + 'longitudinal sector_{} date {}_{}_{}-{}_{}_{} LT_{}-{} Kp_{}-{}'.format(
        fixed_glon, y0, m0, d0, y1, m1, d1, lt[0], lt[1] + 1, Kp[0], Kp[1]))
    """
    return True


def gen_mean_value_at_2lt(f, lt1, lt2, item):
    f1, f2 = f.query('lt == {}'.format(lt1))[item], \
             f.query('lt == {}'.format(lt2))[item]
    # print(len(f1), len(f2))

    ret = {}
    for i in range(len(f1)):
        key = f1.index[i].date()
        if not pd.isna(f1[i]) and not pd.isna(f2[i]):    # 皆非空取均值
            value = np.mean([f1[i], f2[i]])
        elif pd.isna(f2[i]):                              # f2空，取f1值
            value = f1[i]
        else:                                             # f1空，取f2值
            value = f2[i]
        ret[key] = value
        # print(key, f1[i], f2[i], value)
    ret = pd.Series(ret)
    return ret


def MIT_Vs(df, y1, m1, d1, y2, m2, d2, lt):                     # 线性拟合
    df = df.query('lt == {} | lt == {} '.format(lt[0], lt[-1]))  # 限定地方时
    df = df['{}-{}-{}'.format(y1, m1, d1): '{}-{}-{}'.format(y2, m2, d2)]  # 分析时段
    df = df.sort_index()
    df = df[['trough_min_lat', 'Solar_Wind_Speed', 'lt']]

    MIT = gen_mean_value_at_2lt(df, lt[0], lt[1], 'trough_min_lat')
    Vs = gen_mean_value_at_2lt(df, lt[0], lt[1], 'Solar_Wind_Speed')
    print('length of trough or Vs data', len(MIT), len(Vs))

    df1 = pd.DataFrame({'MIT': MIT, 'Vs': Vs})
    df1 = df1.dropna()
    print('length after dropna:', len(df1))
    MIT, Vs = df1['MIT'], df1['Vs']
    print(len(MIT), len(Vs))

    corr = round(MIT.corr(Vs), 2)
    print('corr between MIT & VS', corr)

    figure = plt.figure(figsize=(5, 6))
    ax = figure.add_subplot(111)
    plt.sca(ax)

    plt.plot(Vs, MIT, 'b.', alpha=0.6)

    z = list(np.polyfit(np.array(Vs), np.array(MIT), 1))
    print(z)
    x = list(set(Vs))
    y = [z[0] * _ + z[1] for _ in x]
    ax.plot(x, y, 'r--', label='{:.3f} * x + {:.1f}'.format(round(z[0], 3), round(z[1], 1)))
    ax.text(0.68, 0.83, 'corr: {}'.format(corr), transform=ax.transAxes, color='black')

    plt.ylabel('MIT gdlat (°)')
    plt.xlabel('Vs (km/s)')
    title = 'MIT-Vs {}-{}-{}_{}-{}-{} lt_0'.format(y1, m1, d1, y2, m2, d2)
    plt.title(title)
    plt.legend()
    figure.tight_layout()
    # plt.show()
    figure.savefig(figure_path + title)


def plot_ls_ax(ax, y, yname, window, ls=False, mode=False):
    y = y.dropna()
    # print(len(y), y.head())
    rolmean_y = y.rolling(window=window, center=True).mean()
    if mode:
        y_new = y - rolmean_y
        # y_new = y - np.mean(y)
    else:                                     # 默认取原值
        y_new = y

    if ls == 0:                                # plot scatter & rolling mean
        plt.sca(ax)
        ax.plot(y, 'b.', ms=6, alpha=0.6, label=yname)
        # ax.plot(rolmean_y, 'k.', ms=3, alpha=0.6, label='rolling mean')
        if yname == 'trough_min_lat':
            plt.ylabel('trough_min_lat (°)')
            yticks = ax.get_yticks()
            ax.set_yticks(yticks, ['{}'.format(_) for _ in yticks])
            # xticks = ax.get_xticks()
            # diff = max(y) - min(y)
            ax.set_ylim(min(y) - 1, max(y) + 1)
            ax.yaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_minor_locator(MultipleLocator(1))
        else:                                    # yname == 'Vs'
            plt.ylabel('Vs (km/s)')
            # ax.set_ylim(200, 800)
            ax.yaxis.set_major_locator(MultipleLocator(150))
            ax.yaxis.set_minor_locator(MultipleLocator(50))
        # ax.legend()
    elif ls == 1:                             # plot ls period
        # print(y_new)
        y_new = y_new.dropna()
        t = np.array([(_ - y_new.index[0]).days for _ in y_new.index])
        # print(t, y_new)
        dy = (y_new.max() - y_new.min()) / 10
        # print(dy)
        ls = LombScargle(t, y_new, dy=dy)
        frequency, power = ls.autopower(minimum_frequency=1/36, maximum_frequency=1/2)
        sig10 = ls.false_alarm_level(0.1, minimum_frequency=1/36, maximum_frequency=1/2)
        sig5 = ls.false_alarm_level(0.05, minimum_frequency=1/36, maximum_frequency=1/2)
        # sig15 = ls.false_alarm_level(0.15, minimum_frequency=1/36, maximum_frequency=1/2)
        ax.plot(1/frequency, power, 'rx-', ms=6, alpha=0.5)
        ax.plot([1/frequency[0], 1/frequency[-1]], [sig5, sig5], ':', c='g', label='α= 0.05')
        ax.plot([1/frequency[0], 1/frequency[-1]], [sig10, sig10], ':', c='k', label='α= 0.10')
        # ax.plot([1/frequency[0], 1/frequency[-1]], [sig15, sig15], ':', c='b', label='sig10')
        plt.ylabel(yname)
        # plt.xlim(-2, 38)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        # ax.set_ylim(0, sig5 * 2)
        # ax.legend(loc='right')
        ax.legend(loc=(0.85, 0.55))
        if yname == 'Vs':
            ax.set_xlabel('period(day)')
            # ax.set_ylim(0, sig5 * 2)


def MIT_period(df, y1, m1, d1, y2, m2, d2, lt):
    # print(len(df))
    df = df.query('lt == {} | lt == {} '.format(lt[0], lt[-1]))                 # 限定地方时
    df = df['{}-{}-{}'.format(y1, m1, d1): '{}-{}-{}'.format(y2, m2, d2)]       # 分析时段
    df = df.sort_index()
    df = df[['trough_min_lat', 'Solar_Wind_Speed', 'lt']]

    print(len(df))
    window = 27

    figure = plt.figure(figsize=(12, 8))
    ax1 = figure.add_subplot(411)
    trough_min_lat = gen_mean_value_at_2lt(df, lt[0], lt[1], 'trough_min_lat')
    plot_ls_ax(ax1, trough_min_lat, 'trough_min_lat', window=window, ls=False)
    # plt.setp(ax1.get_xticklabels(), visible=False)
    ax2 = figure.add_subplot(412, sharex=ax1)
    Vs = gen_mean_value_at_2lt(df, lt[0], lt[1], 'Solar_Wind_Speed')
    plot_ls_ax(ax2, Vs, 'Vs', window=window, ls=False)

    ax3 = figure.add_subplot(413)
    plt.sca(ax3)
    plot_ls_ax(ax3, trough_min_lat, 'trough_min_lat', window=window, ls=True)
    # plt.setp(ax3.get_xticklabels(), visible=False)
    ax4 = figure.add_subplot(414, sharex=ax3)
    plt.sca(ax4)
    plot_ls_ax(ax4, Vs, 'Vs', window=window, ls=True)

    plt.gcf()
    title = 'date:{}-{}-{}:{}-{}-{}   lt:{}-{}'.format(y1, m1, d1, y2, m2, d2, lt[0], lt[-1]+1)
    plt.suptitle('Lomb-Scargle Periodogram    ' + title)
    title = title.replace(':', '_')

    figure.subplots_adjust(top=0.93, left=0.08, bottom=0.08, right=0.96)
    outpath = figure_path + 'LS\\lt_{}\\'.format(lt[1])
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    figure.savefig(outpath + title)
    plt.show()
    figure.savefig(figure_path + 'wavelet\\{}'.format(title))
    plt.close()


def table_aggregation(solar_wind_file, mit_file, glon, data_csv):
    # df_mit = pd.read_csv(mit_file, parse_dates=['date'], index_col='date', date_parser=dateparse)
    df_mit = pd.read_csv(mit_file, encoding='gb2312').query('glon == {}'.format(glon))
    print(df_mit.head())
    df_mit['date'] = pd.to_datetime(df_mit['date'])
    df_mit = df_mit.set_index('date').sort_index()
    df_mit = df_mit['2014-9-1': '2017-8-31']
    print(df_mit.head())

    df_solar_wind = pd.read_csv(solar_wind_file, encoding='gb2312').query('lt >= {} | lt <= {}'.format(18, 5))
    print(df_solar_wind.head())

    df = pd.merge(df_mit, df_solar_wind, on=['year', 'doy', 'lt'], how='outer')
    df = df.set_index(['date', 'lt']).sort_index()                    # 设置lt与date为索引
    df.to_csv(data_csv)
    print(df.head(20))


if __name__ == '__main__':
    fixed_glon = -90                                     # 限定longitude sector

    # get data.csv which comprising mit, kp, ae, F10.7, V_solar-wind, etc.
    data_path = "C:\\tmp\\data.csv"
    solar_wind_txt = "C:\\DATA\\omni\\omni_m_years14-17.txt"       # 1小时分辨率
    solar_wind_csv = 'C:\\tmp\\SolarWind.csv'

    # bigtable_SolarWind(solar_wind_txt, solar_wind_csv, fixed_glon)      # 得到solar-wind file，计算lt与date

    mit_csv = "C:\\tmp\\bigtable.csv"                                   # mit file，2014/9/1-2017/8/31
    # table_aggregation(solar_wind_csv, mit_csv, fixed_glon, data_path)   # 聚合solar-wind file and mit file

    # data = pd.read_csv(data_path, parse_dates=['date'], index_col='date', date_parser=dateparse)
    data = pd.read_csv(data_path, parse_dates=['date'], encoding='gb2312', index_col=['date'])
    # index_col=['date', 'lt']
    # print(data.loc(axis=0)['2014-9-1': '2014-9-2', [23, 0]])

    # figure_path = 'C:\\DATA\\GPS_MIT\\millstone\\summary graph\\scatter plot\\lat-solar\\'
    figure_path = 'C:\\tmp\\figure\\solar-wind\\'
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    localtime_list = [[19, 20], [21, 22], [23, 0], [1, 2], [3, 4]]
    kp_range = [0, 9]
    start_end = [[2014, 10, 1, 2015, 1, 30], [2015, 3, 1, 2015, 5, 30], [2015, 9, 1, 2015, 11, 20],
                 [2016, 8, 1, 2016, 10, 31], [2016, 9, 1, 2017, 1, 10], [2017, 2, 1, 2017, 6, 20],
                 [2016, 3, 1, 2016, 5, 15],
                 [2015, 6, 17, 2015, 7, 25], [2016, 1, 10, 2016, 2, 22]]
    for localtime in localtime_list:
        if localtime[0] == 23:
            pass
        else:
            pass
            # continue

        for date12 in start_end:
            MIT_period(data, date12[0], date12[1], date12[2], date12[3], date12[4], date12[5], localtime)
            # MIT_F107(data, kp_range, localtime, date12[0], date12[1], date12[2], date12[3], date12[4], date12[5])
            # MIT_Vs(data, 2015, 6, 17, 2015, 7, 25, localtime_list)  # 线性拟合

    # MIT_doy(data, localtime_list, y1, m1, d1, y2, m2, d2)   # 各参数随时间变化
