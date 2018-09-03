#! python3
# -*- coding: utf-8 -*-

import os
# import gc
# import time
# import datetime
import numpy as np
import pandas as pd
from scipy import interpolate
# from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from astropy.stats import LombScargle


def dateparse(dates):
    return pd.datetime.strptime(dates, '%Y-%m-%d')


def bigtable_SolarWind(txt_path, csv_path, glon):
    """
    get .csv file which save Solar Wind data
    :param txt_path: origin .txt datafile
    :param csv_path: the path that .csv file saved 
    :param glon: for calculating lt for glon
    :return: none, get saved .csv
    """
    col_widths = [4, 4, 3, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 9]  # 定宽数据宽度
    col_names = ['year', 'doy', 'Hour', 'Lat_of_Earth', 'Lon_of_Earth', 'BR', 'BT', 'BN',
                 '|B', 'Solar_Wind_Speed', 'THETA_elevation_angle_of_v', 'PHI_azimuth_angle_of_v',
                 'Ion_Density', 'Temperature']  # 自定义列名

    table = pd.read_fwf(txt_path, widths=col_widths, header=None, names=col_names)
    lt_0 = np.array(table['Hour'])
    lt_glon = (lt_0 + glon // 15) % 24
    table.insert(3, 'lt'.format(fixed_glon), lt_glon)
    table.to_csv(csv_path, index=False)


def MIT_doy(df, lt, y1, m1, d1, y2, m2, d2):
    print(df.columns)
    df = df.sort_index()                                                     # 按照date排序
    df = df['{}-{}-{}'.format(y1, m1, d1): '{}-{}-{}'.format(y2, m2, d2)]    # 取给定时间段数据

    figure = plt.figure(figsize=(10, 6))
    ax1 = figure.add_subplot(411)
    plt.sca(ax1)
    df1 = df.query('lt >= {} | lt <= {}'.format(lt[0], lt[-1]))  # 取午夜地方时，23-1
    ax1.plot(df1['MIT_gdlat'], 'bx', ms=5, alpha=0.5)
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
    y = np.array(df1['MIT_gdlat'])
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


def MIT_F107(df, glon, Kp, lt):
    df = df.query('glon == {} & {} <= kp9 <= {}'.format(glon, Kp[0], Kp[1]))
    if lt[0] <= lt[1]:
        df = df.query('{} <= lt <= {}'.format(lt[0], lt[1]))
    else:
        df = df.query('lt >= {} | lt <= {}'.format(lt[0], lt[1]))
    print(df.columns)

    y_f107, y_mit = df['F10.7'], df['MIT_gdlat']
    fig = plt.figure(figsize=(8, 6))
    plt.gcf()
    ax1 = plt.axes([0.1, 0.15, 0.5, 0.3])
    ax1.plot(y_f107)
    ax2 = plt.axes([0.1, 0.55, 0.5, 0.3])
    ax2.plot(y_mit)
    ax3 = plt.axes([0.7, 0.15, 0.25, 0.6])
    cc = y_f107.corr(y_mit)
    ax3.text(0.6, 0.85, 'count:{}\ncc:{}'.format(len(y_f107), round(cc, 2)), transform=ax3.transAxes, color='black')
    ax3.scatter(y_f107, y_mit, s=3)
    # plt.legend()
    fig.savefig(figure_path + 'f107-mit undetrend')
    plt.show()

    y_f107_rolmean = y_f107.rolling(window=27, center=True).mean()
    y_new = y_f107 - y_f107_rolmean
    fig = plt.figure(figsize=(8, 6))
    plt.gcf()
    ax1 = plt.axes([0.1, 0.15, 0.5, 0.3])
    ax1.plot(y_new)
    ax2 = plt.axes([0.1, 0.55, 0.5, 0.3])
    ax2.plot(y_mit)
    ax3 = plt.axes([0.7, 0.15, 0.25, 0.6])
    print(len(y_new), len(y_mit), y_new.index[0])
    cc = y_new.corr(y_mit)
    ax3.scatter(y_new, y_mit[:], s=3)
    ax3.text(0.6, 0.85, 'count:{}\ncc:{}'.format(len(y_new), round(cc, 2)), transform=ax3.transAxes, color='black')
    # plt.legend()
    fig.savefig(figure_path + 'f107-mit detrend')
    plt.show()

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
        plt.plot(df_ssn['F10.7'], df_ssn['MIT_gdlat'], 'bx')
        cc = df_ssn['F10.7'].corr(df_ssn['MIT_gdlat'])
        count = len(df_ssn)
        plt.xlabel('F10.7')
        plt.xlim(50, 250)
        plt.xticks([100, 150, 200])
        plt.ylim(38, 56)
        plt.yticks([_ for _ in range(40, 56, 2)])
        ax.set_title('{}'.format(ssn))
        plt.text(0.6, 0.85, 'count:{}\ncc:{}'.format(count, round(cc, 2)), transform=ax.transAxes, color='black')
        if idx == 0:
            plt.ylabel('Geographic Latitude')
    title = 'date: 2014/9/1-2017/8/31   longitude-sector: {}°  lt: {}-{}   kp: {}-{}'.format(
        glon, lt[0], lt[1]+1, Kp[0], Kp[1])
    plt.suptitle(title)
    # figure.tight_layout()
    # plt.show()
    figure.savefig(figure_path + '2014-9-1_2017-8-31 longitude sector{}° lt range{}-{} kp range{}-{}'.format(
        glon, lt[0], lt[1] + 1, Kp[0], Kp[1]))
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


def MIT_Vs(df, y1, m1, d1, y2, m2, d2, lt):
    df = df.query('lt == {} | lt == {} '.format(lt[0], lt[-1]))  # 限定地方时
    df = df['{}-{}-{}'.format(y1, m1, d1): '{}-{}-{}'.format(y2, m2, d2)]  # 分析时段
    df = df.sort_index()
    df = df[['MIT_gdlat', 'Solar_Wind_Speed', 'lt']]

    MIT = gen_mean_value_at_2lt(df, lt[0], lt[1], 'MIT_gdlat')
    Vs = gen_mean_value_at_2lt(df, lt[0], lt[1], 'Solar_Wind_Speed')
    print(len(MIT), len(Vs))

    df1 = pd.DataFrame({'MIT': MIT, 'Vs': Vs})
    df1 = df1.dropna()
    print(len(df1))
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
    else:
        y_new = y

    if ls == 0:                                # plot scatter & rolling mean
        plt.sca(ax)
        ax.plot(y, 'bx', ms=6, alpha=0.6, label=yname)
        # ax.plot(rolmean_y, 'k.', ms=3, alpha=0.6, label='rolling mean')
        if yname == 'MIT_gdlat':
            plt.ylabel('MIT_gdlat (°)\n')
            yticks = ax.get_yticks()
            ax.set_yticks(yticks, ['{}'.format(_) for _ in yticks])
            # xticks = ax.get_xticks()
            # ax.set_ylim(40, 55)
        else:                                    # yname == 'Vs'
            plt.ylabel('Vs (km/s)')
            # ax.set_ylim(200, 800)
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
        ax.plot([1/frequency[0], 1/frequency[-1]], [sig5, sig5], ':', c='g', label='sig5')
        ax.plot([1/frequency[0], 1/frequency[-1]], [sig10, sig10], ':', c='k', label='sig10')
        # ax.plot([1/frequency[0], 1/frequency[-1]], [sig15, sig15], ':', c='b', label='sig10')
        plt.ylabel(yname)
        # plt.xlim(-2, 38)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        # ax.set_ylim(0, sig5 * 2)
        # ax.legend(loc='right')
        ax.legend(loc=(0.9, 0.55))
        if yname == 'Vs':
            ax.set_xlabel('period(day)')
            # ax.set_ylim(0, sig5 * 2)


def MIT_period(df, y1, m1, d1, y2, m2, d2, lt):
    # print(len(df))
    df = df.query('lt == {} | lt == {} '.format(lt[0], lt[-1]))                 # 限定地方时
    df = df['{}-{}-{}'.format(y1, m1, d1): '{}-{}-{}'.format(y2, m2, d2)]       # 分析时段
    df = df.sort_index()
    df = df[['MIT_gdlat', 'Solar_Wind_Speed', 'lt']]

    print(len(df))
    window = 27

    figure = plt.figure(figsize=(12, 8))
    ax1 = figure.add_subplot(411)
    MIT_gdlat = gen_mean_value_at_2lt(df, lt[0], lt[1], 'MIT_gdlat')
    plot_ls_ax(ax1, MIT_gdlat, 'MIT_gdlat', window=window, ls=False)

    ax2 = figure.add_subplot(412)
    Vs = gen_mean_value_at_2lt(df, lt[0], lt[1], 'Solar_Wind_Speed')
    plot_ls_ax(ax2, Vs, 'Vs', window=window, ls=False)

    ax3 = figure.add_subplot(413)
    plt.sca(ax3)
    plot_ls_ax(ax3, MIT_gdlat, 'MIT_gdlat', window=window, ls=True)

    ax4 = figure.add_subplot(414)
    plt.sca(ax4)
    plot_ls_ax(ax4, Vs, 'Vs', window=window, ls=True)

    plt.gcf()
    title = 'date:{}-{}-{}:{}-{}-{}   lt:{}h-{}h'.format(y1, m1, d1, y2, m2, d2, lt[0], lt[-1]+1)
    plt.suptitle('\n\nLomb-Scargle Periodogram    ' + title)
    title = title.replace(':', '_')
    # figure.savefig(figure_path + 'LS pgram\\three years\\lt{}\\tm+{}'.format(lt[1], title))
    figure.savefig(figure_path + 'wavelet\\{}'.format(title))
    # plt.show()


def table_aggregation(solar_wind_file, mit_file, glon, data_csv):
    # df_mit = pd.read_csv(mit_file, parse_dates=['date'], index_col='date', date_parser=dateparse)
    df_mit = pd.read_csv(mit_file, encoding='gb2312').query('2014 <= year <= 2017 & glon == {}'.format(glon))
    print(df_mit.head())
    df_solar_wind = pd.read_csv(solar_wind_file).query('lt >= 18 | lt <= 5')
    print(df_solar_wind.head())
    df = pd.merge(df_mit, df_solar_wind, on=['year', 'doy', 'lt'], how='outer')
    df = df.query('glon == {}'.format(glon))             # 只取都有数据的日子
    df = df.set_index(['lt', 'date'])                    # 设置lt与date为索引
    df.to_csv(data_csv)
    print(df.head())


if __name__ == '__main__':
    fixed_glon = -90                                     # 限定longitude sector

    # get data.csv which comprising mit, kp, ae, F10.7, V_solar-wind, etc.
    data_path = "C:\\tmp\\data.csv"
    solar_wind_txt = "C:\\DATA\\omni\\omni_m_years14-17.txt"
    solar_wind_csv = 'C:\\tmp\\SolarWind.csv'
    bigtable_SolarWind(solar_wind_txt, solar_wind_csv, fixed_glon)
    mit_csv = "C:\\tmp\\bigtable.csv"
    # table_aggregation(solar_wind_csv, mit_csv, fixed_glon, data_path)

    data = pd.read_csv(data_path, parse_dates=['date'], index_col='date', date_parser=dateparse)
    data = data.sort_index()
    # print(data)
    data = data['2014': '2017']

    figure_path = 'C:\\DATA\\GPS_MIT\\millstone\\summary graph\\scatter plot\\lat-solar\\'
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    localtime_list = [23, 0]
    kp_range = [0, 9]
    t_list = [[2014, 10, 20, 2015, 7, 1], [2015, 3, 1, 2015, 7, 1], [2015, 6, 17, 2015, 7, 25],
              [2015, 9, 1, 2015, 12, 15], [2016, 1, 10, 2016, 2, 22], [2016, 7, 1, 2017, 6, 1],
              [2016, 8, 1, 2016, 10, 31], [2017, 2, 1, 2017, 5, 31], [2016, 10, 1, 2016, 12, 31]]

    # MIT_period(data, 2016, 7, 1, 2017, 6, 1, [23, 0])
    for t in t_list:
        MIT_Vs(data, t[0], t[1], t[2], t[3], t[4], t[5], localtime_list)

    # MIT_Vs(data, 2015, 6, 17, 2015, 7, 25, localtime_list)
    # MIT_F107(data, -90, kp_range, localtime_list)
    # MIT_doy(data, localtime_list, y1, m1, d1, y2, m2, d2)
