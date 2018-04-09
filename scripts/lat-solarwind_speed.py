#! python3
# -*- coding: utf-8 -*-

import os
# import gc
# import time
# import datetime
import numpy as np
import pandas as pd
from scipy import signal, interpolate
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


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


def MIT_Vs(df, lt):
    figure = plt.figure(figsize=(5, 6))
    ax = figure.add_subplot(111)
    df = df.query('lt == {} | lt == {}'.format(lt[0], lt[1]))
    plt.sca(ax)
    MIT, Vs = np.array(df['MIT_gdlat']), df['Solar_Wind_Speed']
    corr = round(df['MIT_gdlat'].corr(df['Solar_Wind_Speed']), 2)
    plt.plot(MIT, Vs, 'b+', alpha=0.6)
    plt.xlabel('MIT gdlat (°)')
    plt.ylabel('Vs (km/s)')
    plt.title('Vs-MIT \n glon: -90° lt:{}-{}'.format(lt[0], lt[1] + 1))
    ax.text(0.8, 0.9, 'corr: {}'.format(corr), transform=ax.transAxes, color='black')
    # figure.tight_layout()
    # plt.show()
    figure.savefig(figure_path + 'Vs-MIT lt_{}-{}'.format(lt[0], lt[1] + 1))


def MIT_doy(df, lt, y1, m1, d1, y2, m2, d2):
    print(df.columns)
    df = df.sort_index()                                                     # 按照date排序
    df = df['{}-{}-{}'.format(y1, m1, d1): '{}-{}-{}'.format(y2, m2, d2)]    # 取给定时间段数据

    figure = plt.figure(figsize=(10, 6))
    ax1 = figure.add_subplot(411)
    plt.sca(ax1)
    df1 = df.query('lt >= {} | lt <= {}'.format(lt[0], lt[-1]))  # 取午夜地方时，23-1
    x = [int(i) + int(j) / 365 for i, j in zip(df1['year'], df1['doy'])]
    y = [_ for _ in df1['MIT_gdlat']]
    ax1.plot(x, y, 'bx', ms=5, alpha=0.5)
    f = interpolate.splrep(x, y, s=10)                             # 尝试插值未果
    y_new = interpolate.splev(x, f, der=0)
    print(y_new)                                                   # 输出nan
    plt.plot(x, y_new, 'r')
    # plt.plot(x, y, 'bx', x, f, 'r--', alpha=0.5)
    # p = np.polyfit(x, y, deg=3)                                           # 尝试多项式插值未果
    # z = [p[3] * i ** 3 + p[2] * i ** 2 + p[1] * i + p[0] for i in x]
    # ax1.plot(x, z, 'r')
    ax1.set_xticks([])
    plt.yticks([40, 45, 50, 55, 60], [' {}'.format(_) for _ in [40, 45, 50, 55, 60]])
    # ax1.spines['left'].set_color('blue')                                  # 设置子图左轴颜色
    plt.ylabel('MIT gdlat (°)', color='b')                                 # 设置子图label颜色
    # ax1.yaxis.label.set_color('blue')                                     # 设置子图label颜色
    ax1.tick_params(axis='y', colors='blue')                                # 设置子图y刻度颜色
    # plt.ylim(45, 60)
    # yticks = [_ for _ in range(40, 70, 5)]
    # plt.yticks(yticks, ['{}°'.format(_) for _ in yticks])
    plt.title('longitude-sector: {}  {}-{}-{}: {}-{}-{}  lt: {}-{}h'.format(
        fixed_glon, y1, m1, d1, y2, m2, d2, lt[0], lt[-1] + 1))

    ax2 = figure.add_subplot(412)
    plt.sca(ax2)
    plt.plot(df.index, df['Solar_Wind_Speed'], 'r+', ms=3, alpha=0.5)
    plt.ylabel('Vs (km/s)', color='r')
    plt.yticks([300, 400, 500, 600, 700])
    ax2.set_xticks([])
    # ax2.spines['left'].set_color('red')
    # ax2.yaxis.label.set_color('red')
    ax2.tick_params(axis='y', colors='red')

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

    ax4 = figure.add_subplot(414)
    plt.sca(ax4)
    plt.bar(df1.index, df1['kp'], color='black')
    plt.ylim(0, 9)
    plt.yticks([0, 3, 6], ['  {}'.format(_) for _ in [0, 3, 6]])
    ax4.tick_params(axis='x', rotation=15)
    plt.ylabel('Kp index')

    figure.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    # figure.savefig(figure_path + 'longitude-sector-90° lt_{} Kp9_{}-{}'.format(localtime_glon, Kp_low, Kp_high))


def MIT_F107(df, glon, Kp, lt, years):
    df = df.query('glon == {} & {} < kp9 < {}'.format(glon, Kp[0], Kp[1]))
    if lt[0] < lt[1]:
        df = df.query('{} <= lt < {}'.format(lt[0], lt[1]))
    else:
        df = df.query('lt >= {} | lt < {}'.format(lt[0], lt[1]))

    print(df.columns)

    figure = plt.figure(figsize=(9, 5))
    ssn_dict = {'equinox': [3, 4, 9, 10], 'summer': [5, 6, 7, 8], 'winter': [11, 12, 1, 2]}
    for idx, ssn in enumerate(ssn_dict):
        months = ssn_dict[ssn]
        print(ssn, months)
        for year in years:
            for month in months:
                if year == years[0] and month == months[0]:
                    df_ssn = df['{}-{}'.format(year, month)]
                else:
                    # print('df.head: ', df['{}-{}'.format(year, month)].head())
                    df_ssn = df_ssn.append(df['{}-{}'.format(year, month)])

        ax = figure.add_subplot(131 + idx)
        plt.sca(ax)
        print(df_ssn.columns)
        # print(len(df_ssn))
        plt.plot(df_ssn['F10.7'], df_ssn['MIT_gdlat'], 'b+')
        cc = df_ssn['F10.7'].corr(df_ssn['MIT_gdlat'])
        count = len(df_ssn)
        plt.xlabel('F10.7')
        plt.xlim(50, 250)
        plt.xticks([100, 150, 200])
        plt.ylim(42, 62)
        plt.yticks([_ for _ in range(42, 62, 2)])
        if idx == 0:
            plt.ylabel('Geographic Latitude')

        plt.text(0.6, 0.85, '{}\n{} {}'.format(ssn, count, round(cc, 2)), transform=ax.transAxes, color='black')
    title = 'years:{}-{} longitude-sector: {}° lt: {}-{} {} < Kp9 < {}'.format(
        years[0], years[-1], glon, lt[0], lt[1], Kp[0], Kp[1])
    plt.suptitle(title)
    # figure.tight_layout()
    # plt.show()
    figure.savefig(figure_path + 'years{}-{} longitude-sector{}° lt_{}-{} Kp9_{}-{}'.format(
        years[0], years[-1], glon, lt[0], lt[1], Kp[0], Kp[1]))
    pass


def MIT_period(df, y1, m1, d1, y2, m2, d2, lt):
    # print(len(df))
    df = df.query('lt == {} | lt == {} '.format(lt[0], lt[1]))                 # 限定地方时
    df = df['{}-{}-{}'.format(y1, m1, d1): '{}-{}-{}'.format(y2, m2, d2)]      # 分析时段
    print(len(df))

    doy, mit = np.array([float(_) for _ in df['doy']]), np.array([float(_) for _ in df['MIT_gdlat']])
    df1 = df[~np.isnan(mit)]
    print(df['doy'])

    figure = plt.figure(figsize=(15, 8))
    ax1 = figure.add_subplot(411)
    plt.sca(ax1)
    ax1.plot(df1.index, df1['MIT_gdlat'], 'bx', ms=3, alpha=0.6, label='MIT-gdlat')
    plt.ylabel('MIT_gdlat (°)')
    # plt.subplots_adjust(wspace=0, hspace=0)

    ax2 = figure.add_subplot(412)
    plt.sca(ax2)
    Vs = np.array([float(_) for _ in df['Solar_Wind_Speed']])
    ax2.plot(df.index, df['Solar_Wind_Speed'], 'k+', ms=5, alpha=0.6, label='Vs')
    plt.ylabel('Vs (km/s)')
    plt.xlabel('date')

    ax3 = figure.add_subplot(413)
    plt.sca(ax3)
    x = doy[~np.isnan(mit)]
    y = mit[~np.isnan(mit)]
    x_normval = x.shape[0]
    f = np.linspace(0.1, 30, 3000)                                         # 频率
    pgram_MIT = signal.lombscargle(x, y, f)
    ax3.plot(f, np.sqrt(4 * (pgram_MIT / x_normval)), label='MIT period')
    plt.legend()
    plt.ylabel('MIT_gdlat (°)')
    ax3.xaxis.set_major_locator(MultipleLocator(5))
    ax3.xaxis.set_minor_locator(MultipleLocator(1))

    ax4 = figure.add_subplot(414)
    plt.sca(ax4)
    doy_normval = doy.shape[0]
    # f = np.linspace(0.1, 30, 1000)
    pgram_Vs = signal.lombscargle(doy, Vs, f)
    ax4.plot(f, np.sqrt(4 * (pgram_Vs / doy_normval)), label='Vs period')
    plt.ylabel('Vs (km/s)')
    ax4.xaxis.set_major_locator(MultipleLocator(5))
    ax4.xaxis.set_minor_locator(MultipleLocator(1))

    plt.legend()
    title = 'date:{}-{}-{}:{}-{}-{}   lt:{}h-{}h'.format(y1, m1, d1, y2, m2, d2, lt[0], lt[1]+1)
    plt.suptitle('\n\nLomb-Scargle Periodogram    ' + title)
    title = 'date_{}-{}-{}_{}-{}-{} lt_{}h-{}h'.format(y1, m1, d1, y2, m2, d2, lt[0], lt[1]+1)
    figure.savefig(figure_path + 'LS pgram\\{}'.format(title))
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
    localtime_list = [23, 0]
    figure_path = 'C:\\DATA\\GPS_MIT\\millstone\\summary graph\\scatter plot\\lat-solar\\'
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    MIT_period(data, 2015, 12, 1, 2016, 2, 1, localtime_list)
    # MIT_Vs(data, localtime_list)
    # MIT_doy(data, localtime_0_list, 2015, 11, 1, 2016, 3, 1)                  #
    # MIT_F107(data, -90, [0, 3], [23, 0], [2014, 2015, 2016, 2017])
