#! python3
# -*- coding: utf-8 -*-

import os
# import gc
import datetime
import time
import numpy as np
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# import matplotlib.ticker as ticker
import h5py
import aacgmv2


def print_time_cost(name, start):
    print('time cost - {}: '.format(name), time.clock() - start)
    start = time.clock()
    return start


def read_map_hdf5(month, filepath):
    file = h5py.File(filepath)
    HMB_mlats = np.array(list(file['HMB_mlats'][:]), dtype=np.float)
    # print(HMB_mlats[1][35:50])
    print('HMB_mlats dtype: ', HMB_mlats.dtype)
    CPCP = np.array([_[0] for _ in file['CPCP']], dtype=np.float)
    mlons = np.array(list(range(0, 365, 5)))           # 减少计算
    # HMB_mlons = np.array([_ for _ in file['HMB_mlons']], dtype=np.float)
    Times = np.array(list(map(lambda _: int(_[0]), file['Times'])), dtype=np.int)
    Nvecs = np.array(list(map(lambda _: int(_[0]), file['Nvecs'])), dtype=np.int)
    date_time = np.array(list(map(lambda x: datetime.datetime.fromtimestamp(x), Times)))

    arr_Timestamp, arr_datetime, arr_date, arr_ut_hour, arr_lt = [], [], [], [], []
    arr_mlat, arr_mlon, arr_glon, arr_CPCP, arr_Nvecs = [], [], [], [], []

    start = time.clock()
    for idx in range(len(HMB_mlats)):
        if len(HMB_mlats[0]) != len(mlons):
            raise RuntimeError(
                'len(HMB[0]) != len(mlons): len(HMB[0]) == {} & len(mlons) == {}'.format(len(HMB_mlats[0]), len(mlons)))
        # start = time.clock()
        glon = list(map(lambda x, y, z: np.round(aacgmv2.convert(x, y, 350, z)[1], 1),
                        HMB_mlats[idx], mlons, date_time))
        # start = print_time_cost('cal glon', start)         # the longest time cost, about 8ms once

        UT_hour = round(date_time[idx].hour + date_time[idx].minute / 60, 2)
        # start = print_time_cost('cal ut', start)         # about 15μs

        lt = list(map(lambda x: round((x / 15 + UT_hour) % 24, 2), glon))
        # start = print_time_cost('cal lt', start)         # about 1 ms

        for i, item in enumerate(zip(glon, lt)):  # item: (glon, lt), i: index for glons
            if -95 < item[0] < -85 and (19 < item[1] or item[1] < 6):  # at -90° & midnight
                arr_Timestamp.append(Times[idx])
                # start = print_time_cost('append ts', start)
                arr_datetime.append(date_time[idx])
                # start = print_time_cost('append dt', start)
                arr_date.append(date_time[idx].date())
                arr_ut_hour.append(UT_hour)
                # start = print_time_cost('append ut', start)
                arr_lt.append(item[1])  # 即lt[i]
                # start = print_time_cost('append lt', start)
                arr_mlat.append(HMB_mlats[idx][i])  # 3 * 10^(-6)s, longest append time cost
                arr_mlon.append(mlons[i])
                arr_glon.append(item[0])  # glon[i]
                arr_CPCP.append(CPCP[idx])
                arr_Nvecs.append(Nvecs[idx])

                # start = print_time_cost('append all', start)   # totally cost about 5*10^(-6) s
                # print(idx, date_time[idx], Nvecs[idx], UT_hour, item[0], glon[i], mlons[i], lt[i], item[1])
            else:
                pass

    columns = ['Timestamp', 'datetime', 'date', 'UT_hour', 'lt', 'HMB_mlat', 'HMB_mlon', 'HMB_glon', 'CPCP', 'Nvecs']
    df = pd.DataFrame(columns=columns)
    df['Timestamp'], df['datetime'], df['date'], df['UT_hour'] = arr_Timestamp, arr_datetime, arr_date, arr_ut_hour
    df['lt'], df['Nvecs'], df['CPCP'] = arr_lt, arr_Nvecs, arr_CPCP
    df['HMB_mlat'], df['HMB_mlon'], df['HMB_glon'] = arr_mlat, arr_mlon, arr_glon
    print('df_{}.__len__(): '.format(month), len(df))
    print(df.head())
    print('time cost for month_{} data:'.format(month), time.clock() - start)

    return df


def map_pot_hdf5_aggregation(map_datapath):
    columns = ['Timestamp', 'datetime', 'date', 'UT_hour', 'lt', 'HMB_mlat', 'HMB_mlon', 'HMB_glon', 'CPCP', 'Nvecs']
    df = pd.DataFrame(columns=columns)   # 保存所有-90°经度区域，23-01 LT的数据
    for filename in os.listdir(map_datapath):
        try:
            string = filename.split('.')
            print(string)
            if string[1] == 'hdf5' and string[0][:3] == '201':
                month = string[0].split('_')[0]
                print('----------------------- {} start : --------------------'.format(month))
                df_tmp = read_map_hdf5(month, map_datapath + filename)
                df = df.append(df_tmp)
                print('df.__len__() == ', df.__len__())
            else:
                continue
        finally:
            pass
    df.to_csv(map_datapath + 'mapex_data.csv', encoding='gb2312')
    return df


def gen_mean_HMB(df, lt):     #
    df = df[['Timestamp', 'date', 'lt', 'LT', 'HMB_mlat', 'HMB_mlon', 'HMB_glon', 'CPCP', 'Nvecs']]
    if lt[0] > lt[1]:
        pass
        # df = df.query('lt >= {} | lt <= {}'.format(lt[0], lt[1]))
    else:
        pass
        # df = df.query('{} <= lt <= {}'.format(lt[0], lt[1]))

    df_averaged = df.groupby(by=['date', 'lt']).mean().reset_index(['date', 'lt'])     # cal mean values
    """
    date_list = []  # get date list
    date_start = datetime.datetime.strptime(t1, '%Y-%m-%d').date()
    date_end = datetime.datetime.strptime(t2, '%Y-%m-%d').date()
    while date_start <= date_end:
        date_list.append(date_start)
        date_start += datetime.timedelta(days=1)
        
    df = df.set_index('date')
    data_averaged = []
    count_empty = 0           # 没有HMB数据的 date & lt
    for date in date_list:          # 按照地方时取平均,1小时分辨率
        for lt in [23, 0]:
            df_tmp = df[date:date].query('lt == {}'.format(lt))
            try:
                if df_tmp.empty:
                    # print('empty:', date, lt)
                    count_empty += 1
                    continue
                HMB_tmp = np.int(np.array(df_tmp['HMB_mlat']).mean())
                CPCP_tmp = np.int(np.array(df_tmp['CPCP']).mean() / 1000)  # kV
                Nvecs_tmp = np.int(np.array(df_tmp['Nvecs']).mean())
                data_averaged.append([date, lt, HMB_tmp, CPCP_tmp, Nvecs_tmp])
            except ValueError:
                print(date, lt)
            else:
                continue
    print('There are {} empty date & lt couples in {} days'.format(count_empty, len(date_list)))
    df_averaged = pd.DataFrame(data_averaged, columns=['date', 'lt', 'HMB_ave', 'CPCP_ave', 'Nvecs_ave'])
    df_averaged['date'] = pd.to_datetime(df_averaged['date'])
    """
    # rename columns
    df_averaged = df_averaged.rename(columns={'HMB_mlat': 'HMB_ave', 'CPCP': 'CPCP_ave', 'Nvecs': 'Nvecs_ave'})
    # print(df_averaged.columns)
    return df_averaged[['date', 'lt', 'HMB_ave', 'CPCP_ave', 'Nvecs_ave']]


def data_aggregation(path_HMB, path_MIT, n_threshold, lt):
    df_HMB = pd.read_csv(path_HMB, encoding='gb2312').query('Nvecs > {}'.format(n_threshold))   # 筛除Nvc小的数据
    df_HMB['date'] = pd.to_datetime(df_HMB['date'])

    df_HMB['LT'] = df_HMB['lt']                         # 地方时原值复制为新的一列
    df_HMB['lt'] = df_HMB['lt'].astype(np.int)          # 地方时取整

    df_HMB_new = gen_mean_HMB(df_HMB, lt=lt)            # 相同地方时，求平均

    df_MIT = pd.read_csv(path_MIT, encoding='gb2312').query('glon == {}'.format(-90))
    if lt[0] > lt[1]:
        # df_MIT = df_MIT.query('lt >= {} | lt <= {}'.format(lt[0], lt[1]))
        pass
    else:
        pass
        # df_MIT = df_MIT.query('{} <= lt <= {}'.format(lt[0], lt[1]))

    df_MIT['date'] = pd.to_datetime(df_MIT['date'])
    df_MIT = df_MIT.set_index('date')[time_from:time_to].reset_index('date')
    df_MIT = df_MIT[['date', 'lt', 'trough_min_lat', 'Solar_Wind_Speed', 'Ion_Density', 'Kp', 'Kp9', 'AE', 'AE6',
                     'BR', 'BT', 'BN', '|B', 'THETA_elevation_angle_of_v', 'PHI_azimuth_angle_of_v', 'Temperature']]
    df_MIT.rename(columns={'THETA_elevation_angle_of_v': 'theta', 'PHI_azimuth_angle_of_v': 'phi'}, inplace=True)

    # print(df_HMB_new.head(8))
    # print(df_MIT.head(10))

    df = pd.merge(df_MIT, df_HMB_new, on=['date', 'lt'], how='outer')    # 不足的补nan
    # print('before merge:', len(df_HMB_new), len(df_MIT))
    # print('after merge:', len(df))
    # print(df.head(10))
    # print(df[['date', 'lt', 'HMB_ave', 'CPCP_ave', 'Nvecs_ave', 'trough_min_lat', 'Kp9']].head(10))
    # print(df[df['lt'] == 0]['lt'].describe())
    # print(df[df['lt'] == 23]['lt'].describe())
    if df.empty:
        raise RuntimeError('merge Error: empty dataframe gotten!')
    return df, df_HMB_new, df_MIT


def fitted_1d(data_x, data_y):
    idx = np.isfinite(data_x) & np.isfinite(data_y)
    z = list(np.polyfit(np.array(data_x[idx]), np.array(data_y[idx]), 1))
    # for item in zip(data_x, data_y):
    #     print(item)
    # print(data_x, data_y)
    print(z)
    x = list(set(data_x))
    y = [z[0] * _ + z[1] for _ in x]
    return x, y, z


def plot_MIT_HMBandCPCP(df, figure_path, threshold=150, lt=0):
    # print('data length of df: {}'.format(len(df)))
    # df = df.groupby(by='date').mean().reset_index()       # 相同日期求平均
    df['trough_min_lat'] = df['trough_min_lat'] + 9         # 槽极小位置转换为磁纬
    # print(df.head())
    df1 = df.query('Nvecs_ave >= {}'.format(threshold))
    df2 = df.query('Nvecs_ave < {}'.format(threshold))

    MIT1, MIT2 = df1['trough_min_lat'], df2['trough_min_lat']
    HMB1, HMB2 = df1['HMB_ave'], df2['HMB_ave']

    MIT, Vsw = df['trough_min_lat'], df['Solar_Wind_Speed']
    HMB, CPCP, Nvc = df['HMB_ave'], df['CPCP_ave'] / 1000, df['Nvecs_ave']

    figure1 = plt.figure(figsize=(7, 4))
    ax1 = figure1.add_subplot(111)
    p1, = ax1.plot(HMB2, MIT2, 'x', c='lightskyblue', label=None)
    p2, = ax1.plot(HMB1, MIT1, 'b.', label=None)
    l1 = plt.legend([p2, p1], ['Nvecs >={}'.format(threshold), '{} < Nvecs < {}'.format(min_nvecs, threshold)],
                    loc='lower right', fontsize=10)

    print('the data of which MIT > HMB: ----------------------------')
    print(df.query('trough_min_lat + 9 > HMB_ave'))
    print('data end. -----------------------------------------------')

    p3, = ax1.plot(list(set(MIT)), list(set(MIT)), 'k--', label='MIT == $Λ_{HM}$', linewidth=0.5)

    try:
        x1, y1, z1 = fitted_1d(HMB1, MIT1)
        print(x1, y1, z1)
        p4, = ax1.plot(x1, y1, 'r--', label='MIT = {:.2f} * r"$Λ$" + {:.2f}'.format(round(z1[0], 2), round(z1[1], 2)),
                       linewidth=0.8)

        x2, y2, z2 = fitted_1d(HMB2, MIT2)
        # print(x2, y2, z2)
        p5, = ax1.plot(x2, y2, ls='--', c='lightsalmon', linewidth=0.7,
                       label='MIT = {:.2f} * r"$Λ$" + {:.2f}'.format(round(z2[0], 2), round(z2[1], 2)))
        ax1.legend([p3, p4, p5], ['MIT == $Λ_{HM}$',
                                  'MIT = {:.2f} * Λ + {:.2f}'.format(round(z1[0], 2), round(z1[1], 2)),
                                  'MIT = {:.2f} * Λ + {:.2f}'.format(round(z2[0], 2), round(z2[1], 2))],
                   loc='upper left', scatterpoints=1, fontsize=10)
    except TypeError as e:
        print(e)
        pass

    plt.gca().add_artist(l1)

    coe1 = round(HMB1.corr(MIT1), 2)
    coe2 = round(HMB2.corr(MIT2), 2)
    ax1.text(0.02, 0.45, 'c.c_o={}\ncount_o={}\n\nc.c_x={}\ncount_x={}'.format(coe1, len(HMB1), coe2, len(HMB2)),
             transform=ax1.transAxes, color='k', fontsize=10)

    ax1.set_xlabel(r'$Λ_{HM}$(°)')
    ax1.set_ylabel('MIT Gm_lat(°)')
    # ax1.set_xlim(50, 75)
    # ax1.set_ylim(53, 63)
    # ax1.legend()
    plt.suptitle('date_{}:{}  lt_{}'.format(time_from, time_to, lt))
    plt.subplots_adjust(top=0.92, bottom=0.14, left=0.1, right=0.95)
    # plt.show()
    figure1.savefig(figure_path + '{}-{}_lt-{:2d}_MIT-HMB-fit'.format(time_from, time_to, lt))
    plt.close()

    """
    figure2 = plt.figure(figsize=(7, 4))
    ax2 = figure2.add_subplot(111)
    ax2.plot(CPCP, MIT, 'b.', label='Nvecs > 80')
    coe2 = round(CPCP.corr(MIT), 2)
    x2, y2, z2 = fitted_1d(CPCP, MIT)
    print(x2, y2, z2)
    ax2.plot(x2, y2, 'r--', label='MIT  = {:.2f} * CPCP + {:.2f}'.format(round(z2[0], 2), round(z2[1], 2)))
    ax2.text(0.1, 0.1, 'c.c={}'.format(coe2), transform=ax2.transAxes, color='k', fontsize=10)
    ax2.set_xlabel(r'$Φ_{pc}$(kV)')
    ax2.set_ylabel('MIT Gm_lat(°)')
    # ax2.set_xlim(15, 80)
    # ax2.set_ylim(53, 63)
    ax2.legend(loc='upper right', fontsize=10)
    plt.suptitle('date_{}:{}  lt_{}'.format(time_from, time_to, lt))
    plt.subplots_adjust(top=0.92, bottom=0.14, left=0.1, right=0.95)
    # plt.show()
    # figure2.savefig(figure_path + '{}-{}_lt-{:2d}_MIT-CPCP-fit'.format(time_from, time_to, lt))
    plt.close()
    """

    """
    figure3 = plt.figure(figsize=(7, 4))
    ax3 = figure3.add_subplot(111)
    ax3.plot(CPCP, HMB, 'b.', label='Nvecs > 80')
    coe3 = round(CPCP.corr(HMB), 2)
    x3, y3, z3 = fitted_1d(CPCP, HMB)
    ax3.plot(x3, y3, 'r--', label='HMB  = {:.2f} * CPCP + {:.2f}'.format(round(z2[0], 2), round(z2[1], 2)))
    ax3.text(0.1, 0.1, 'c.c={}'.format(coe3), transform=ax3.transAxes, color='k', fontsize=10)
    ax3.set_xlabel(r'$Φ_{pc}$(kV)')
    ax3.set_ylabel(r'$Λ_{HM}$(°)')
    ax3.legend(loc='upper right', fontsize=10)
    plt.suptitle('date_{}:{}  lt_{}'.format(time_from, time_to, lt))
    plt.subplots_adjust(top=0.92, bottom=0.14, left=0.1, right=0.95)
    # plt.show()
    figure3.savefig(figure_path + '{}-{}_lt-{:2d}_HMB-CPCP-fit'.format(time_from, time_to, lt))
    plt.close()
    """
    return True


def plot_VSW_HMBandCPCP():
    return


def plot_MIT_VSW():
    return


def interpolate_3(data_x, data_y, k, s):
    x = np.array([(_ - data_x[0]).days for _ in data_x])
    y = np.array([int(_) if not np.isnan(_) else np.nan for _ in data_y])
    # print(x)
    # print(x.dtype, y.dtype)
    # print(len(data_x), len(data_y))

    w = np.isnan(y)
    y[w] = 0

    sx = np.linspace(x[0], x[-1], 5 * len(x))

    y = interpolate.UnivariateSpline(x, y, w=~w, k=k, s=s)(sx)  # 能外推的样条插值, 默认k=3,即三阶样条曲线

    dt0 = datetime.datetime.combine(data_x[0], datetime.time.min)
    dx = np.array(list(map(lambda _: dt0 + datetime.timedelta(_), sx)))
    print(dx.dtype)
    return sx, y


def plot_timeSeries_individual_panel(df, df_E_Bz, figure_path, threshold=150, lt=0, ):
    """
    :param df:
    :param df_E_Bz:
    :param figure_path:
    :param threshold:
    :param lt:
    :return:
    """

    s_CPCP, s_HMB, s_MIT = 2000, 120, 40
    if time_from == '2015-6-17' or time_from == '2016-1-10':
        s_CPCP, s_HMB, s_MIT = 250, 50, 10
    print('start plot timeSeries figure: -----------------------')
    # df = df.groupby(by='date').mean().reset_index('date')
    df = df.query('lt == {}'.format(lt)).reset_index(drop=True)
    df['trough_min_lat'] = df['trough_min_lat'] + 9             # 槽极小位置转换为磁纬
    df['CPCP_ave'] = df['CPCP_ave'] / 1000
    # print(df_E_Bz.head())

    """
    df1 = df.query('Nvecs_ave >= {}'.format(threshold))
    df1 = df1.reset_index(drop=True)
    df2 = df.query('Nvecs_ave < {}'.format(threshold))
    df2 = df2.reset_index(drop=True)
    print(len(df), len(df1), len(df2))
    print(df1.head())
    """

    date1 = df_E_Bz['date'][0]
    x_date_E = np.array(list(map(lambda dt, h: (dt - date1).days + (h - lt) / 24, df_E_Bz['date'], df_E_Bz['lt'])))
    # print(x_date_E)

    figure = plt.figure(figsize=(15, 12))
    plt.gcf()

    ax1 = plt.subplot(511)               # first panel: SW_Speed,
    plt.sca(ax1)
    ax1.bar(x_date_E, df_E_Bz['Bz GSM'], width=0.04, color='k', alpha=0.8)
    ax1.set_ylabel('Bz_GSM(nT)')
    ax1.set_yticks([-15, 0, 15])

    ax12 = ax1.twinx()
    ax12.plot(x_date_E, df_E_Bz['Plasma flow speed'], 'o-', ms=0.5, color='r', linewidth=0.2, alpha=1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax12.set_ylabel(r'$V_{SW}$(km)', color='r')
    ax12.set_yticks([300, 450, 600])
    ax12.set_ylim(200, )
    ax12.tick_params(axis='y', colors='r')
    ax12.spines['right'].set_color('r')
    plt.title('{}:{} {}:00 -{}:00 LT'.format(time_from, time_to, lt, lt + 1))

    ax2 = figure.add_subplot(512, sharex=ax1)           # second panel:
    plt.sca(ax2)
    ax2.bar(x_date_E, df_E_Bz['E'], color='k', width=0.04, alpha=0.8)
    # ax2.spines['bottom'].set_position(('data', 0))
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set_ylabel('Esw(mV/m)')
    # ax2.set_ylim(df_E_Bz['E'].min() - 1, df_E_Bz['E'].max() + 1)

    date0 = df['date'][0]
    x_date = np.array([(_ - date0).days for _ in df['date']])
    # print(x_date)
    ax3 = figure.add_subplot(513, sharex=ax1)           # thrid panel: CPCP
    ax3.bar(x_date, df['AE6'], color='k', alpha=0.5)
    ax3.set_ylabel('AE6 index')
    ax3.set_yticks([250, 500])
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax32 = ax3.twinx()
    ax32.plot(x_date, df['CPCP_ave'], '.', color='r', ms=10, linewidth=0.5)
    ax32.set_ylabel(r'$Φ_{pc}$(kV)', color='r')
    ax32.tick_params(axis='y', colors='r')
    ax32.spines['right'].set_color('r')
    ax32.set_yticks([20, 40, 60])

    # x, y = interpolate_3(df['date'], df['CPCP_ave'], k=3, s=s_CPCP)
    # ax32.plot(x, y, 'r--', linewidth=0.5)
    try:
        ax32.set_ylim(df['CPCP_ave'].min() - 5, df['CPCP_ave'].max() + 5)
    except ValueError:
        ax32.set_ylim(10, 80)

    ax4 = figure.add_subplot(514, sharex=ax1)           # forth panel:
    # ax4.fill_between(x_date, df['Nvecs_ave'], threshold, color='k', alpha=0.5)
    ax4.plot(x_date, df['Nvecs_ave'], color='k', alpha=0.5, linewidth=0.5)
    ax4.set_ylabel('Nvecs')
    ax4.set_yticks([threshold, 300, 600 - threshold])
    # ax4.plot(x_date, [150] * len(x_date), 'r--', linewidth=0.3)
    plt.setp(ax4.get_xticklabels(), visible=False)

    ax42 = ax4.twinx()
    ax42.plot(x_date, df['HMB_ave'], '.', color='r', ms=10, linewidth=0.5)
    ax42.set_ylabel(r'$Λ_{HM}$(°)', color='r')
    ax42.tick_params(axis='y', colors='r')
    ax42.spines['right'].set_color('r')
    ax42.set_yticks([55, 60, 65, 70])

    # x, y = interpolate_3(df['date'], df['HMB_ave'], k=3, s=s_HMB)
    # ax42.plot(x, y, 'r--', linewidth=0.8)
    try:
        ax42.set_ylim(df['HMB_ave'].min() - 3, df['HMB_ave'].max() + 3)
    except ValueError:
        ax42.set_ylim(55, 75)

    ax5 = figure.add_subplot(515, sharex=ax1)           # forth panel:
    # ax42.plot(x_date, df['Kp9'], 'o-', ms=2, alpha=0.5, color='slategrey', linewidth=0.5)
    ax5.bar(x_date, df['Kp9'], color='k', alpha=0.5)
    ax5.set_ylabel('Kp9 index')
    ax5.set_yticks([0, 2, 4, 6])
    ax5.set_xlabel('date')
    x_ticks = x_date[::(len(x_date) // 6)]
    plt.xticks(x_ticks, [r'{}'.format((df['date'][0] + datetime.timedelta(int(_))).date()) for _ in x_ticks])

    ax52 = ax5.twinx()
    ax52.plot(x_date, df['trough_min_lat'], 'x', color='b', ms=10, linewidth=0.8)
    ax52.set_ylabel(r'$MIT_{gmlat}$(°)', color='b')
    ax52.tick_params(axis='y', colors='b')
    ax52.spines['right'].set_color('b')
    try:
        ax52.set_ylim(df['trough_min_lat'].min() - 2, df['trough_min_lat'].max() + 2)
    except ValueError:
        pass
    # ax52.set_yticks([50, 55, 60])

    # x, y = interpolate_3(df['date'], df['trough_min_lat'], k=2, s=s_MIT)
    # ax52.plot(x, y, 'b--', linewidth=0.8)

    figure.tight_layout(pad=2)       # 调整整体空白
    plt.subplots_adjust(hspace=0)  # 调整子图间距
    # plt.show()
    figure.savefig(figure_path + '{}_{} lt-{:2d} timeSeries'.format(time_from, time_to, lt))
    plt.close()
    return figure


def plot_for_vt(df):
    df1 = df.query('Nvc >= 0')
    # df2 = df.query('Nvc < 100')

    figure = plt.figure(figsize=(10, 10))
    plt.gcf()
    ax1 = plt.subplot(311)
    plt.sca(ax1)
    ax1.plot(df1['date'], df1['trough_min_lat'] + 10, '.')
    y = interpolate_3(df1['date'], df1['trough_min_lat'] + 10, k=3, s=80)
    ax1.plot(df1['date'], y, 'k--', linewidth=0.5)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylabel(r'$GMlat_{MIT}$(°)')
    plt.title('date: {}_{}'.format(time_from, time_to))

    ax3 = figure.add_subplot(312, sharex=ax1)
    ax3.plot(df1['date'], df1['HMB'], '.')
    y = interpolate_3(df1['date'], df1['HMB'], k=3, s=1500)
    ax3.plot(df1['date'], y, 'k--', linewidth=0.5)
    ax3.set_ylabel(r'$Λ_{HM}$(°)')
    plt.setp(ax3.get_xticklabels(), visible=False)

    ax2 = figure.add_subplot(313, sharex=ax1)
    ax2.plot(df1['date'], df1['CPCP'], '.')
    y = interpolate_3(df1['date'], df1['CPCP'], k=3, s=8000)
    ax2.plot(df1['date'], y, 'k--', linewidth=0.5)
    ax2.set_ylabel(r'$Φ_{pc}$(kV)')
    ax2.set_xlabel('date')

    figure.tight_layout(pad=2)  # 调整整体空白
    plt.subplots_adjust(hspace=0.05)  # 调整子图间距
    plt.show()

    return figure


def plot_figures(df, t_from, t_to, threshold=150, lt=0):
    print(t_from, t_to, lt, '-------------------------------------')
    df = df.query('lt == {}'.format(lt))
    if lt > 18:                                # 对-90°经度，地方时落后0°子午线6小时；
        df['date'] -= datetime.timedelta(1)                        # date_lt = date_ut - 1
    df = df.set_index('date')[t_from: t_to].reset_index('date')    # 给定时间内的数据

    SW_new_path = 'C:\\DATA\\omni\\SW_tmp.csv'  # 包括E与Bz，date与地方时一致
    df_E = pd.read_csv(SW_new_path)            # .query('lt == {}'.format(lt))
    df_E['date'] = pd.to_datetime(df_E['date'])
    df_E = df_E.set_index('date')[t_from: t_to].reset_index('date')
    # print(df_E.head())

    # df = df[['date', 'lt', 'trough_min_lat', 'Solar_Wind_Speed', 'HMB_ave', 'CPCP_ave', 'Nvecs_ave', 'Kp9']]

    figure_path = 'C:\\Users\\user\\Desktop\\MIT-SW\\figures\\new\\lt_{}\\'.format(((lt + 1) // 2 * 2) % 24)
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    # 线性拟合
    plot_MIT_HMBandCPCP(df, figure_path, threshold=threshold, lt=lt)

    # time_series
    plot_timeSeries_individual_panel(df, df_E, figure_path, threshold=threshold, lt=lt)
    plt.close()

    # figure3 = plot_for_vt(df)
    # figure3.savefig(figure_path + 'timeSeries MIT-ConvPattern {}_{}'.format(time_from, time_to))
    return True


def statistics(df):
    df = df[['CPCP_ave', 'HMB_ave', 'trough_min_lat', 'AE', 'AE6', 'Kp', 'Kp9',
             'Solar_Wind_Speed', 'Temperature', 'BR', 'BT', 'BN', '|B']]
    coe = df.corr()
    print(coe)
    return coe


if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 12})  # 修改所有label和tick的大小
    mapex_data_path = 'C:\\Users\\user\\Desktop\\MIT-SW\\data\\'
    # HMB_path = mapex_data_path + 'CPCP-HMB.xlsx'
    # df_mapex = map_pot_hdf5_aggregation(mapex_data_path)       # 合并各月mapex_hdf5数据，保存在csv中；
    HMB_path = mapex_data_path + 'mapex_data.csv'                # 聚合后的mapex数据；date与世界时一致
    MIT_path = 'C:\\tmp\\data.csv'                               # MIT数据，date与世界时一致
    SW_bigtable_path = 'C:\\DATA\\omni\\SW_tmp.csv'              # 新SW数据，包括E与Bz，date与地方时一致

    list_time = [['2014-10-1', '2015-1-30'], ['2015-3-1', '2015-5-30'], ['2015-9-1', '2015-11-20'],
                 ['2016-8-1', '2016-10-31'], ['2016-9-1', '2016-12-31'], ['2017-2-1', '2017-6-20'],
                 ['2016-3-1', '2016-5-15'],
                 ['2015-6-17', '2015-7-25'], ['2016-1-10', '2016-2-22']]

    list_time_short = [['2015-6-17', '2015-7-25'], ['2016-1-10', '2016-2-22']]

    min_nvecs = 100
    LocalTimes = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
    nvecs_threshold = 150
    for time in list_time_short:
        time_from, time_to = time[0], time[1]
        # mapex的HMB数据限制最小Nvc，然后取平均，得1小时分辨率后，与MIT聚合
        DF, DF_HMB, DF_MIT = data_aggregation(HMB_path, MIT_path, n_threshold=min_nvecs, lt=LocalTimes)
        for localtime in LocalTimes:
            plot_figures(DF, time_from, time_to, threshold=nvecs_threshold, lt=localtime)
            pass
        # 某些特殊值，如HMB低于MIT， HMB<60等；
        # statistics(DF)
