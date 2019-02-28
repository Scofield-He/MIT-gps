#! python3
# -*- coding: utf-8 -*-

# import os
# import gc
# import time
import datetime
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import pycwt as wavelet
import matplotlib
# from matplotlib.ticker import MultipleLocator

# from pycwt.helpers import find


def dateparse(dates):
    return pd.datetime.strptime(dates, '%Y-%m-%d')


def interpolate_mit(ds):
    ds.index = pd.to_datetime(ds.index)
    index = [_.date() for _ in ds.index]
    # print(ds.index)
    # print(index)
    x_new = [(_ - ds.index[0]).days for _ in ds.index]
    # print(x_new)

    ds = ds.dropna()
    x = [(_ - ds.index[0]).days for _ in ds.index]           # 非空数据的x
    # x_new = x_new[x_new.index(min(x)): x_new.index(max(x))]         # 最小到最大无间隔取x
    index_new = [index[0] + datetime.timedelta(_) for _ in x_new]     # 对应index
    y = np.array(ds)
    # print(x)

    tck = interpolate.splrep(x, y, s=8)                      # k=3 for default
    y_bspline = interpolate.splev(x_new, tck, der=0)         # der=0, 对0阶导，即对自身插值

    ret = pd.Series(data=y_bspline, index=index_new)
    return ret


def plot_TroughAndVs(y1, y1_new, y2, lt):
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(211)
    plt.sca(ax1)
    # ax1.plot(y1, 'o', ms=2)
    ax1.plot(y1_new, 'k*--', ms=2, alpha=1, linewidth=0.5)
    # ax1.legend(['data', 'b_spline'], loc='best')
    ax1.set_ylabel('trough_min_gdlat(°)')
    # ax1.set_xlabel('date')

    ax2 = fig.add_subplot(212)
    plt.sca(ax2)
    ax2.plot(y2, 'o--', ms=2, linewidth=0.5)
    ax2.set_ylabel('Vs (km/s)')
    ax2.set_xlabel('date')
    # ax2.legend(['Vs'])

    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.97, top=0.95)
    plt.show()
    fig.savefig(fig_path + 'mit interpolate & Vs at LT-{}'.format(lt[1]))  # 中间地方时
    plt.close()


def gen_mean_value_at_2lt(f, lt, item):
    f1, f2 = f.query('lt == {}'.format(lt[0]))[item], \
             f.query('lt == {}'.format(lt[1]))[item]
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


def wavelet_analysis(f, item, lt):
    N = len(f)
    dt = 1
    t = np.array([(_ - f.index[0]).days for _ in f.index])
    dat = np.array(f - f.mean())                                  # 减去均值
    print(t, '\nlength of t: ', len(t), '\nlength of dat: ', len(dat))
    std = dat.std()
    var = dat.var()
    dat_norm = dat / std

    mother = wavelet.Morlet(6)
    s0 = 2                   # starting scale, 2 days here
    dj = 1/3
    J = 4/dj
    alpha, _, _ = wavelet.ar1(dat)

    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J, mother)
    print('wave:', wave)
    print('fft:', fft)
    iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std

    power = np.abs(wave) ** 2
    fft_power = np.abs(fft) ** 2
    period = 1 / freqs
    # print(period)
    print(np.log2(period))

    signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                             significance_level=0.95,
                                             wavelet=mother)
    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = power / sig95

    glbl_power = power.mean(axis=1)
    dof = N - scales
    glbl_signif, tmp = wavelet.significance(var, dt, scales, 1, alpha,
                                            significance_level=0.95, dof=dof, wavelet=mother)

    plt.close('all')
    plt.ioff()
    # figprops = dict(figsize=(11, 8), dpi=72)
    fig = plt.figure(figsize=(15, 8), dpi=72)
    plt.gcf()

    ax1 = plt.axes([0.1, 0.65, 0.65, 0.25])
    ax1.plot(f.index, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
    ax1.plot(f.index, dat, 'k', linewidth=1.5)
    ax1.set_title('a) {}'.format(item))
    ax1.set_ylabel('{}(km/s)'.format(item))

    ax2 = plt.axes([0.1, 0.15, 0.65, 0.4])
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    # levels = [_ ** 2 for _ in range(2, 16, 2)]
    cs = ax2.contourf(f.index, period, np.log2(power), np.log2(levels), extend='both',
                      cmap=plt.cm.jet)
    # extent = [t.min(), t.max(), 0, max(period)]
    extent = [f.index.min(), f.index.max(), 0, max(period)]
    ax2.contour(f.index, period, sig95, [-99, 1], colors='k', linewidths=2, extent=extent)
    ax2.set_title('b)Wavelet Power Spectrum')
    ax2.set_ylabel('Period (days)')
    # ax2.plot(t, [9] * len(t), 'k--', t, [13.5] * len(t), 'k--', t, [27] * len(t), 'k--')
    ax2.plot([f.index[0], f.index[-1]], [9, 9], 'k--',
             [f.index[0], f.index[-1]], [13.5, 13.5], 'k--',
             [f.index[0], f.index[-1]], [27, 27], 'k--', )
    plt.sca(ax2)
    # plt.subplots_adjust(bottom=0.1, right=0.9, top=0.7)  # color bar
    cax = plt.axes([0.1, 0.1, 0.65, 0.01])
    plt.colorbar(cs, cax=cax, orientation='horizontal', label='log2(power)')

    ax3 = plt.axes([0.8, 0.15, 0.15, 0.4])
    ax3.plot(glbl_signif, period, 'b-')
    # print('glbl_signif: ', glbl_signif)
    # ax3.plot(var * fft_theor, period, '.-', color='#cccccc')
    ax3.plot(var * fft_power, 1./fftfreqs, '-', color='#cccccc', linewidth=1.)
    ax3.plot(var * glbl_power, period, 'b--', linewidth=1.5)
    # ax3.set_ylim(period.min(), period.max())
    ax3.set_ylim(min(period), max(period))

    ax3.set_title('c) wavelet spectrum')
    if item == 'Vs':
        xlabel_3 = r'Power (km/s)2'
    elif item == 'trough_min_gdlat':
        xlabel_3 = r'Power (degree)2'
    else:
        raise Exception('Input Error; input item here is: {}'.format(item))
    ax3.set_xlabel(xlabel_3)
    # ax3.set_xlim(0, glbl_power.max() + var)

    fig.savefig(fig_path + 'wavelet {} LT-{}'.format(item, lt[1]))
    plt.show()


def wavelet_f(f):
    N = len(f)
    dt = 1
    # t = np.array([(_ - f.index[0]).days for _ in f.index])
    dat = np.array(f - f.mean())
    std = dat.std()
    var = dat.var()
    dat_norm = dat / std                # 归一化？

    mother = wavelet.Morlet(6)
    s0 = 2
    dj = 1/3
    J = 4/dj
    alpha, _, _ = wavelet.ar1(dat)

    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J, mother)
    # iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std
    power = np.abs(wave) ** 2
    fft_power = np.abs(fft) ** 2
    period = 1 / freqs

    signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                             significance_level=0.95, wavelet=mother)
    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = power / sig95

    glbl_power = power.mean(axis=1)
    dof = N - scales
    glbl_signif, tmp = wavelet.significance(var, dt, scales, 1, alpha,
                                            significance_level=0.95, dof=dof, wavelet=mother)
    return f, period, power, sig95, glbl_power, glbl_signif, fft_power, var, fftfreqs


def plot_wavelet(y1, y2, lt):

    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_axes([0.06, 0.12, 0.72, 0.37])
    plt.sca(ax1)
    f, period, power, sig95, glbl_power, glbl_signif, fft_power, var, fftfreqs = wavelet_f(y1)
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    cs = ax1.contourf(f.index, period, np.log2(power), np.log2(levels), extend='both', cmap=plt.cm.jet)
    extent = [f.index.min(), f.index.max(), 0, max(period)]
    ax1.contour(f.index, period, sig95, [-99, 1], colors='k', linewidths=2, extent=extent)
    ax1.set_ylabel('Period (days)')
    ax1.set_title('a) Trough_min_gdlat wavelet-Power-Spectrum at LT:{}-{}'.format(lt[0], lt[1] + 1), loc='left')
    ax1.plot([f.index[0], f.index[-1]], [9, 9], 'k--',
             [f.index[0], f.index[-1]], [13.5, 13.5], 'k--',
             [f.index[0], f.index[-1]], [27, 27], 'k--', )
    cax = plt.axes([0.06, 0.07, 0.72, 0.01])
    plt.colorbar(cs, cax=cax, orientation='horizontal', label='log2(power)')

    ax2 = fig.add_axes([0.83, 0.12, 0.15, 0.37])
    plt.sca(ax2)
    ax2.plot(glbl_signif, period, 'b-')
    ax2.plot(var * fft_power, 1./fftfreqs, '-', color='#cccccc', linewidth=1.)
    ax2.plot(var * glbl_power, period, 'b--', linewidth=1.5)
    ax2.set_ylim(min(period), max(period))
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set_xlabel('Power')
    ax2.set_ylabel('Period')

    ax3 = fig.add_axes([0.06, 0.58, 0.72, 0.37])
    plt.sca(ax3)
    f, period, power, sig95, glbl_power, glbl_signif, fft_power, var, fftfreqs = wavelet_f(y2)
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    ax3.contourf(f.index, period, np.log2(power), np.log2(levels), extend='both', cmap=plt.cm.jet)
    extent = [f.index.min(), f.index.max(), 0, max(period)]
    ax3.contour(f.index, period, sig95, [-99, 1], colors='k', linewidths=2, extent=extent)
    ax3.set_ylabel('Period (days)')
    ax3.set_title('b) Solar-Wind-Speed wavelet-Power-Spectrum at LT:{}-{}'.format(lt[0], lt[1] + 1), loc='left')
    ax3.plot([f.index[0], f.index[-1]], [9, 9], 'k--',
             [f.index[0], f.index[-1]], [13.5, 13.5], 'k--',
             [f.index[0], f.index[-1]], [27, 27], 'k--', )

    ax4 = fig.add_axes([0.83, 0.58, 0.15, 0.37])
    plt.sca(ax2)
    ax4.plot(glbl_signif, period, 'b-')
    ax4.plot(var * fft_power, 1. / fftfreqs, '-', color='#cccccc', linewidth=1.)
    ax4.plot(var * glbl_power, period, 'b--', linewidth=1.5)
    ax4.set_ylim(min(period), max(period))
    plt.setp(ax4.get_xticklabels(), visible=False)
    ax4.set_xlabel('Power')
    ax4.set_ylabel('Period')

    plt.gcf()
    fig.tight_layout()  # 调整整体空白
    plt.subplots_adjust(hspace=0)  # 调整子图间距
    # plt.show()
    # plt.suptitle('Wavelet Power Spectrum at LT:{0:2d}-{0:2d}'.format(lt[0], lt[1] + 1))
    fig.savefig(fig_path + 'wavelet power spectrum at LT-{}'.format(lt[1]))
    plt.close()


def wavelet_xwt(gdlat_mit, V_solar_wind, lt):
    if len(gdlat_mit) != len(V_solar_wind):
        raise Exception("data error")
    dt = 1
    t = np.array([(_ - gdlat_mit.index[0]).days for _ in gdlat_mit.index])
    N = len(t)
    f1 = np.array(gdlat_mit - gdlat_mit.mean())
    f2 = np.array(V_solar_wind - V_solar_wind.mean())

    s0, dj = 2, 1/3
    J = 4/dj

    xwt, coi, freqs, signif = wavelet.xwt(y1=f1, y2=f2, dt=dt, dj=dj, s0=s0, J=J)
    # 默认morlet
    period = 1/freqs
    power = np.abs(xwt) ** 2

    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = power / sig95

    plt.close('all')
    plt.ioff()
    fig = plt.figure(figsize=(12, 4.8), dpi=72)
    plt.gcf()

    ax1 = plt.axes([0.06, 0.11, 0.84, 0.8])
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    cs = ax1.contourf(gdlat_mit.index, period, np.log2(power), np.log2(levels), extend='both', cmap=plt.cm.jet)
    extent = [gdlat_mit.index.min(), gdlat_mit.index.max(), 0, max(period)]
    ax1.contour(gdlat_mit.index, period, sig95, [-99, 1], colors='k', linewidths=2, extent=extent)
    ax1.plot([gdlat_mit.index.min(), gdlat_mit.index.max()], [9, 9], 'k--',
             [gdlat_mit.index.min(), gdlat_mit.index.max()], [13.5, 13.5], 'k--',
             [gdlat_mit.index.min(), gdlat_mit.index.max()], [27, 27], 'k--')
    ax1.set_title('cross wavelet transform of MIT & Vs at LT: {}-{}'.format(lt[0], lt[1] + 1))
    ax1.set_ylabel('Period (days)')
    ax1.set_xlabel('date')
    cax = plt.axes([0.92, 0.11, 0.01, 0.8])
    plt.colorbar(cs, cax=cax, orientation='vertical', label='log2(power)')
    # plt.show()
    fig.savefig(fig_path + 'trough-vs_xwt LT-{}'.format(lt[1]))
    plt.close()


if __name__ == '__main__':
    data_path = "C:\\tmp\\data.csv"
    # fig_path = 'C:\\DATA\\GPS_MIT\\millstone\\summary graph\\scatter plot\\lat-solar\\'
    fig_path = "C:\\tmp\\figure\\solar-wind\\wavelet\\"
    glon = -90
    localtime_list = [[19, 20], [21, 22], [23, 0], [1, 2], [3, 4]]
    data_all_lt = pd.read_csv(data_path, parse_dates=['date'], encoding='gb2312', index_col=['date'])
    # data = pd.read_csv(data_path, parse_dates=['date'], index_col='date', date_parser=dateparse)

    date_list = [['2015-6-17', '2015-7-25'], ]
    matplotlib.rcParams.update({'font.size': 12})  # 修改所有label和tick的大小

    for localtime in localtime_list:
        data = data_all_lt.query('glon == {} & (lt == {} | lt == {})'.format(glon, localtime[0], localtime[1]))
        # print(data['trough_min_lat'])
        print(len(data))
        mit = gen_mean_value_at_2lt(data[['lt', 'trough_min_lat']], localtime, 'trough_min_lat')  # 两地方时求均值
        Vs = gen_mean_value_at_2lt(data[['lt', 'Solar_Wind_Speed']], localtime, 'Solar_Wind_Speed')
        # print(mit.index[:5])
        if localtime[0] == 19 or localtime[0] == 21:
            print('localtime: {}-{}-------------------------------------'.format(
                localtime[0], localtime[1] + 1))
            print(data['2017-01-15': '2017-02-14'][['lt', 'trough_min_lat', 'Kp', 'Kp9']])
            # print(mit['2016-12-01': '2017-02-28'])
        mit_new = interpolate_mit(mit)  # 对mit做三次b样条曲线拟合，插值
        # print(mit)
        plot_TroughAndVs(mit, mit_new, Vs, localtime)                   # 随时间变化

        # wavelet_analysis(mit_new, 'trough_min_gdlat', localtime)        # 单画中纬槽小波
        # wavelet_analysis(Vs, 'Vs', localtime)                           # 单画太阳风速度小波
        plot_wavelet(mit_new, Vs, localtime)                            # 一幅图，两个小波
        wavelet_xwt(mit_new, Vs, localtime)                             # xwt
