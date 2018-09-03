#! python3
# -*- coding: utf-8 -*-

# import os
# import gc
# import time
# import datetime
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import pycwt as wavelet
# from matplotlib.ticker import MultipleLocator

# from pycwt.helpers import find


def dateparse(dates):
    return pd.datetime.strptime(dates, '%Y-%m-%d')


def interpolate_mit(ds):
    x_new = np.array([(_ - ds.index[0]).days for _ in ds.index])
    # print(x_new)
    index = ds.index

    ds = ds.dropna()
    x = np.array([(_ - ds.index[0]).days for _ in ds.index])
    x_new = x_new[x_new <= x.max()]
    # print(x_new)
    # print(x)
    y = np.array(ds)

    tck = interpolate.splrep(x, y, s=5)                     # k=3 for default
    y_bspline = interpolate.splev(x_new, tck, der=0)         # der=0, 对0阶导，即对自身插值
    fig = plt.figure(figsize=(12, 4))
    plt.plot(x, y, 'o', ms=3)
    plt.plot(x_new, y_bspline, 'kx-', ms=2, alpha=0.6)
    plt.legend(['data', 'b_spline'], loc='best')
    fig.savefig(figpath + 'mit interpolate')
    plt.show()
    ret = pd.Series(data=y_bspline, index=index)
    return ret


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


def wavelet_analysis(f, item):
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
    iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std

    power = np.abs(wave) ** 2
    fft_power = np.abs(fft) ** 2
    period = 1 / freqs
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

    ax2 = plt.axes([0.1, 0.15, 0.65, 0.4], sharex=ax1)
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
    ax3.set_ylim(period.min(), period.max())

    ax3.set_title('c) wavelet spectrum')
    if item == 'Vs':
        xlabel_3 = r'Power (km/s)2'
    elif item == 'mit gdlat':
        xlabel_3 = r'Power (degree)2'
    else:
        raise Exception('Input Error; input item here is: {}'.format(item))
    ax3.set_xlabel(xlabel_3)
    # ax3.set_xlim(0, glbl_power.max() + var)

    fig.savefig(figpath + item)
    plt.show()


def wavelet_xwt(gdlat_mit, V_solar_wind):
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

    ax1 = plt.axes([0.08, 0.18, 0.84, 0.7])
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    cs = ax1.contourf(gdlat_mit.index, period, np.log2(power), np.log2(levels), extend='both', cmap=plt.cm.jet)
    extent = [gdlat_mit.index.min(), gdlat_mit.index.max(), 0, max(period)]
    ax1.contour(gdlat_mit.index, period, sig95, [-99, 1], colors='k', linewidths=2, extent=extent)
    ax1.plot([gdlat_mit.index.min(), gdlat_mit.index.max()], [9, 9], 'k--',
             [gdlat_mit.index.min(), gdlat_mit.index.max()], [13.5, 13.5], 'k--',
             [gdlat_mit.index.min(), gdlat_mit.index.max()], [27, 27], 'k--')
    ax1.set_title('a) cross wavelet transform of MIT & Vs')
    ax1.set_ylabel('Period (days)')
    cax = plt.axes([0.08, 0.1, 0.84, 0.01])
    plt.colorbar(cs, cax=cax, orientation='horizontal')
    fig.savefig(figpath + 'gdlat-vs_xwt')
    plt.show()


if __name__ == '__main__':
    data_path = "C:\\tmp\\data.csv"
    figpath = 'C:\\DATA\\GPS_MIT\\millstone\\summary graph\\scatter plot\\lat-solar\\'
    glon, lt_list = -90, [23, 0]
    data = pd.read_csv(data_path, parse_dates=['date'], index_col='date', date_parser=dateparse)
    data = data.query('glon == {} & (lt == {} | lt == {})'.format(glon, lt_list[0], lt_list[1])).sort_index()
    data = data['2014-9-1': '2017-9-1']
    mit = gen_mean_value_at_2lt(data[['lt', 'MIT_gdlat']], lt_list, 'MIT_gdlat')           # 两地方时槽极小位置求均值
    Vs = gen_mean_value_at_2lt(data[['lt', 'Solar_Wind_Speed']], lt_list, 'Solar_Wind_Speed')
    # print(mit)
    mit = interpolate_mit(mit)                                                # 插值，补全
    # print(mit)

    wavelet_analysis(mit, 'mit gdlat')
    wavelet_analysis(Vs, 'Vs')
    wavelet_xwt(mit, Vs)
