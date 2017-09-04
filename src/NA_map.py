#! python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import h5py
import aacgmv2
import datetime

data_type = np.dtype({
    "names": ['mlat', 'mlon', 'mlt'],
    "formats": ['<f4', '<f4', '<f4']
})


def plot_map_tec(geographic_data, geomagnetic_data, start, end, ax):
    plt.sca(ax)
    fig = Basemap(projection='stere', width=12000000, height=8000000,
                  lat_0=55, lon_0=-107, lat_ts=50, resolution='l')

    fig.drawcoastlines(linewidth=0.5)
    fig.drawmeridians(np.arange(-200., -20., 20.), labels=[0, 0, 0, 1])
    fig.drawparallels(np.arange(30, 81, 10), labels=[1, 0, 0, 0])

    glat1, glat2 = 30, 75
    glon1, glon2 = -180, -60
    glat_NA, glon_NA, tec_NA, mlt_NA = [], [], [], 0
    index = start
    # print(geographic_data["gdlat"][start:end])
    print("start-index == {}".format(start))
    time1 = time.time()
    tec_global = geographic_data["tec"]                # 避免重复调用参数
    mlt_array = geomagnetic_data["mlt"]
    for (i, j) in zip(geographic_data["gdlat"][start:end], geographic_data["glon"][start:end]):
        if (glat1 < i < glat2) and (glon1 < j < glon2):
            glat_NA.append(i)
            glon_NA.append(j)
            tec_NA.append(tec_global[index])
            if i == 50.0 and -130 < j < -120:
                mlt_NA = mlt_array[index].round(2)
                # print(mlt_NA)
        index += 1
    else:
        print("check all data end--index == {}".format(end))
        print("len of data is --> {} {}".format(len(glat_NA), len(tec_NA)))
        print(time.time() - time1)

    tec = np.array(tec_NA)
    mask = tec < 1
    mask_tec = np.ma.array(tec, mask=mask)
    color = np.log10(mask_tec)

    date = datetime.date(geographic_data[start]["year"], geographic_data[start]["month"], geographic_data[start]["day"])
    date_time = datetime.datetime(geographic_data[start]["year"], geographic_data[start]["month"],
                                  geographic_data[start]["day"], geographic_data[start]["hour"],
                                  geographic_data[start]["min"], geographic_data[start]["sec"])

    fig.scatter(glon_NA, glat_NA, c=color, latlon=True, vmin=0.0, vmax=1.7, marker='s', s=5, alpha=1, edgecolors='face',
                cmap=plt.cm.jet)
    fig.colorbar(pad='5%').set_label('log10(TEC)')

    y1, x1 = aacgmv2.convert(45, np.arange(-120, 50), 0, date, a2g=True)
    y2, x2 = aacgmv2.convert(70, np.arange(-120, 50), 0, date, a2g=True)
    y3, x3 = aacgmv2.convert(60, np.arange(-120, 50), 0, date, a2g=True)
    fig.plot(x1, y1, latlon=True, color='b', label='mlat = 45°')
    fig.plot(x2, y2, latlon=True, color='b', label='mlat = 70°')
    fig.plot(x3, y3, latlon=True, color='r', label='mlat = 60°')

    lat1, lon1 = aacgmv2.convert(np.arange(40, 75), -75, 0, date, a2g=True)
    lat2, lon2 = aacgmv2.convert(np.arange(40, 75), -60, 0, date, a2g=True)
    fig.plot(lon1, lat1, latlon=True, color='k', label='mlon = -65°')
    fig.plot(lon2, lat2, latlon=True, color='k', label='mlon = -60°')

    ax.legend(loc="best")

    title = 'tec_map of NA at {} '.format(date_time)
    plt.title(title)
    plt.text(0.8, 0.9, "MLT = %.2f" % mlt_NA, transform=ax.transAxes, color="black")
    return True


def plot_profile_tec(geographic_data, geomagnetic_data, start, end, ax):
    mlat1, mlat2 = 40, 75
    mlon1, mlon2 = -75, -60
    mlat_NA, mlon_NA, tec_NA, mlt = [], [], [], 0
    tec_global = geographic_data["tec"]
    index = start
    # 过滤得到NA内给定区域 [[mlat1, mlat2],[mlon1, mlon2]]的mlon_NA，mlat_NA,tec_NA
    for (i, j) in zip(geomagnetic_data["mlon"][start: end], geomagnetic_data["mlat"][start: end]):
        if (mlon1 <= i <= mlon2) and (mlat1 <= j <= mlat2):
            mlon_NA.append(i)
            mlat_NA.append(j)
            tec_NA.append(tec_global[index])
        index += 1
    else:
        print("geo_magnetic data filtered done")

    # 计算给定磁纬范围[mlat1, mlat2]中相应磁纬对应的，[mlon1, mlon2]经度范围的tec平均值
    tec = []
    for i in range(mlat2 - mlat1 + 1):
        tec.append([])
    for (m, n) in zip(mlat_NA, tec_NA):
        tec[int(m) - mlat1].append(n)
    print(len(tec) == mlat2 - mlat1 + 1, len(tec))

    plt.sca(ax)
    x = range(mlat1, mlat2+1)
    y = []
    for i in range(len(tec)):
        y.append(np.mean(tec[i]))
    plt.plot(x, y, 'k')
    plt.ylim(0, 20)
    plt.xlabel("mlat")
    plt.ylabel("mean_tec")
    # ax.legend()
    return True


def map_tec_north_american(file_path, file, outpath):
    start_time = time.time()

    f = h5py.File(file_path + file)
    rawData = f["Data"]["Table Layout"]
    mData = f["Data"]["mData"]

    fig_path = os.path.join(outpath, "{}-{}\\".format(rawData[0]["month"], rawData[0]["day"]))
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    data_length = len(mData)

    first, last, count, index = 0, 0, 0, 0
    while index < data_length:
        first = index
        minute = rawData[first]["min"]
        while index < data_length and rawData[index]["min"] == minute:
            index += 1
        else:
            last = index - 1

        if first == 0:
            print("{} -->   first = {}, last = {}".format(file_path + file, first, last))

        figure = plt.figure(1)
        ax1 = figure.add_subplot(211)
        ax2 = figure.add_subplot(212)

        plt.sca(ax1)
        plot_profile_tec(rawData, mData, first, last, ax1)

        plt.sca(ax2)
        plot_map_tec(rawData, mData, first, last, ax2)

        count += 1
        print("figure " + "%03d" % count + "cost {}".format(time.time() - start_time))
        plt.figure(1)
        figure.set_size_inches(8, 10)
        figure.savefig(fig_path + '{:03d}'.format(count), fmt='jpg', dpi=80)
        plt.close()
    print("{} cost --> {}".format(fig_path, time.time() - start_time))
    return True


fi_path = "C:\\DATA\\GPS_MIT\\2013\\data\\"
out_path = "C:\\DATA\\GPS_MIT\\2013\\figure\\NA-map\\"
if not os.path.isdir(out_path):
    os.makedirs(out_path)
for fi in os.listdir(fi_path):
    if ".hdf5" in fi:
        print("{} begins: -- > ".format(fi))
        map_tec_north_american(fi_path, fi, out_path)
else:
    print("All work have been done!")
