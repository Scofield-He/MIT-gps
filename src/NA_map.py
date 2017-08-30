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
    "names": ['UT', 'lt', 'tec', 'mlat', 'mlon', 'mlt'],
    "formats": ['<f8', '<f8', '<f8', '<f8', '<f8', '<f8']
})


def plot_map_tec(geographic_data, geomagnetic_data, start, end, ax):
    fig = Basemap(projection='stere', width=12000000, height=8000000,
                  lat_0=55, lon_0=-107, lat_ts=50, resolution='l')

    fig.drawcoastlines(linewidth=0.5)
    fig.drawmeridians(np.arange(-200., -20., 20.), labels=[0, 0, 0, 1])
    fig.drawparallels(np.arange(30, 81, 10), labels=[1, 0, 0, 0])

    mlat, mlon = geomagnetic_data["mlat"][start:end], geomagnetic_data["mlon"][start:end]
    tec = geographic_data["tec"][start:end]

    mask = tec < 1
    mask_tec = np.ma.array(tec, mask=mask)
    color = np.log10(mask_tec)

    date = datetime.date(geographic_data[0]["year"], geographic_data[0]["month"], geographic_data[0]["day"])
    # glat, glon = aacgmv2.convert(mlat, mlon, 0, date, a2g=True)
    glat, glon = geographic_data["gdlat"][start:end], geographic_data["glon"][start:end]
    x, y = fig(glon, glat)

    fig.scatter(x, y, c=color, latlon=False, vmin=0, vmax=1.5, marker='s', s=5, alpha=1, edgecolors='face',
                cmap=plt.cm.jet)
    fig.colorbar(pad='5%').set_label('log10(TEC)')

    y1, x1 = aacgmv2.convert(45, np.arange(-120, 50), 0, date, a2g=True)
    y2, x2 = aacgmv2.convert(70, np.arange(-120, 50), 0, date, a2g=True)
    y3, x3 = aacgmv2.convert(60, np.arange(-120, 50), 0, date, a2g=True)
    fig.plot(x1, y1, latlon=True, color='b', label='mlat = 45°')
    fig.plot(x2, y2, latlon=True, color='r', label='mlat = 70°')
    fig.plot(x3, y3, latlon=True, color='k', label='mlat = 60°')

    title = 'tec_map of NA at {} '.format(date, )
    plt.title(title)
    plt.text(0.8, 0.9, "MLT = {}".format(round(geomagnetic_data[end]["mlt"], 2)), transform=ax.transAxes, color="black")
    return True


def plot_profile_tec(rawData, mData, first, last, ax2):
    rawData += 1

    plt.text()
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

        plt.figure(figsize=(8, 10))
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)

        plt.sca(ax1)
        plot_profile_tec(rawData, mData, first, last, ax1)

        plt.sca(ax2)
        plot_map_tec(rawData, mData, first, last, ax2)

        count += 1
        print("figure " + "%03d" % count + "cost {}".format(time.time() - start_time))
        plt.legend()    # 显示图示
        plt.savefig(fig_path + '{:03d}'.format(count), fmt='jpg', dpi=80)
        plt.close()
    print("{} cost --> {}".format(fig_path, time.time() - start_time))
    return True


fi_path = "C:\\DATA\\GPS_MiT\\2011\\"
out_path = fi_path + "figure\\NA_map\\"
if not os.path.isdir(out_path):
    os.makedirs(out_path)
for fi in os.listdir(fi_path):
    if ".hdf5" in fi:
        print("{} begins: -- > ".format(fi))
        map_tec_north_american(fi_path, fi, out_path)
else:
    print("All work have been done!")
