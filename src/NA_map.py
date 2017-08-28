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


def map_tec_north_american(file_path, file, outpath):
    start_time = time.time()

    f = h5py.File(file_path + file)
    d = f["Data"]["Table Layout"]
    data = f["Data"]["Post NA Data"]
    data_length = len(data)

    fig_path = os.path.join(outpath, "{}-{}\\".format(d[0]["month"], d[0]["day"]))
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    first, last, count, i = 0, 0, 0, 0
    while i < data_length:
        first = i
        h = data[first]["UT"]
        while i < data_length and data[i]["UT"] == h:
            i += 1
        last = i - 1
        if first == 0:
            print("{} -->   first = {}, last = {}".format(file_path + file, first, last))

        plt.figure(figsize=(10, 5))
        ax = plt.subplot(111)
        fig = Basemap(projection='stere', width=12000000, height=8000000,
                      lat_0=55, lon_0=-107, lat_ts=50, resolution='l')

        fig.drawcoastlines(linewidth=0.5)
        fig.drawmeridians(np.arange(-200., -20., 20.), labels=[0, 0, 0, 1])
        fig.drawparallels(np.arange(30, 81, 10), labels=[1, 0, 0, 0])

        mlat, mlon, tec = data["mlat"][first:last], data["mlon"][first:last], data["tec"][first:last]

        mask = tec < 1
        mask_tec = np.ma.array(tec, mask=mask)
        color = np.log10(mask_tec)

        date = datetime.date(d[0]["year"], d[0]["month"], d[0]["day"])
        glat, glon = aacgmv2.convert(mlat, mlon, 0, date, a2g=True)
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

        title = 'tec_map of NA at {} {}'.format(date, h)
        plt.title(title)
        plt.text(0.8, 0.9, "MLT = {}".format(round(data[last]["mlt"], 2)), transform=ax.transAxes, color="black")

        count += 1
        # print("%03d" % count, time.time() - start_time)
        plt.legend()

        plt.savefig(fig_path + '{:03d}'.format(count), fmt='jpg', dpi=80)
        plt.close()
    print("{} figures cost time --> {}".format(fig_path, time.time() - start_time))
    return True


fi_path = "C:\\DATA\\GPS_MIT\\2011\\"
out_path = fi_path + "figure\\NA_map\\"
if not os.path.isdir(out_path):
    os.makedirs(out_path)
for fi in os.listdir(fi_path):
    if ".hdf5" in fi:
        print("{} begins: -- > ".format(fi))
        map_tec_north_american(fi_path, fi, out_path)
print("Work Done")
