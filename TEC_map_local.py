#! python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import h5py
# import aacgmv2
import datetime


def gen_local_data(global_data, h, m):
    return global_data.query("hour == {} & minute == {} & {} <= gdlat <= {} & {} <= glon <= {}"
                             "& dtec < tec / 3".format(h, m, glat1, glat2, glon1, glon2))


def plot_local_map(local_data, t):
    fig = Basemap(projection='cyl', llcrnrlat=glat1, urcrnrlat=glat2, llcrnrlon=glon1, urcrnrlon=glon2, resolution='l')
    fig.drawcoastlines(linewidth=0.3)
    fig.drawmeridians(np.arange(glon1, glon2, 5.), labels=[0, 0, 0, 1])
    fig.drawparallels(np.arange(glat1, glat2, 5.), labels=[1, 0, 0, 0])

    x, y, color = np.array(local_data["glon"]), np.array(local_data["gdlat"]), np.array(local_data["tec"])
    print(list(color))
    print(min(color), max(color))
    # color = np.log10(color)
    # fig.scatter(x, y, c=color, latlon=True, vmin=3, vmax=18,
    #             marker='s', s=35, alpha=1, edgecolors='face', cmap=plt.cm.jet)
    print(x, y, color)
    print(len(x), len(y), len(color))
    fig.contourf(x, y, color, latlon=True, tri=True, cmap=plt.cm.jet)
    fig.colorbar(pad='5%').set_label('TEC')
    plt.title(t)
    print(t)
    # plt.savefig(outpath + '{} {} {}'.format(t.date(), t.hour, t.minute))
    plt.show()
    plt.close()


if __name__ == '__main__':
    filepath = 'C:\\Users\\tmp\\gps110218g.002.hdf5'
    outpath = 'C:\\tmp\\zhao\\'
    glat1, glat2 = 50, 80
    glon1, glon2 = -110, -60
    hour, minute1, minute2 = 11, 2, 57

    file = h5py.File(filepath)
    data = file["Data"]["Table Layout"]
    hours, minutes = data['hour'], data['min']
    gdlat, glon, tec, dtec = data['gdlat'], data['glon'], data['tec'], data['dtec']

    df = pd.DataFrame({'hour': hour, 'minute': minutes, 'gdlat': gdlat, 'glon': glon, 'tec': tec, 'dtec': dtec})
    for minute in range(minute1, minute2, 5):
        local_df = gen_local_data(df, hour, minute)
        time = datetime.datetime(2011, 2, 18, hour, minute, 30)
        plot_local_map(local_df, time)
        break
