#! python3
# -*- coding: utf-8 -*-
import os
import os.path
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import h5py
import aacgmv2
from datetime import date

"""
def data_parameters(file):
    for item in file['Metadata']['Data Parameters']:
        print(item)
    test = file['Data']['Table Layout'][:10]
    print(test)

    return True
"""


def a2g(mlat, mlon, alt, date_time):
    """
    :param mlat: int, °
    :param mlon: int, °
    :param alt: int, km
    :param date_time: datetime.date, tuple like (2013, 1, 1)
    :return: list, glon and glat for fixed mlat
    """
    glat, glon = aacgmv2.convert(mlat, mlon, alt, date_time, a2g=True)
    # print('a2g done', 'len(glat) = {}'.format(len(glat)))
    return glon, glat


def map_tec(datafile, outpath):
    """
    :param datafile: h5py file, read by h5py.File
    :param outpath: str, path to output figure
    :return: bool
    """
    data = datafile['Data']['Table Layout']
    data_length = len(data)

    index, first, last, count = 0, 0, 0, 0
    while index < data_length:
        first = index
        minute = data[first]['min']
        while index < data_length and data[index]['min'] == minute:
            index += 1
        last = index - 1
        if first == 0:
            print('first = {}, last = {}'.format(first, last))

        gdlat = data['gdlat'][first:last]
        glon = data['glon'][first:last]
        tec = data['tec'][first:last]
        # print('max(tec) = {}, min(tec) = {}'.format(np.max(tec), np.min(tec)))

        plt.figure(figsize=(10, 6))  # 默认大小？
        fig = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='l')
        fig.drawcoastlines(linewidth=0.5)
        # draw lat/lon grid lines and lables, lables = [left, right, top, bottom]
        fig.drawmeridians(np.arange(-180., 181., 60.), labels=[False, False, False, True])
        fig.drawparallels(np.arange(-90., 91., 30.), labels=[True, False, False, False])

        x1, y1 = a2g(45, np.arange(-113, 246), 350, date(2013, 2, 20))
        x2, y2 = a2g(70, np.arange(-124, 236), 350, date(2013, 3, 10))
        x3, y3 = a2g(60, np.arange(-118, 241), 350, date(2010, 3, 20))
        fig.plot(x1, y1, latlon=False, color='b')
        fig.plot(x2, y2, latlon=False, color='b')
        fig.plot(x3, y3, latlon=False, color='r')

        x, y = (glon, gdlat)
        mask = tec < 1
        mtec = np.ma.array(tec, mask=mask)
        color = np.log10(mtec)
        # min_tec, max_tec = np.min(color), np.max(color)
        fig.scatter(x, y, c=color, latlon=False, vmin=0, vmax=2, marker='s', s=1, alpha=1, edgecolors='face',
                    cmap=plt.cm.jet)
        fig.colorbar(pad='5%').set_label('log10(TEC)/TECU')
        ymdhms = '{}-{:2}-{:2}-{:2}-{:2}-{:2}'.format(data[last]['year'], data[last]['month'], data[last]['day'],
                                                      data[last]['hour'], data[last]['min'], data[last]['sec'])
        plt.title("Geodetic median vertical TEC at " + ymdhms)

        output_path = outpath + '{}-{}\\'.format(data[last]['month'], data[last]['day'])
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        count += 1
        # figure_name = '{}-{}'.format(data[last]['hour'], data[last]['min'])
        figure_name = '%03d' % count
        plt.savefig(output_path + figure_name, fmt='jpg', dpi=150)
        plt.close()
    return True


file_path = 'C:\\DATA\\GPS_MIT\\2006\\'
dat, year = [], 6
for month in [3, 6, 9, 12]:
    for day in np.arange(21, 26):
        dat.append('gps{:02d}{:02d}{:02d}g'.format(year, month, day))

for item in os.listdir(file_path):
    i = item.split('.')
    if i[0] in dat and i[2] == 'hdf5':
        start = time.clock()
        f = h5py.File(file_path + item)
        fig_path = 'C:\\DATA\\GPS_MIT\\2006\\figure\\global_map\\'
        map_tec(f, fig_path)
        print(time.clock() - start)

print('Work Done')
