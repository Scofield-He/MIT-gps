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
import matplotlib


matplotlib.rcParams.update({'font.size': 12})


def data_parameters(file):
    for parameter in file['Metadata']['Data Parameters']:
        print(parameter)
    test = file['Data']['Table Layout'][:10]
    print(test)
    return True


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
        if first == 0 or first:
            print('first = {}, last = {}'.format(first, last))

        hour = data[first]['hour']
        if hour < 7:
            print(hour)
            continue
        elif hour > 7:
            break
        elif minute < 30 or minute > 35:
            continue
        else:
            print(hour, minute)

        gdlat = data['gdlat'][first:last]
        glon = data['glon'][first:last]
        tec = data['tec'][first:last]
        # print('max(tec) = {}, min(tec) = {}'.format(np.max(tec), np.min(tec)))

        plt.figure(figsize=(10, 5))
        fig = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='l')
        fig.drawcoastlines(linewidth=0.3)
        # draw lat/lon grid lines and lables, lables = [left, right, top, bottom]
        fig.drawmeridians(np.arange(-180., 181., 60.), labels=[False, False, False, True])
        fig.drawparallels(np.arange(-90., 91., 20.), labels=[True, False, False, False])

        x1, y1 = a2g(45, np.arange(-113, 246), 350, date(2013, 2, 20))
        x2, y2 = a2g(70, np.arange(-124, 236), 350, date(2013, 3, 10))
        x3, y3 = a2g(60, np.arange(-118, 241), 350, date(2010, 3, 20))
        fig.plot(x1, y1, latlon=False, color='b')
        fig.plot(x2, y2, latlon=False, color='b')
        fig.plot(x3, y3, latlon=False, color='r')
        
        for _ in [-120, -90, 0, 30]:
            gdlat1 = np.array([_ for _ in range(30, 80)])
            glon1, glon2 = np.array([_ - 5] * len(gdlat1)), np.array([_ + 5] * len(gdlat1))
            fig.plot(glon1, gdlat1, 'k')
            fig.plot(glon2, gdlat1, 'k')
            glons = np.array([_ for _ in range(_ - 5, _ + 6)])
            fig.plot(glons, np.array([30] * len(glons)), 'k')
            fig.plot(glons, np.array([80] * len(glons)), 'k')

        x, y = (glon, gdlat)
        mask = tec < 1
        mtec = np.ma.array(tec, mask=mask)
        color = np.log10(mtec)
        # color = mtec
        # min_tec, max_tec = np.min(color), np.max(color)
        fig.scatter(x, y, c=color, latlon=False, vmin=0, vmax=2, marker='s', s=0.65, alpha=1, edgecolors='face',
                    cmap=plt.cm.jet)
        fig.colorbar(pad='5%').set_label('log10(TEC)/TECU')
        # fig.colorbar(pad='5%').set_label('TEC(TECU)')
        plt.ylabel('Geographical Latitude\n\n\n')
        plt.xlabel('\n Geographical Longitude')
        plt.subplots_adjust(top=0.94, left=0.09, right=0.92, bottom=0.06)

        ymdhms = '{}-{:2}-{:2}-{:2}-{:2}-{:2}'.format(data[last]['year'], data[last]['month'], data[last]['day'],
                                                      data[last]['hour'], data[last]['min'], data[last]['sec'])
        plt.title("Geodetic median vertical TEC at " + ymdhms).set_size(15)

        # output_path = outpath + 'to_lj {}-{}\\'.format(data[last]['month'], data[last]['day'])
        # output_path = outpath + '{}-{}_new\\'.format(data[last]['month'], data[last]['day'])
        output_path = outpath + '{}-{}_new\\'.format(data[last]['month'], data[last]['day'])
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        count += 1
        figure_name = 'figure1_{}-{}'.format(data[last]['hour'], data[last]['min'])
        # figure_name = '%03d' % count
        plt.savefig(output_path + figure_name + '.jpg', fmt='jpg', dpi=150)
        plt.savefig(output_path + figure_name + '.eps', fmt='eps', dpi=150)
        # plt.show()
        # plt.close()
    return True


file_path = 'C:\\DATA\\GPS_MIT\\millstone\\2016\\data\\'
year = 16

for item in os.listdir(file_path):
    i = item.split('.')
    day, form = int(i[0][-3: -1]), i[2]
    if day == 18 and form == 'hdf5':
        start = time.clock()
        f = h5py.File(file_path + item)
        # fig_path = 'C:\\DATA\\GPS_MIT\\millstone\\2016\\figure\\global_map\\'
        fig_path = 'C:\\tmp\\figure\\eps\\maps\\'
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        map_tec(f, fig_path)
        print(time.clock() - start)

print('Work Done')
