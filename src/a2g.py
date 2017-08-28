#! python3
#  -*- coding: utf-8 -*-
import time
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from datetime import date
import aacgmv2


def a2g(mlat, mlon, alt, dat):
    """
    :param mlat: int, °
    :param mlon: int, °
    :param alt: int, km
    :param dat: datetime.date, tuple like (2013, 1, 1)
    :return: list, glon and glat for fixed mlat
    """
    glat, glon = aacgmv2.convert(mlat, mlon, alt, dat, a2g=True)
    print('a2g done', 'len(glat) = {}'.format(len(glat)))
    return glon, glat


def base_map():
    start_time = time.clock()
    plt.figure(figsize=(20, 10))
    figure = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='l')
    figure.drawcoastlines(linewidth=0.5)
    figure.drawmeridians(np.arange(-180., 181., 60.), labels=[False, False, False, True])
    figure.drawparallels(np.arange(-90., 91., 30.), labels=[True, False, False, False])

    x1, y1 = a2g(45, np.arange(-113, 246), 350, date(2013, 2, 20))
    x2, y2 = a2g(70, np.arange(-124, 236), 350, date(2013, 3, 10))
    x3, y3 = a2g(60, np.arange(-118, 241), 350, date(2010, 3, 20))
    figure.plot(x1, y1, latlon=False, color='b')
    figure.plot(x2, y2, latlon=False, color='b')
    figure.plot(x3, y3, latlon=False, color='r')

    print('cost time --> {}'.format(time.clock() - start_time))

    plt.show()


base_map()
