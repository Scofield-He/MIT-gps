#! python3
# -*- coding: utf-8 -*-
"""
read omni2_years14-17.dat to get BZ and E data of Solar Wind.
the date is consistent with localtime instead of ut
"""

import os
# import gc
# import time
import datetime
import numpy as np
import pandas as pd
from scipy import interpolate
# from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from astropy.stats import LombScargle
import matplotlib


matplotlib.rcParams.update({'font.size': 12})        # 修改所有label和tick的大小


def bigtable_SolarWind(txt_path, csv_path, glon):
    """
    get .csv file which save Solar Wind data
    :param txt_path: origin .txt datafile
    :param csv_path: the path that .csv file saved
    :param glon: for calculating lt for glon
    :return: none, get saved .csv
    """
    col_widths = [4, 4, 3, 5, 3, 3, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 6, 6, 6, 6, 6, 6,
                  9, 6, 6, 6, 6, 6, 7, 7, 6, 3, 4, 6, 5, 10, 9, 9, 9, 9, 9, 3, 4, 6, 6, 6, 6, 5]  # 定宽数据宽度
    col_names = list(range(1, 56))                    # 自定义列名

    table = pd.read_fwf(txt_path, widths=col_widths, header=None, names=col_names)
    table = table[[1, 2, 3, 10, 17, 23, 24, 25, 36, 39, 42]]
    print(table.columns)
    table.columns = ['year', 'doy', 'Hour', 'B', 'Bz GSM', 'Proton temp', 'Proton Dens',
                     'Plasma flow speed', 'E', 'Kp', 'AE']
    print(table.columns)
    print(table.head(3))
    hour = np.array(table['Hour'])
    lt_glon = (hour + glon // 15) % 24                       # 计算地方时
    doy_bias = (hour + glon // 15) // 24
    year = np.array(table['year'])
    doy = np.array(table['doy'])
    date = np.array(list(map(lambda y, d, b: datetime.date(y, 1, 1) + datetime.timedelta(int(d) - 1 + int(b)),
                             year, doy, doy_bias)))          # 计算date，与地方时保持一致；
    # print(date[:5])
    table.insert(0, 'date', date)
    table.insert(4, 'lt'.format(fixed_glon), lt_glon)

    table['date'] = pd.to_datetime(table['date'])
    table = table.set_index('date')
    table = table['2014-9-1': '2017-8-31']
    print(table.head())
    table.to_csv(csv_path, index=True)


def compare_sw(old_path, new_path, date1, date2, lt):
    df_old = pd.read_csv(old_path, encoding='gb2312').query('lt == {}'.format(lt))
    df_old['date'] = pd.to_datetime(df_old['date'])
    df_old = df_old.set_index('date')[date1: date2]
    print(df_old.head())

    df = pd.read_csv(new_path).query('lt == {}'.format(lt))
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')[date1: date2]
    print(df.head())


if __name__ == '__main__':
    fixed_glon = -90
    SW_txt_path = 'C:\\DATA\\omni\\omni2_years14-17.dat'
    SW_bigtable_path = 'C:\\DATA\\omni\\SW_tmp.csv'
    bigtable_SolarWind(SW_txt_path, SW_bigtable_path, fixed_glon)
    data_path = 'C:\\tmp\\data.csv'
    t1, t2 = '2014-9-1', '2017-8-31'
    localtime = 22
    compare_sw(data_path, SW_bigtable_path, t1, t2, localtime)
