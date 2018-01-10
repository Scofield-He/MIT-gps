#! python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import h5py
import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,  FuncFormatter


def lt_formatter(x1, _):
    lon_local = (lon_left + lon_right) / 2
    m = round((x1 + lon_local / 15) % 24, 2)
    x1 = "{:2d}:00".format(int(x1))
    return "{}\n{}".format(x1, m)


def read_C9_index_of_one_day(dat):
    """
    ----------
    get C9 index of one day given.
    
    Parameters
    ----------
    dat: datetime.date(), input date.
    
    Returns
    ----------
    C9 index value
    """
    file_path = "C:\\DATA\\index\\Kp-ap-Flux_01-17.dat"
    with open(file_path, 'rb') as fp:
        index_value = -1
        for line in fp:
            line = line.strip().decode('utf-8')
            if line[:2] == str(dat.year)[2:] and int(line[2:4]) == dat.month and int(line[4:6]) == dat.day:
                index_value = int(line[61])
                break
        if index_value == -1:
            raise ValueError("not find index value of the day: {}".format(dat))
        else:
            return index_value


def read_t(t_string):
    t = [int(_) for _ in t_string.split(',')]
    if len(t) == 1:
        return t[0], 0, 0
    elif len(t) == 2:
        return t[0], t[1], 0
    elif len(t) == 3:
        return t[0], t[1], t[2]
    else:
        print("input t error")
        return 0, 0, 0


def gen_tec_profile_of_current_ut(glat, glon, tec, cur_datalist, glat1, glat2, glon1, glon2):
    """
    Usage
    ----------
    generate and record data of one tec profile at one given ut;

    Parameters
    ----------
    glat, glon, tec: list, all global data in at one moment;
    cur_datalist: list, [[], []], record gdlat and correspond mean_tec values
    glat1, glat2: int, lat range for given area;
    glon1, glon2: int, lon range for given area;

    Return
    ----------
    none. append lat value and its correspond mean tec value to data[ut];

    Notes
    ----------
    """
    tec_value = {}
    for idx in range(glat1, glat2 + 1):
        tec_value[idx] = []
    for (j, k, l) in zip(glon, glat, tec):
        if glon1 < j < glon2 and glat1 < k < glat2:
            tec_value[int(k)].append(l)

    latitude, mean_tec = [], []
    for idx in range(glat1, glat2 + 1):
        if tec_value[idx]:
            latitude.append(idx)
            mean_tec.append(np.mean(tec_value[idx]))
    cur_datalist.append(latitude)
    cur_datalist.append(mean_tec)


def gen_data_of_current_day(path, file):
    table_layout = h5py.File(path + file)["Data"]["Table Layout"]
    hours, minutes = table_layout["hour"], table_layout["min"]
    lat, lon = table_layout["gdlat"], table_layout["glon"]
    tec_data, dtec = table_layout["tec"], table_layout["dtec"]
    data_length = len(table_layout)

    i, h0, m0 = 0, hours[0], minutes[0]
    data = {}
    while i < data_length:
        index_begin, index_end, ut = -1, -1, 0
        while i < data_length and minutes[i] == m0:
            if index_begin < 0:
                index_begin = i
                ut = datetime.time(h0, m0)
                data[ut] = []
            i += 1
        else:
            index_end = i - 1
            cur_tec, cur_glat, cur_glon = tec_data[index_begin: i], lat[index_begin: i], lon[index_begin: i]
            gen_tec_profile_of_current_ut(cur_glat, cur_glon, cur_tec, data[ut], lat_low, lat_high, lon_left, lon_right)

            if i < data_length:
                h0, m0 = hours[i], minutes[i]
            else:
                print("last index: {}, data length: {}".format(index_end, data_length))
    return data


def plot_heatmap_of_current_day(data, C9, figure_name, fo_path):
    x, y, color = [], [], []  # ut、纬度、mean_tec
    for key in data.keys():
        if data[key]:
            for _ in range(len(data[key][0])):
                x.append(key)                   # ut
                y.append(data[key][0][_])       # lat
                color.append(data[key][1][_])   # mean_tec
    x = [round(_.hour + _.minute / 60, 2) for _ in x]
    print('length of lt: {}; lat: {};  mean_tec: {}'.format(len(x), len(y), len(color)))

    fig = plt.figure(figsize=(14, 3))
    ax1 = fig.add_subplot(111)
    plt.gca()
    # plt.scatter(x, y, c=color, vmin=0, vmax=15, marker='s', s=4, alpha=1, cmap=plt.cm.jet)
    plt.scatter(x, y, c=color, vmin=0, vmax=15, marker='s', s=4, alpha=1, cmap=plt.cm.jet)
    ax1.xaxis.set_major_locator(MultipleLocator(2))
    ax1.xaxis.set_major_formatter(FuncFormatter(lt_formatter))
    ax1.xaxis.set_minor_locator(MultipleLocator(1))

    plt.xticks(fontsize=8, color='k', rotation=0)
    plt.yticks(fontsize=8, color='k')

    ax1.set_xlabel("UT/LT", fontsize=8, rotation=0)
    ax1.set_ylabel("gdlat", fontsize=8)
    plt.title(figure_name + '  NA  {}'.format(C9))

    plt.subplots_adjust(bottom=0.18, left=0.05, right=0.92, top=0.85)
    cax = plt.axes([0.94, 0.18, 0.02, 0.67])
    plt.colorbar(cax=cax).set_label('TEC/TECU')

    fig.savefig(fo_path + figure_name, dpi=500)
    plt.close()
    return True


def main(data_site, t_str, lon1, lon2):
    year, month, day = read_t(t_str)

    data_path = "C:\\DATA\\GPS_MIT\\{}\\{}\\data\\".format(data_site, year)
    fo_path = "C:\\DATA\\GPS_MIT\\{}\\{}\\figure\\daily_lt_changes_{}_{}\\".format(data_site, year, lon1, lon2)
    if not os.path.isdir(fo_path):
        os.mkdir(fo_path)

    for filename in os.listdir(data_path):
        if (not month or filename[5:7] == str(month)) and (not day or filename[7:9] == str(day)) \
                                                       and ".hdf5" in filename and filename[3:5] == str(year)[-2:]:
            date = datetime.date(year, int(filename[5:7]), int(filename[7:9]))
            doy = date.timetuple().tm_yday

            figure_name = '{} {}'.format(date, doy)
            if os.path.exists(fo_path + figure_name + '.png'):
                print("picture {} has already been drawn".format(figure_name))
                continue

            print("{} begins: -------------------------------------".format(date))
            time0 = time.time()

            C9 = read_C9_index_of_one_day(date)
            print("C9 index: {}".format(C9))

            data = gen_data_of_current_day(data_path, filename)

            plot_heatmap_of_current_day(data, C9, figure_name, fo_path)
            print("date {} cost: {} sec".format(date, round(time.time() - time0, 2)))

    return True


site, str_year, lon_left, lon_right = 'millstone', '2015', -125, -115
lat_low, lat_high = 30, 80
main(site, str_year, lon_left, lon_right)
