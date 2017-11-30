#! python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import h5py
import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,  FuncFormatter


def tec_profile_gen(glat, glon, tec, lst):

    glon1, glon2 = -162, -158         # (-162, -158)
    glat1, glat2 = 45, 75             # [45, 75]

    tec_value = {}
    for idx in range(glat1, glat2 + 1):
        tec_value[idx] = []
    for (j, k, l) in zip(glon, glat, tec):
        if glon1 < j < glon2 and glat1 < k < glat2:
            tec_value[int(k)].append(l)

    # print(tec_value)

    latitude, mean_tec = [], []
    for idx in range(glat1, glat2 + 1):
        if tec_value[idx]:
            latitude.append(idx)
            mean_tec.append(np.mean(tec_value[idx]))

    lst.append(latitude)
    lst.append(mean_tec)


def lt_formatter(x1, _):
    lon_local = -160
    m = round((x1 + lon_local / 15) % 24, 2)
    return "{}".format(m)


year = 13
data_path = "C:\\DATA\\GPS_MIT\\20{}\\DATA\\".format(year)
fo_path = "C:\\DATA\\GPS_MIT\\2013\\figure\\lt_3\\"

for file in os.listdir(data_path):
    if ".hdf5" in file and file[3:5] == str(year) and int(file[5:7]) > 0:

        date = datetime.date(int('20' + file[3:5]), int(file[5:7]), int(file[7:9]))
        doy = date.timetuple().tm_yday
        title = '{} {}'.format(date, doy)
        print(title)

        if os.path.exists(fo_path + title + '.png'):
            print("picture has already been plotted")
            continue

        C9 = -1
        with open("C:\\DATA\\index\\Kp-ap-Flux_01-17.dat", 'rb') as fp:
            for line in fp:
                line_data = line.strip().decode('utf-8')
                if line_data[:2] == str(date.year)[2:] and int(line_data[2:4]) == date.month \
                        and int(line_data[4:6]) == date.day:
                    C9 = int(line_data[61])
        print('C9: {}'.format(C9))

        print("{} begins: -------------------------------------".format(date))
        time0 = time.time()
        f = h5py.File(data_path + file)
        table_layout, gm_data = f["Data"]["Table Layout"], f["Data"]["mData"]
        hours, minutes = table_layout["hour"], table_layout["min"]
        lat, lon, tec_data = table_layout["gdlat"], table_layout["glon"], table_layout["tec"]

        data_length = len(table_layout)
        lon_center = -160

        i = 0
        h0, m0 = hours[i], minutes[i]
        # print(h0, m0)
        data = {}
        while 1:
            if i == data_length:
                break
            index_begin, index_end, ut = -1, -1, -1

            while i < data_length and minutes[i] == m0:
                if index_begin < 0:
                    index_begin = i
                    ut = round(h0 + m0 / 60, 2)
                    # lt = round((ut + lon_center / 15) % 24, 2)
                    data[ut] = []
                    # print("ut: {}-{}, lt: {}".format(h0, m0, lt))
                i += 1
                # print(i, minutes[i], m0, index_begin)
            else:
                index_end = i - 1
                # print(" ut: {:2d}-{:2d}; -------- begin: {:7d} ; end: {:7d}".format(h0, m0, index_begin, index_end))

                tec_ut = tec_data[index_begin: i]
                gdlat = lat[index_begin: i]
                gdlon = lon[index_begin: i]

                tec_profile_gen(gdlat, gdlon, tec_ut, data[ut])

                if i < data_length:
                    h0, m0 = hours[i], minutes[i]
                    # print(h0, m0)
                else:
                    print("data length: {}".format(data_length))
                    # print(data)

        print("time cost: {}".format(round(time.time() - time0, 2)))
        x, y, color = [], [], []         # ut、纬度、mean_tec
        for key in data.keys():
            if data[key]:
                for _ in range(len(data[key][0])):
                    x.append(key)
                    y.append(data[key][0][_])
                    color.append(data[key][1][_])

        print('length of lt: {}; lat: {};  mean_tec: {}'.format(len(x), len(y), len(color)))

        fig = plt.figure(figsize=(15, 1.5))

        plt.scatter(x, y, c=color, vmin=0, vmax=15, marker='s', s=4, alpha=1, cmap=plt.cm.jet)

        plt.title(title + '  {}'.format(C9))

        ax = plt.gca()

        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.xaxis.set_major_formatter(FuncFormatter(lt_formatter))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        # x_ticks = [round(_ + lon_center / 15, 2) % 24 for _ in range(0, 25, 2)]
        # plt.xticks(x_ticks)
        plt.xlabel('lt')

        plt.ylabel('latitude')

        plt.subplots_adjust(bottom=0.15, right=0.9, top=0.85)
        cax = plt.axes([0.93, 0.15, 0.02, 0.7])
        plt.colorbar(cax=cax).set_label('TEC/TECU')

        fig.savefig(fo_path + title, dpi=500)
        plt.close()

        # plt.show()
        # break
