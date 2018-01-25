#! python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import h5py
import datetime
import pandas as pd

year, data_site = 2013, 'millstone'
lt1, lt2 = 23.0, 24


def tec_gen_write2txt(year_datapath, gdlat1, gdlat2, glon):
    misplaced_files = []
    year_data = []
    for filename in os.listdir(year_datapath):
        if ".hdf5" in filename and filename[3:5] == str(year)[-2:]:
            time0 = time.time()
            date = datetime.date(int(year), int(filename[5:7]), int(filename[7:9]))
            cur_doy = date.timetuple().tm_yday
            year_data.append([cur_doy])                                     # 当天列表第一个元素记录年积日doy

            file = h5py.File(year_datapath + filename)
            Table = file["Data"]["Table Layout"]
            hour, minute = Table["hour"], Table["min"]
            gdlat, glon, tec, dtec = Table["gdlat"], Table["glon"], Table["tec"], Table["dtec"]

            data_df = pd.DataFrame(
                {'hour': hour, 'minute': minute, 'gdlat': gdlat, 'glon': glon, 'tec': tec, 'dtec': dtec})

            data_df["lt"] = round((data_df['hour'] + data_df['minute'] / 60 + data_df['glon'] / 15) % 24, 2)

            tec_120W, tec_90W, tec_0, tec_30E = {}, {}, {}, {}
            for _ in range(gdlat1, gdlat2 + 1):
                tec_120W[_] = []
                tec_90W[_] = []
                tec_0[_] = []
                tec_30E[_] = []

            # 取均值，必须

            """
            for i in range(len(data_array[0])):
                item = data_array[:, i]
                if is_in[i]:
                    if -114 > item[3] > -126:                                               # 四个经度带
                        if lt1 < (item[0] + item[1] / 60 + item[3] / 15) % 24 < lt2:
                            tec_120W[int(item[2])].append(item[4])
                    elif -84 > item[3] > -96:
                        if lt1 < (item[0] + item[1] / 60 + item[3] / 15) % 24 < lt2:
                            tec_90W[int(item[2])].append(item[4])
                    elif 6 > item[3] > -6:
                        if lt1 < (item[0] + item[1] / 60 + item[3] / 15) % 24 < lt2:
                            tec_0[int(item[2])].append(item[4])
                    elif 24 < item[3] < 36:
                        if lt1 < (item[0] + item[1] / 60 + item[3] / 15) % 24 < lt2:
                            tec_30E[int(item[2])].append(item[4])
            """
            print("the process of data at {} cost: {:.2f}".format(date, time.time()-time0))

        else:
            misplaced_files.append((year, filename))

    else:
        print('misplaced_files:\n')
        if misplaced_files:
            print(misplaced_files)
        else:
            print("None")
