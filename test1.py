#! python3
# -*- coding: utf-8 -*-

import os
import time
import datetime
import numpy as np
import h5py
import aacgmv2


data_type = np.dtype({
    "names": ['year', 'month', 'day', 'hour', 'min', 'sec', 'gdlat', 'glon', 'lt', 'tec', 'mlat', 'mlon', 'mlt'],
    "formats": ['<i8', '<i8', '<i8', '<i8', '<i8', '<i8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8']
})


def calculate(data, L1):
    for item in data:
        if -180 < item["glon"] < -60 and 40 < item["gdlat"] < 70:
            UT = item['hour'] + item['min'] / 60 + item['sec'] / 3600
            LT = (item['glon'] / 15 + UT) % 24                                 # 计算地方时
            MLAT, MLON = aacgmv2.convert(item["gdlat"], item["glon"], alt=350,
                                         date=datetime.date(item["year"], item["month"], item["day"]), a2g=False)
            MLT = aacgmv2.convert_mlt(MLON, datetime.datetime(item["year"], item["month"], item["day"],
                                                              item["hour"], item["min"], item["sec"]), m2a=False)

            a = np.array((item["year"], item["month"], item["day"], item["hour"], item["min"], item["sec"],
                          item["gdlat"], item["glon"], LT, item["tec"], MLAT, MLON, MLT), dtype=data_type)
            L1.append(a)
    print("calculation work done!")


file_path = "C:\\DATA\\GPS_MIT\\2011\\"
out_path = file_path + "data\\"
for fi in os.listdir(file_path):
    if ".hdf5" in fi:
        file = h5py.File(file_path + fi)
        data_set = file["Data"]["Table Layout"]
        data_list = []

        start_time = time.clock()
        calculate(data_set, data_list)
        print("len of data: --> {}".format(len(data_list)))
        print("time cost --> {}".format(time.clock() - start_time))

        f = h5py.File(out_path + "NA_" + fi)
        f.create_group("Parameter")
        f.create_group("Data")
        f["Data"].create_dataset("Table Layout", data=data_list, chunks=True)

print("work done!")
