#! python3
# -*- coding: utf-8 -*-

import os
import time
import datetime
import numpy as np
import h5py
import aacgmv2


data_type = np.dtype({
    "names": ['UT', 'lt', 'tec', 'mlat', 'mlon', 'mlt'],
    "formats": ['<f8', '<f8', '<f8', '<f8', '<f8', '<f8']
})


def convert(raw_data, post_data):
    for item in raw_data:
        if -180 < item["glon"] < -60 and 30 < item["gdlat"] < 75:
            UT = round(item['hour'] + item['min'] / 60 + item['sec'] / 3600, 2)
            LT = round((item['glon'] / 15 + UT) % 24, 2)                                 # 计算地方时
            MLAT, MLON = aacgmv2.convert(item["gdlat"], item["glon"], alt=350,
                                         date=datetime.date(item["year"], item["month"], item["day"]), a2g=False)
            MLT = aacgmv2.convert_mlt(MLON, datetime.datetime(item["year"], item["month"], item["day"],
                                                              item["hour"], item["min"], item["sec"]), m2a=False)

            a = np.array((UT, LT, item["tec"], MLAT, MLON, MLT), dtype=data_type)
            post_data.append(a)
    print("calculation work done!")


file_path = "C:\\DATA\\GPS_MIT\\2013\\data\\"

count = 0
for fi in os.listdir(file_path):
    if ".hdf5" in fi:
        file = h5py.File(file_path + fi)
        count += 1
    else:
        continue
    if "Post data" in file["Data"]:
        del file["Data"]["Post data"]
    if "Post NA Data" in file["Data"]:
        print("{} has been writen".format(fi))
        continue
    start_time = time.time()
    raw_dat = file["Data"]["Table Layout"]
    post_dat = []
    convert(raw_dat, post_dat)
    file["Data"].create_dataset("Post NA Data", data=post_dat, chunks=True)
    print("len of data: --> {}".format(len(post_dat)), end=" ")
    print("{} cost --> {}".format(fi, round(time.time() - start_time, 2)))
    print("count = {}".format(count))

print("work done!")
