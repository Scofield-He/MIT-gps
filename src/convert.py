#! python3
# -*- coding: utf-8 -*-

import os
import time
import datetime
import numpy as np
import h5py
import aacgmv2


data_type = np.dtype({
    "names": ['mlat', 'mlon', 'mlt'],
    "formats": ['<f4', '<f4', '<f4']
})


def convert(raw_data, post_data):
    for item in raw_data:
        mlat, mlon = aacgmv2.convert(item["gdlat"], item["glon"], alt=350,
                                     date=datetime.date(item["year"], item["month"], item["day"]), a2g=False)
        mlt = aacgmv2.convert_mlt(mlon, datetime.datetime(item["year"], item["month"], item["day"],
                                                          item["hour"], item["min"], item["sec"]), m2a=False)

        a = np.array((mlat, mlon, mlt), dtype=data_type)
        post_data.append(a)
    print("convert work done!")


file_path = "C:\\DATA\\GPS_MIT\\2013\\data\\"

count = 0
for fi in os.listdir(file_path):
    if ".hdf5" in fi:
        file = h5py.File(file_path + fi)
        count += 1
    else:
        continue

    print(file)
    for item in file:
        print(item)

    if "Post NA Data" in file["Data"]:
        print("{} have 'Post NA Data' and will be deleted".format(file))
        del file["Data"]["Post NA Data"]

    start_time = time.time()
    raw_dat = file["Data"]["Table Layout"]
    post_dat = []
    convert(raw_dat, post_dat)
    file["Data"].create_dataset("Post NA Data", data=post_dat, chunks=True)
    print("len of converted data: --> {}".format(len(post_dat)))
    print("len of pre-convert data: --> {}".format(len(raw_dat)))
    print("{} convert work cost --> {}".format(fi, round(time.time() - start_time, 2)))
    print("count = {}".format(count))

print("work done!")
