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
    "formats": ['<f8', '<f8', '<f8']
})


def convert(raw_data, post_data):
    """
    calculate mlat, mlon and mlt for every data raw.
    :param raw_data: dataset of raw hdf5 files
    :param post_data: dataset conclude mlat,mlon and mlt
    :return: post_data
    """
    # start_time = time.time()
    for raw in raw_data:
        mlat, mlon = aacgmv2.convert(raw["gdlat"], raw["glon"], alt=350,
                                     date=datetime.date(raw["year"], raw["month"], raw["day"]), a2g=False)
        mlt = aacgmv2.convert_mlt(mlon, datetime.datetime(raw["year"], raw["month"], raw["day"],
                                                          raw["hour"], raw["min"], raw["sec"]), m2a=False)

        a = np.array((mlat, mlon, mlt), dtype=data_type)
        post_data.append(a)
    else:
        print("length of aac = {}, length of cvt = {}".format(len(raw_data), len(post_data)))
        # print("it cost {}".format(round((time.time() - start_time), 2)))
    return post_data


file_path = "C:\\DATA\\GPS_MIT\\2006\\data\\"
count = 0
for fi in os.listdir(file_path):
    if ".hdf5" in fi:
        file = h5py.File(file_path + fi)
        print("time = {}".format(datetime.datetime.now()), "file = {}".format(file), "count = {}".format(count))
        count += 1
    else:
        continue
    if "mData" in file["Data"]:
        print(fi)
        continue

    time0 = time.time()
    raw_dat = file["Data"]["Table Layout"]
    cvt_data = []
    convert(raw_dat, cvt_data)
    file["Data"].create_dataset("mData", data=cvt_data, dtype=data_type, chunks=(391, ), compression="gzip")
    print(datetime.datetime.now(), "converting of this file cost: {}".format(round(time.time() - time0, 2)))

print("all files in {} done!".format(file_path))
