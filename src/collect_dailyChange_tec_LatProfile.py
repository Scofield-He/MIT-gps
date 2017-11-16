#! python3
# -*- coding: utf-8 -*-

# import sys

import os
import time
import numpy as np
import h5py
import datetime
# import matplotlib.pyplot as plt
# from scipy import interpolate

m_data_type = np.dtype({  # 地磁坐标下的经纬度与地方时
    "names": ['mlat', 'mlon', 'mlt'],
    "formats": ['<f4', '<f4', '<f4']
})

print("now time: {}".format(datetime.datetime.now()))
Year = '13'
fi_path = "C:\\DATA\\GPS_MIT\\20{}\\data\\".format(Year)
if not os.path.isdir(fi_path):
    print("there is no fi_path")

doy = 0
data = {}
for file in os.listdir(fi_path):
    if ".hdf5" in file and file[3:5] == Year:
        time0 = time.time()
        doy += 1
        print('doy = {}'.format(doy))
        data[doy] = []
        '''
        if 21 > int(file[7:9]) or int(file[7:9]) > 25:
            continue
        if int(file[5:7]) not in [3, 6, 9, 12]:
            continue
        '''
        print("{} begins: -------------------------------------------------".format(file))
        f = h5py.File(fi_path + file)
        gp_data, gm_data = f["Data"]["Table Layout"], f["Data"]["mData"]

        index_begin, index_end, i = 0, 0, 0          # 得到世界时10h57min的数据下标范围
        for line in gp_data:
            if line["hour"] == 8 and line["min"] == 57:
                if not index_begin:
                    index_begin = i
                index_end = i
            i += 1
            if line["hour"] == 9:
                break

        tec_0857 = gp_data["tec"][index_begin:index_end + 1]
        mlat_0857, mlon_0857 = gm_data["mlat"][index_begin: index_end + 1], gm_data["mlon"][index_begin: index_end + 1]
        # mlt_0857 = gm_data["mlt"][index_begin: index_end + 1]

        print(len(tec_0857))

        mlat1, mlat2 = 40, 80                         # tec数据取值范围
        mlon1, mlon2 = -105, -95

        tec_value = {}                                # 记录tec，以磁纬为key，同一磁纬下的tec值记在value对应的列表中
        for i in range(mlat1, mlat2 + 1):
            tec_value[i] = []
        for (i, j, k) in zip(mlat_0857, mlon_0857, tec_0857):
            if mlon1 < j < mlon2 and mlat1 < i < mlat2:
                tec_value[int(i)].append(k)

        x, y = [], []                                 # x为磁纬，y记录tec均值
        for i in range(mlat1, mlat2 + 1):
            if tec_value[i]:                          # tec非空
                x.append(i)
                y.append(np.mean(tec_value[i]))
        """
        mlat = list(range(mlat1, mlat2 + 1))           # 45-70°磁纬范围，连续取值
        tec = interpolate.UnivariateSpline(x, y, s=8)(mlat)   # 插值得到tec纬度剖面
        tec = list(tec)
        """
        data[doy].append(x)
        data[doy].append(y)
        print('file of {} cost {}'.format(doy, time.time() - time0))
else:
    print("All files have been processed!")

out_path = "C:\\code\\MIT-gps\\resources\\"
if not os.path.exists(out_path):
    os.mkdir(out_path)

lat_center, lon_center = 60, -160             # 中心点的位置，地方时取此处值
lt = round((lon_center / 15 + 8 + 57 / 60) % 24, 2)  # 地方时

with open(out_path + '2013-{}-{}.txt'.format(lt, lon_center), 'w') as fo:
    fo.write('{} {} {}\n'.format(Year, lt, lon_center))
    for key in data.keys():
        if data[key]:
            fo.write('{:3d};'.format(key))
            for item in data[key][0]:
                fo.write('{:3d}'.format(item))
            fo.write(';')
            for item in data[key][1]:
                fo.write('{:7.2f}'.format(item))
            fo.write('\n')
        else:
            fo.write('{:3d} \n'.format(key))

print("write done!")
