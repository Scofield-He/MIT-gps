#! python3
# -*- coding: utf-8 -*-

# import sys

import os
# import time
import numpy as np
# import matplotlib.pyplot as plt
import h5py
import datetime
from scipy import interpolate

m_data_type = np.dtype({  # 地磁坐标下的经纬度与地方时
    "names": ['mlat', 'mlon', 'mlt'],
    "formats": ['<f4', '<f4', '<f4']
})

feature_type = np.dtype({  # 地磁坐标下的经纬度与地方时
    "names": ['UT', 'LT', 'mlt', 'gdlat', 'glon', 'mlat', 'mlon', 'tec'],
    "formats": [datetime.datetime, '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4']
})


def parameter_gen(lat, TEC):

    return False


def tec_profile_gen(gp_data, gm_data, index_begin, index_end):  # 计算每个时刻，给定范围的tec纬度剖面
    """
    :param gp_data: fi["Data"]["Table Layout"]
    :param gm_data: fi["Data"]["mData"]
    :param index_begin: 某时刻数据的起始index
    :param index_end: 某时刻数据的结尾index 
    :return: 磁纬范围，mlat，1°分辨率；对应的tec纬度剖面，list；
    """
    feature = []
    ut = datetime.datetime(gp_data[index_begin]["year"], gp_data[index_begin]["month"], gp_data[index_begin]["day"],
                           gp_data[index_begin]["hour"], gp_data[index_begin]["min"], gp_data[index_begin]["sec"])
    t = round(gp_data[index_begin]["hour"] + gp_data[index_begin]["min"] / 60 + gp_data[index_begin]["sec"] / 3600, 2)
    lat_center, lon_center = 60, -120                            # 中心点的位置，地方时取此处值
    mlat1, mlat2 = 45, 75                                        # tec数据取值范围
    mlon1, mlon2 = -105, -95
    lt = round(t + lon_center / 15, 2)                           # 中心点处的地方时

    tec_global = gp_data["tec"][index_begin: index_end]          # ut时刻的全球数据
    mlat_global = gm_data["mlat"][index_begin: index_end]
    mlon_global = gm_data["mlon"][index_begin: index_end]
    mlt_global = gm_data["mlat"][index_begin: index_end]

    index = index_begin
    gdlat, glon, tec, mlat, mlon, mlt = [], [], [], [], [], []  # 记录给定区域的数据，以磁经纬度为筛选条件
    for (i, j) in zip(mlat_global, mlon_global):
        if mlon1 < j < mlon2 and mlat1 < i < mlat2:
            mlat.append(i)
            mlon.append(j)
            tec.append(tec_global[index])
            mlt.append(mlt_global[index])
        index += 1
    else:
        print("data of {} filtered done".format(ut))

    feature.extend([ut, t, lat_center, lon_center, lt, ])

    # 计算给定区域经度范围各磁纬的的tec算术平均值，并采用样条插值以1°间隔得到tec纬度剖面
    tec_value = {}
    for i in range(mlat1, mlat2):
        tec_value[i] = []
    for (i, j) in zip(mlat, tec):                               # 相同磁纬的tec记录在一个列表中
        tec_value[int(i)].append(j)
    x, y = [], []
    for i in range(mlat1, mlat2):
        if tec_value[i]:
            x.append(i)
            y.append(np.mean(tec_value[i]))
    mlat = list(range(np.min(x), np.max(x) + 1))                # 磁纬范围
    tec = interpolate.UnivariateSpline(x, y, s=8)(mlat)         # 插值得到的tec纬度剖面
    return mlat, tec, feature


def main(fi, fo):
    f = h5py.File(fi)
    data = f["Data"]
    gp_data, gm_data = data["Table Layout"], data["mData"]
    if len(gp_data) != len(gm_data):
        raise ValueError("data length of gp_data and gm_data is not equal!")
    else:
        data_length = len(gm_data)

    index_begin, index_end, index, count = 0, 0, 0, 0
    features = []  # 创建列表，记录每个时刻的槽特征信息
    while index < data_length:
        index_begin = index
        while index < data_length and gp_data[index]["min"] == gp_data[index_begin]["min"]:
            index += 1
        else:
            index_end = index - 1

        print("index_begin = {}, index_end = {}".format(index_begin, index_end))
        mlat, tec, fea = tec_profile_gen(gp_data=gp_data, gm_data=gm_data, index_begin=index_begin, index_end=index_end)
        parameter_gen(mlat, tec)

    with open(fo, 'a') as f:
        for line in features:
            f.write(line)


print("now time: {}".format(datetime.datetime.now()))
Year = '13'
fi_path = "C:\\DATA\\GPS_MIT\\20{}\\data\\".format(Year)
out_path = "C:\\code\\MIT-gps\\resources\\20{}.txt".format(Year)
if not fi_path:
    print("there is no fi_path")
if not out_path:
    print("there is no out_path folder")
for file in os.listdir(fi_path):
    if ".hdf5" in file and file[3:5] == Year:
        print("{} begins: ---------------------------".format(file))
        main(fi=file, fo=out_path)
else:
    print("All work have been done!")
