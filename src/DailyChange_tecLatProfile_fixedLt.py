#! python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import h5py
import datetime
# import matplotlib.pyplot as plt

m_data_type = np.dtype({                   # mData数据格式
    "names": ['mlat', 'mlon', 'mlt'],
    "formats": ['<f4', '<f4', '<f4']
})


def data_collect_and_write2txt(data_path, out_path, year, lt1, lt2, gdlat1, gdlat2, gdlon1, gdlon2):
    lt_mid, glon = (lt1 + lt2) / 2, (gdlon1 + gdlon2) // 2
    print("lt range from {} to {} & mid lt: {}".format(lt1, lt2, lt_mid))

    data = {}                                                            # data记录数据

    for file in os.listdir(data_path):
        if ".hdf5" in file and file[3:5] == str(year)[-2:]:
            time0 = time.time()
            date = datetime.date(int(year), int(file[5:7]), int(file[7:9]))
            doy = date.timetuple().tm_yday
            data[doy] = []                                              # 重复日期，刷新data
            print("{} begins: {} -------------------------------------------------".format(date, doy))

            f = h5py.File(data_path + file)
            gp_data = f["Data"]["Table Layout"]
            hours, minutes = gp_data["hour"], gp_data["min"]            # 避免循环内重复调用
            gdlat, gdlon, tec_data, dtec = gp_data["gdlat"], gp_data["glon"], gp_data["tec"], gp_data["dtec"]
            len_data = len(tec_data)

            tec_value = {}                                              # key:glat; value: tec_list
            for _ in range(gdlat1, gdlat2 + 1):
                tec_value[_] = []
            for _ in range(len_data):
                i, j, k = gdlat[_], gdlon[_], tec_data[_]
                if gdlat1 < i < gdlat2 and gdlon1 < j < gdlon2:        # 根据给定范围筛选数据
                    if dtec[_] > k / 2:                                # 过滤dtec超过tec值一半的数据
                        continue
                    lt = (hours[_] + minutes[_] / 60 + j / 15) % 24
                    if lt1 < lt < lt2:
                        tec_value[int(i)].append(k)

            glat, tec = [], []                                          # glat为地理纬度，tec记录对应纬度的tec均值
            for _ in range(gdlat1, gdlat2 + 1):
                if tec_value[_]:  # tec非空
                    glat.append(_)
                    tec.append(np.mean(tec_value[_]))

            data[doy].append(glat)
            data[doy].append(tec)

            print('file of {} cost {}'.format(doy, time.time() - time0))
    else:
        print("All files have been processed!")

    # 将得到的data数据保存在out_path，以年份-地方时-经度值命名
    out_path = out_path + '{}_{}-{}_{}-{}.txt'.format(year, lt1, lt2, gdlon1, gdlon2)

    with open(out_path, 'w') as fo:
        fo.write('year:{}  lt1:{} lt2:{}  glon1:{} glon2:{}\n'.format(year, lt1, lt2, gdlon1, gdlon2))
        for key in data.keys():
            if data[key]:
                fo.write('{:3d};'.format(key))
                for item in data[key][0]:                              # 写入地理纬度数据
                    fo.write('{:3d}'.format(item))
                fo.write(';')
                for item in data[key][1]:                              # 写入对应纬度的tec均值
                    fo.write('{:7.2f}'.format(item))
                fo.write('\n')
            else:
                fo.write('{:3d} \n'.format(key))
    print("write done!")
    return out_path                                                    # 返回输出文件路径


def read_tec_profile(file_path):
    doy, mlat, tec = [], [], []
    with open(file_path, 'r') as fi:
        for line in fi:
            items = line.strip().split(';')
            if len(items) == 3:                                         # mlat;tec
                # for lat in range(45, 71):
                for lat in items[1].split(' '):
                    if lat:
                        doy.append(int(items[0]))
                        mlat.append(lat)
                cnt = 0
                values = items[2].split(' ')
                if values:
                    values = [float(v) for v in values if v]
                    # print(values)
                    mini, maxi = min(values), max(values)
                    values = [int(20 * (v - mini) / (maxi - mini)) for v in values]    # 归一化同一时刻的tec
                for i in values:
                    tec.append(float(i))
                    cnt += 1

    print(len(doy), len(mlat), len(tec))
    return doy, mlat, tec


def read_mag_index(file_path):
    year = '2013'
    doy, c9, Kp = [], [], []
    i = 0
    with open(file_path, 'rb') as file:
        for line in file:
            line_data = line.strip().decode('utf-8')
            # print(line_data)
            # print((line_data[:2]))
            if line_data[:2] == year[-2:]:
                i += 1
                doy.append(i)  # 按顺序加一即得年积日，无需根据日期计算
                c9.append(int(line_data[61]))
                Kp.append(int(line_data[16:18]))          # 3 hours: 06-09UT
    return doy, c9, Kp


print("now time: {}".format(datetime.datetime.now()))
data_year, site = 2016, 'millstone'
fi_path = "C:\\DATA\\GPS_MIT\\{}\\{}\\data\\".format(site, data_year)
path = "C:\\code\\MIT-gps\\resources\\{}\\{}\\".format(data_year, site)

lat1, lat2 = 30, 80
lon1, lon2 = -125, -115
if not os.path.exists(path):
    os.mkdir(path)

# localtime = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
localtime = [22, 23, 0, 1, 2, 3, 4, 5, 18, 19, 20, 21]

for localtime1 in localtime:
    localtime2 = localtime1 + 1                                          # 18-19, ……, 23-24, 0-1, ……
    fo_path = data_collect_and_write2txt(fi_path, path, data_year, localtime1, localtime2, lat1, lat2, lon1, lon2)
    print(fo_path)

print("work done!")
