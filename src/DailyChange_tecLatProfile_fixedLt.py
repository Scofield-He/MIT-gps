#! python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import h5py
import datetime
import matplotlib.pyplot as plt

m_data_type = np.dtype({                   # mData数据格式
    "names": ['mlat', 'mlon', 'mlt'],
    "formats": ['<f4', '<f4', '<f4']
})


def data_collect_and_write2txt(data_path, out_path, year, hour, minute, mlat1, mlat2, mlon1, mlon2, lon_center=-160,
                               lat_center=60):
    lt = round((lon_center / 15 + 8 + 57 / 60) % 24, 2)                 # 地方时
    print("Local time at lat={}, lon={}".format(lat_center, lon_center))
    doy, data = 0, {}                                                   # doy年积日,data记录数据
    for file in os.listdir(data_path):
        if ".hdf5" in file and file[3:5] == str(year):
            time0 = time.time()
            doy += 1
            data[doy] = []                                              # 以doy为key，对应tec与磁纬value记在列表中
            print("{} begins: {} -------------------------------------------------".format(file, doy))
            f = h5py.File(data_path + file)
            gp_data, gm_data = f["Data"]["Table Layout"], f["Data"]["mData"]
            # print(len(gp_data))

            index_begin, index_end, i = 0, 0, 0                         # 得到某一世界时的数据下标范围
            hours, minutes = gp_data["hour"], gp_data["min"]            # 避免循环内重复调用
            end_hour, end_minute = hour + 1, minute + 5
            for i in range(len(hours)):
                if hours[i] == hour and minutes[i] == minute:           # 选定的时刻
                    if not index_begin:
                        index_begin = i
                elif hours[i] == end_hour or minutes[i] == end_minute:
                    index_end = i - 1
                    break

            tec = gp_data["tec"][index_begin:index_end + 1]             # 选定时刻的数据
            mlat = gm_data["mlat"][index_begin: index_end + 1]
            mlon = gm_data["mlon"][index_begin: index_end + 1]
            # mlt_0857 = gm_data["mlt"][index_begin: index_end + 1]

            print(len(tec))

            tec_value = {}                                              # key:mlat; value: tec_list
            for i in range(mlat1, mlat2 + 1):
                tec_value[i] = []
            for (i, j, k) in zip(mlat, mlon, tec):
                if mlon1 < j < mlon2 and mlat1 < i < mlat2:            # 根据给定范围筛选数据
                    tec_value[int(i)].append(k)

            mlat, tec = [], []                                          # mlat为磁纬，tec记录对应磁纬的tec均值
            for i in range(mlat1, mlat2 + 1):
                if tec_value[i]:  # tec非空
                    mlat.append(i)
                    tec.append(np.mean(tec_value[i]))
            data[doy].append(mlat)
            data[doy].append(tec)
            print('file of {} cost {}'.format(doy, time.time() - time0))
    else:
        print("All files have been processed!")

    # 将得到的data数据保存在out_path，以年份-地方时-经度值命名
    out_path = out_path + '20{}-{}-{}.txt'.format(year, lt, lon_center)
    with open(out_path, 'w') as fo:
        fo.write('{} {} {}\n'.format(year, lt, lon_center))
        for key in data.keys():
            if data[key]:
                fo.write('{:3d};'.format(key))
                for item in data[key][0]:                              # 写入磁纬数据
                    fo.write('{:3d}'.format(item))
                fo.write(';')
                for item in data[key][1]:                              # 写入对应磁纬的tec均值
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
            if len(items) == 3:
                # for lat in range(45, 71):
                for lat in items[1].split(' '):
                    if lat:
                        doy.append(int(items[0]))
                        mlat.append(lat)
                cnt = 0
                for i in items[2].split(' '):
                    if i:
                        tec.append(float(i))
                        cnt += 1
                print('tec value count {}: for doy {}'.format(cnt, doy))
    print(len(doy), len(mlat), len(tec))
    return doy, mlat, tec


def read_mag_index(file_path):
    year = '13'
    doy, c9, Kp = [], [], []
    i = 0

    with open(file_path, 'rb') as file:
        for line in file:
            line_data = line.strip().decode('utf-8')
            # print(line_data)
            # print((line_data[:2]))
            if line_data[:2] == year:
                i += 1
                doy.append(i)  # 按顺序加一即得年积日，无需根据日期计算
                c9.append(int(line_data[61]))
                Kp.append(int(line_data[16:18]))          # 3 hours: 06-09UT
    return doy, c9, Kp


print("now time: {}".format(datetime.datetime.now()))
data_year = 13
fi_path = "C:\\DATA\\GPS_MIT\\20{}\\data\\".format(data_year)
path = "C:\\code\\MIT-gps\\resources\\20{}\\".format(data_year)
if not os.path.exists(path):
    os.mkdir(path)

fo_path = data_collect_and_write2txt(fi_path, path, 13, 8, 57, 40, 80, -105, -95)

print("start plot:")
fig = plt.figure(figsize=(30, 6))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)

plt.sca(ax1)
x, y, color = read_tec_profile(file_path=fo_path)
plt.scatter(x, y, c=color, vmin=0, vmax=20, marker='s', s=6, alpha=1, cmap=plt.cm.jet)
plt.ylabel('mlat')
plt.xlim(0, 366)
new_ticks = list(map(int, np.linspace(0, 366, 25)))
print(new_ticks)
plt.xticks(new_ticks)

plt.title('year=2013  lt=22.28  lon_center=-160°')

plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)           # plot color bar
cax = plt.axes([0.92, 0.54, 0.01, 0.35])
plt.colorbar(cax=cax).set_label('TEC/TECU')

plt.sca(ax2)
day_of_year, C9_index, Kp_index = read_mag_index(file_path="C:\\DATA\\index\\Kp-ap-Flux_01-17.dat")
width = 1
plt.bar(day_of_year, Kp_index, width, color='b', label='Kp index')
plt.ylabel('Kp index')
plt.xlabel('doy')
plt.ylim(0, 90)
plt.xlim(0, 366)
new_ticks = list(map(int, np.linspace(0, 366, 25)))
print(new_ticks)
plt.xticks(new_ticks)

days = [82, 173, 266, 356]
date = 180
plt.sca(ax1)
for day in days:
    plt.plot([day, day, ], [45, 70, ], 'w--', linewidth=1)
    print(day)
plt.plot([date, date, ], [45, 70, ], 'w--', linewidth=1)

plt.sca(ax2)
for day in days:
    plt.plot([day, day, ], [0, 90, ], 'g--', linewidth=1)
    print(day)
plt.plot([date, date, ], [0, 90, ], 'k--', linewidth=1)
plt.show()
# plt.savefig(out_path + '2013-tec_profile_daily_change', dpi=200)
