#! python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def profile_daily_change():
    Year = '13'
    file_path = "C:\\code\\MIT-gps\\resources\\20{}.txt".format(Year)
    doy = []
    mlat = []
    tec = []
    with open(file_path, 'r') as fi:
        for line in fi:
            items = line.strip().split(';')
            if len(items) == 3:
                for lat in range(45, 71):
                    doy.append(int(items[0]))
                    mlat.append(lat)
                cnt = 0
                for i in items[2].split(' '):
                    if i:
                        try:
                            if float(i) < 0:
                                tec.append(1.0)
                            else:
                                tec.append(float(i))
                            cnt += 1
                        except ValueError:
                            tec.append(float(i.split('-')[0]))
                            tec.append(float(i.split('-')[0]))
                            cnt += 2

                print('tec value count for one doy: {}'.format(cnt))

    print(len(doy), len(mlat), len(tec))
    return doy, mlat, tec


def mag_index_daily_change():
    file_path = "C:\\DATA\\index\\Kp-ap-Flux_01-17.dat"
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


fig = plt.figure(figsize=(30, 6))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)

plt.sca(ax1)
x, y, color = profile_daily_change()
plt.scatter(x, y, c=color, vmin=0, vmax=20, marker='s', s=25, alpha=1, cmap=plt.cm.jet)
plt.ylabel('mlat')
plt.xlim(0, 366)
new_ticks = list(map(int, np.linspace(0, 366, 25)))
print(new_ticks)
plt.xticks(new_ticks)

plt.title('year=2013  lt=22.28  lon_center=-160°')

plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)     # colorbar
cax = plt.axes([0.92, 0.54, 0.01, 0.35])
plt.colorbar(cax=cax).set_label('TEC/TECU')

plt.sca(ax2)
day_of_year, C9_index, Kp_index = mag_index_daily_change()
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
# plt.show()
out_path = 'C:\\code\\MIT-gps\\resources\\2013\\'
plt.savefig(out_path + '2013-tec_profile_daily_change', dpi=200)
