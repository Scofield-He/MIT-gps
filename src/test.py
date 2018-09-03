#! python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def profile_daily_change(file_path):
    doy, glat, tec = [], [], []
    xdoy, position = [], []
    with open(file_path, 'r') as fi:
        for line in fi:
            items = line.strip().split(';')
            if len(items) == 3:
                # for lat in range(45, 71):
                lats = items[1].split(' ')
                for lat in lats:
                    if lat:
                        doy.append(int(items[0]))
                        glat.append(lat)
                cnt = 0
                values = items[2].split(' ')
                if values:
                    values = [float(v) for v in values if v]
                    lats = [_ for _ in lats if _]

                    """
                    if int(items[0]) == 275:
                        print("tec value of doy 275: \n", lats, "\n", values)
                    elif int(items[0]) == 216:
                        print("tec value of doy 216: \n", lats, "\n", values)
                    """

                    index_45, index_65 = lats.index('45'), lats.index('60')
                    lats = lats[index_45: index_65]
                    value = values[index_45: index_65]

                    xdoy.append(int(items[0]))
                    position.append(lats[value.index(min(value))])

                    maxi = max(values)
                    # mini = min(values)
                    # values = [int(20 * (v - mini) / (maxi - mini)) for v in values]  # 归一化
                    values = [int(20 * v / maxi) for v in values]
                    # else:
                    #     values = [10 for _ in values]
                for _ in values:
                    tec.append(float(_))
                    cnt += 1

                    # print('tec value count for one doy: {}'.format(cnt))

    print(len(doy), len(glat), len(tec))
    print("trough mini loc: {}".format(position))
    return doy, glat, tec, xdoy, position


def mag_index_daily_change(year):
    file_path = "C:\\DATA\\index\\Kp-ap-Flux_01-17.dat"
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
                Kp.append(int(line_data[16:18]))  # 3 hours: 06-09UT
    return doy, c9, Kp


Year, lt1, lt2, glon1, glon2 = 2013, 23, 1, -125, -115
datapath = "C:\\code\\MIT-gps\\resources\\{}\\{}_{}-{}_{}-{}.txt".format(Year, Year, lt1, lt2, glon1, glon2)

fig = plt.figure(figsize=(30, 6))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)

plt.sca(ax1)
x, y, color, x_doy, loc_mini = profile_daily_change(datapath)
print("length of xday: {}".format(x_doy))
print("length of loc_mini: {}".format(loc_mini))

doy1, doy2 = 0, 366
x = [_ for _ in x if doy1 < _ < doy2]
y, color = y[:len(x)], color[:len(x)]
plt.scatter(x, y, c=color, vmin=0, vmax=22, marker='s', s=8, alpha=1, cmap=plt.cm.jet)
plt.plot(x_doy, loc_mini, 'r')
plt.ylabel('mlat')
axis = plt.gca().xaxis
for label in axis.get_ticklabels():
    label.set_rotation(45)
    label.set_fontsize(8)

plt.title(' ')
plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)  # color bar
cax = plt.axes([0.92, 0.54, 0.01, 0.35])
plt.colorbar(cax=cax).set_label('TEC/TECU')

plt.sca(ax2)
day_of_year, C9_index, Kp_index = mag_index_daily_change("2013")
C9_index = [10 * _ for _ in C9_index]
width = 1
plt.bar(day_of_year, Kp_index, width, color='b', label='Kp index')
day_of_year = [_ + 0.5 for _ in day_of_year]
plt.plot(day_of_year, C9_index, '.', color='r', label='C9 index')
plt.ylabel('Kp index')
plt.xlabel('doy')
plt.ylim(0, 90)
plt.xlim(doy1 - 2, doy2 + 2)
new_ticks = list(map(int, np.linspace(doy1, doy2, int((doy2 - doy1) / 7))))
print(new_ticks)
plt.xticks(new_ticks)
axis = plt.gca().xaxis
for label in axis.get_ticklabels():
    label.set_rotation(45)
    label.set_fontsize(8)

days = [82, 173, 266, 356]
dates = [20, 26, 60, 76, 88, 121, 152, 158, 180, 187, 191, 217, 228, 256, 262, 275, 282, 287, 313, 327, 342, 348]
plt.sca(ax1)
for day in days:
    # plt.plot([day, day, ], [45, 70, ], 'w--', linewidth=1)
    # print(day)
    pass
for date in dates:
    plt.plot([date, date, ], [45, 70, ], 'w--', linewidth=1)

plt.sca(ax2)
for _ in days:
    # plt.plot([day, day, ], [0, 90, ], 'g--', linewidth=1)
    # print(day)
    pass
for date in dates:
    plt.plot([date, date, ], [0, 90, ], 'k--', linewidth=1)
plt.show()
out_path = 'C:\\code\\MIT-gps\\resources\\2013\\'
# plt.savefig(out_path + '2013-tec_prof_lt1-lt2_glon1-glon2_regular', dpi=500)
plt.close()

# 拟合槽极小的位置与磁场活动指数的关系
Kp_index = [int(_) for _ in Kp_index]
loc_mini = [int(_) for _ in loc_mini]
coe = np.corrcoef(np.array([Kp_index, loc_mini]))
print("coe = {}".format(coe))
r = round(coe[0][1], 2)

x = np.array([[_] for _ in Kp_index])
y = np.array([[_] for _ in loc_mini])
# print(x)

import statsmodels.formula.api as sm

linear_model = sm.OLS(y, x)
results = linear_model.fit()
print(results.summary())

"""
lr = LinearRegression()
lr.fit(x, y)

print(lr.intercept_, lr.coef_)

fig2 = plt.figure(figsize=(8, 6))
ax = plt.gca()

yhat = lr.predict(x)
plt.scatter(x, y, color='k')
plt.plot(x, yhat, 'b--', label='{} * x + {}'.format(round(lr.coef_[0][0], 3), round(lr.intercept_[0], 2)))
mse = round(mean_squared_error(y, yhat), 2)
print('mse = {}'.format(mse))
plt.text(0.71, 0.88, 'mse= {}   r= {}'.format(mse, r), transform=ax.transAxes, color="black")
plt.legend()
plt.show()
"""
