#! python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

file_path = "C:\\DATA\\index\\Kp-ap-Flux_01-17.dat"
year = "13"
doy, c9, Kp = [], [], []
i = 0

with open(file_path, 'rb') as file:
    for line in file:
        line_data = line.strip().decode('utf-8')
        # print(line_data)
        # print((line_data[:2]))
        if line_data[:2] == year:
            i += 1
            doy.append(i)             # 按顺序加一即得年积日，无需根据日期计算
            c9.append(int(line_data[61]))
            Kp.append(int(line_data[18:20]))

print("days of {}: {}".format(year, len(doy)))
print("index count: {}".format(len(c9)))
print(doy[:20])
print(c9[:20])
print(Kp[:20])

fig = plt.figure(figsize=(12, 5))
left_axis = fig.add_subplot(111)

width = 1
left_axis.bar(doy[:], c9, width, color="b", label="C9")
# plt.bar(doy[:], c9, width, color="b", label="C9 index")
left_axis.set_label('C9 index')
left_axis.set_ylabel("C9 index")
left_axis.set_ylim(0, 9)
left_axis.set_xlabel("doy")

doys = [82, 173, 266, 356]
for x in doys:
    plt.plot([x, x, ], [0, 9, ], 'g--', linewidth=2)

right_axis = left_axis.twinx()
# right_axis.plot(doy, Kp, color='k')
right_axis.scatter(doy, Kp, s=5, marker='o', color='k', label='Kp')
right_axis.set_ylabel("Kp at midnight")
right_axis.set_ylim(0, 90)

plt.xlim(0, 366)
new_ticks = list(map(int, np.linspace(0, 366, 25)))
print(new_ticks)
plt.xticks(new_ticks)
plt.title("magnetic activity of 2013")
plt.legend(loc='best')
plt.show()
