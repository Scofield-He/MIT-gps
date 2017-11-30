#! python3
# -*- coding: utf-8 -*-

import os
import time
import h5py
import datetime

print("now time: {}".format(datetime.datetime.now()))
data_year = 13
fi_path = "C:\\DATA\\GPS_MIT\\20{}\\data\\".format(data_year)
path = "C:\\code\\MIT-gps\\resources\\20{}\\".format(data_year)
if not os.path.exists(path):
    os.mkdir(path)

fo_path = path + "20{}-gps-index.txt".format(data_year)

index = []                              # 记录一年中所有天里，相同时刻的起止index
for file in os.listdir(fi_path):
    if '.hdf5' in file and file[3:5] == str(data_year):
        time0 = time.time()
        date = datetime.date(int('20' + file[3:5]), int(file[5:7]), int(file[7:9]))
        doy = date.timetuple().tm_yday
        print("-------------- {} --------------".format(date))

        f = h5py.File(fi_path + file)
        table_layout = f["Data"]["Table Layout"]
        hours, minutes = table_layout["hour"], table_layout["min"]

        table_length = len(table_layout)
        i = 0
        h0, m0 = hours[i], minutes[i]
        while 1:
            if i == table_length:
                break
            index_begin, index_end, ut = -1, -1, -1

            while i < table_length and minutes[i] == m0:
                if index_begin < 0:
                    index_begin = i
                i += 1
            else:
                index_end = i - 1
                line = [date, doy, h0, m0, index_begin, index_end]
                # print(line)
                index.append(line)
                if i < table_length:
                    h0, m0 = hours[i], minutes[i]
        print('doy:{} cost {}'.format(doy, round(time.time() - time0, 2)))


"""
index_data_type = np.dtype({                   
    "names": ['date', 'doy', 'hour', 'min',  'index_begin', 'index_end],
    "formats": ['datetime.date', 'int', 'int', int', 'int', 'int]
})
"""

with open(fo_path, 'w') as fo:
    for line in index:
        fo.write('{0}{1:4d}{2:3d}{3:3d}{4:8d}{5:8d}\n'.format(line[0], line[1], line[2],
                                                              line[3], line[4], line[5]))

print(datetime.datetime.now())
print("work done!")
