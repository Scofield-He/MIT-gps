#! python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import datetime
import pandas as pd
# import gc


def read_kp_index(year, glon_list, lt_list):
    kp_index_filepath = "C:\\DATA\\index\\Kp-ap-Flux_01-17.dat"
    index_of_1year = []

    with open(kp_index_filepath, 'rb') as fi:
        last_line = ""
        for line in fi:
            line = line.strip().decode('utf-8')
            if line[:6] == "{}1231".format(str(year-1)[-2:]):
                last_line = line
            if line[:2] == str(year)[-2:]:
                date = datetime.date(year, int(line[2:4]), int(line[4:6]))
                time0 = time.time()
                print(date, end=':  ')
                cur_doy = date.timetuple().tm_yday

                for glon in glon_list:
                    for lt in lt_list:
                        index_of_1year.append([year, date, cur_doy, glon, lt])

                        sum_8kp, mean_8ap, C9 = int(line[28:31])/10, int(line[55:58]), int(line[61])
                        Cp = float(line[58:61])
                        if line[65:70] != '     ':
                            f107 = float(line[65:70])
                        else:
                            f107 = np.nan

                        day_diff, ut = divmod(lt - glon // 15, 24)
                        begin_digit = 12 + 2 * int(ut // 3)
                        # end_digit = begin_digit + 2
                        kp_index = []                                    # 取当前3小时kp值及前2个3小时kp值
                        if begin_digit >= 16:
                            kp_index.append(int(line[begin_digit - 4: begin_digit - 2]) / 10)
                            kp_index.append(int(line[begin_digit - 2: begin_digit]) / 10)
                            kp_index.append(int(line[begin_digit:begin_digit+2]) / 10)
                        elif begin_digit == 14:
                            kp_index.append(int(last_line[26:28]) / 10)
                            kp_index.append(int(line[12: 14]) / 10)
                            kp_index.append(int(line[14: 16]) / 10)
                        elif begin_digit == 12:
                            kp_index.append(int(last_line[24: 26]) / 10)
                            kp_index.append(int(last_line[26: 28]) / 10)
                            kp_index.append(int(line[12: 14]) / 10)

                        kp = kp_index[-1]
                        kp9_sum, weight_sum = 0, 0
                        for i in range(3):
                            weight_sum += np.e ** (-i)
                            kp9_sum += kp_index.pop() * np.e ** (-i)
                        kp9 = round(kp9_sum / weight_sum, 1)

                        index_of_1year[-1].extend([kp, kp9, Cp, C9, sum_8kp, mean_8ap, f107])
                print("{:.2f}s".format(time.time()-time0))
            last_line = line

    # year, date, cur_doy, glon, lt
    df_columns_name = ["year", "date", "doy", "glon", "lt", "kp", "kp9", "Cp", "C9", "sum_8kp", "mean_8ap", "F10.7"]
    df_cur_year = pd.DataFrame(index_of_1year, columns=df_columns_name)
    path_tmp = "C:\\DATA\\index\\kp_index.csv"
    if not os.path.exists(path_tmp):
        df_cur_year.to_csv(path_tmp, index=False, encoding='gb2312')
    else:
        df = pd.read_csv(path_tmp, encoding='gb2312')
        new_df = df.append(df_cur_year, ignore_index=True)
        new_df.to_csv(path_tmp, index=False, encoding='gb2312')

    return True


data_site = 'millstone'
Year_list = [2017, 2016, 2015, 2014, 2013, 2012]
Glon_c_list = [-120, -90, 0, 30]
Lt_list = [22, 23, 0, 1, 2, 3, 4, 5, 18, 19, 20, 21]
for Year in Year_list:
    read_kp_index(year=Year, glon_list=Glon_c_list, lt_list=Lt_list)
