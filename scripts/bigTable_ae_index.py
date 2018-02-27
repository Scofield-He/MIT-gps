#! python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import datetime
import pandas as pd
# import gc


def read_kp_index(year, glon_list, lt_list):
    kp_index_filepath = "C:\\DATA\\index\\AE index_0101-1706_WDC-like format.txt"
    index_of_1year = []

    with open(kp_index_filepath, 'rb') as fi:
        last_line = ""
        for line in fi:
            line = line.strip().decode('utf-8')
            if line[:2] != 'AE':
                continue
            if line[:6] == "{}1231".format(str(year-1)[-2:]):
                last_line = line
            if line[14:16] + line[3:5] == str(year):
                time0 = time.time()
                date = datetime.date(year, int(line[5:7]), int(line[8:10]))
                print(date, end=':  ')
                cur_doy = date.timetuple().tm_yday

                for glon in glon_list:
                    for lt in lt_list:
                        index_of_1year.append([year, date, cur_doy, glon, lt])
                        day_diff, ut = divmod(lt - glon // 15, 24)
                        ae_index = []
                        if ut >= 6:                                          # 7个ae值在同一行
                            begin_col_index = 20 + ut * 4                    # 第20位为ut 0-1的ae值
                            for _ in range(7):                              # 当前时刻ae值与前6个ae值
                                ae_index.append(int(line[begin_col_index: begin_col_index + 4]))
                                begin_col_index -= 4

                            ae6_sum, weight_sum = 0, 0
                            for i in range(7):
                                ae6_sum += ae_index[i] * np.e ** (-i)       # 加权求和
                                weight_sum += np.e ** (-i)                  # 权重和
                            ae6 = round(ae6_sum / weight_sum, 1)            # ae6指数值，加权平均
                            ae = ae_index[0]                                # 当前时刻ae值
                            index_of_1year[-1].extend([ae, ae6])            # 记录当前ae与ae6指数
                        else:
                            # 用列表栈记录若干ae指数值，当前时刻在栈顶
                            last_col_index = 24 + ut * 4                    # ut 0-1为20-24
                            for _ in range(88, 116, 4):
                                ae_index.append(int(last_line[_: _+4]))
                            for _ in range(20, last_col_index, 4):
                                ae_index.append(int(line[_: _+4]))

                            ae = ae_index[-1]
                            ae6_sum, weight_sum = 0, 0
                            for i in range(7):
                                weight_sum += np.e ** (-i)
                                ae6_sum += ae_index.pop() * np.e ** (-i)
                            ae6 = round(ae6_sum / weight_sum, 1)
                            index_of_1year[-1].extend([ae, ae6])

                print("{:.2f}s".format(time.time()-time0))
            last_line = line

    # year, date, cur_doy, glon, lt
    df_columns_name = ["year", "date", "doy", "glon", "lt", "AE", "AE6"]
    df_cur_year = pd.DataFrame(index_of_1year, columns=df_columns_name)
    path_tmp = "C:\\DATA\\index\\AE_index.csv"
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
