#! python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import h5py
import datetime
import pandas as pd
import gc


def tec_gen_write2txt(year_datapath, glon_slt, lt_list):
    misplaced_files = []
    data_of_1year = []
    print("{} beginning ---------------------------------------------".format(year))
    for filename in os.listdir(year_datapath):
        if ".hdf5" in filename and filename[3:5] == str(year)[-2:]:
            time0 = time.time()
            date = datetime.date(int(year), int(filename[5:7]), int(filename[7:9]))
            cur_doy = date.timetuple().tm_yday
            print("{} begins -------------- {}".format(filename, cur_doy))

            # if cur_doy < 30:
            #     continue
            file = h5py.File(year_datapath + filename)
            Table = file["Data"]["Table Layout"]
            hour, minute = Table["hour"], Table["min"]
            gdlat, glon, tec, dtec = Table["gdlat"], Table["glon"], Table["tec"], Table["dtec"]
            file.close()
            # time1 = time.time()
            # print("time -- read data: {}".format(round(time1 - time0, 2)), end='        ')

            data_df = pd.DataFrame(
                {'hour': hour, 'minute': minute, 'gdlat': gdlat, 'glon': glon, 'tec': tec, 'dtec': dtec})

            data_df["lt"] = round((data_df['hour'] + data_df['minute'] / 60 + data_df['glon'] / 15) % 24, 2)

            # print(data_df.loc[1])
            # time2 = time.time()
            # print("time -- calculate lt: {}".format(round(time2 - time1, 2)), end='       ')

            for lon_c in glon_slt:
                lon1, lon2 = lon_c - 5, lon_c + 5                           # 经度范围，取10
                for lt1 in lt_list:
                    lt2 = lt1 + 1
                    df_cur = data_df.query("{} < glon < {} & {} < lt < {} & {} <= gdlat <= {} & tec > 2 * dtec".format(
                        lon1, lon2, lt1, lt2, gdlat_low, gdlat_high
                    ))

                    """
                    [year, lon_c, lt1, doy, trough_min_lat, mean_tec_30, mean_tec_31, ……， mean_tec_80]
                    """
                    data_of_1year.append([year, date, cur_doy, lon_c, lt1])

                    lats = []                                               # 得到记录对应纬度tec均值的列表
                    for _ in range(gdlat_low, gdlat_high + 1):
                        lats.append(df_cur.query("gdlat=={}".format(_))["tec"].mean())

                    # 除去极端值
                    for _ in range(5, len(lats) - 5):                      # 大于1.5倍左右10个值均值视为异常，记为nan
                        if lats[_] > 1.5 * np.nanmean(lats[_ - 5: _+5]):
                            lats[_] = np.nan

                    # 找到极小的纬度, lats[0]-30°, 于是lats[15]对应45°，15:40对应[45:70]纬度范围
                    mean_tec, min_tec = np.nanmean(lats[15:40]), np.nanmin(lats[15:40])
                    if min_tec < mean_tec * 0.8:
                        min_value_loc = np.nanargmin(lats[15:40]) + 45
                        data_of_1year[-1].append(min_value_loc)
                    else:
                        data_of_1year[-1].append(np.nan)                    # 极小大于均值80%的，写入nan

                    # 记录纬度剖面
                    data_of_1year[-1].extend([round(_, 2) for _ in lats])
                    print("len: ", len(data_of_1year))
                    del lats, df_cur
            del Table, hour, minute, gdlat, glon, tec, dtec, data_df
            gc.collect()
            # time3 = time.time()
            # print("time -- get tec lat prof & trough min loc: {}".format(round(time3 - time2, 2)))
            print("the process of data at {} cost: {:.2f}".format(date, round(time.time()-time0, 2)))
        else:
            misplaced_files.append((year, filename))
    else:
        print('misplaced_files:\n')
        if misplaced_files:
            print(misplaced_files)
        else:
            print("None")

    df_columns_name = ["year", "date", "doy", "glon", "lt", "trough_min_lat"] + \
                      ["gdlat-{}°".format(_) for _ in range(30, 81)]
    df_cur_year = pd.DataFrame(data_of_1year, columns=df_columns_name)
    # print("count of df_cur_year: ---------------------\n{}".format(df_cur_year.count()))
    path_tmp = "C:\\DATA\\GPS_MIT\\{}\\tec_profile_gdlat.csv".format(data_site)
    if not os.path.exists(path_tmp):
        df_cur_year.to_csv(path_tmp, index=False, encoding='gb2312')
    else:
        df = pd.read_csv(path_tmp, encoding='gb2312')
        # print("read df: ")
        # print(df.loc[1:3])
        new_df = df.append(df_cur_year, ignore_index=True)
        # print("new_df: -------------------------------\n{}".format(new_df))
        new_df.to_csv(path_tmp, index=False, encoding='gb2312')


year_list = [2017, 2016, 2015, 2014, 2013, 2012]
data_site = 'millstone'
glon_c = [-120, -90, 0, 30]
gdlat_low, gdlat_high = 30, 80
lt = [22, 23, 0, 1, 2, 3, 4, 5, 18, 19, 20, 21]
for year in year_list:
    # datapath = "C:\\DATA\\GPS_MIT\\{}\\{}\\data\\".format(data_site, year)
    datapath = "G:\\MIT research\\DATA\\GPS_MIT\\{}\\{}\\".format(year, data_site)
    tec_gen_write2txt(datapath, glon_c, lt)
