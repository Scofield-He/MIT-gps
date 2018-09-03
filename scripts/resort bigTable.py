#! python3
# -*- coding: utf-8 -*-


import time
import pandas as pd
import gc


# bigTable_path = "C:\\tmp\\tec_profile_gdlat.csv"
bigTable_path = "C:\\tmp\\kp_index.csv"
bigTable = pd.read_csv(bigTable_path, encoding='gb2312')

year_list = [2017, 2016, 2015, 2014, 2013, 2012]
glon_list = [-120, -90, 0, 30]
lt_list = [22, 23, 0, 1, 2, 3, 4, 5, 18, 19, 20, 21]
df = pd.DataFrame()

time0 = time.time()
for year in year_list:
    for glon in glon_list:
        for lt in lt_list:
            new_df = bigTable.query("year == {} & glon == {} & lt == {}".format(year, glon, lt))
            new_df.sort_values(by='doy')        # 按照doy升序
            df = df.append(new_df, ignore_index=True)
            del new_df
            gc.collect()

# df.to_csv("C:\\tmp\\sorted tec_profile_gdlat.csv", index=False, encoding='gb2312')
df.to_csv("C:\\tmp\\sorted kp_index.csv", index=False, encoding='gb2312')
print('time cost: {}s'.format(round(time.time()-time0, 2)))
print("done")
