#! python3
# -*- coding: utf-8 -*-

import pandas as pd


def data_aggregation(y1, m1, d1, y2, m2, d2):
    """
    :return: 2014-9-1:2017-8-31日，包含槽极小，Kp，AE，F107，等的bigTable
    date: 聚合前后的数据都与世界时的date保持一致，而不是地方时date；
    """
    mit_filepath = "C:\\tmp\\sorted tec_profile_gdlat.csv"
    Kp_filepath = "C:\\tmp\\sorted kp_index.csv"
    AE_filepath = "C:\\tmp\\sorted AE_index.csv"

    table_mit = pd.read_csv(mit_filepath, encoding='gb2312')[['year', 'doy', 'date', 'glon', 'lt', 'trough_min_lat']]
    table_mit["date"] = pd.to_datetime(table_mit["date"])
    table_kp = pd.read_csv(Kp_filepath, encoding='gb2312')
    #  [['year', 'doy', 'date', 'glon', 'lt', 'kp', 'kp9', 'Cp', 'C9', 'sum_8kp', 'mean_8ap', 'F10.7']]
    table_kp["date"] = pd.to_datetime(table_kp["date"])
    table_ae = pd.read_csv(AE_filepath, encoding='gb2312')    # [['year', 'doy', 'date', 'glon', 'lt', 'AE', 'AE6']]
    table_ae["date"] = pd.to_datetime(table_ae["date"])

    print(len(table_mit) == len(table_kp) == len(table_ae))

    # mit = table_mit["trough_min_lat"]
    # table_kp.insert(5, "MIT_gdlat", mit)
    print(len(table_kp), len(table_ae), len(table_mit))

    df_tmp = pd.merge(table_mit, table_ae, on=['year', 'doy', 'date', 'glon', 'lt'], how='outer')
    print(len(df_tmp))
    df_rt = pd.merge(df_tmp, table_kp, on=['year', 'doy', 'date', 'glon', 'lt'], how='outer')

    if len(df_rt) != len(table_kp):      # 仅AE指数数据少两个月
        print(len(df_rt))
        raise Exception("df merge Error: table lengths do not equal")

    df_rt = df_rt.set_index('date')
    df_rt = df_rt['{}-{}-{}'.format(y1, m1, d1): '{}-{}-{}'.format(y2, m2, d2)]
    print(len(df_rt))
    return df_rt


if __name__ == '__main__':
    df = data_aggregation(2014, 9, 1, 2017, 8, 31)
    print(len(df))
    df.to_csv("C:\\tmp\\bigtable.csv", index=True, encoding='gb2312')
