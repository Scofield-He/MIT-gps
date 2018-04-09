#! python3
# -*- coding: utf-8 -*-

import pandas as pd


def data_aggregation():
    mit_filepath = "C:\\tmp\\sorted tec_profile_gdlat.csv"
    Kp_filepath = "C:\\tmp\\sorted kp_index.csv"
    AE_filepath = "C:\\tmp\\sorted AE_index.csv"

    table_mit = pd.read_csv(mit_filepath, encoding='gb2312')
    table_kp = pd.read_csv(Kp_filepath, encoding='gb2312')
    table_ae = pd.read_csv(AE_filepath, encoding='gb2312').drop(['date'], axis=1)

    mit = table_mit["trough_min_lat"]

    table_kp.insert(5, "MIT_gdlat", mit)
    print(len(table_kp), len(table_ae), len(table_mit))
    df_rt = pd.merge(table_kp, table_ae, on=['year', 'doy', 'glon', 'lt'], how='outer')

    if len(df_rt) != len(table_kp):
        raise Exception("df merge Error: table lengths do not equal")

    return df_rt


if __name__ == '__main__':
    df = data_aggregation()
    print(len(df))
    df.to_csv("C:\\tmp\\bigtable.csv", index=False, encoding='gb2312')
