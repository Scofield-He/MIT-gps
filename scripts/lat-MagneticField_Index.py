#! python3
# -*- coding: utf-8 -*-

import os
# import gc
# import time
# import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12})


def lat_MagneticField_Index(df, lts, magnetic_indexes, y1, m1, d1, y2, m2, d2):
    df = df["{}-{}-{}".format(y1, m1, d1):"{}-{}-{}".format(y2, m2, d2)].dropna()
    for lt in lts:
        output_path = figure_path + 'lat-index_lt{}\\'.format(lt)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        df_lt = df.query("lt == {}".format(lt))
        for magnetic_index in magnetic_indexes:
            index = df_lt[magnetic_index]
            trough_min_lat = df_lt['trough_min_lat']
            corrcoef = round(index.corr(trough_min_lat), 2)      # 求相关系数

            # 统计每个点的个数
            points = list(zip(index, trough_min_lat))
            counts = []
            for _ in points:
                counts.append(points.count(_))

            # 作散点图
            figure = plt.figure(figsize=(6, 5))
            ax = figure.add_subplot(111)
            plt.sca(ax)
            plt.scatter(index, trough_min_lat, color='b', s=counts, label=None)
            # 做最小二乘线性拟合
            z = list(np.polyfit(np.array(index), np.array(trough_min_lat), 1))   # 斜率与截距
            print(glon, lt, magnetic_index, z)
            x = list(set(index))
            y = [z[0] * _ + z[1] for _ in x]
            plt.plot(x, y, 'r--', label='a={:.2f} b={:.2f} cc={}'.format(round(z[0], 2), round(z[1], 2), corrcoef))
            # figure参数设置
            plt.xlabel(magnetic_index + ' index')
            plt.ylabel('Trough_min gdlat(°)')
            plt.ylim()
            plt.legend(loc='upper right')
            title = '2014/9/1-2017/8/31 glon:-90 lt:{}'.format(lt)
            plt.title(title)
            # plt.show()
            figure.savefig(output_path + 'lt_{} {}'.format(lt, magnetic_index))
            plt.close()
    return True


def lat_4index_in_1figure(df, lt, y1, m1, d1, y2, m2, d2):
    df = df["{}-{}-{}".format(y1, m1, d1):"{}-{}-{}".format(y2, m2, d2)].dropna()
    df = df.query("lt == {}".format(lt))
    figure = plt.figure(figsize=(10, 8))
    indexes = ['AE', 'AE6', 'Kp', 'Kp9']
    for idx, MF_Index in zip(list(range(4)), indexes):
        ax = figure.add_subplot(221 + idx)
        plt.sca(ax)
        index, trough_min_lat = df[MF_Index], df['trough_min_lat']
        corrcoef = round(index.corr(trough_min_lat), 2)  # 求相关系数

        # 统计每个点的个数
        points = list(zip(index, trough_min_lat))
        counts = []
        for _ in points:
            counts.append(points.count(_))

        # 作散点图
        plt.scatter(index, trough_min_lat, color='b', s=counts, label=None)
        # 做最小二乘线性拟合
        z = list(np.polyfit(np.array(index), np.array(trough_min_lat), 1))  # 斜率与截距
        print(glon, lt, index, z)
        x = list(set(index))
        y = [z[0] * _ + z[1] for _ in x]
        plt.plot(x, y, 'r--', label='a={:.2f} b={:.2f} r={}'.format(round(z[0], 2), round(z[1], 2), corrcoef))
        # figure参数设置
        plt.xlabel(MF_Index + ' index')
        plt.ylabel('Trough_min gdlat(°)')
        # plt.ylim()
        plt.legend(loc='upper right')
    plt.gcf()
    title = '2014/9/1-2017/8/31 glon:-90 lt:{}'.format(lt)
    figure.suptitle(title)
    plt.subplots_adjust(top=0.93)
    # figure.tight_layout()
    figure.savefig(figure_path + 'lt-{}h'.format(lt))
    plt.show()
    return True


def plot_4ccOfIndex():
    data_path = 'C:\\tmp\\lat-MF_index corrcoef.csv'
    data = pd.read_excel(data_path).transpose()
    print(data)
    figure = plt.figure(figsize=(6, 4))
    ax = figure.add_subplot(111)
    plt.sca(ax)
    plt.plot(data["AE index"], 'b.-')
    plt.plot(data["AE6 index"], 'r.-')
    plt.plot(data['Kp index'], 'g.-')
    plt.plot(data["Kp9 index"], 'k.-')
    plt.ylim(-1, -0.3)
    plt.xlabel('LT')
    plt.ylabel('r')
    plt.title('Linear Correlation Coefficient')
    plt.legend()
    figure.savefig(figure_path + 'LCC of 4 indexes within LTs')
    plt.show()
    return True


bigtable_path = "C:\\tmp\\bigtable.csv"
bigtable = pd.read_csv(bigtable_path, encoding='gb2312')
bigtable["date"] = pd.to_datetime(bigtable["date"])      # 格式转换为datetime.date
bigtable = bigtable.set_index("date").sort_index()       # 按照时间排序
figure_path = "C:\\tmp\\figure\\lat-MagneticField_index\\"

if __name__ == '__main__':
    glon = -90
    bigtable = bigtable.query("glon == {}".format(glon))
    year0, month0, day0, year, month, day = 2014, 9, 1, 2017, 8, 31
    LocalTimes = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
    index_items = ['Kp', 'Kp9', 'AE', 'AE6', 'Cp', 'C9', 'sum_8kp', 'mean_8ap']
    # lat_MagneticField_Index(bigtable, LocalTimes, index_items, year0, month0, day0, year, month, day)
    # lat_4index_in_1figure(bigtable, 0, year0, month0, day0, year, month, day)
    plot_4ccOfIndex()
