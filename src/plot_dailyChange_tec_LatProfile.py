#! python3
# -*- coding: utf-8 -*-

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


def plot_daily_tec_profile(cur_doy, lats, tec, is_min, lat_min, min_tec, C9):
    print('plot tec profile  {}  --------------------'.format(cur_doy))
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    plt.sca(ax)

    plt.xlim(25, 85)
    # plt.ylim(0, 30)

    plt.plot(lats, tec, 'bo', label='TEC')                      # 蓝色○，每个纬度的TEC均值
    lat_max = max(lats)
    x = list(range(30, lat_max))
    y = interpolate.UnivariateSpline(lats, tec, s=5)(x)         # 插值
    plt.plot(x, y, 'k--', label='interpolate profile')

    if is_min:                                                  # 这天被判定为槽发生，且记录了槽极小的位置
        plt.plot(lat_min, min_tec, 'r^', label='trough mini')
    title = 'mean TEC lat profile  \n lt:{}-{}  glon:{}-{}  year:{}  doy:{:3d}  C9 index:{}'.\
        format(lt1, lt2, glon1, glon2, year, cur_doy, C9)
    plt.title(title)
    plt.legend()

    savefig_path = out_path + 'TEC profiles\\'
    if not os.path.exists(savefig_path):
        os.mkdir(savefig_path)
    fig.savefig(savefig_path + '{:3d}'.format(cur_doy))
    plt.close()
    return True


def get_profiles_one_year(file_path, list_C9):
    """
    """
    """
    error_days_2013 = [_ for _ in range(214, 218)] + [_ for _ in range(309, 317)] + \
                      [_ for _ in range(326, 339)] + []
    error_days_2014 = [_ for _ in range(61, 68)] + [_ for _ in range(87, 237)]
    if year == 2014:
        error_days = error_days_2014
    elif year == 2013:
        error_days = error_days_2013
    else:
        error_days = []
    """
    doy, glat, tec = [], [], []
    xdoy, position = [], []
    error_days = []
    with open(file_path, 'r') as fi:
        for line in fi:
            items = line.strip().split(';')
            if len(items) == 3:                                     # doy, glat, mean_tec
                cur_doy = int(items[0])
                lats = items[1].split(' ')
                values = items[2].split(' ')

                if values:
                    lats = [int(_) for _ in lats if _]
                    values = [float(v) for v in values if v]

                    # for i, j in zip(lats, values):
                    #     pass

                    low_lat_limit, high_lat_limit = 45, 70
                    while high_lat_limit not in lats:                                        # 高纬区域有时没有数据
                        high_lat_limit -= 1
                    else:
                        index_65 = lats.index(high_lat_limit)
                        index_45 = lats.index(low_lat_limit)
                    lats_cur, values_cur = lats[index_45: index_65], values[index_45: index_65]
                    mean_tec, min_tec = np.mean(values_cur), np.min(values_cur)
                    is_min, lat_min = 0, 0
                    if cur_doy not in error_days and min_tec < mean_tec * 0.8:
                        lat_min = lats_cur[values_cur.index(min_tec)]
                        if low_lat_limit + 1 < lat_min < high_lat_limit - 1:  # 边界
                            is_min = 1
                            xdoy.append(cur_doy)
                            position.append(lat_min)
                    else:
                        pass

                    # 作图画出每天lats-tec_values
                    C9 = list_C9[cur_doy - 1]
                    plot_daily_tec_profile(cur_doy, lats, values, is_min, lat_min, min_tec, C9)

                    maxi = max(values)
                    # print("values: ", maxi, values)
                    values = [float(20 * v / maxi) for v in values if v]                       # 归一化

                for lat in lats:
                    if lat:
                        doy.append(cur_doy)
                        glat.append(lat)
                for _ in values:
                    tec.append(float(_))
                    # print('tec value count for one doy: {}'.format(cnt))
    print('\n', len(doy), len(glat), len(tec))
    print("count of days that occur trough: {}".format(len(xdoy)))
    print("trough mini doy: {}".format(xdoy))
    print("trough mini loc: {}".format(position))
    return doy, glat, tec, xdoy, position


def read_kp_index(glon):
    file_path = "C:\\DATA\\index\\Kp-ap-Flux_01-17.dat"
    doy, c9, Kp = [], [], []
    day_diff, ut = divmod(lt1 - glon // 15, 24)
    begin_digit = 12 + 2 * int(ut // 3)
    print("digit: ", begin_digit)
    end_digit = begin_digit + 2

    with open(file_path, 'rb') as file:
        for line in file:
            line_data = line.strip().decode('utf-8')
            # print(line_data)
            # print((line_data[:2]))
            if line_data[:2] == str(year)[-2:]:
                dat = datetime.date(year, int(line_data[2:4]), int(line_data[4:6]))
                doy.append(dat.timetuple().tm_yday)                  # 根据年月日计算doy
                c9.append(int(line_data[61]))                        # 第61位，当日的C9指数值，0-9
                Kp.append(int(line_data[begin_digit:end_digit]))     # 3 hours: 00-03ut对应12-13位，03-06对应14-15位

    return doy, c9, Kp


def read_ae_index(glon):
    file_path = "C:\\DATA\\index\\AE index_0101-1706_WDC-like format.txt"
    doy, ae, ae6 = [], [], []
    day_diff, ut = divmod(lt1 - glon // 15, 24)

    div = 0
    for _ in range(7):                                                       # 计算ae6的负e指数权重
        div += np.e ** (-_)

    if ut >= 6:                                                              # ae6中7个数在同一行
        with open(file_path, 'rb') as file:
            for line in file:
                line = line.strip().decode('utf-8')
                if line[:2] == 'AE' and line[14:16] + line[3:5] == str(year):
                    dat = datetime.date(year, int(line[5:7]), int(line[8:10]))
                    doy.append(dat.timetuple().tm_yday)                       # 根据日期计算doy

                    ae_index = []
                    begin_col_index = 20 + ut * 4
                    for _ in range(7):
                        ae_index.append(int(line[begin_col_index: begin_col_index + 4]))
                        begin_col_index -= 4
                    # print('count of ae_index', len(ae_index))               # 1 + 6 = 7个时刻
                    ae.append(ae_index[0])                                    # 取出第一个数，为当前时刻的ae指数
                    ae6_index = 0                                             # 计算ae6 index的值
                    for i in range(7):
                        ae6_index += ae_index[i] * np.e ** (-i)
                    ae6.append(round(ae6_index / div, 1))
    else:
        with open(file_path, 'rb') as file:
            last_index_string = ''
            for line in file:
                line = line.strip().decode('utf-8')
                if line[:2] == 'AE' and line[14:16] + line[3:5] == str(year - 1):
                    if line[5:7] == '12' and line[8:10] == '31':
                        last_index_string += line[88:116]                      # 取ut倒数7个时刻的ae指数
                elif line[:2] == 'AE' and line[14:16] + line[3:5] == str(year):
                    dat = datetime.date(year, int(line[5:7]), int(line[8:10]))
                    doy.append(dat.timetuple().tm_yday)                        # 根据日期计算doy
                    ae_index = []
                    # print("doy:{}, last_index_string: {}".format(doy[-1], last_index_string))
                    """
                    for _ in last_index_string.strip().split(' '):
                        if _:
                            ae_index.append(int(_))         
                    """
                    i = 0
                    while i <= 24:
                        ae_index.append(int(last_index_string[i: i+4]))
                        i += 4
                    begin_col_index, end_col_index = 20, 20 + ut * 4
                    while begin_col_index <= end_col_index:
                        ae_index.append(int(line[begin_col_index: begin_col_index + 4]))
                        begin_col_index += 4

                    ae_index = ae_index[-7:]
                    # print("doy: {}, len of ae_index: {}".format(doy[-1], len(ae_index)))
                    ae.append(ae_index[-1])
                    ae6_index = 0                                              # 计算ae6 index的值
                    for i in range(7):
                        ae6_index += ae_index[i] * np.e ** (i - 6)
                    ae6.append(round(ae6_index / div, 1))
                    last_index_string = line[88:116]

    print('count of ae index of {}: '.format(year), len(ae))
    return doy, ae, ae6


def plot_tec_profile_with_mag_index():

    def plot_fig1_profile():
        datapath = "C:\\code\\MIT-gps\\resources\\{}\\{}\\{}_{}-{}_{}-{}.txt".format(year, site, year, lt1, lt2,
                                                                                     glon1, glon2)

        day_of_year, C9_index, Kp_index = read_kp_index(glon_center)

        fig1 = plt.figure(figsize=(30, 6))  # plot figure1
        # plt.subplots_adjust(hspace=0)
        ax1 = plt.subplot(211)
        plt.sca(ax1)  # plot ax1 in fig1
        x, y, color, x_doy, loc_mini = get_profiles_one_year(datapath, C9_index)
        doy1, doy2 = 0, 366
        x = [_ for _ in x if doy1 < _ < doy2]
        y, color = y[:len(x)], color[:len(x)]
        plt.scatter(x, y, c=color, vmin=0, vmax=20, marker='s', s=8, alpha=1, cmap=plt.cm.jet)
        plt.plot(x_doy, loc_mini, 'r')

        # print("count of doy: {}, loc_mini: {}".format(len(x_doy), len(loc_mini)))
        plt.ylabel('gdlat')
        axis = plt.gca().xaxis
        for label in axis.get_ticklabels():
            label.set_rotation(45)
            label.set_fontsize(8)

        plt.title('{} TEC profile variance between lt {}-{} on glon {}-{}'.format(year, lt1, lt2, glon1, glon2))
        plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)  # color bar
        cax = plt.axes([0.92, 0.54, 0.01, 0.35])
        plt.colorbar(cax=cax).set_label('TEC/TECU')

        ax2 = plt.subplot(212, sharex=ax1)
        width = 1
        plt.sca(ax2)  # plot ax2 in fig1
        # print("len of x_doy:{}".format(len(x_doy)))
        for i in range(1, 366):
            if i not in x_doy:
                plt.bar(i, 10, width, color='0.8')
        plt.yticks(())

        ax4 = ax2.twinx()
        plt.sca(ax4)
        C9_index = [_ for _ in C9_index]
        # plt.bar(day_of_year, Kp_index, width, color='b', label='Kp index')
        plt.bar(day_of_year, C9_index, width, color='b', label='C9 index')
        # day_of_year = [_ + 0.5 for _ in day_of_year]
        # plt.plot(day_of_year, C9_index, '.', color='r', label='C9 index')
        plt.ylabel('C9 index')
        plt.xlabel('doy')
        # plt.xlim(-1, 367)
        plt.ylim(0, 9)
        plt.legend(bbox_to_anchor=(0.976, 0.8))

        ax3 = ax2.twinx()
        ax3.plot(x_doy, loc_mini, 'r', label='trough min lat')
        ax3.set_ylabel("lat of trough mini")
        ax3.set_ylim(40, 65)
        # ax3.set_ylim(30, 60)
        plt.legend(bbox_to_anchor=(0.993, 0.95))

        new_ticks = list(map(int, np.linspace(doy1, doy2, int((doy2 - doy1) / 7))))
        # print(new_ticks)
        plt.xticks(new_ticks)
        axis = plt.gca().xaxis
        for label in axis.get_ticklabels():
            label.set_rotation(45)
            label.set_fontsize(8)

        fig1.savefig(out_path + '{}-tec profile and Kp index daily variance between LT {}-{} and longitude {}-{} '
                                'regular'.format(year, lt1, lt2, glon1, glon2), dpi=500)
        plt.close()
        return x_doy, Kp_index, C9_index, loc_mini

    doy, Kp, C9, trough_mini = plot_fig1_profile()
    print("count of x_doy: {}, count of Kp: {}, count of C9: {}, count of trough_mini: {}".
          format(len(doy), len(Kp), len(C9), len(trough_mini)))

    def plot_fig2_k(x_doy, k_index, loc_mini):

        if k_index == Kp:
            x_label = 'Kp index'
            mag_index = [k_index[_ - 1] / 10 for _ in x_doy]
        else:
            x_label = 'C9 index'
            mag_index = [k_index[_ - 1] for _ in x_doy]

        fig2 = plt.figure(figsize=(15, 6))                         # figure2
        ax1 = fig2.add_subplot(121)                                # ax1 in fig2

        lat_index = list(zip(mag_index, loc_mini))
        count_of_point = []
        for _ in lat_index:
            count_of_point.append(lat_index.count(_))
        plt.sca(ax1)
        plt.scatter(mag_index, loc_mini, color='b', s=[3 * _ for _ in count_of_point])

        coe = np.corrcoef(np.array([mag_index, loc_mini]))        # 线性回归，相关系数
        corrcoef = round(coe[0][1], 2)
        print('corrcoef: ', corrcoef)

        z1 = list(np.polyfit(mag_index, loc_mini, 1))             # 线性拟合， 即一阶多项式拟合
        print("z1: ", z1)
        x = list(set(mag_index))
        y = [z1[0] * _ + z1[1] for _ in x]
        plt.plot(x, y, 'r--', label='{:.2f} * x + {:.2f}'.format(round(z1[0], 2), round(z1[1], 2)))

        plt.text(0.68, 0.88, 'r= {} count: {}'.format(corrcoef, len(loc_mini)),
                 transform=ax1.transAxes, color="black")
        plt.xlabel(x_label)
        plt.ylabel("Geographic Latitude")
        plt.ylim(40, 65)
        # plt.ylim(30, 60)
        plt.xlim(-1, 9)
        plt.legend()
        plt.title("trough mini lat - {} - {} lt:{:2d}-{:2d}".format(year, x_label, lt1, lt2))

        ax2 = fig2.add_subplot(122)                                # ax2 in fig2
        plt.sca(ax2)

        data_dict, para_dict = {}, {}
        for i, j in zip(mag_index, loc_mini):
            if i not in data_dict.keys():
                data_dict[i] = []
                para_dict[i] = []
                data_dict[i].append(j)
            else:
                data_dict[i].append(j)

        print("keys:", data_dict.keys())
        print(data_dict)

        for _ in para_dict.keys():                             # 求同一Kp指数下槽极小位置的中值与上下四分位数
            para_dict[_] = [np.median(data_dict[_]), np.percentile(data_dict[_], 25), np.percentile(data_dict[_], 75)]
        para_dict = {k: para_dict[k] for k in sorted(para_dict.keys())}  # 按照key值大小，即Kp指数大小排序

        x1 = [_ for _ in para_dict.keys()]
        x2 = np.array([para_dict[_][0] for _ in para_dict.keys()])
        percentile_25 = np.array([para_dict[_][0] - para_dict[_][1] for _ in para_dict.keys()])
        percentile_75 = np.array([para_dict[_][2] - para_dict[_][0] for _ in para_dict.keys()])
        # plt.errorbar(x1, x2, yerr=[percentile_25, percentile_75], fmt='.-')
        for _ in range(len(x1)):
            plt.errorbar(x1[_], x2[_], yerr=[[percentile_25[_]], [percentile_75[_]]], fmt='.-', color='b')
        plt.xlabel(x_label)
        plt.ylabel("Geographic Latitude")
        plt.ylim(40, 65)
        # plt.ylim(30, 60)
        plt.xlim(-1, 9)

        # 对中值做线性拟合
        z = np.polyfit(x1, x2, 1)
        print("z: ", z)
        p1 = np.poly1d(z)
        print("p1: ", p1)
        y = [z[0] * _ + z[1] for _ in x1]
        z = [round(_, 2) for _ in z]
        plt.plot(x1, y, 'r--', label='{:.2f} * x + {:.2f}'.format(z[0], z[1]))

        r = round(np.corrcoef(np.array([x1, x2]))[0][1], 2)
        plt.text(0.68, 0.88, 'r= {}'.format(r), transform=ax2.transAxes, color="black")
        plt.legend()
        plt.title("median of trough mini lat - {} - lt:{:2d}-{:2d}".format(x_label, lt1, lt2))
        return fig2

    fig_name_kp = 'min loc_Kp index LT_{}-{} glon_{}-{}_regular'.format(year, lt1, lt2, glon1, glon2)
    print('plot trough mini loc_Kp:  {}  --------------------'.format(year))
    fig_kp = plot_fig2_k(doy, Kp, trough_mini)
    fig_kp.savefig(out_path + fig_name_kp)
    plt.close()

    fig_name_c9 = 'min loc_C9 index LT_{}-{} glon_{}-{}_regular'.format(year, lt1, lt2, glon1, glon2)
    print('plot trough mini loc_C9:  {}  --------------------'.format(year))
    fig_c9 = plot_fig2_k(doy, C9, trough_mini)
    fig_c9.savefig(out_path + fig_name_c9)
    plt.close()

    def plot_fig3_ae(x_doy, ae_index, ae6_index, loc_mini):
        ae_index = [ae_index[_ - 1] for _ in x_doy]
        # print("count of ae index:", len(ae_index))

        fig3 = plt.figure(figsize=(15, 6))                                 # figure3
        ax1 = fig3.add_subplot(121)                                        # ax1 in fig3

        lat_index = list(zip(ae_index, loc_mini))
        count_of_point = []
        for _ in lat_index:
            count_of_point.append(lat_index.count(_))
        plt.sca(ax1)
        plt.scatter(ae_index, loc_mini, color='b', s=[3 * _ for _ in count_of_point])

        coe = np.corrcoef(np.array([ae_index, loc_mini]))                  # 线性回归，相关系数
        corrcoef = round(coe[0][1], 2)
        print('AE corrcoef: ', corrcoef)

        z1 = list(np.polyfit(ae_index, loc_mini, 1))
        print("z1: ", z1)
        x = list(set(ae_index))
        y = [z1[0] * _ + z1[1] for _ in x]
        plt.plot(x, y, 'r--', label='{:.4f} * x + {}'.format(round(z1[0], 4), round(z1[1], 2)))
        plt.text(0.65, 0.88, 'r= {} count: {}'.format(corrcoef, len(loc_mini)),
                 transform=ax1.transAxes, color="black")
        plt.xlabel("AE_index")
        plt.ylabel("Geographic Latitude")
        plt.ylim(40, 65)
        # plt.ylim(30, 60)
        plt.xlim(0, 1000)
        plt.legend()
        plt.title("trough mini lat - AE index - lt:{:2d}-{:2d}".format(lt1, lt2))

        ax2 = fig3.add_subplot(122)                                      # ax2 in fig3
        plt.sca(ax2)
        ae6_index = [ae6_index[_ - 1] for _ in x_doy]
        lat_index = list(zip(ae6_index, loc_mini))
        count_of_point = []
        for _ in lat_index:
            count_of_point.append(lat_index.count(_))
        plt.scatter(ae6_index, loc_mini, color='b', s=[3 * _ for _ in count_of_point])

        coe = np.corrcoef(np.array([ae6_index, loc_mini]))  # 线性回归，相关系数
        corrcoef = round(coe[0][1], 2)
        print('AE6 corrcoef: ', corrcoef)

        z1 = list(np.polyfit(ae6_index, loc_mini, 1))
        print("z1: ", z1)
        x = list(set(ae6_index))
        y = [z1[0] * _ + z1[1] for _ in x]
        plt.plot(x, y, 'r--', label='{:.4f} * x + {}'.format(round(z1[0], 4), round(z1[1], 2)))
        plt.text(0.65, 0.88, 'r= {} count: {}'.format(corrcoef, len(loc_mini)),
                 transform=ax2.transAxes, color="black")
        plt.xlabel("AE_index")
        plt.ylabel("Geographic Latitude")
        plt.ylim(40, 65)
        plt.xlim(0, 1000)
        plt.legend()
        plt.title("trough mini lat - AE6 index - lt:{:2d}-{:2d}".format(lt1, lt2))
        return fig3

    fig_name_AE = 'min loc_AE-AE6 index LT_{}-{} glon_{}-{}_regular'.format(year, lt1, lt2, glon1, glon2)
    _, ae, ae6 = read_ae_index(glon_center)
    print('plot trough mini loc_Kp:  {}  --------------------'.format(year))
    fig_ae = plot_fig3_ae(doy, ae, ae6, trough_mini)
    fig_ae.savefig(out_path + fig_name_AE)
    plt.close()

    return True


year, site = 2016, 'millstone'
list_lt = [22, 23, 0, 1, 2, 3, 4, 5, 18, 19, 20, 21]              # 18, 19, 20, ……, 4, 5
# list_lt = [2, 3, 4, 5]
list_lon = [-95, -5, 25, -125]
# list_lon = [25]
for glon1 in list_lon:
    glon2 = glon1 + 10
    glon_center = (glon1 + glon2) // 2

    for lt1 in list_lt:
        lt2 = lt1 + 1
        out_path = 'C:\\code\\MIT-gps\\resources\\{}\\{}\\{}-{}\\glon_mid_{}\\'.format(year, site, lt1, lt2,
                                                                                       glon_center)
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        plot_tec_profile_with_mag_index()

# plot_tec_profile_with_mag_index(2013, 23, 1, -125, -115)
# plot_tec_profile_with_mag_index(2014, 23, 0, -125, -115)
# a, b, c = read_ae_index(2013, 0, -120)
