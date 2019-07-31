import time
import numpy as np
import sys
import datetime
import pandas as pd
import os
from Config import config

# 用字典查询代替类型转换，可以减少一部分计算时间
date2position = {}
datestr2dateint = {}
str2int = {}
date2int = {}
idx2date = {}
for i in range(24):
    str2int[str(i).zfill(2)] = i


# 访问记录内的时间从2018年10月1日起，共182天
# 将日期按日历排列
for i in range(182):
    date = datetime.date(day=1, month=10, year=2018) + datetime.timedelta(days=i)
    date_int = int(date.__str__().replace("-", ""))
    date2position[date_int] = [i % 7, i // 7]
    date2int[str(date_int)] = i
    datestr2dateint[str(date_int)] = date_int
    idx2date[i] = str(date_int)


def get_statistic_variable(tmp):
    tmp = np.array(tmp).flatten()
    if len(tmp) > 0:
        return [np.sum(tmp), tmp.mean(), tmp.std(), tmp.max(), tmp.min()] + list(
            np.percentile(tmp, [25, 50, 75]))  # shape = (8, )
    else:
        return list(np.zeros((8,)) - 0)


def relative_ratio(A, B, ep=1):
    return list((np.array(A) + ep) / (np.array(B) + ep))

def judge_date(date):
    now = datestr2dateint[date]
    if now == 20181111:  # 光棍节
        return 1
    elif now == 20181225:  # 圣诞节
        return 2
    elif now == 20190214:  # 情人节
        return 3
    elif now == 20181229:  # 周六调休（元旦）
        return 4
    elif now < 20181008:  # 国庆
        return 5
    elif now >= 20181230 and now <= 20190101:  # 元旦放假
        return 6
    elif  now >= 20190128  and now <= 20190203:  # 小年 -- 除夕前一天
        return 7
    elif now >= 20190204  and now <= 20190210:  # 除夕 -- 初6
        return 8
    elif now >= 20190211 and now <= 20190219:  # 初7 -- 元宵
        return 9
    else:
        return 0


def get_24_ratio(tmp):
    res = []
    for i in range(24):
        res.append((tmp[i] + 1) / (tmp[(i + 1) % 24] + 1))
    return res


def get_fine_feature_cishu(data):
    dim_last = data.shape[-1]
    data_transfrom = data.reshape((7 * 26, -1))
    feature = []
    
    feature += [data_transfrom.mean(), data_transfrom.std(), data_transfrom.max(),
                np.argmax(data_transfrom) // dim_last, np.argmax(data_transfrom) % dim_last]
    feature += list(np.percentile(data_transfrom, [_ * 10 for _ in range(1, 10)]))

    tmp_12 = [[] for _ in range(14)]
    for i in range(182):
        date = idx2date[i]
        day_type = judge_date(date)
        x, y = i % 7, i // 7
        tmp_12[day_type].append(data_transfrom[i, :])
        if day_type == 0:
            if x < 5:
                tmp_12[10].append(data_transfrom[i, :])
            else:
                tmp_12[11].append(data_transfrom[i, :])
                if x == 5:
                    tmp_12[12].append(data_transfrom[i, :])
                else:
                    tmp_12[13].append(data_transfrom[i, :])

    cmp_sun_mean_day = np.sum(tmp_12[13]) / len(tmp_12[13])   # 平均周日的人数
    cmp_sat_mean_day = np.sum(tmp_12[12]) / len(tmp_12[12])  # 平均周六的人数
    cmp_work_mean_day = np.sum(tmp_12[10]) / len(tmp_12[10])
    for day_type, tmp in enumerate(tmp_12):
        ttt = np.array(tmp)
        if day_type >= 1 and day_type <= 4:
            tt = ttt.flatten()
            if day_type == 4:
                feature += [np.sum(tt), tt.std(), tt.max(), np.argmax(tt), (np.sum(tt) + 1) / (cmp_sat_mean_day + 1)]
            else:
                feature += [np.sum(tt), tt.std(), tt.max(), np.argmax(tt), (np.sum(tt) + 1) / (cmp_work_mean_day + 1)]
        else:
            tt_day_mean = np.mean(ttt, axis=1)
            tt_day_std = np.std(ttt, axis=1)

            feature += [np.mean(tt_day_mean), tt_day_mean.std(), tt_day_mean.max(), np.argmax(tt_day_mean)]
            feature += [np.mean(tt_day_std), tt_day_std.std(), tt_day_std.max(), np.argmax(tt_day_std)]
            if  day_type >= 5 and day_type <= 9:
                feature += list(tt_day_mean) + list(tt_day_std)

            tt = np.mean(ttt, axis=0)
            tt_std = np.std(ttt, axis=0)

            feature += [np.sum(tt), tt.std(), tt.max(), np.argmax(tt),
                        (np.sum(tt) + 1) / (cmp_work_mean_day + 1)]
            feature += [np.sum(tt_std), tt_std.std(), tt_std.max(), np.argmax(tt_std),
                        (np.sum(tt_std) + 1) / (cmp_work_mean_day + 1)]
            if dim_last == 24:
                feature += list(tt_std)

        if dim_last == 24:
            feature += list(tt) + get_24_ratio(tt)
    
    return feature

def get_feature_cishu(data):
    feature = []
    feature += get_fine_feature_cishu(data[:, :, :8])
    feature += get_fine_feature_cishu(data[:, :, 8:18])
    feature += get_fine_feature_cishu(data[:, :, 18:])
    feature += get_fine_feature_cishu(data)
    return feature

def get_feature_reshu(data):
    assert data.shape == (7, 26)

    data = data.flatten()
    feature = []
    feature += [data.mean(), data.std(), data.max(), np.argmax(data)]
    feature += list(np.percentile(data, [_ * 10 for _ in range(1, 10)]))
    feature += list(data)

    tmp_12 = [[] for _ in range(12)]
    for i in range(182):
        date = idx2date[i]
        day_type = judge_date(date)
        x, y = i % 7, i // 7
        tmp_12[day_type].append(data[i])
        if day_type == 0:
            if x < 5:
                tmp_12[10].append(data[i])
            else:
                tmp_12[11].append(data[i])

    cmp_mean = np.mean(tmp_12[10])
    for day_type, tmp in enumerate(tmp_12):
        ttt = np.array(tmp)
        if day_type >= 1 and day_type <= 4:
            feature += [(ttt[0] + 1) / (cmp_mean + 1)]
        else:
            feature += [np.mean(ttt), ttt.std(), ttt.max(), np.argmax(ttt), (np.mean(ttt) + 1) / (cmp_mean + 1)]
            if day_type >= 5 and day_type <= 9:
                pass
            else:
                feature += list(np.percentile(data, [_ * 15 for _ in range(1, 6)]))

    return feature

def get_jieri_feature(strings, jieri_dict, name_jieri='guoqing'):
    num_jieri = len(jieri_dict)

    FEATURE = []

    zaoshang_hour_dao = [[] for _ in range(num_jieri)]
    wanshang_hour_zou = [[] for _ in range(num_jieri)]
    zaowanshang_hour_daozou = [[] for _ in range(num_jieri)]
    kuadu = [[] for _ in range(num_jieri)]

    for user_idx, string in enumerate(strings):
        temp = [[item[0:8], item[9:].split("|")] for item in string.split(',')]

        for date, visit_lst in temp:
            if date not in jieri_dict.keys():  # 不再节日范围内
                continue
            idx = jieri_dict[date]  # 10月1日到8日
            zaoqi_hour = str2int[visit_lst[0]]
            wanshang_hour = str2int[visit_lst[-1]]
            zaoshang_hour_dao[idx].append(zaoqi_hour)
            wanshang_hour_zou[idx].append(wanshang_hour)
            zaowanshang_hour_daozou.append(zaoqi_hour * 24 + wanshang_hour)
            kuadu.append(wanshang_hour - zaoqi_hour)

    for idx in range(num_jieri):
        FEATURE += get_statistic_variable(zaoshang_hour_dao[idx])
        FEATURE += get_statistic_variable(wanshang_hour_zou[idx])
        FEATURE += get_statistic_variable(zaowanshang_hour_daozou[idx])
        FEATURE += get_statistic_variable(kuadu[idx])

    if name_jieri == 'guoqing':
        tmp1, tmp2, tmp3, tmp4 = [], [], [], []
        for idx in range(5):
            tmp1 += zaoshang_hour_dao[idx]
            tmp2 += wanshang_hour_zou[idx]
            tmp3 += zaowanshang_hour_daozou[idx]
            tmp4 += kuadu[idx]
        FEATURE += get_statistic_variable(tmp1) + get_statistic_variable(tmp2) \
                   + get_statistic_variable(tmp3) + get_statistic_variable(tmp4)

        tmp1, tmp2, tmp3, tmp4 = [], [], [], []
        for idx in range(5, 7):
            tmp1 += zaoshang_hour_dao[idx]
            tmp2 += wanshang_hour_zou[idx]
            tmp3 += zaowanshang_hour_daozou[idx]
            tmp4 += kuadu[idx]
        FEATURE += get_statistic_variable(tmp1) + get_statistic_variable(tmp2) \
                   + get_statistic_variable(tmp3) + get_statistic_variable(tmp4)

    if name_jieri == 'guonian_chunyun':
        tmp1, tmp2, tmp3, tmp4 = [], [], [], []
        for idx in range(7):  # 20190121 - 20190127 春运
            tmp1 += zaoshang_hour_dao[idx]
            tmp2 += wanshang_hour_zou[idx]
            tmp3 += zaowanshang_hour_daozou[idx]
            tmp4 += kuadu[idx]
        FEATURE += get_statistic_variable(tmp1) + get_statistic_variable(tmp2) \
                    + get_statistic_variable(tmp3) + get_statistic_variable(tmp4)

        tmp1, tmp2, tmp3, tmp4 = [], [], [], []
        for idx in range(7, 15): # 小年到除夕
            tmp1 += zaoshang_hour_dao[idx]
            tmp2 += wanshang_hour_zou[idx]
            tmp3 += zaowanshang_hour_daozou[idx]
            tmp4 += kuadu[idx]
        FEATURE += get_statistic_variable(tmp1) + get_statistic_variable(tmp2) \
                   + get_statistic_variable(tmp3) + get_statistic_variable(tmp4)

        tmp1, tmp2, tmp3, tmp4 = [], [], [], []
        for idx in range(15, 21):  # 初6 到 元宵
            tmp1 += zaoshang_hour_dao[idx]
            tmp2 += wanshang_hour_zou[idx]
            tmp3 += zaowanshang_hour_daozou[idx]
            tmp4 += kuadu[idx]
        FEATURE += get_statistic_variable(tmp1) + get_statistic_variable(tmp2) \
                    + get_statistic_variable(tmp3) + get_statistic_variable(tmp4)

        tmp1, tmp2, tmp3, tmp4 = [], [], [], []
        for idx in range(21, 39):  # 初6 到 元宵
            tmp1 += zaoshang_hour_dao[idx]
            tmp2 += wanshang_hour_zou[idx]
            tmp3 += zaowanshang_hour_daozou[idx]
            tmp4 += kuadu[idx]
        FEATURE += get_statistic_variable(tmp1) + get_statistic_variable(tmp2) \
                   + get_statistic_variable(tmp3) + get_statistic_variable(tmp4)

    if name_jieri == 'yuandan':
        tmp1, tmp2, tmp3, tmp4 = [], [], [], []
        for idx in range(1, num_jieri):   #元旦放假期间
            tmp1 += zaoshang_hour_dao[idx]
            tmp2 += wanshang_hour_zou[idx]
            tmp3 += zaowanshang_hour_daozou[idx]
            tmp4 += kuadu[idx]
        FEATURE += get_statistic_variable(tmp1) + get_statistic_variable(tmp2) \
                   + get_statistic_variable(tmp3) + get_statistic_variable(tmp4)

    return FEATURE

def get_work_rest_feature(strings):
    # work day rest day satedat sunday
    FEATURE = []

    zaoshang_hour_workday_dao = []
    wanshang_hour_workday_zou = []
    zaowanshang_hour_workday_daozou = []
    work_day_kuadu = []

    zaoshang_hour_restday_dao = []
    wanshang_hour_restday_zou = []
    zaowanshang_hour_restday_daozou = []
    rest_day_kuadu = []

    zaoshang_hour_restday_dao_sat = []
    wanshang_hour_restday_zou_sat = []
    zaowanshang_hour_restday_daozou_sat = []
    sat_day_kuadu = []

    zaoshang_hour_restday_dao_sun = []
    wanshang_hour_restday_zou_sun = []
    zaowanshang_hour_restday_daozou_sun = []
    sun_day_kuadu = []

    for user_idx, string in enumerate(strings):
        temp = [[item[0:8], item[9:].split("|")] for item in string.split(',')]

        for date, visit_lst in temp:
            x, y = date2position[datestr2dateint[date]]
            zaoqi_hour = str2int[visit_lst[0]]
            wanshang_hour = str2int[visit_lst[-1]]

            if x < 5:  # workday
                zaoshang_hour_workday_dao.append(zaoqi_hour)
                wanshang_hour_workday_zou.append(wanshang_hour)
                zaowanshang_hour_workday_daozou.append(zaoqi_hour * 24 + wanshang_hour)
                work_day_kuadu.append(wanshang_hour - zaoqi_hour)
            if x >= 5:
                zaoshang_hour_restday_dao.append(zaoqi_hour)
                wanshang_hour_restday_zou.append(wanshang_hour)
                zaowanshang_hour_restday_daozou.append(zaoqi_hour * 24 + wanshang_hour)
                rest_day_kuadu.append(wanshang_hour - zaoqi_hour)
                if x == 5:
                    zaoshang_hour_restday_dao_sat.append(zaoqi_hour)
                    wanshang_hour_restday_zou_sat.append(wanshang_hour)
                    zaowanshang_hour_restday_daozou_sat.append(zaoqi_hour * 24 + wanshang_hour)
                    sat_day_kuadu.append(wanshang_hour - zaoqi_hour)
                else:
                    zaoshang_hour_restday_dao_sun.append(zaoqi_hour)
                    wanshang_hour_restday_zou_sun.append(wanshang_hour)
                    zaowanshang_hour_restday_daozou_sun.append(zaoqi_hour * 24 + wanshang_hour)
                    sun_day_kuadu.append(wanshang_hour - zaoqi_hour)

    for tmp in [zaoshang_hour_workday_dao, wanshang_hour_workday_zou, zaowanshang_hour_workday_daozou, work_day_kuadu,
                zaoshang_hour_restday_dao, wanshang_hour_restday_zou, zaowanshang_hour_restday_daozou, rest_day_kuadu,
                zaoshang_hour_restday_dao_sat, wanshang_hour_restday_zou_sat, zaowanshang_hour_restday_daozou_sat, sat_day_kuadu,
                zaoshang_hour_restday_dao_sun, wanshang_hour_restday_zou_sun, zaowanshang_hour_restday_daozou_sun, sun_day_kuadu]:
        FEATURE += get_statistic_variable(tmp)

    return FEATURE

def get_f_1(visit_lst):
    zaoqi_hour = str2int[visit_lst[0]]
    wanshang_hour = str2int[visit_lst[-1]]
    kuadu = wanshang_hour - zaoqi_hour
    work_hour = len(visit_lst)
    if len(visit_lst) == 1:
        max_jiange = 0.01
    else:
        max_jiange = np.max([str2int[visit_lst[i + 1]] - str2int[visit_lst[i]] for i in range(len(visit_lst) - 1)])

    return [zaoqi_hour, wanshang_hour, kuadu, work_hour, max_jiange]

def get_feature(table):

    feature = []
    init_cishu = np.zeros((7, 26, 24))  # 统计26周每天每小时的用户人数
    init_renshu = np.zeros((7, 26))  #
    
    strings = table[1]
    Num_users = len(strings)  # 统计用户人数
    feature += [Num_users]

    f_n_user = []
    for user_idx, string in enumerate(strings):
        temp = [[item[0:8], item[9:].split("|")] for item in string.split(',')]
        for date, visit_lst in temp:
            x, y = date2position[datestr2dateint[date]]
            init_renshu[x][y] += 1  # 统计每小时的到访的总人数   7 * 26
            for visit in visit_lst:
                init_cishu[x][y][str2int[visit]] += 1  # 统计每天到访的总人数 7 * 26 * 24

        if len(temp) < 10:
            continue

        f_now = [[] for _ in range(14)]
        f_now_24 = [[] for _ in range(14)]
        all_f_1 = []
        all_f_24 = []
        for date, visit_lst in temp:
            day_type = judge_date(date)
            x, y = date2position[datestr2dateint[date]]
            f_1 = get_f_1(visit_lst)
            f_24 = np.zeros((24, ))
            for visit in visit_lst:
                f_24[str2int[visit]] += 1

            f_now[day_type].append(f_1)
            f_now_24[day_type].append(f_24)
            all_f_1.append(f_1)
            all_f_24.append(f_24)
            if day_type == 0:
                if x < 5:
                    f_now[10].append(f_1)
                    f_now_24[10].append(f_24)
                else:
                    f_now[11].append(f_1)
                    f_now_24[11].append(f_24)
                    if x == 5:
                        f_now[12].append(f_1)
                        f_now_24[12].append(f_24)
                    else:
                        f_now[13].append(f_1)
                        f_now_24[13].append(f_24)
        all_f_1 = np.array(all_f_1)
        all_f_24 = np.array(all_f_24)
        cmp_1_mean = np.mean(all_f_1, axis=0)
        cmp_24_mean = np.mean(all_f_24, axis=0)
        feature_now = []
        for day_type, tmp in enumerate(f_now):
            tmp_24 = f_now_24[day_type]
            if day_type >= 1 and day_type <= 4:
                if len(tmp) == 0:
                    feature_now += [0, 0, 0, 0, 0]
                    feature_now += [0 for _ in range(48)]
                else:
                    tmp = np.array(tmp).flatten()
                    tmp_24 = np.array(tmp_24).flatten()
                    feature_now += list(tmp)
                    feature_now += list(tmp_24) + list(tmp_24 / (cmp_24_mean + 1))
            else:
                if len(tmp) == 0:
                    feature_now += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0]
                    feature_now += [0 for _ in range(72)]
                else:
                    tmp = np.array(tmp)
                    tmp_24 = np.array(tmp_24)
                    assert tmp.shape[-1] == 5
                    feature_now += list(np.mean(tmp, axis=0)) + list(np.std(tmp, axis=0)) + [len(tmp)]
                    feature_now += list(np.mean(tmp_24, axis=0)) + list(np.std(tmp_24, axis=0)) \
                                   + list(np.mean(tmp_24, axis=0) / (cmp_24_mean + 1))

        f_n_user.append(feature_now)
        # print('feature_now len:', len(feature_now))
        assert_num = 130 + 4 * 48 + 10 * 72
        assert len(feature_now) == assert_num

    assert_num = 130 + 4 * 48 + 10 * 72
    Num_users_select = len(f_n_user)
    f_n_user = np.array(f_n_user)
    for index in range(assert_num):
        if Num_users_select == 0:
            feature += [0, 0] + [0, 0, 0]
        else:
            feature += [np.mean(f_n_user[:, index]), np.std(f_n_user[:, index])]
            feature += list(np.percentile(f_n_user[:, index], [20, 50, 80]))

    feature += [Num_users_select / Num_users]
    
    return init_cishu, init_renshu, feature


def visit2array(table):
    init_cishu, init_renshu, feature_3 = get_feature(table)
    feature_1 = get_feature_cishu(init_cishu)
    feature_2 = get_feature_reshu(init_renshu)
    
#     print(len(FEATURE_1), len(FEATURE_2), len(FEATURE_3))
    feature = feature_1 + feature_2 + feature_3
    shape = np.array([len(feature_1), len(feature_2), len(feature_3)])
    return feature, shape


