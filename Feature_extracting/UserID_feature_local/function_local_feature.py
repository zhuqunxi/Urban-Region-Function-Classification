# -*- coding: utf-8 -*-
import numpy as np
import datetime

# 用字典查询代替类型转换，可以减少一部分计算时间
date2position = {}
datestr2dateint = {}
str2int = {}
for i in range(24):
    str2int[str(i).zfill(2)] = i


# 访问记录内的时间从2018年10月1日起，共182天
# 将日期按日历排列
for i in range(182):
    date = datetime.date(day=1, month=10, year=2018) + datetime.timedelta(days=i)
    date_int = int(date.__str__().replace("-", ""))
    date2position[date_int] = [i % 7, i // 7]
    datestr2dateint[str(date_int)] = date_int

def judge_date(date, x):
    now = datestr2dateint[date]
    if now < 20181008:  # 国庆
        return 0
    elif now >= 20181230 and now <= 20190101:  # 元旦放假
        return 1
    elif  now >= 20190128  and now <= 20190203:  # 小年 -- 除夕前一天
        return 2
    elif now >= 20190204  and now <= 20190210:  # 除夕 -- 初6
        return 3
    elif now >= 20190211 and now <= 20190219:  # 初7 -- 元宵
        return 4
    else:
        if x < 5:
            return 5
        return 6

def get_local_feature_1(temp):
    day = [1, len(temp)]
    feature = np.array(day)
    return feature, 1, len(feature)


def map_250_to_num(x, jinzhi=250):
    res = 0
    for i in range(len(x)):
        res += x[i] * (jinzhi ** i)
    return res


def map_num_to_250(res, k=None, jinzhi=250):
    assert not (k==None)
    x = []
    for i in range(k):
        x.append(res % jinzhi)
        res = res // jinzhi
    return x


def get_local_feature_2(temp):
    def get_f_1(visit_lst):
        zaoqi_hour = str2int[visit_lst[0]]
        wanshang_hour = str2int[visit_lst[-1]]
        kuadu = wanshang_hour - zaoqi_hour
        work_hour = len(visit_lst)
        if len(visit_lst) == 1:
            max_jiange = 0
        else:
            max_jiange = np.max([str2int[visit_lst[i + 1]] - str2int[visit_lst[i]] for i in range(len(visit_lst) - 1)])

        return [zaoqi_hour, wanshang_hour, kuadu, work_hour, max_jiange]

    hour = []
    for date, visit_lst in temp:
        f_1 = get_f_1(visit_lst)
        hour.append(f_1)
    hour = np.array(hour)
    feature = list(np.mean(hour, axis=0) * 10 // 1) #+ list(np.std(hour, axis=0))
    k_wei = len(feature)
    feature = np.array([map_250_to_num(feature)])  # k = 5, jinzhi = 250
    return feature, k_wei, len(feature)

def get_local_feature_3(temp):
    feature = np.zeros(7)

    for date, visit_lst in temp:
        x, y = date2position[datestr2dateint[date]]
        day_type = judge_date(date, x)
        feature[day_type] += 1

    # feature = np.array(feature) + 0.0
    # return feature, len(feature)

    k_wei = len(feature)
    feature = np.array([map_250_to_num(feature, jinzhi=185)]) # k = 7, jinzhi = 185
    return feature, k_wei, len(feature)


def get_local_feature_4(temp):
    def get_f_1(visit_lst):
        zaoqi_hour = str2int[visit_lst[0]]
        wanshang_hour = str2int[visit_lst[-1]]
        kuadu = wanshang_hour - zaoqi_hour
        work_hour = len(visit_lst)
        ff = [kuadu, work_hour]
        return np.array(ff)

    tmp = np.zeros((7, 2))
    day_fangjia = np.zeros(7)
    for date, visit_lst in temp:
        x, y = date2position[datestr2dateint[date]]
        day_type = judge_date(date, x)
        tmp[day_type] += get_f_1(visit_lst)
        day_fangjia[day_type] += 1

    # feature = list(tmp[:, 1]) + list((tmp[:, 0] + 0.001) / (day_fangjia + 0.1))
    feature = list((tmp[:, 0] + 0.001) / (day_fangjia + 0.1) * 5 // 1)

    k_wei = len(feature)
    feature = np.array([map_250_to_num(feature, jinzhi=150)]) # k = 5, jinzhi = 250
    return feature, k_wei, len(feature)

    # feature = np.array(feature)
    # return feature, len(feature)

def get_local_feature_5(temp):
    def get_f_1(visit_lst):
        zaoqi_hour = str2int[visit_lst[0]]
        wanshang_hour = str2int[visit_lst[-1]]
        kuadu = wanshang_hour - zaoqi_hour
        work_hour = len(visit_lst)
        if len(visit_lst) == 1:
            max_jiange = 0
        else:
            max_jiange = np.max([str2int[visit_lst[i + 1]] - str2int[visit_lst[i]] for i in range(len(visit_lst) - 1)])

        return [zaoqi_hour, wanshang_hour, kuadu, work_hour, max_jiange]

    hour = []
    for date, visit_lst in temp:
        f_1 = get_f_1(visit_lst)
        hour.append(f_1)
    hour = np.array(hour)
    # feature = list(np.mean(hour, axis=0) * 10 // 1) #+ list(np.std(hour, axis=0))
    feature = list(np.std(hour, axis=0) * 10 // 1)  # + list(np.std(hour, axis=0))
    k_wei = len(feature)
    feature = np.array([map_250_to_num(feature)])  # k = 5, jinzhi = 250
    return feature, k_wei, len(feature)

def get_local_feature_6(temp):
    def get_f_1(visit_lst):
        zaoqi_hour = str2int[visit_lst[0]]
        wanshang_hour = str2int[visit_lst[-1]]
        kuadu = wanshang_hour - zaoqi_hour
        return kuadu

    tmp = [[] for _ in range(7)]
    for date, visit_lst in temp:
        x, y = date2position[datestr2dateint[date]]
        day_type = judge_date(date, x)
        tmp[day_type].append(get_f_1(visit_lst))

    # feature = list(tmp[:, 1]) + list((tmp[:, 0] + 0.001) / (day_fangjia + 0.1))

    feature = []
    for i in range(7):
        tmp_ = tmp[i]
        if tmp_ == []:
            feature += [0]
        else:
            feature += [np.std(tmp_) * 5 // 1] #= list((tmp[:, 0] + 0.001) / (day_fangjia + 0.1) * 5 // 1)

    k_wei = len(feature)
    feature = np.array([map_250_to_num(feature, jinzhi=150)]) # k = 5, jinzhi = 250
    return feature, k_wei, len(feature)