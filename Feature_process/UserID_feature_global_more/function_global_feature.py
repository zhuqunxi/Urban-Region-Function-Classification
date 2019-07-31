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

# def get_global_feature_1(table):
#     users = table[0]
#     strings = table[1]
#     feature = [len(users)]
#     day = []
#     for user, string in zip(users, strings):
#         temp = []
#         for item in string.split(','):
#             temp.append([item[0:8], item[9:].split("|")])
#         day.append(len(temp))
#
#     day = np.array(day).flatten()
#     feature += [np.sum(day), day.mean(), day.std(), day.max()] + list(np.percentile(day, [10, 25, 50, 75, 90]))
#     return feature

def get_global_feature_1(table):
    def get_f_1(visit_lst):
        zaoqi_hour = str2int[visit_lst[0]]
        wanshang_hour = str2int[visit_lst[-1]]
        kuadu = wanshang_hour - zaoqi_hour
        work_hour = len(visit_lst)
        if len(visit_lst) == 1:
            max_jiange = 0
        else:
            max_jiange = np.max([str2int[visit_lst[i + 1]] - str2int[visit_lst[i]] for i in range(len(visit_lst) - 1)])

        hour_cnt = np.zeros(4)
        day_cnt = np.zeros(4)
        for visit in visit_lst:
            hour_cnt[str2int[visit] // 6] += 1
            day_cnt[str2int[visit] // 6] = 1

        return list(day_cnt) + list(hour_cnt) + [zaoqi_hour, wanshang_hour, kuadu, work_hour, max_jiange]

    users = table[0]
    strings = table[1]
    feature = [len(users)]
    day = []
    hour = []
    for user, string in zip(users, strings):
        temp = []
        for item in string.split(','):
            temp.append([item[0:8], item[9:].split("|")])
        day.append(len(temp))
        for date, visit_lst in temp:
            f_1 = get_f_1(visit_lst)
            hour.append(f_1)

    day = np.array(day).flatten()
    feature += [np.sum(day), day.mean(), day.std(), day.max()] + list(np.percentile(day, [10, 25, 50, 75, 90]))

    hour = np.array(hour)
    feature += list(np.mean(hour, axis=0)) + list(np.std(hour, axis=0)) + list(np.percentile(hour[:, -2], [10, 40, 60, 90]))
    return feature


def get_global_feature_2(table):
    users = table[0]
    strings = table[1]
    feature_1 = np.zeros(7)
    feature_2 = np.zeros(7)
    for user, string in zip(users, strings):
        temp = []
        for item in string.split(','):
            temp.append([item[0:8], item[9:].split("|")])

        for date, visit_lst in temp:
            x, y = date2position[datestr2dateint[date]]
            day_type = judge_date(date, x)
            feature_1[day_type] += 1
            feature_2[day_type] += len(visit_lst)
    feature_1 = list(feature_1) + list(feature_1 / feature_1.sum())
    feature_2 = list(feature_2) + list(feature_2 / feature_2.sum())
    return feature_1 + feature_2


# def get_global_feature_4(table):
#     users = table[0]
#     strings = table[1]
#     feature = np.zeros(7)
#     for user, string in zip(users, strings):
#         temp = []
#         for item in string.split(','):
#             temp.append([item[0:8], item[9:].split("|")])
#
#         for date, visit_lst in temp:
#             x, y = date2position[datestr2dateint[date]]
#             day_type = judge_date(date, x)
#             # feature[day_type] += 1
#             feature[day_type] += len(visit_lst)
#
#     feature = list(feature) + list(feature / feature.sum())
#     return feature

