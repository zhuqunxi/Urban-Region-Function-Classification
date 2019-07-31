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


def get_statistic_variable(tmp):
    tmp = np.array(tmp).flatten()
    if len(tmp) > 0:
        return [np.sum(tmp), tmp.mean(), tmp.std(), tmp.max(), tmp.min()] + list(
            np.percentile(tmp, [25, 50, 75]))  # shape = (8, )
    else:
        return list(np.zeros((8,)) - 0)


def relative_ratio(A, B, ep=1):
    return list((np.array(A) + ep) / (np.array(B) + ep))

def get_fine_feature_cishu(data):
    #  待加的特征： 1） 平常以及节日每天人数最多的是几点钟  2)  8点到17点， 17点24点， 0点到8点
    # assert data.shape == (7, 26, 24)
    # data shape: (7, 26, 24), (7, 26, 8)
    
    fr24_to_num = data.shape[-1]
    
    feature = []
    
    feature += [data.mean(), data.std()]
    
    # 国庆节  1. 每天的平均人数  2. 均值 3. 方差
    guoqing_day = np.mean(data[:, 0, :], axis=-1)
    assert len(guoqing_day) == 7
    feature += list(guoqing_day)
    feature += [guoqing_day.mean(), guoqing_day.std()]
    feature += [(guoqing_day.mean() + 1) / (data.mean()+ 1), (guoqing_day.std() + 1) / (data.std() + 1)]
    feature += list(np.argmax(data[:, 0, :], axis=-1))  # 每天中人数最多的时间
    feature += list(np.argmin(data[:, 0, :], axis=-1))  # 每天中人数最多的时间
    
    assert len(guoqing_day) == 7
    
    # 工作日 和 休息日
    hour_24_work = np.array([np.sum(data[:5, :, i]) / 26 / 5 for i in range(fr24_to_num)])
    hour_24_rest = np.array([np.sum(data[5:, :, i]) / 26 / 2 for i in range(fr24_to_num)])
    
    
    assert len(hour_24_work) == fr24_to_num
    assert len(hour_24_rest) == fr24_to_num
    feature += list((hour_24_work + 1) / (data.mean() + 1))
    feature += list((hour_24_rest + 1) / (data.mean() + 1))
    feature += [(hour_24_work.mean() + 1) / (data.mean() + 1), (hour_24_work.std() + 1) / (data.std() + 1),
                (hour_24_rest.mean() + 1) / (data.mean() + 1), (hour_24_rest.std() + 1) / (data.std() + 1)]
    
    # 工作日 和 休息日 后一小时和前一小时的比值
    if fr24_to_num==24:
        hour_24_work_ratio = np.array([(hour_24_work[(i + 1) % 24] + 1) / (hour_24_work[i] + 1) for i in range(24)])
        hour_24_rest_ratio = np.array([(hour_24_rest[(i + 1) % 24] + 1) / (hour_24_rest[i] + 1) for i in range(24)])
    
        feature += list(hour_24_work_ratio)
        feature += list(hour_24_rest_ratio)
        # 工作日 和 休息日 的比值
        feature += list(hour_24_rest_ratio / (hour_24_work_ratio + 0.1))
    
    # 工作日 和 休息日
    day_work = np.array([np.sum(data[i, :, :]) / 26 / fr24_to_num for i in range(5)])
    day_rest = np.array([np.sum(data[i, :, :]) / 26 / fr24_to_num for i in range(5, 7)])
    
    assert len(day_work) == 5
    assert len(day_rest) == 2
    feature += list(day_work)
    feature += list(day_rest)
    feature += [(day_work.mean() + 1) / (data.mean() + 1), (day_work.std() + 1) / (data.std() + 1),
                (day_rest.mean() + 1) / (data.mean() + 1), (day_rest.std() + 1) / (data.std() + 1)]
    
    # 过年
    guonian_day = list(np.array([np.sum(data[i, 18, :]) / fr24_to_num for i in range(7)])) + list(
        np.array([np.sum(data[i, 19, :]) / fr24_to_num for i in range(7)])) + list(
        np.array([np.sum(data[i, 20, :]) / fr24_to_num for i in range(3)]))
    guonian_day = np.array(guonian_day)
    
    guonian_hour_24_chu_1_7 = np.array([np.sum(data[:, 18, i]) / 14 for i in range(fr24_to_num)])
    guonian_hour_24_chu_8_15 = np.array([np.sum(data[:, 19, i]) / 14 for i in range(fr24_to_num)])
    assert len(guonian_hour_24_chu_1_7) == fr24_to_num
    
    feature += list((guonian_day + 1) / (data.mean() + 1))
    feature += [(guonian_day.mean() + 1) / (data.mean() + 1), (guonian_day.std() + 1) / (data.std() + 1)]
    feature += list((guonian_hour_24_chu_1_7 + 1) / (data.mean() + 1))
    feature += list((guonian_hour_24_chu_8_15 + 1) / (data.mean() + 1))
    feature += [(guonian_hour_24_chu_1_7.mean() + 1) / (data.mean() + 1), (guonian_hour_24_chu_1_7.std() + 1) / (data.std() + 1)]
    feature += [(guonian_hour_24_chu_8_15.mean() + 1) / (data.mean() + 1), (guonian_hour_24_chu_8_15.std() + 1) / (data.std() + 1)]
    
    if fr24_to_num==24:
        guonian_hour_24_chu_1_7_ratio = np.array(
            [(guonian_hour_24_chu_1_7[(i + 1) % 24] + 1) / (guonian_hour_24_chu_1_7[i] + 1) for i in range(24)])
        guonian_hour_24_chu_8_15_ratio = np.array(
            [(guonian_hour_24_chu_8_15[(i + 1) % 24] + 1) / (guonian_hour_24_chu_8_15[i] + 1) for i in range(24)])
        guonian_hour_24_ratio_chu_1_7_relative_work = (guonian_hour_24_chu_1_7 + 1) / (hour_24_work + 1)
        guonian_hour_24_ratio_chu_8_15_relative_work = (guonian_hour_24_chu_8_15 + 1) / (hour_24_work + 1)
        feature += list(guonian_hour_24_chu_1_7_ratio)
        feature += list(guonian_hour_24_chu_8_15_ratio)
        feature += list(guonian_hour_24_ratio_chu_1_7_relative_work)
        feature += list(guonian_hour_24_ratio_chu_8_15_relative_work)
    
    # 春运
    
    chunyun_day = list(np.array([np.sum(data[i, 16, :]) / fr24_to_num for i in range(7)])) + list(
        np.array([np.sum(data[i, 17, :]) / fr24_to_num for i in range(7)]))
    chunyun_day = np.array(chunyun_day)
    
    if fr24_to_num ==24:
        chunyun_hour_24 = np.array([np.sum(data[:, 16:18, i]) / 14 for i in range(fr24_to_num)])
        chunyun_hour_24_ratio = np.array(
            [(chunyun_hour_24[(i + 1) % 24] + 1) / (chunyun_hour_24[i] + 1) for i in range(24)])
        chunyun_hour_24_relative_work = (chunyun_hour_24 + 1) / (hour_24_work + 1)
        feature += list((chunyun_hour_24 + 1)/ (data.mean() + 1))
        feature += list(chunyun_hour_24_ratio)
        feature += list(chunyun_hour_24_relative_work)
        
    feature += list((chunyun_day + 1) / (data.mean() + 1))
    feature += [(chunyun_day.mean() + 1)/ (data.mean() + 1), (chunyun_day.std() + 1) / (data.std() + 1)]
    
    #     guonian_day_danian30_chu6 = np.array([np.sum(data[i, 18, :])/24 for i in range(7)])
    #     guonian_day_chu6_zhengyue15 = np.array([np.sum(data[i, 19, :])/24 for i in range(7)])
    
    if fr24_to_num == 24:
        guonian_hour_24_danian30_chu6 = np.array([np.sum(data[:, 18, i]) / 7 for i in range(fr24_to_num)])
        guonian_hour_24_chu6_zhengyue15 = np.array([np.sum(data[:, 19, i]) / 7 for i in range(fr24_to_num)])
    
        assert len(guonian_hour_24_danian30_chu6) == 24
        feature += list(guonian_hour_24_danian30_chu6)
        feature += [(guonian_hour_24_danian30_chu6.mean() + 1) / (data.mean() + 1), (guonian_hour_24_danian30_chu6.std() + 1) / (data.std() + 1)]
        assert len(guonian_hour_24_chu6_zhengyue15) == 24
        feature += list(guonian_hour_24_chu6_zhengyue15)
        feature += [(guonian_hour_24_chu6_zhengyue15.mean() +1)  / (data.mean() + 1),
                    (guonian_hour_24_chu6_zhengyue15.std() + 1) / (data.std() + 1)]
    
    # 元旦
    yuandan_day = np.array(
        [data[5, 12, :].mean(), data[6, 12, :].mean(), data[0, 13, :].mean(), data[1, 13, :].mean()]) / (data.mean() + 1)
    yuandan_hour_24 = (data[6, 12, :] + data[0, 13, :] + data[1, 13, :]) / 3 / (data.mean() + 1)
    
    feature += list(yuandan_day)
    feature += list(yuandan_hour_24)
    
    # 双11  圣诞节  情人节
    jieri_day = np.array([data[6, 5, :].mean(), data[1, 12, :].mean(), data[3, 19, :].mean()]) / (data.mean() + 1)
    jieri_hour_24 = (data[6, 5, :] + data[1, 12, :] + data[3, 19, :]) / 3 / (data.mean() + 1)
    
    feature += list(jieri_day)
    feature += list(jieri_hour_24)
    
    return feature

def get_feature_cishu(data):
    feature = []
    feature += get_fine_feature_cishu(data[:, :, :8])
    feature += get_fine_feature_cishu(data[:, :, 11:14])
    feature += get_fine_feature_cishu(data[:, :, 8:17])
    feature += get_fine_feature_cishu(data[:, :, 17:])

    feature += get_fine_feature_cishu(data[:, :, 8:11])
    feature += get_fine_feature_cishu(data[:, :, 14:17])
    feature += get_fine_feature_cishu(data[:, :, 17:20])
    feature += get_fine_feature_cishu(data[:, :, 20:24])

    feature += get_fine_feature_cishu(data)
    return feature


def get_feature_reshu(data):
    assert data.shape == (7, 26)

    feature = []

    feature += list(data.flatten())
    feature += get_statistic_variable(data)

    # 国庆节  1. 每天的平均人数  2. 均值 3. 方差
    guoqing_day = data[:, 0]
    assert len(guoqing_day) == 7
    feature += get_statistic_variable(guoqing_day)
    feature += list(relative_ratio(get_statistic_variable(guoqing_day), get_statistic_variable(data), ep=1))
    feature += list([np.argmax(guoqing_day), np.argmin(guoqing_day)])

    # 每天平均人数
    day_renshu = np.mean(data, axis=-1)
    feature += list(day_renshu)
    feature += list([np.argmax(day_renshu), np.argmin(day_renshu)])

    # 每周人数
    week_renshu = np.mean(data, axis=0)
    feature += list(week_renshu)
    feature += list([np.argmax(week_renshu), np.argmin(week_renshu)])

    # 工作日 和 休息日
    day_work = np.array([np.sum(data[i, :, ]) / 26 for i in range(5)])
    day_rest = np.array([np.sum(data[i, :, ]) / 26 for i in range(5, 7)])

    assert len(day_work) == 5
    assert len(day_rest) == 2

    feature += [day_work.mean() / (data.mean() + 1), day_rest.mean() / (data.mean() + 1), ]

    # 过年
    guonian_day = list(data[:, 18]) + list(data[:, 19]) + list(data[:3, 20])
    guonian_day = np.array(guonian_day)

    feature += list([np.argmax(data[:, 18]), np.argmin(data[:, 18])])
    feature += list([np.argmax(data[:, 19]), np.argmin(data[:, 19])])

    feature += list(guonian_day / (data.mean() + 1))
    feature += [guonian_day.mean() / (data.mean() + 1), guonian_day.std() / (data.std() + 1)]

    # 春运

    chunyun_day = list(data[:, 16]) + list(data[:, 17])
    chunyun_day = np.array(chunyun_day)
    feature += list([np.argmax(data[:, 16]), np.argmin(data[:, 16])])
    feature += list([np.argmax(data[:, 17]), np.argmin(data[:, 17])])

    feature += list(chunyun_day / (data.mean() + 1))
    feature += [chunyun_day.mean() / (data.mean() + 1), chunyun_day.std() / (data.std() + 1)]

    #     guonian_day_danian30_chu6 = np.array([np.sum(data[i, 18, :])/24 for i in range(7)])
    #     guonian_day_chu6_zhengyue15 = np.array([np.sum(data[i, 19, :])/24 for i in range(7)])

    # 元旦
    yuandan_day = np.array([data[5, 12], data[6, 12], data[0, 13], data[1, 13]])
    yuandan_day_relative = np.array([data[5, 12], data[6, 12], data[0, 13], data[1, 13]]) / (data.mean() + 1)
    feature += list(yuandan_day) + list(yuandan_day_relative)

    # 双11  圣诞节  情人节

    jieri_day = np.array([data[6, 5], data[1, 12], data[3, 19]])
    jieri_day_relative = np.array([data[6, 5], data[1, 12], data[3, 19]]) / (data.mean() + 1)
    feature += list(jieri_day) + list(jieri_day_relative)

    return feature



# def get_feature_1(table):
#     # 用户时间纵向特征，看用户在时间轴上的变化
#
#     # 编号 (星期i, 星期j) : k
#     # 可以考虑下将礼拜1到5进行合并！！！！  能减少几百个特征
#     #     dict_day2day = {(1, 2): 0, (2, 3): 1, (3, 4): 2, (4, 5): 3, (5, 6): 4, (6, 7): 5,
#     #                    (7, 1): 6, (5, 1): 7, (6, 1): 8}
#     dict_day2day = {(1, 2): 0, (2, 3): 0, (3, 4): 0, (4, 5): 0, (5, 6): 1, (6, 7): 2,
#                     (7, 1): 3, (5, 1): 4, (6, 1): 5}
#     dict_num = 6
#
#     strings = table[1]
#
#     # shape = (1, )
#     # ok到FEATURE
#     Num_users = len(strings)  # 统计用户人数
#
#     # shape = (用户人数, )
#     # ok到FEATURE                                             #  天   小时   每天几小时   每天小时的std  做多一天几小时，  做少一天几小时, 25， 50 ， 75， 分位数
#     Num_users_day_hour_information = np.zeros(
#         (Num_users, 9))  # 10   100     10             0.5        16                3
#
#     #     Num_users_day = np.zeros((Num_users, )) # 统计每个用户访问天数
#     #     Num_users_hour = np.zeros((Num_users, )) # 统计每个用户访问小时数
#     #     Num_users_hour_day_mean = [] #统计每个用户每天工作的小时数的均值
#     #     Num_users_hour_day_std = [] #统计每个用户每天工作的小时数的方差
#     #     Num_users_hour_day_per_25 = [] #统计每个用户每天工作的小时数的分位数25%
#     #     Num_users_hour_day_per_50 = [] #统计每个用户每天工作的小时数的中位数
#     #     Num_users_hour_day_per_75 = [] #统计每个用户每天工作的小时数的分位数75%
#     #     Num_users_hour_day_max = [] #统计每个用户每天工作的小时数的max
#
#
#
#     # ok到FEATURE                                            #  总共的差值   差值平均值   插值的std  最大插值，最小插值,  25， 50 ， 75 分位数
#     Num_users_chazhi_information = np.zeros((Num_users, 8))  # 跨度100天     2.5天        0.5天       100天
#
#     #     Num_users_chazhi_day_mean = [] # 统计每个用户差值的均值
#     #     Num_users_chazhi_day_std = [] # 统计每个用户差值的方差
#     #     Num_users_chazhi_day_per_25 = [] # 统计每个用户差值的分位数25%
#     #     Num_users_chazhi_day_per_50 = [] # 统计每个用户差值的中位数
#     #     Num_users_chazhi_day_per_75 = [] # 统计每个用户差值的分位数75%
#     #     Num_users_chazhi_day_max = [] # 统计每个用户差值的max
#
#
#
#     # shape = (>用户人数, )
#     # ok到FEATURE
#     Num_chazhi_Day = []  # 统计相邻天数的差值
#     Num_users_hour_day = []  # 统计每天工作的小时数
#
#     # ok到FEATURE
#     Num_users_aoye = np.zeros((Num_users,))  # 统计用户熬夜个数，  7点以后
#     Num_users_zaoqi = np.zeros((Num_users,))  # 统计用户早起个数， 6点之前
#     Num_users_tongxiao = np.zeros((Num_users,))  # 统计用户通宵个数， 前一天晚上7点以后， 到次日6点之前
#
#     # ok到FEATURE
#     Num_day2day = np.zeros((dict_num, Num_users))
#
#     #     Num_fri2mon = [] #统计每个用户礼拜5到礼拜1的差值个数
#     #     Num_sat2mon = [] #统计每个用户礼拜6到礼拜1的差值个数
#     #     Num_sun2mon = [] #统计每个用户礼拜7到礼拜1的差值个数
#     #     Num_mon2tue = [] #统计每个用户礼拜1到礼拜2的差值个数
#     #     Num_tue2wen = [] #统计每个用户礼拜2到礼拜3的差值个数
#     #     Num_wen2thr = [] #统计每个用户礼拜3到礼拜4的差值个数
#     #     Num_thr2fri = [] #统计每个用户礼拜4到礼拜5的差值个数
#     #     Num_fri2sat = [] #统计每个用户礼拜5到礼拜6的差值个数
#     #     Num_sat2sun = [] #统计每个用户礼拜6到礼拜7的差值个数
#
#
#     # shape = (24, )   24 * 12 特征
#     # ok到FEATURE
#
#     # Num_day2day_hour_pre = np.zeros((dict_num, 24))  # dict_num = 6
#     # Num_day2day_hour_next = np.zeros((dict_num, 24))
#
#
#     #     Num_fri2mon_24hour_for_pre = [] #统计礼拜5到礼拜1的差值个数, 且礼拜5最晚几点走
#     #     Num_fri2mon_24hour_for_next = [] #统计礼拜5到礼拜1的差值个数， 且礼拜1几点到
#
#     #     Num_sat2mon_24hour_for_pre = [] #统计礼拜6到礼拜1的差值个数, 且礼拜6最晚几点走
#     #     Num_sat2mon_24hour_for_next = [] #统计礼拜6到礼拜1的差值个数， 且礼拜1几点到
#
#     #     Num_sun2mon_24hour_for_pre = [] #统计礼拜7到礼拜1的差值个数, 且礼拜7最晚几点走
#     #     Num_sun2mon_24hour_for_next = [] #统计礼拜7到礼拜1的差值个数， 且礼拜1几点到
#
#     #     Num_mon2tue_24hour_for_pre = [] #统计礼拜1到礼拜2的差值个数, 且礼拜1最晚几点走
#     #     Num_mon2tue_24hour_for_next = [] #统计礼拜1到礼拜2的差值个数， 且礼拜2几点到
#
#     #     Num_tue2wen_24hour_for_pre = [] #统计礼拜2到礼拜3的差值个数, 且礼拜2最晚几点走
#     #     Num_tue2wen_24hour_for_next = [] #统计礼拜2到礼拜3的差值个数， 且礼拜3几点到
#
#     #     Num_wen2thr_24hour_for_pre = [] #统计礼拜3到礼拜4的差值个数, 且礼拜3最晚几点走
#     #     Num_wen2thr_24hour_for_next = [] #统计礼拜3到礼拜4的差值个数， 且礼拜4几点到
#
#     #     Num_thr2fri_24hour_for_pre = [] #统计礼拜4到礼拜5的差值个数, 且礼拜4最晚几点走
#     #     Num_thr2fri_24hour_for_next = [] #统计礼拜4到礼拜5的差值个数， 且礼拜5几点到
#
#     #     Num_fri2sat_24hour_for_pre = [] #统计礼拜5到礼拜6的差值个数, 且礼拜5最晚几点走
#     #     Num_fri2sat_24hour_for_next = [] #统计礼拜5到礼拜6的差值个数， 且礼拜6几点到
#
#     #     Num_sat2sun_24hour_for_pre = [] #统计礼拜6到礼拜7的差值个数, 且礼拜6最晚几点走
#     #     Num_sat2sun_24hour_for_next = [] #统计礼拜6到礼拜7的差值个数， 且礼拜7几点到
#
#
#     init_cishu = np.zeros((7, 26, 24))  # 统计26周每天每小时的用户人数
#     init_renshu = np.zeros((7, 26))  #
#
#     zaoshang_hour_workday_dao = []
#     wanshang_hour_workday_zou = []
#     zaoshang_hour_restday_dao = []
#     wanshang_hour_restday_zou = []
#     zaoshang_hour_restday_dao_sat = []
#     wanshang_hour_restday_zou_sat = []
#     zaoshang_hour_restday_dao_sun = []
#     wanshang_hour_restday_zou_sun = []
#     work_day_kuadu = []
#     rest_day_kuadu = []
#     sat_day_kuadu = []
#     sun_day_kuadu = []
#
#     #     print('\n 用户数目：', len(strings))
#
#     for user_idx, string in enumerate(strings):
#         temp = [[item[0:8], item[9:].split("|")] for item in string.split(',')]
#
#         cnt_day, cnt_hour = len(temp), 0  # 统计工作天数，和工作小时数
#         tmp = np.zeros((cnt_day,))
#         for i, (date, visit_lst) in enumerate(temp):
#             tmp[i] = len(visit_lst)
#             cnt_hour += len(visit_lst)
#             Num_users_hour_day.append(len(visit_lst))
#
#
#             #  天   小时   每天几小时   每天小时的std     做多一天几小时  最少多少小时
#         Num_users_day_hour_information[user_idx, :] = np.array(
#             [cnt_day, cnt_hour, tmp.mean(), tmp.std(), tmp.max(), tmp.min()] + list(np.percentile(tmp, [25, 50, 75])))
#         #         Num_users_day[user_idx] = cnt_day
#         #         Num_users_hour[user_idx] = np.sum(tmp)
#
#
#         jiange_day = []
#         jiange_flag = 0
#         for date, visit_lst in temp:
#             x, y = date2position[datestr2dateint[date]]
#             init_renshu[x][y] += 1  # 统计每小时的到访的总人数   7 * 26
#             for visit in visit_lst: init_cishu[x][y][str2int[visit]] += 1  # 统计每天到访的总人数 7 * 26 * 24
#
#             zaoqi_hour = str2int[visit_lst[0]]
#             wanshang_hour = str2int[visit_lst[-1]]
#
#             if x < 5:  # workday
#                 zaoshang_hour_workday_dao.append(zaoqi_hour)
#                 wanshang_hour_workday_zou.append(wanshang_hour)
#                 work_day_kuadu.append(wanshang_hour - zaoqi_hour)
#             if x >= 5:
#                 zaoshang_hour_restday_dao.append(zaoqi_hour)
#                 wanshang_hour_restday_zou.append(wanshang_hour)
#                 rest_day_kuadu.append(wanshang_hour - zaoqi_hour)
#                 if x == 5:
#                     zaoshang_hour_restday_dao_sat.append(zaoqi_hour)
#                     wanshang_hour_restday_zou_sat.append(wanshang_hour)
#                     wanshang_hour_restday_zou_sat.append(wanshang_hour - zaoqi_hour)
#                 else:
#                     zaoshang_hour_restday_dao_sun.append(zaoqi_hour)
#                     wanshang_hour_restday_zou_sun.append(wanshang_hour)
#                     sun_day_kuadu.append(wanshang_hour - zaoqi_hour)
#
#             if zaoqi_hour <= 6:  # 早起
#                 Num_users_zaoqi[user_idx] += 1
#
#             if jiange_flag:  # 涉及到插值，昨天和今天的关系
#                 jiange_flag = 1
#
#                 day_cha = date2int[date] - date2int[pre_date]
#                 jiange_day.append(day_cha)
#
#                 if day_cha == 1:
#                     idx = dict_day2day[(pre_x + 1, x + 1)]  # 前一天， 后一天
#                     Num_day2day[idx, user_idx] += 1  # (6, 用户数)
#                     # Num_day2day_hour_pre[idx, pre_aoye_hour] += 1  # 前一天 几点走
#                     # Num_day2day_hour_next[idx, zaoqi_hour] += 1  # 后一天 几点到
#
#                     if zaoqi_hour <= 6 and pre_aoye_hour >= 7:  # 通宵
#                         Num_users_tongxiao[user_idx] += 1
#
#                 if day_cha == 2 or day_cha == 3:
#                     if (pre_x + 1, x + 1) == (5, 1) or (pre_x + 1, x + 1) == (6, 1):  # 礼拜五（六）晚上走  礼拜1早上几点到
#                         idx = dict_day2day[(pre_x + 1, x + 1)]  # 前一天， 后一天
#                         # Num_day2day_hour_pre[idx, pre_aoye_hour] += 1
#                         # Num_day2day_hour_next[idx, zaoqi_hour] += 1
#
#             pre_date = date
#             pre_x, pre_y = x, y
#             pre_aoye_hour = str2int[visit_lst[-1]]
#
#             if pre_aoye_hour >= 7:  # 熬夜
#                 Num_users_aoye[user_idx] += 1
#
#         Num_chazhi_Day += jiange_day
#         jiange_day = np.array(jiange_day)
#
#         if len(jiange_day) > 0:
#             Num_users_chazhi_information[user_idx, :] = np.array(
#                 [np.sum(jiange_day), jiange_day.mean(), jiange_day.std(), jiange_day.max(), jiange_day.min()] + list(
#                     np.percentile(jiange_day, [25, 50, 75])))
#
#     Num_chazhi_Day = np.array(Num_chazhi_Day)  # 统计相邻天数的差值
#     Num_users_hour_day = np.array(Num_users_hour_day)  # 统计每天工作的小时数
#
#     #     print(Jiange_Day)
#
#     # 特征个数=1
#     FEATURE = [Num_users]
#
#     # 特征个数= 1 + 18= 19
#     FEATURE += list(np.mean(Num_users_day_hour_information, axis=0)) + list(
#         np.std(Num_users_day_hour_information, axis=0))
#     # Num_users_day_hour_information: (Num_users,  9)
#     #     print(len(FEATURE))
#
#     # 特征个数= 19 + 8 * 3 = 43
#     FEATURE += [np.sum(Num_users_aoye), Num_users_aoye.mean(), Num_users_aoye.std(), Num_users_aoye.max(),
#                 Num_users_aoye.min()] + list(np.percentile(Num_users_aoye, [25, 50, 75]))  # 统计用户熬夜个数，  7点以后
#     FEATURE += [np.sum(Num_users_zaoqi), Num_users_zaoqi.mean(), Num_users_zaoqi.std(), Num_users_zaoqi.max(),
#                 Num_users_zaoqi.min()] + list(np.percentile(Num_users_aoye, [25, 50, 75]))  # 统计用户早起个数， 6点之前
#     FEATURE += [np.sum(Num_users_tongxiao), Num_users_tongxiao.mean(), Num_users_tongxiao.std(),
#                 Num_users_tongxiao.max(), Num_users_tongxiao.min()] + list(
#         np.percentile(Num_users_tongxiao, [25, 50, 75]))  # 统计用户通宵个数， 前一天晚上7点以后， 到次日6点之前
#     #     print(len(FEATURE))
#
#     # 特征个数 = 43 + 12 + 16 = 71
#     FEATURE += list(np.mean(Num_day2day, axis=1)) + list(
#         np.std(Num_day2day, axis=1))  # Num_day2day = np.zeros((9, Num_users))
#     FEATURE += list(np.mean(Num_users_chazhi_information, axis=0)) + list(
#         np.std(Num_users_chazhi_information, axis=0))  # Num_users_chazhi_information = np.zeros((Num_users,  8))
#     #     print(len(FEATURE))
#
#     # 特征个数 = 71 + 12 * 24  = 359
#     # FEATURE += list(Num_day2day_hour_pre.flatten()) + list(Num_day2day_hour_next.flatten())  # (6, 24)
#     #     print(len(FEATURE))
#
#
#     # 特征个数 = 71 + 16 = 87
#     if len(Num_chazhi_Day) > 0:
#         FEATURE += [np.sum(Num_chazhi_Day), Num_chazhi_Day.mean(), Num_chazhi_Day.std(), Num_chazhi_Day.max(),
#                     Num_chazhi_Day.min()] + list(np.percentile(Num_chazhi_Day, [25, 50, 75]))
#     else:
#         FEATURE += list(np.zeros((8,)))
#     if len(Num_users_hour_day) > 0:
#         FEATURE += [np.sum(Num_users_hour_day), Num_users_hour_day.mean(), Num_users_hour_day.std(),
#                     Num_users_hour_day.max(), Num_users_hour_day.min()] + list(
#             np.percentile(Num_users_hour_day, [25, 50, 75]))  # np.sum(Num_users_hour_day) 之前算cnt_hour已经算过
#     else:
#         FEATURE += list(np.zeros((8,)))
#
#     # print(len(FEATURE))
#     # 特征个数 = 87 + 12 * 8 = 183
#
#     for tmp in [zaoshang_hour_workday_dao, wanshang_hour_workday_zou, zaoshang_hour_restday_dao,
#                 wanshang_hour_restday_zou,
#                 zaoshang_hour_restday_dao_sat, wanshang_hour_restday_zou_sat,
#                 zaoshang_hour_restday_dao_sun, wanshang_hour_restday_zou_sun,
#                 work_day_kuadu, rest_day_kuadu, sat_day_kuadu, sun_day_kuadu]:
#         tmp = np.array(tmp)
#         if len(tmp) > 0:
#             FEATURE += [np.sum(tmp), tmp.mean(), tmp.std(), tmp.max(), tmp.min()] + list(
#                 np.percentile(tmp, [25, 50, 75]))
#         else:
#             FEATURE += list(np.zeros((8,)))
#
#     FEATURE = np.array(FEATURE)
#     #     print('feature num =', len(FEATURE))
#     #     print('FEATURE max:', FEATURE.max())
#
#     assert len(FEATURE) == 183
#     return init_cishu, init_renshu, FEATURE


# def get_feature_1_1(table):
#     # 用户时间纵向特征，看用户在时间轴上的变化,  与上面的区别，进行相对归一化！！！
#
#     # 编号 (星期i, 星期j) : k
#     # 可以考虑下将礼拜1到5进行合并！！！！  能减少几百个特征
#     #     dict_day2day = {(1, 2): 0, (2, 3): 1, (3, 4): 2, (4, 5): 3, (5, 6): 4, (6, 7): 5,
#     #                    (7, 1): 6, (5, 1): 7, (6, 1): 8}
#
#     dict_day2day = {(1, 2): 0, (2, 3): 0, (3, 4): 0, (4, 5): 0, (5, 6): 1, (6, 7): 2,
#                     (7, 1): 3, (5, 1): 4, (6, 1): 5}
#
#     dict_num = 6
#
#     strings = table[1]
#
#     # shape = (1, )
#     Num_users = len(strings)  # 统计用户人数
#
#
#     # shape = (用户人数, )
#                                                                #  天   小时   每天几小时   每天小时的std  做多一天几小时，  做少一天几小时, 25， 50 ， 75， 分位数
#     Num_users_day_hour_information = np.zeros((Num_users, 9))  # 10   100     10             0.5        16                3
#     #     Num_users_day = np.zeros((Num_users, )) # 统计每个用户访问天数
#     #     Num_users_hour = np.zeros((Num_users, )) # 统计每个用户访问小时数
#     #     Num_users_hour_day_mean = [] #统计每个用户每天工作的小时数的均值
#     #     Num_users_hour_day_std = [] #统计每个用户每天工作的小时数的方差
#     #     Num_users_hour_day_per_25 = [] #统计每个用户每天工作的小时数的分位数25%
#     #     Num_users_hour_day_per_50 = [] #统计每个用户每天工作的小时数的中位数
#     #     Num_users_hour_day_per_75 = [] #统计每个用户每天工作的小时数的分位数75%
#     #     Num_users_hour_day_max = [] #统计每个用户每天工作的小时数的max
#
#
#
#     # ok到FEATURE                                            #  总共的差值   差值平均值   插值的std  最大插值，最小插值,  25， 50 ， 75 分位数
#
#     Num_users_chazhi_information = np.zeros((Num_users, 8))  # 跨度100天     2.5天        0.5天       100天
#     #     Num_users_chazhi_day_mean = [] # 统计每个用户差值的均值
#     #     Num_users_chazhi_day_std = [] # 统计每个用户差值的方差
#     #     Num_users_chazhi_day_per_25 = [] # 统计每个用户差值的分位数25%
#     #     Num_users_chazhi_day_per_50 = [] # 统计每个用户差值的中位数
#     #     Num_users_chazhi_day_per_75 = [] # 统计每个用户差值的分位数75%
#     #     Num_users_chazhi_day_max = [] # 统计每个用户差值的max
#
#
#
#     # shape = (>用户人数, )
#     # ok到FEATURE
#
#     Num_chazhi_Day = []  # 统计相邻天数的差值
#     Num_users_hour_day = []  # 统计每天工作的小时数
#
#     Num_users_aoye = np.zeros((Num_users,))  # 统计用户熬夜个数，  7点以后
#     Num_users_zaoqi = np.zeros((Num_users,))  # 统计用户早起个数， 6点之前
#     Num_users_tongxiao = np.zeros((Num_users,))  # 统计用户通宵个数， 前一天晚上7点以后， 到次日6点之前
#
#
#     Num_day2day = np.zeros((dict_num, Num_users))
#     #     Num_fri2mon = [] #统计每个用户礼拜5到礼拜1的差值个数
#     #     Num_sat2mon = [] #统计每个用户礼拜6到礼拜1的差值个数
#     #     Num_sun2mon = [] #统计每个用户礼拜7到礼拜1的差值个数
#     #     Num_mon2tue = [] #统计每个用户礼拜1到礼拜2的差值个数
#     #     Num_tue2wen = [] #统计每个用户礼拜2到礼拜3的差值个数
#     #     Num_wen2thr = [] #统计每个用户礼拜3到礼拜4的差值个数
#     #     Num_thr2fri = [] #统计每个用户礼拜4到礼拜5的差值个数
#     #     Num_fri2sat = [] #统计每个用户礼拜5到礼拜6的差值个数
#     #     Num_sat2sun = [] #统计每个用户礼拜6到礼拜7的差值个数
#
#
#     # shape = (24, )   24 * 12 特征
#     # ok到FEATURE
#
#     Num_day2day_hour_pre_specific = [[] for i in range(dict_num)]
#     Num_day2day_hour_next_specific = [[] for i in range(dict_num)]
#     # Num_day2day_hour_pre = np.zeros((dict_num, 24))  # dict_num = 6
#     # Num_day2day_hour_next = np.zeros((dict_num, 24))
#     #     Num_fri2mon_24hour_for_pre = [] #统计礼拜5到礼拜1的差值个数, 且礼拜5最晚几点走
#     #     Num_fri2mon_24hour_for_next = [] #统计礼拜5到礼拜1的差值个数， 且礼拜1几点到
#
#     #     Num_sat2mon_24hour_for_pre = [] #统计礼拜6到礼拜1的差值个数, 且礼拜6最晚几点走
#     #     Num_sat2mon_24hour_for_next = [] #统计礼拜6到礼拜1的差值个数， 且礼拜1几点到
#
#     #     Num_sun2mon_24hour_for_pre = [] #统计礼拜7到礼拜1的差值个数, 且礼拜7最晚几点走
#     #     Num_sun2mon_24hour_for_next = [] #统计礼拜7到礼拜1的差值个数， 且礼拜1几点到
#
#     #     Num_mon2tue_24hour_for_pre = [] #统计礼拜1到礼拜2的差值个数, 且礼拜1最晚几点走
#     #     Num_mon2tue_24hour_for_next = [] #统计礼拜1到礼拜2的差值个数， 且礼拜2几点到
#
#     #     Num_tue2wen_24hour_for_pre = [] #统计礼拜2到礼拜3的差值个数, 且礼拜2最晚几点走
#     #     Num_tue2wen_24hour_for_next = [] #统计礼拜2到礼拜3的差值个数， 且礼拜3几点到
#
#     #     Num_wen2thr_24hour_for_pre = [] #统计礼拜3到礼拜4的差值个数, 且礼拜3最晚几点走
#     #     Num_wen2thr_24hour_for_next = [] #统计礼拜3到礼拜4的差值个数， 且礼拜4几点到
#
#     #     Num_thr2fri_24hour_for_pre = [] #统计礼拜4到礼拜5的差值个数, 且礼拜4最晚几点走
#     #     Num_thr2fri_24hour_for_next = [] #统计礼拜4到礼拜5的差值个数， 且礼拜5几点到
#
#     #     Num_fri2sat_24hour_for_pre = [] #统计礼拜5到礼拜6的差值个数, 且礼拜5最晚几点走
#     #     Num_fri2sat_24hour_for_next = [] #统计礼拜5到礼拜6的差值个数， 且礼拜6几点到
#
#     #     Num_sat2sun_24hour_for_pre = [] #统计礼拜6到礼拜7的差值个数, 且礼拜6最晚几点走
#     #     Num_sat2sun_24hour_for_next = [] #统计礼拜6到礼拜7的差值个数， 且礼拜7几点到
#
#
#     init_cishu = np.zeros((7, 26, 24))  # 统计26周每天每小时的用户人数
#     init_renshu = np.zeros((7, 26))  #
#
#     zaoshang_hour_workday_dao = []
#     wanshang_hour_workday_zou = []
#     zaoshang_hour_restday_dao = []
#     wanshang_hour_restday_zou = []
#     zaoshang_hour_restday_dao_sat = []
#     wanshang_hour_restday_zou_sat = []
#     zaoshang_hour_restday_dao_sun = []
#     wanshang_hour_restday_zou_sun = []
#     work_day_kuadu = []
#     rest_day_kuadu = []
#     sat_day_kuadu = []
#     sun_day_kuadu = []
#
#     ########################################################################################################
#     #  国庆节
#
#
#
#     ########################################################################################################
#
#     #     print('\n 用户数目：', len(strings))
#
#     for user_idx, string in enumerate(strings):
#         temp = [[item[0:8], item[9:].split("|")] for item in string.split(',')]
#
#         cnt_day, cnt_hour = len(temp), 0  # 统计工作天数，和工作小时数
#         tmp = np.zeros((cnt_day,))
#         for i, (date, visit_lst) in enumerate(temp):
#             tmp[i] = len(visit_lst)
#             cnt_hour += len(visit_lst)
#             Num_users_hour_day.append(len(visit_lst))
#
#
#             #  天   小时   每天几小时   每天小时的std     做多一天几小时  最少多少小时
#         Num_users_day_hour_information[user_idx, :] = np.array(
#             [cnt_day, cnt_hour, tmp.mean(), tmp.std(), tmp.max(), tmp.min()] + list(np.percentile(tmp, [25, 50, 75])))
#         #         Num_users_day[user_idx] = cnt_day
#         #         Num_users_hour[user_idx] = np.sum(tmp)
#
#
#         jiange_day = []
#         jiange_flag = 0
#         for date, visit_lst in temp:
#             x, y = date2position[datestr2dateint[date]]
#             init_renshu[x][y] += 1  # 统计每小时的到访的总人数   7 * 26
#             for visit in visit_lst: init_cishu[x][y][str2int[visit]] += 1  # 统计每天到访的总人数 7 * 26 * 24
#
#             zaoqi_hour = str2int[visit_lst[0]]
#             wanshang_hour = str2int[visit_lst[-1]]
#
#             if x < 5:  # workday
#                 zaoshang_hour_workday_dao.append(zaoqi_hour)
#                 wanshang_hour_workday_zou.append(wanshang_hour)
#                 work_day_kuadu.append(wanshang_hour - zaoqi_hour)
#             if x >= 5:
#                 zaoshang_hour_restday_dao.append(zaoqi_hour)
#                 wanshang_hour_restday_zou.append(wanshang_hour)
#                 rest_day_kuadu.append(wanshang_hour - zaoqi_hour)
#                 if x == 5:
#                     zaoshang_hour_restday_dao_sat.append(zaoqi_hour)
#                     wanshang_hour_restday_zou_sat.append(wanshang_hour)
#                     wanshang_hour_restday_zou_sat.append(wanshang_hour - zaoqi_hour)
#                 else:
#                     zaoshang_hour_restday_dao_sun.append(zaoqi_hour)
#                     wanshang_hour_restday_zou_sun.append(wanshang_hour)
#                     sun_day_kuadu.append(wanshang_hour - zaoqi_hour)
#
#             if zaoqi_hour <= 6:  # 早起
#                 Num_users_zaoqi[user_idx] += 1
#
#             if jiange_flag:  # 涉及到插值，昨天和今天的关系
#                 jiange_flag = 1
#
#                 day_cha = date2int[date] - date2int[pre_date]
#                 jiange_day.append(day_cha)
#
#                 if day_cha == 1:
#                     idx = dict_day2day[(pre_x + 1, x + 1)]  # 前一天， 后一天
#                     Num_day2day[idx, user_idx] += 1  # (6, 用户数)
#                     # Num_day2day_hour_pre[idx, pre_aoye_hour] += 1  # 前一天 几点走
#                     # Num_day2day_hour_next[idx, zaoqi_hour] += 1  # 后一天 几点到
#                     Num_day2day_hour_pre_specific[idx].append(pre_aoye_hour)
#                     Num_day2day_hour_next_specific[idx].append(zaoqi_hour)
#
#                     if zaoqi_hour <= 6 and pre_aoye_hour >= 7:  # 通宵
#                         Num_users_tongxiao[user_idx] += 1
#
#                 if day_cha == 2 or day_cha == 3:
#                     if (pre_x + 1, x + 1) == (5, 1) or (pre_x + 1, x + 1) == (6, 1):  # 礼拜五（六）晚上走  礼拜1早上几点到
#                         idx = dict_day2day[(pre_x + 1, x + 1)]  # 前一天， 后一天
#                         # Num_day2day_hour_pre[idx, pre_aoye_hour] += 1
#                         # Num_day2day_hour_next[idx, zaoqi_hour] += 1
#                         Num_day2day_hour_pre_specific[idx].append(pre_aoye_hour)
#                         Num_day2day_hour_next_specific[idx].append(zaoqi_hour)
#
#             pre_date = date
#             pre_x, pre_y = x, y
#             pre_aoye_hour = str2int[visit_lst[-1]]
#
#             if pre_aoye_hour >= 7:  # 熬夜
#                 Num_users_aoye[user_idx] += 1
#
#         Num_chazhi_Day += jiange_day
#         jiange_day = np.array(jiange_day)
#         Num_users_chazhi_information[user_idx, :] = np.array(get_statistic_variable(jiange_day))
#
#     Num_chazhi_Day = np.array(Num_chazhi_Day)  # 统计相邻天数的差值
#     Num_users_hour_day = np.array(Num_users_hour_day)  # 统计每天工作的小时数
#
#     #     print(Jiange_Day)
#
#     # 特征个数=1
#     FEATURE = [Num_users]
#
#     # 特征个数= 1 + 18= 19
#     FEATURE += list(np.mean(Num_users_day_hour_information, axis=0)) + list(
#         np.std(Num_users_day_hour_information, axis=0))
#     # Num_users_day_hour_information: (Num_users,  9)
#     #     print(len(FEATURE))
#
#     # 特征个数= 19 + 8 * 3 = 43
#     FEATURE += [np.sum(Num_users_aoye), Num_users_aoye.mean(), Num_users_aoye.std(), Num_users_aoye.max(),
#                 Num_users_aoye.min()] + list(np.percentile(Num_users_aoye, [25, 50, 75]))  # 统计用户熬夜个数，  7点以后
#     FEATURE += [np.sum(Num_users_zaoqi), Num_users_zaoqi.mean(), Num_users_zaoqi.std(), Num_users_zaoqi.max(),
#                 Num_users_zaoqi.min()] + list(np.percentile(Num_users_aoye, [25, 50, 75]))  # 统计用户早起个数， 6点之前
#
#     FEATURE += [np.sum(Num_users_tongxiao), Num_users_tongxiao.mean(), Num_users_tongxiao.std(),
#                 Num_users_tongxiao.max(), Num_users_tongxiao.min()] + list(
#         np.percentile(Num_users_tongxiao, [25, 50, 75]))  # 统计用户通宵个数， 前一天晚上7点以后， 到次日6点之前
#     #     print(len(FEATURE))
#
#     # 特征个数 = 43 + 12 + 16 = 71
#     FEATURE += list(np.mean(Num_day2day, axis=1)) + list(np.std(Num_day2day, axis=1))  # Num_day2day = np.zeros((9, Num_users))
#     FEATURE += list(np.mean(Num_users_chazhi_information, axis=0)) + list(
#         np.std(Num_users_chazhi_information, axis=0))  # Num_users_chazhi_information = np.zeros((Num_users,  8))
#     #     print(len(FEATURE))
#
#     # 特征个数 = 71 + 12 * 24  = 359
#     # FEATURE += list(Num_day2day_hour_pre.flatten()) + list(Num_day2day_hour_next.flatten())  # (6, 24)
#     for i in range(dict_num):
#         FEATURE += get_statistic_variable(Num_day2day_hour_pre_specific[i])  # 8
#         FEATURE += get_statistic_variable(Num_day2day_hour_next_specific[i])   # 8
#     #     print(len(FEATURE))
#
#     # 特征个数 = 359 + 16 = 375
#     FEATURE += get_statistic_variable(Num_chazhi_Day)
#     FEATURE += get_statistic_variable(Num_users_hour_day)
#
#     # print(len(FEATURE))
#     # 特征个数 = 375 + 12 * 8 = 471   下面的是重要的性质  importance比较大
#     for tmp in [zaoshang_hour_workday_dao, wanshang_hour_workday_zou, zaoshang_hour_restday_dao,
#                 wanshang_hour_restday_zou,
#                 zaoshang_hour_restday_dao_sat, wanshang_hour_restday_zou_sat,
#                 zaoshang_hour_restday_dao_sun, wanshang_hour_restday_zou_sun,
#                 work_day_kuadu, rest_day_kuadu, sat_day_kuadu, sun_day_kuadu]:
#         tmp = np.array(tmp)
#         FEATURE += get_statistic_variable(tmp)
#
#     FEATURE = np.array(FEATURE)
#
#     # print('feature num =', len(FEATURE))
#     # print('FEATURE max:', FEATURE.max())
#
#     # assert len(FEATURE) == 471
#
#     return init_cishu, init_renshu, list(FEATURE)

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



def get_feature_1_2(table):
    # 用户时间纵向特征，看用户在时间轴上的变化,  与上面的区别，进行相对归一化！！！
    # 编号 (星期i, 星期j) : k
    # 可以考虑下将礼拜1到5进行合并！！！！  能减少几百个特征
    #     dict_day2day = {(1, 2): 0, (2, 3): 1, (3, 4): 2, (4, 5): 3, (5, 6): 4, (6, 7): 5,
    #                    (7, 1): 6, (5, 1): 7, (6, 1): 8}

    FEATURE = []

    dict_day2day = {(1, 2): 0, (2, 3): 0, (3, 4): 0, (4, 5): 0, (5, 6): 1, (6, 7): 2,
                    (7, 1): 3, (5, 1): 4, (6, 1): 5}
    
    dict_num = 6
    
    strings = table[1]
    
    # shape = (1, )
    Num_users = len(strings)  # 统计用户人数
    FEATURE += [Num_users]

    name_jieri = 'guoqing'
    ########################################################################################################
    #  国庆节
    jieri_dict = {}
    dates = [str(_) for _ in range(20181001, 20181009)]
    for idx, date in enumerate(dates):
        jieri_dict[date] = idx
    FEATURE += get_jieri_feature(strings, jieri_dict, name_jieri=name_jieri)

    # Guoqing_zaoshang_hour_dao = [[], [], [], [], [], [], [], []]  # 10月1日到8日
    # Guoqing_wanshang_hour_zou = [[], [], [], [], [], [], [], []]  # 10月1日到8日
    # Guoqing_zaowanshang_hour_daozou = [[], [], [], [], [], [], [], []]  # 10月1日到8日   zaoshang * 24 + wanshang
    # Guoqing_kuadu = [[], [], [], [], [], [], [], []]  # 10月1日到8日
    #
    # for date, visit_lst in temp:
    #     idx = date2int[date]  # 10月1日到8日
    #     if idx > 7:
    #         continue
    #     zaoqi_hour = str2int[visit_lst[0]]
    #     wanshang_hour = str2int[visit_lst[-1]]
    #     Guoqing_zaoshang_hour_dao[idx].append(zaoqi_hour)
    #     Guoqing_wanshang_hour_zou[idx].append(wanshang_hour)
    #     Guoqing_zaowanshang_hour_daozou.append(zaoqi_hour * 24 + wanshang_hour)
    #     Guoqing_kuadu.append(wanshang_hour - zaoqi_hour)
    #
    # for idx in enumerate(8):
    #     FEATURE += get_statistic_variable(Guoqing_zaoshang_hour_dao[idx])
    #     FEATURE += get_statistic_variable(Guoqing_wanshang_hour_zou[idx])
    #     FEATURE += get_statistic_variable(Guoqing_zaowanshang_hour_daozou[idx])
    #     FEATURE += get_statistic_variable(Guoqing_kuadu[idx])
    #
    # tmp1, tmp2, tmp3, tmp4 = [], [], [], []
    # for idx in enumerate(5):
    #     tmp1 += Guoqing_zaoshang_hour_dao[idx]
    #     tmp2 += Guoqing_wanshang_hour_zou[idx]
    #     tmp3 += Guoqing_zaowanshang_hour_daozou[idx]
    #     tmp4 += Guoqing_kuadu[idx]
    # FEATURE += get_statistic_variable(tmp1) + get_statistic_variable(tmp2) \
    #            + get_statistic_variable(tmp3) + get_statistic_variable(tmp4)
    #
    # tmp1, tmp2, tmp3, tmp4 = [], [], [], []
    # for idx in enumerate(5, 7):
    #     tmp1 += Guoqing_zaoshang_hour_dao[idx]
    #     tmp2 += Guoqing_wanshang_hour_zou[idx]
    #     tmp3 += Guoqing_zaowanshang_hour_daozou[idx]
    #     tmp4 += Guoqing_kuadu[idx]
    # FEATURE += get_statistic_variable(tmp1) + get_statistic_variable(tmp2) \
    #            + get_statistic_variable(tmp3) + get_statistic_variable(tmp4)

    name_jieri = 'guonian_chunyun'
    ########################################################################################################
    #  过年和春运
    jieri_dict = {}
    dates = [str(_) for _ in range(20190121, 20190132)] + [str(_) for _ in range(20190201, 20190229)]
    for idx, date in enumerate(dates):
        jieri_dict[date] = idx
    FEATURE += get_jieri_feature(strings, jieri_dict, name_jieri=name_jieri)

    # st_date, ed_date = date2int['20190127'], date2int['20190220']  # 20190128 小年， 20190219 元宵
    # num_guonian = ed_date - st_date + 1
    # Guoqnian_zaoshang_hour_dao = [[] for i in range(num_guonian)]
    # Guoqnian_wanshang_hour_zou = [[] for i in range(num_guonian)]
    # Guoqnian_zaowanshang_hour_daozou = [[] for i in range(num_guonian)]    zaoshang * 24 + wanshang
    # Guoqnian_kuadu = [[] for i in range(num_guonian)]
    #
    # for date, visit_lst in temp:
    #     idx = date2int[date]
    #     if idx < st_date or idx > ed_date:
    #         continue
    #     zaoqi_hour = str2int[visit_lst[0]]
    #     wanshang_hour = str2int[visit_lst[-1]]
    #     Guoqing_zaoshang_hour_dao[idx].append(zaoqi_hour)
    #     Guoqing_wanshang_hour_zou[idx].append(wanshang_hour)
    #     Guoqing_zaowanshang_hour_daozou.append(zaoqi_hour * 24 + wanshang_hour)
    #     Guoqing_kuadu.append(wanshang_hour - zaoqi_hour)
    #
    # for idx in enumerate(8):
    #     FEATURE += get_statistic_variable(Guoqing_zaoshang_hour_dao[idx])
    #     FEATURE += get_statistic_variable(Guoqing_wanshang_hour_zou[idx])
    #     FEATURE += get_statistic_variable(Guoqing_zaowanshang_hour_daozou[idx])
    #     FEATURE += get_statistic_variable(Guoqing_kuadu[idx])
    #
    # tmp1, tmp2, tmp3, tmp4 = [], [], [], []
    # for idx in enumerate(5):
    #     tmp1 += Guoqing_zaoshang_hour_dao[idx]
    #     tmp2 += Guoqing_wanshang_hour_zou[idx]
    #     tmp3 += Guoqing_zaowanshang_hour_daozou[idx]
    #     tmp4 += Guoqing_kuadu[idx]
    # FEATURE += get_statistic_variable(tmp1) + get_statistic_variable(tmp2) \
    #            + get_statistic_variable(tmp3) + get_statistic_variable(tmp4)
    #
    # tmp1, tmp2, tmp3, tmp4 = [], [], [], []
    # for idx in enumerate(5, 7):
    #     tmp1 += Guoqing_zaoshang_hour_dao[idx]
    #     tmp2 += Guoqing_wanshang_hour_zou[idx]
    #     tmp3 += Guoqing_zaowanshang_hour_daozou[idx]
    #     tmp4 += Guoqing_kuadu[idx]
    # FEATURE += get_statistic_variable(tmp1) + get_statistic_variable(tmp2) \
    #            + get_statistic_variable(tmp3) + get_statistic_variable(tmp4)

    name_jieri = 'yuandan'
    ########################################################################################################
    #  元旦
    jieri_dict = {}
    dates = [str(_) for _ in range(20181229, 20181232)] + ['20190101']
    for idx, date in enumerate(dates):
        jieri_dict[date] = idx
    FEATURE += get_jieri_feature(strings, jieri_dict, name_jieri=name_jieri)

    name_jieri = 'shengdan'
    ########################################################################################################
    #  圣诞节
    jieri_dict = {}
    dates = [str(_) for _ in range(20181224, 20181226)]
    for idx, date in enumerate(dates):
        jieri_dict[date] = idx
    FEATURE += get_jieri_feature(strings, jieri_dict, name_jieri=name_jieri)



    work_rest = 'work_rest'
    ########################################################################################################
    #  work day rest day sat day sun day
    FEATURE += get_work_rest_feature(strings)


    # shape = (用户人数, )
    #  天   小时   每天几小时   每天小时的std  做多一天几小时，  做少一天几小时, 25， 50 ， 75， 分位数
    Num_users_day_hour_information = np.zeros(
        (Num_users, 9))  # 10   100     10             0.5        16                3
    #     Num_users_day = np.zeros((Num_users, )) # 统计每个用户访问天数
    #     Num_users_hour = np.zeros((Num_users, )) # 统计每个用户访问小时数
    #     Num_users_hour_day_mean = [] #统计每个用户每天工作的小时数的均值
    #     Num_users_hour_day_std = [] #统计每个用户每天工作的小时数的方差
    #     Num_users_hour_day_per_25 = [] #统计每个用户每天工作的小时数的分位数25%
    #     Num_users_hour_day_per_50 = [] #统计每个用户每天工作的小时数的中位数
    #     Num_users_hour_day_per_75 = [] #统计每个用户每天工作的小时数的分位数75%
    #     Num_users_hour_day_max = [] #统计每个用户每天工作的小时数的max
    
    
    
    # ok到FEATURE                                            #  总共的差值   差值平均值   插值的std  最大插值，最小插值,  25， 50 ， 75 分位数
    
    Num_users_chazhi_information = np.zeros((Num_users, 8))  # 跨度100天     2.5天        0.5天       100天
    #     Num_users_chazhi_day_mean = [] # 统计每个用户差值的均值
    #     Num_users_chazhi_day_std = [] # 统计每个用户差值的方差
    #     Num_users_chazhi_day_per_25 = [] # 统计每个用户差值的分位数25%
    #     Num_users_chazhi_day_per_50 = [] # 统计每个用户差值的中位数
    #     Num_users_chazhi_day_per_75 = [] # 统计每个用户差值的分位数75%
    #     Num_users_chazhi_day_max = [] # 统计每个用户差值的max
    
    
    
    # shape = (>用户人数, )
    # ok到FEATURE
    
    Num_chazhi_Day = []  # 统计相邻天数的差值
    Num_users_hour_day = []  # 统计每天工作的小时数
    
    Num_users_aoye = np.zeros((Num_users,))  # 统计用户熬夜个数，  7点以后
    Num_users_zaoqi = np.zeros((Num_users,))  # 统计用户早起个数， 6点之前
    Num_users_tongxiao = np.zeros((Num_users,))  # 统计用户通宵个数， 前一天晚上7点以后， 到次日6点之前
    
    Num_day2day = np.zeros((dict_num, Num_users))
    #     Num_fri2mon = [] #统计每个用户礼拜5到礼拜1的差值个数
    #     Num_sat2mon = [] #统计每个用户礼拜6到礼拜1的差值个数
    #     Num_sun2mon = [] #统计每个用户礼拜7到礼拜1的差值个数
    #     Num_mon2tue = [] #统计每个用户礼拜1到礼拜2的差值个数
    #     Num_tue2wen = [] #统计每个用户礼拜2到礼拜3的差值个数
    #     Num_wen2thr = [] #统计每个用户礼拜3到礼拜4的差值个数
    #     Num_thr2fri = [] #统计每个用户礼拜4到礼拜5的差值个数
    #     Num_fri2sat = [] #统计每个用户礼拜5到礼拜6的差值个数
    #     Num_sat2sun = [] #统计每个用户礼拜6到礼拜7的差值个数
    
    
    # shape = (24, )   24 * 12 特征
    # ok到FEATURE
    
    Num_day2day_hour_pre_specific = [[] for i in range(dict_num)]
    Num_day2day_hour_next_specific = [[] for i in range(dict_num)]
    # Num_day2day_hour_pre = np.zeros((dict_num, 24))  # dict_num = 6
    # Num_day2day_hour_next = np.zeros((dict_num, 24))
    #     Num_fri2mon_24hour_for_pre = [] #统计礼拜5到礼拜1的差值个数, 且礼拜5最晚几点走
    #     Num_fri2mon_24hour_for_next = [] #统计礼拜5到礼拜1的差值个数， 且礼拜1几点到
    
    #     Num_sat2mon_24hour_for_pre = [] #统计礼拜6到礼拜1的差值个数, 且礼拜6最晚几点走
    #     Num_sat2mon_24hour_for_next = [] #统计礼拜6到礼拜1的差值个数， 且礼拜1几点到
    
    #     Num_sun2mon_24hour_for_pre = [] #统计礼拜7到礼拜1的差值个数, 且礼拜7最晚几点走
    #     Num_sun2mon_24hour_for_next = [] #统计礼拜7到礼拜1的差值个数， 且礼拜1几点到
    
    #     Num_mon2tue_24hour_for_pre = [] #统计礼拜1到礼拜2的差值个数, 且礼拜1最晚几点走
    #     Num_mon2tue_24hour_for_next = [] #统计礼拜1到礼拜2的差值个数， 且礼拜2几点到
    
    #     Num_tue2wen_24hour_for_pre = [] #统计礼拜2到礼拜3的差值个数, 且礼拜2最晚几点走
    #     Num_tue2wen_24hour_for_next = [] #统计礼拜2到礼拜3的差值个数， 且礼拜3几点到
    
    #     Num_wen2thr_24hour_for_pre = [] #统计礼拜3到礼拜4的差值个数, 且礼拜3最晚几点走
    #     Num_wen2thr_24hour_for_next = [] #统计礼拜3到礼拜4的差值个数， 且礼拜4几点到
    
    #     Num_thr2fri_24hour_for_pre = [] #统计礼拜4到礼拜5的差值个数, 且礼拜4最晚几点走
    #     Num_thr2fri_24hour_for_next = [] #统计礼拜4到礼拜5的差值个数， 且礼拜5几点到
    
    #     Num_fri2sat_24hour_for_pre = [] #统计礼拜5到礼拜6的差值个数, 且礼拜5最晚几点走
    #     Num_fri2sat_24hour_for_next = [] #统计礼拜5到礼拜6的差值个数， 且礼拜6几点到
    
    #     Num_sat2sun_24hour_for_pre = [] #统计礼拜6到礼拜7的差值个数, 且礼拜6最晚几点走
    #     Num_sat2sun_24hour_for_next = [] #统计礼拜6到礼拜7的差值个数， 且礼拜7几点到
    
    
    init_cishu = np.zeros((7, 26, 24))  # 统计26周每天每小时的用户人数
    init_renshu = np.zeros((7, 26))  #
    
    # if True:
    #     zaoshang_hour_workday_dao = []
    #     wanshang_hour_workday_zou = []
    #     zaowanshang_hour_workday_daozou =[]
    #
    #     zaoshang_hour_restday_dao = []
    #     wanshang_hour_restday_zou = []
    #     zaowanshang_hour_restday_daozou = []
    #
    #     zaoshang_hour_restday_dao_sat = []
    #     wanshang_hour_restday_zou_sat = []
    #     zaowanshang_hour_restday_daozou_sat = []
    #
    #     zaoshang_hour_restday_dao_sun = []
    #     wanshang_hour_restday_zou_sun = []
    #     zaowanshang_hour_restday_daozou_sun = []
    #
    #     work_day_kuadu = []
    #     rest_day_kuadu = []
    #     sat_day_kuadu = []
    #     sun_day_kuadu = []
    #




    
    ########################################################################################################
    
    #     print('\n 用户数目：', len(strings))
    
    for user_idx, string in enumerate(strings):
        temp = [[item[0:8], item[9:].split("|")] for item in string.split(',')]
        
        cnt_day, cnt_hour = len(temp), 0  # 统计工作天数，和工作小时数
        tmp = np.zeros((cnt_day,))
        for i, (date, visit_lst) in enumerate(temp):
            tmp[i] = len(visit_lst)
            cnt_hour += len(visit_lst)
            Num_users_hour_day.append(len(visit_lst))
            
            
            #  天   小时   每天几小时   每天小时的std     做多一天几小时  最少多少小时
        Num_users_day_hour_information[user_idx, :] = np.array(
            [cnt_day, cnt_hour, tmp.mean(), tmp.std(), tmp.max(), tmp.min()] + list(np.percentile(tmp, [25, 50, 75])))
        #         Num_users_day[user_idx] = cnt_day
        #         Num_users_hour[user_idx] = np.sum(tmp)
        
        
        jiange_day = []
        jiange_flag = 0
        for date, visit_lst in temp:
            x, y = date2position[datestr2dateint[date]]
            init_renshu[x][y] += 1  # 统计每小时的到访的总人数   7 * 26
            for visit in visit_lst: init_cishu[x][y][str2int[visit]] += 1  # 统计每天到访的总人数 7 * 26 * 24
            
            zaoqi_hour = str2int[visit_lst[0]]
            wanshang_hour = str2int[visit_lst[-1]]
            
            # if x < 5:  # workday
            #     zaoshang_hour_workday_dao.append(zaoqi_hour)
            #     wanshang_hour_workday_zou.append(wanshang_hour)
            #     work_day_kuadu.append(wanshang_hour - zaoqi_hour)
            # if x >= 5:
            #     zaoshang_hour_restday_dao.append(zaoqi_hour)
            #     wanshang_hour_restday_zou.append(wanshang_hour)
            #     rest_day_kuadu.append(wanshang_hour - zaoqi_hour)
            #     if x == 5:
            #         zaoshang_hour_restday_dao_sat.append(zaoqi_hour)
            #         wanshang_hour_restday_zou_sat.append(wanshang_hour)
            #         wanshang_hour_restday_zou_sat.append(wanshang_hour - zaoqi_hour)
            #     else:
            #         zaoshang_hour_restday_dao_sun.append(zaoqi_hour)
            #         wanshang_hour_restday_zou_sun.append(wanshang_hour)
            #         sun_day_kuadu.append(wanshang_hour - zaoqi_hour)
            
            if zaoqi_hour <= 6:  # 早起
                Num_users_zaoqi[user_idx] += 1
            
            if jiange_flag:  # 涉及到插值，昨天和今天的关系
                jiange_flag = 1
                
                day_cha = date2int[date] - date2int[pre_date]
                jiange_day.append(day_cha)
                
                if day_cha == 1:
                    idx = dict_day2day[(pre_x + 1, x + 1)]  # 前一天， 后一天
                    Num_day2day[idx, user_idx] += 1  # (6, 用户数)
                    # Num_day2day_hour_pre[idx, pre_aoye_hour] += 1  # 前一天 几点走
                    # Num_day2day_hour_next[idx, zaoqi_hour] += 1  # 后一天 几点到
                    Num_day2day_hour_pre_specific[idx].append(pre_aoye_hour)
                    Num_day2day_hour_next_specific[idx].append(zaoqi_hour)
                    
                    if zaoqi_hour <= 6 and pre_aoye_hour >= 7:  # 通宵
                        Num_users_tongxiao[user_idx] += 1
                
                if day_cha == 2 or day_cha == 3:
                    if (pre_x + 1, x + 1) == (5, 1) or (pre_x + 1, x + 1) == (6, 1):  # 礼拜五（六）晚上走  礼拜1早上几点到
                        idx = dict_day2day[(pre_x + 1, x + 1)]  # 前一天， 后一天
                        # Num_day2day_hour_pre[idx, pre_aoye_hour] += 1
                        # Num_day2day_hour_next[idx, zaoqi_hour] += 1
                        Num_day2day_hour_pre_specific[idx].append(pre_aoye_hour)
                        Num_day2day_hour_next_specific[idx].append(zaoqi_hour)
            
            pre_date = date
            pre_x, pre_y = x, y
            pre_aoye_hour = str2int[visit_lst[-1]]
            
            if pre_aoye_hour >= 7:  # 熬夜
                Num_users_aoye[user_idx] += 1
        
        Num_chazhi_Day += jiange_day
        jiange_day = np.array(jiange_day)
        Num_users_chazhi_information[user_idx, :] = np.array(get_statistic_variable(jiange_day))
    
    Num_chazhi_Day = np.array(Num_chazhi_Day)  # 统计相邻天数的差值
    Num_users_hour_day = np.array(Num_users_hour_day)  # 统计每天工作的小时数
    
    #     print(Jiange_Day)

    
    # 特征个数= 1 + 18= 19
    FEATURE += list(np.mean(Num_users_day_hour_information, axis=0)) + list(
        np.std(Num_users_day_hour_information, axis=0))
    # Num_users_day_hour_information: (Num_users,  9)
    #     print(len(FEATURE))
    
    # 特征个数= 19 + 8 * 3 = 43
    FEATURE += [np.sum(Num_users_aoye), Num_users_aoye.mean(), Num_users_aoye.std(), Num_users_aoye.max(),
                Num_users_aoye.min()] + list(np.percentile(Num_users_aoye, [25, 50, 75]))  # 统计用户熬夜个数，  7点以后
    FEATURE += [np.sum(Num_users_zaoqi), Num_users_zaoqi.mean(), Num_users_zaoqi.std(), Num_users_zaoqi.max(),
                Num_users_zaoqi.min()] + list(np.percentile(Num_users_aoye, [25, 50, 75]))  # 统计用户早起个数， 6点之前
    FEATURE += [np.sum(Num_users_tongxiao), Num_users_tongxiao.mean(), Num_users_tongxiao.std(),
                Num_users_tongxiao.max(), Num_users_tongxiao.min()] + list(
        np.percentile(Num_users_tongxiao, [25, 50, 75]))  # 统计用户通宵个数， 前一天晚上7点以后， 到次日6点之前
    #     print(len(FEATURE))
    
    # 特征个数 = 43 + 12 + 16 = 71
    FEATURE += list(np.mean(Num_day2day, axis=1)) + list(
        np.std(Num_day2day, axis=1))  # Num_day2day = np.zeros((9, Num_users))
    FEATURE += list(np.mean(Num_users_chazhi_information, axis=0)) + list(
        np.std(Num_users_chazhi_information, axis=0))  # Num_users_chazhi_information = np.zeros((Num_users,  8))
    #     print(len(FEATURE))
    
    # 特征个数 = 71 + 12 * 24  = 359
    # FEATURE += list(Num_day2day_hour_pre.flatten()) + list(Num_day2day_hour_next.flatten())  # (6, 24)
    for i in range(dict_num):
        FEATURE += get_statistic_variable(Num_day2day_hour_pre_specific[i])  # 8
        FEATURE += get_statistic_variable(Num_day2day_hour_next_specific[i])  # 8
    # print(len(FEATURE))
    
    # 特征个数 = 359 + 16 = 375
    FEATURE += get_statistic_variable(Num_chazhi_Day)
    FEATURE += get_statistic_variable(Num_users_hour_day)
    
    # print(len(FEATURE))
    # 特征个数 = 375 + 12 * 8 = 471
    # for tmp in [zaoshang_hour_workday_dao, wanshang_hour_workday_zou, zaoshang_hour_restday_dao,
    #             wanshang_hour_restday_zou,
    #             zaoshang_hour_restday_dao_sat, wanshang_hour_restday_zou_sat,
    #             zaoshang_hour_restday_dao_sun, wanshang_hour_restday_zou_sun,
    #             work_day_kuadu, rest_day_kuadu, sat_day_kuadu, sun_day_kuadu]:
    #     tmp = np.array(tmp)
    #     FEATURE += get_statistic_variable(tmp)
    
    FEATURE = np.array(FEATURE)
    
    # print('feature num =', len(FEATURE))
    # print('FEATURE max:', FEATURE.max())
    
    # assert len(FEATURE) == 471
    
    return init_cishu, init_renshu, list(FEATURE)


def visit2array(table):
    FEATURE = []
    # init_cishu, init_renshu, FEATURE_3 = get_feature_1_1(table)
    init_cishu, init_renshu, FEATURE_3 = get_feature_1_2(table)
    FEATURE_1 = get_feature_cishu(init_cishu)
    FEATURE_2 = get_feature_reshu(init_renshu)
    
#     print(len(FEATURE_1), len(FEATURE_2), len(FEATURE_3))
    FEATURE = FEATURE_1 + FEATURE_2 + FEATURE_3
    shape = np.array([len(FEATURE_1), len(FEATURE_2), len(FEATURE_3)])
    # print(len(FEATURE), shape)
    return FEATURE, init_cishu, shape


