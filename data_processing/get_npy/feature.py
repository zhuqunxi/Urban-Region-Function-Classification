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


def get_init_cishu(table):
    strings = table[1]
    init_cishu = np.zeros((7, 26, 24))  # 统计26周每天每小时的用户人数
    for user_idx, string in enumerate(strings):
        temp = [[item[0:8], item[9:].split("|")] for item in string.split(',')]
        for date, visit_lst in temp:
            x, y = date2position[datestr2dateint[date]]
            for visit in visit_lst:
                init_cishu[x][y][str2int[visit]] += 1  # 统计每天到访的总人数 7 * 26 * 24
    return init_cishu

def visit2array(table):
    init_cishu = get_init_cishu(table)

    return init_cishu


