# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import time
import os
from Config import config
from function_global_feature import  get_global_feature_1, get_global_feature_2,\
    get_global_feature_3, get_global_feature_4

if not os.path.exists("./data/tmp/"):
    os.makedirs("./data/tmp/")
if not os.path.exists("./feature/"):
    os.makedirs("./feature/")
if not os.path.exists("./output/"):
    os.makedirs("./output/")

main_data_path = config.main_data_path

train_feature_out_path = config.train_feature_out_path  # './feature/train/'
train_table_path = config.train_table_path  # main_data_path + 'train.txt'
train_main_visit_path = config.train_main_visit_path  # main_data_path + "train_visit/train/"

test_feature_out_path = config.test_feature_out_path  # './feature/test/'
test_table_path = config.test_table_path  # main_data_path + 'test.txt'
test_main_visit_path = config.test_main_visit_path  # main_data_path + "test_visit/test/"


def golbal_feature(table, num=None):
    if num == 1:
        return get_global_feature_1(table)
    if num == 2:
        return get_global_feature_2(table)
    if num == 3:
        return get_global_feature_3(table)
    if num == 4:
        return get_global_feature_4(table)

def get_statistic_variable(tmp):
    n_out = 8
    tmp = np.array(tmp).flatten()
    if len(tmp) > 0:
        # return [np.sum(tmp), tmp.mean(), tmp.std(), tmp.max(), tmp.min()] + list(
        #     np.percentile(tmp, [50]))  # shape = (8, )
        return [np.sum(tmp), tmp.mean(), tmp.std(), tmp.max(), tmp.min()] + list(
            np.percentile(tmp, [25, 50, 75]))  # shape = (8, )
    else:
        return list(np.zeros((8,)) - 0)

def user_information(table, user_place_visit_num, num=None, label=None):
    users = table[0]
    strings = table[1]
    global_feature, len_feature = golbal_feature(table, num=num)

    f_n_user = []

    for user, string in zip(users, strings):
        user_place_visit_num_ = np.zeros((len_feature, 9))

        temp = []
        for item in string.split(','):
            temp.append([item[0:8], item[9:].split("|")])

        if user in user_place_visit_num:
            user_place_visit_num_ += user_place_visit_num[user]
        ###########################################################Important
        if label is not None:
            user_place_visit_num_[:, label] -= global_feature
        ###########################################################
        f_n_user.append(user_place_visit_num_.flatten())

    features = []
    f_n_user = np.array(f_n_user)
    assert f_n_user.shape[-1] == len_feature * 9
    for index in range(len_feature * 9):
        features += get_statistic_variable(f_n_user[:, index])

    return features, users


def static_user_place_num(num=None):
    user_place_visit_num = {}
    table = pd.read_csv(train_table_path, header=None)
    filenames = [a[0].split("/")[-1].split('.')[0] for a in table.values]
    length = len(filenames)
    start_time = time.time()
    cnt_users = 0
    for index, filename in enumerate(filenames):
        table = pd.read_table(train_main_visit_path + filename + ".txt", header=None)
        label = int(filename.split("_")[1]) - 1
        users = table[0]
        strings = table[1]
        cnt_users += len(users)
        global_feature, len_feature = golbal_feature(table, num=num)
        for user, string in zip(users, strings):
            temp = []
            for item in string.split(','):
                temp.append([item[0:8], item[9:].split("|")])

            if user not in user_place_visit_num:
                user_place_visit_num[user] = np.zeros((len_feature, 9))
            user_place_visit_num[user][:, label] += global_feature
        sys.stdout.write(
            '\r>> Processing visit data %d/%d --- global feature len = %d' % (index + 1, length, len_feature))
        sys.stdout.flush()
    sys.stdout.write('\n')
    print('train users:', cnt_users)
    print("using time:%.2fs" % (time.time() - start_time))
    print('totoal users:', cnt_users)
    return user_place_visit_num


def visit2array_train(user_place_visit_num, num=None, stop_num=2, flag=True):
    table = pd.read_csv(train_table_path, header=None)
    filenames = [a[0].split("/")[-1].split('.')[0] for a in table.values]
    length = len(filenames)
    start_time = time.time()
    total_users = []
    all_features = []
    cnt_users = 0
    for index, filename in enumerate(filenames):
        table = pd.read_table(train_main_visit_path + filename + ".txt", header=None)
        # array = visit2array(table)
        features, users = user_information(table, user_place_visit_num, num=num, label=int(filename[-1]) - 1)

        cnt_users += len(users)
        total_users += list(users.values)
        all_features.append(features)
        sys.stdout.write(
            '\r>> Processing train visit data %d/%d --- feature len = %d' % (index + 1, length, len(features)))
        sys.stdout.flush()

        if flag and index >= stop_num:
            break

    sys.stdout.write('\n')
    print('train users:', cnt_users)
    print("using time:%.2fs" % (time.time() - start_time))
    all_features = np.array(all_features)
    return all_features, total_users

def visit2array_test(user_place_visit_num, num=None, stop_num=2, flag=True):
    table = pd.read_csv(test_table_path, header=None)
    filenames = [a[0].split("/")[-1].split('.')[0] for a in table.values]
    length = len(filenames)
    start_time = time.time()
    total_users = []
    all_features = []
    cnt_user = 0
    for index, filename in enumerate(filenames):
        table = pd.read_table(test_main_visit_path + filename + ".txt", header=None)
        features, users = user_information(table, user_place_visit_num, num=num)

        cnt_user += len(users)
        total_users += list(users.values)
        all_features.append(features)
        sys.stdout.write(
            '\r>> Processing test  visit data %d/%d --- feature len = %d' % (index + 1, length, len(features)))
        sys.stdout.flush()

        if flag and index >= stop_num:
            break
    sys.stdout.write('\n')
    print('test users:', cnt_user)
    print("using time:%.2fs" % (time.time() - start_time))
    all_features = np.array(all_features)
    return all_features, total_users


def main(num, flag=True):
    global_local = 'global'
    data_name = ['day', 'hour', 'work_rest_fangjia_day', 'work_rest_fangjia_hour']

    train_data_name = 'train_X_UserID_normal_%s_%s.npy' % (global_local, data_name[num - 1])
    test_data_name = 'test_X_UserID_normal_%s_%s.npy' % (global_local, data_name[num - 1])

    print('start: num', num)
    user_place_visit_num = static_user_place_num(num=num)
    print('data process done!')
    print(train_data_name)
    print(test_data_name)

    train_features, train_user = visit2array_train(user_place_visit_num, num=num, flag=flag)
    np.save('./feature/' + train_data_name, train_features)

    test_feature, test_user = visit2array_test(user_place_visit_num, num=num, flag=flag)
    np.save('./feature/' + test_data_name, test_feature)

    print('train_features & test_feature shape:', train_features.shape, test_feature.shape)

    print('(train_users, test_users) = (%d, %d)' % (len(train_user), len(test_user)))
    train_user, test_user = set(train_user), set(test_user)
    print('unique -- (train_users, test_users) = (%d, %d)' % (len(train_user), len(test_user)))

    common_users = train_user & test_user
    different_users = test_user - common_users
    print('common users:', len(common_users))
    print('different users:', len(different_users))

    print('train users:', len(train_user))
    print('test user:', len(test_user))
    print('common user:', len(train_user & test_user))

    print('all done!')