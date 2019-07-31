# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import time
import os
from Config import config
from function_global_feature import get_global_feature_1, get_global_feature_2
from multiprocessing import Process


import pickle
print('start read pkl: ')
user_place_visit_num = pickle.load(open('./data/tmp/user_place_visit_num.pkl', 'rb'))
print('user_place_visit_num read done!')

# All_global_feature = np.load('./feature/train_global_feature.npy')
# All_global_feature = np.load('/home/For_U_44w/' + 'train_basic_13_RF_106.npy')
All_global_feature = np.load('/home/For_U_44w/' + 'train_basic_13_RF_21.npy')
len_feature = All_global_feature.shape[-1]
print('All_global_feature loaded -- shape =', All_global_feature.shape)

y_train = np.load('/home/For_U_44w/' + 'y_train_44w.npy')
print('y_train loaded -- shape = ', y_train.shape)


main_data_path = config.main_data_path

train_feature_out_path = config.train_feature_out_path  # './feature/train/'
train_table_path = config.train_table_path  # main_data_path + 'train.txt'
train_main_visit_path = config.train_main_visit_path  # main_data_path + "train_visit/train/"

test_feature_out_path = config.test_feature_out_path  # './feature/test/'
test_table_path = config.test_table_path  # main_data_path + 'test.txt'
test_main_visit_path = config.test_main_visit_path  # main_data_path + "test_visit/test/"

TEST_FLAG = False

train_num = 440000
test_num = 100000
# test_select_num = 10000
file_num_each_job_train = 110000
file_num_each_job_test = 100000
# file_num_each_job_test_select = 200
workers_train = int(train_num/file_num_each_job_train)
workers_test = int(test_num/file_num_each_job_test)


def get_global_feature(table):
    feature = []
    feature += get_global_feature_1(table)
    feature += get_global_feature_2(table)

    return feature

# def get_statistic_variable(tmp):
#     n_out = 8
#     tmp = np.array(tmp).flatten()
#     if len(tmp) > 0:
#         # return [np.sum(tmp), tmp.mean(), tmp.std(), tmp.max(), tmp.min()] + list(
#         #     np.percentile(tmp, [50]))  # shape = (8, )
#         return [np.sum(tmp), tmp.mean(), tmp.std(), tmp.max(), tmp.min()] + list(
#             np.percentile(tmp, [25, 50, 75]))  # shape = (8, )
#     else:
#         return list(np.zeros((8,)) - 0)


def get_statistic_variable(tmp):
    n_out = 4
    tmp = np.array(tmp).flatten()
    if len(tmp) > 0:
        # return [np.sum(tmp), tmp.mean(), tmp.std(), tmp.max(), tmp.min()] + list(
        #     np.percentile(tmp, [50]))  # shape = (8, )
        # return [tmp.mean(), tmp.std()] #, np.sum(tmp),
        return list(np.percentile(tmp, [25, 50, 75])) + [np.sum(tmp)]
    else:
        return list(np.zeros((n_out,)) - 0)


def get_user_stat(user):
    tmp = np.zeros((len_feature, 9))
    lst = user_place_visit_num[user]
    for index in lst:
        label = y_train[index] - 1
        tmp[:, label] += All_global_feature[index]

    return tmp

def user_information(table, label=None, index=None):
    users = table[0]
    strings = table[1]

    if label is not None:
        global_feature = All_global_feature[index]

    f_n_user = []

    for user, string in zip(users, strings):
        user_place_visit_num_ = np.zeros((len_feature, 9))

        temp = []
        for item in string.split(','):
            temp.append([item[0:8], item[9:].split("|")])

        if user in user_place_visit_num:
            # user_place_visit_num_ += user_place_visit_num[user]
            user_place_visit_num_ += get_user_stat(user)
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

    return features

def static_user_place_num():
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
        cnt_users += len(users)
        # global_feature, len_feature = golbal_feature(table, num=num)
        for user in users:
            if user not in user_place_visit_num:
                user_place_visit_num[user] = []
            user_place_visit_num[user].append(index)
        sys.stdout.write(
            '\r>> Processing visit data %d/%d' % (index + 1, length))
        sys.stdout.flush()
    sys.stdout.write('\n')
    print('totoal users:', cnt_users)
    print("using time:%.2fs" % (time.time() - start_time))

    return user_place_visit_num


def visit2array_train(num):
    table = pd.read_csv(train_table_path, header=None)
    filenames = [a[0].split("/")[-1].split('.')[0] for a in table.values]
    length = len(filenames)
    start_time = time.time()
    all_features = []
    cnt_users = 0

    # for index, filename in enumerate(filenames):
    for ind, filename in enumerate(filenames[num * file_num_each_job_train: (num + 1) * file_num_each_job_train]):
        index = ind + num * file_num_each_job_train
        table = pd.read_table(train_main_visit_path + filename + ".txt", header=None)
        users = table[0]
        cnt_users += len(users)
        features = user_information(table, label=int(filename[-1]) - 1, index=index)
        # features = get_global_feature(table)

        all_features.append(features)
        sys.stdout.write(
            '\r>> Processing train visit data %d/%d --- feature len = %d, num = %d' % (ind + 1, length, len(features), num))
        sys.stdout.flush()


    sys.stdout.write('\n')
    print('train users:', cnt_users)
    print("using time:%.2fs" % (time.time() - start_time))
    all_features = np.array(all_features)
    return all_features


def visit2array_test(num):
    table = pd.read_csv(test_table_path, header=None)
    filenames = [a[0].split("/")[-1].split('.')[0] for a in table.values]
    length = len(filenames)
    start_time = time.time()
    all_features = []
    cnt_user = 0
    # for index, filename in enumerate(filenames):
    for ind, filename in enumerate(filenames[num * file_num_each_job_test: (num + 1) * file_num_each_job_test]):
        index = ind + num * file_num_each_job_test
        table = pd.read_table(test_main_visit_path + filename + ".txt", header=None)
        users = table[0]
        cnt_user += len(users)
        features = user_information(table)
        # features = get_global_feature(table)

        all_features.append(features)
        sys.stdout.write(
            '\r>> Processing test  visit data %d/%d --- feature len = %d, num = %d' % (ind + 1, length, len(features), num))
        sys.stdout.flush()

    sys.stdout.write('\n')
    print('test users:', cnt_user)
    print("using time:%.2fs" % (time.time() - start_time))
    all_features = np.array(all_features)
    return all_features

def write_pkl(data, fname):
    import pickle
    pickle.dump(data, open(fname, 'wb'))

def write_npy(data, fname):
    np.save(fname, data)

def run_train_feature(num):
    train_features = visit2array_train(num)
    write_npy(data=train_features, fname='./data/tmp_1/train_feature_user_{}.npy'.format(num))

def run_test_feature(num):
    test_feature = visit2array_test(num)
    write_npy(data=test_feature, fname='./data/tmp_1/test_feature_user_{}.npy'.format(num))

def run():

    threads = []
    # lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    #        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41]
    cnt = 0
    for i in range(workers_train):
        # if i in lst:
        #     continue
        # cnt += 1
        # if cnt > 8:
        #     break
        p = Process(target=run_train_feature, args=[i])
        threads.append(p)
        p.start()

    # lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22,
    #        23, 24, 25, 26, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    for i in range(workers_test):
        # if i in lst:
        #     continue
        p = Process(target=run_test_feature, args=[i])
        threads.append(p)
        p.start()

def get_final_train_features(data_name, workers=workers_train):
    for i in range(workers):
        tmp = np.load('./data/tmp_1/train_feature_user_{}.npy'.format(i))
        if i == 0:
            features = tmp
        else:
            features = np.concatenate((features, tmp))

    print('train_features shape =', features.shape)
    np.save('./feature/' + data_name, features)

def get_final_test_features(data_name, workers=workers_test):
    for i in range(workers):
        tmp = np.load('./data/tmp_1/test_feature_user_{}.npy'.format(i))
        if i == 0:
            features = tmp
        else:
            features = np.concatenate((features, tmp))

    print('test_features shape =', features.shape)
    np.save('./feature/' + data_name, features)



if __name__ == '__main__':
    if not os.path.exists("./data/tmp_1/"):
        os.makedirs("./data/tmp_1/")
    if not os.path.exists("./feature/"):
        os.makedirs("./feature/")

    global_local = 'global'
    train_data_name = 'train_X_UserID_normal_%s_%s_more.npy' % (global_local, 'feature')
    test_data_name = 'test_X_UserID_normal_%s_%s_more.npy' % (global_local, 'feature')
    print(train_data_name)
    print(test_data_name)

    Is_Run = True
    if Is_Run:
        print('Run run feature:')
        run()
        print('run feature done!')
    else:
        print('start saving feature:')
        get_final_train_features(train_data_name, workers=workers_train)
        get_final_test_features(test_data_name, workers=workers_test)
        print('feature save done!')