# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import time
import os
from Config import config
from function_global_feature import get_global_feature_1, get_global_feature_2
from multiprocessing import Process

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

TEST_FLAG = False

train_num = 440000
test_num = 100000
# test_select_num = 10000
file_num_each_job_train = 10000
file_num_each_job_test = 2500
# file_num_each_job_test_select = 200
workers_train = int(train_num/file_num_each_job_train)
workers_test = int(test_num/file_num_each_job_test)


def get_global_feature(table):
    feature = []
    feature += get_global_feature_1(table)
    feature += get_global_feature_2(table)

    return feature

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

def visit2array_train(num):
    table = pd.read_csv(train_table_path, header=None)
    filenames = [a[0].split("/")[-1].split('.')[0] for a in table.values]
    length = len(filenames)
    start_time = time.time()
    all_features = []
    cnt_users = 0

    # for index, filename in enumerate(filenames):
    for index, filename in enumerate(filenames[num * file_num_each_job_train: (num + 1) * file_num_each_job_train]):
        table = pd.read_table(train_main_visit_path + filename + ".txt", header=None)
        users = table[0]
        cnt_users += len(users)
        # array = visit2array(table)
        # features, users = user_information(table, user_place_visit_num, num=num, label=int(filename[-1]) - 1)
        features = get_global_feature(table)

        all_features.append(features)
        sys.stdout.write(
            '\r>> Processing train visit data %d/%d --- feature len = %d' % (index + 1, length, len(features)))
        sys.stdout.flush()


    sys.stdout.write('\n')
    print('train users:', cnt_users)
    print("using time:%.2fs" % (time.time() - start_time))
    all_features = np.array(all_features)
    return all_features

# def visit2array_test(num):
#     table = pd.read_csv(test_table_path, header=None)
#     filenames = [a[0].split("/")[-1].split('.')[0] for a in table.values]
#     length = len(filenames)
#     start_time = time.time()
#     all_features = []
#     cnt_user = 0
#     # for index, filename in enumerate(filenames):
#     for index, filename in enumerate(filenames[num * file_num_each_job_test: (num + 1) * file_num_each_job_test]):
#         table = pd.read_table(test_main_visit_path + filename + ".txt", header=None)
#         users = table[0]
#         cnt_user += len(users)
#
#         features = get_global_feature(table)
#         all_features.append(features)
#         sys.stdout.write(
#             '\r>> Processing test  visit data %d/%d --- feature len = %d' % (index + 1, length, len(features)))
#         sys.stdout.flush()
#
#     sys.stdout.write('\n')
#     print('test users:', cnt_user)
#     print("using time:%.2fs" % (time.time() - start_time))
#     all_features = np.array(all_features)
#     return all_features

def write_pkl(data, fname):
    import pickle
    pickle.dump(data, open(fname, 'wb'))

def write_npy(data, fname):
    np.save(fname, data)

def run_train_feature(num):
    train_features, train_users = visit2array_train(num)
    write_npy(data=train_features, fname='./data/tmp/train_feature_user_{}.npy'.format(num))

# def run_test_feature(num):
#     test_feature, test_users = visit2array_test(num)
#     write_npy(data=test_feature, fname='./data/tmp/test_feature_user_{}.npy'.format(num))

def run():

    threads = []
    for i in range(workers_train):
        p = Process(target=run_train_feature, args=[i])
        threads.append(p)
        p.start()

    # for i in range(workers_test):
    #     p = Process(target=run_test_feature, args=[i])
    #     threads.append(p)
    #     p.start()

def get_final_train_features(workers=workers_train):
    for i in range(workers):
        tmp = np.load('./data/tmp/train_feature_user_{}.npy'.format(i))
        if i == 0:
            features = tmp
        else:
            features = np.concatenate((features, tmp))

    print('train_features shape =', features.shape)
    write_npy(data=features, fname='./feature/train_global_feature.npy')

# def get_final_test_features(workers=workers_test):
#     for i in range(workers):
#         tmp = np.load('./data/tmp/test_feature_user_{}.npy'.format(i))
#         if i == 0:
#             features = tmp
#         else:
#             features = np.concatenate((features, tmp))
#
#     print('test_features shape =', features.shape)
#     write_npy(data=features, fname='./feature/test_basic_feature.npy')
#

def main():
    if not os.path.exists("./data/tmp/"):
        os.makedirs("./data/tmp/")
    if not os.path.exists("./feature/"):
        os.makedirs("./feature/")

    print('start mian_2: ')

    Is_Run = True
    if Is_Run:
        print('Run:')
        run()
        print('run feature done!')
    else:
        print('start:')
        get_final_train_features(workers=workers_train)
        # get_final_test_features(workers=workers_test)
        print('feature save done!')

    print('all done!')

if __name__ == '__main__':
    main()