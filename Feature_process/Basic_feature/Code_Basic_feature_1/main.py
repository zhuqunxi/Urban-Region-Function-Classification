import time
import numpy as np
import sys
import datetime
import pandas as pd
import os
from Config import config
from feature import visit2array # config #visit2array
from multiprocessing import Process

flag = config.True_Small  # True: 只测试小数据
main_data_path = config.main_data_path

train_feature_out_path = config.train_feature_out_path  # './feature/train/'
if flag:
    train_feature_out_path = train_feature_out_path[:-1] + '_small/'

train_table_path = config.train_table_path  # main_data_path + 'train.txt'
train_main_visit_path = config.train_main_visit_path  # main_data_path + "train_visit/train/"

test_feature_out_path = config.test_feature_out_path  # './feature/test/'
if flag:
    test_feature_out_path = test_feature_out_path[:-1] + '_small/'

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
# workers_test_select = int(test_select_num/file_num_each_job_test_select)


def visit2array_train(num):
    table = pd.read_csv(train_table_path, header=None)
    filenames = [a[0].split("/")[-1].split('.')[0] for a in table.values]
    length = len(filenames)
    start_time = time.time()
    total_users = []
    All_feature = []
    cnt_users = 0
    for index, filename in enumerate(filenames[num*file_num_each_job_train: (num+1) * file_num_each_job_train]):
        if TEST_FLAG:
            if index > 5:
                break
        table = pd.read_table(train_main_visit_path + filename + ".txt", header=None)
        feature, init_cishu, f_shape = visit2array(table)
        users = table[0]

        # np.save('/home/download-20190701/npy/train_visit/' + filename + '.npy', init_cishu)

        cnt_users += len(users)
        total_users += list(users.values)
        All_feature.append(feature)
        # users = table[0]
        # total_users += list(users.values)
        sys.stdout.write('\r>> Processing train visit data %d/%d' % (index + 1, length))
        sys.stdout.flush()

    sys.stdout.write('\n')
    print('train users:', cnt_users)
    print("using time:%.2fs" % (time.time() - start_time))
    return All_feature, total_users

def visit2array_test(num):
    table = pd.read_csv(test_table_path, header=None)
    filenames = [a[0].split("/")[-1].split('.')[0] for a in table.values]
    length = len(filenames)
    start_time = time.time()
    total_users = []
    All_feature = []
    cnt_user = 0
    for index, filename in enumerate(filenames[num*file_num_each_job_test: (num+1) * file_num_each_job_test]):
        if TEST_FLAG:
            if index > 5:
                break
        table = pd.read_table(test_main_visit_path + filename + ".txt", header=None)
        feature, init_cishu, f_shape = visit2array(table)
        users = table[0]

        # np.save('/home/download-20190701/npy/test_visit/' + filename + '.npy', init_cishu)

        cnt_user += len(users)
        total_users += list(users.values)
        All_feature.append(feature)
        # users = table[0]
        # total_users += list(users.values)
        sys.stdout.write('\r>> Processing test visit data %d/%d' % (index + 1, length))
        sys.stdout.flush()

    sys.stdout.write('\n')
    print('test users:', cnt_user)
    print("using time:%.2fs" % (time.time() - start_time))
    return All_feature, total_users

def write_npy(data, fname):
    np.save(fname, data)

def write_pkl(data, fname):
    import pickle
    pickle.dump(data, open(fname, 'wb'))
    # np.savetxt(fname, feature, delimiter=',', fmt='%.4e')

def run_train_feature(num):
    train_features, train_users = visit2array_train(num)
    # write_pkl(data=train_features, fname='./data/tmp/train_feature_user_{}.pkl'.format(num))
    write_npy(data=train_features, fname='./data/tmp/train_feature_user_{}.npy'.format(num))
    write_pkl(data=train_users, fname='./data/tmp/train_users_{}.pkl'.format(num))

def run_test_feature(num):
    test_feature, test_users = visit2array_test(num)
    # write_pkl(data=test_feature, fname='./data/tmp/test_feature_user_{}.pkl'.format(num))
    write_npy(data=test_feature, fname='./data/tmp/test_feature_user_{}.npy'.format(num))
    write_pkl(data=test_users, fname='./data/tmp/test_users_{}.pkl'.format(num))

def run():

    threads = []
    for i in range(workers_train):
        p = Process(target=run_train_feature, args=[i])
        threads.append(p)
        p.start()

    for i in range(workers_test):
        p = Process(target=run_test_feature, args=[i])
        threads.append(p)
        p.start()


def get_final_train_features(workers=workers_train):
    import pickle
    for i in range(workers):
        tmp = np.load('./data/tmp/train_feature_user_{}.npy'.format(i))
        tmp_user = pickle.load(open('./data/tmp/train_users_{}.pkl'.format(i), 'rb'))
        if i == 0:
            features = tmp
            users = tmp_user
        else:
            features = np.concatenate((features, tmp))
            users += tmp_user

    print('train_features shape =', features.shape)
    write_npy(data=features, fname='./feature/train_basic_feature.npy')

    return users


def get_final_test_features(workers=workers_test):
    import pickle
    for i in range(workers):
        tmp = np.load('./data/tmp/test_feature_user_{}.npy'.format(i))
        tmp_user = pickle.load(open('./data/tmp/test_users_{}.pkl'.format(i), 'rb'))
        if i == 0:
            features = tmp
            users = tmp_user
        else:
            features = np.concatenate((features, tmp))
            users += tmp_user

    print('test_features shape =', features.shape)
    write_npy(data=features, fname='./feature/test_basic_feature.npy')

    return users

if __name__ == '__main__':
    if not os.path.exists("./data/tmp/"):
        os.makedirs("./data/tmp/")
    if not os.path.exists("./feature/"):
        os.makedirs("./feature/")

    """
    step 1: Is_Run = True, 跑分布式的特征
    step 2: Is_Run = False, 合并分布式的特征
    """
    Is_Run = True
    if Is_Run:
        run()
        print('run feature done!')
    else:
        print('start:')
        train_user = get_final_train_features(workers=workers_train)
        test_user = get_final_test_features(workers=workers_test)

        print('feature save done!')

        print('(train_users, test_users) = (%d, %d)' % (len(train_user), len(test_user)))
        train_user, test_user = set(train_user), set(test_user)
        print('unique -- (train_users, test_users) = (%d, %d)' % (len(train_user), len(test_user)))

        common_users = train_user & test_user
        different_users = test_user - common_users
        print('common users:', len(common_users))
        print('different users:', len(different_users))

        # print('train users:', len(train_user))
        # print('test user:', len(test_user))
        # print('common user:', len(train_user & test_user))

        print('all done!')