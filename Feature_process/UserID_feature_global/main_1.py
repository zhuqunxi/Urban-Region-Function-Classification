# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import time
import os
from Config import config
from function_global_feature import get_global_feature_1, get_global_feature_2

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



def write_pkl(data, fname):
    import pickle
    pickle.dump(data, open(fname, 'wb'))

def main():
    if not os.path.exists("./data/tmp/"):
        os.makedirs("./data/tmp/")
    if not os.path.exists("./feature/"):
        os.makedirs("./feature/")

    print('start: ')
    user_place_visit_num = static_user_place_num()
    write_pkl(data=user_place_visit_num, fname='./data/tmp/user_place_visit_num.pkl')

    print('user_place_visit_num process done!')

    # global_local = 'global'
    # train_data_name = 'train_X_UserID_normal_%s_%s.npy' % (global_local, 'feature')
    # test_data_name = 'test_X_UserID_normal_%s_%s.npy' % (global_local, 'feature')
    # print(train_data_name)
    # print(test_data_name)
    #
    # train_features, train_user = visit2array_train(user_place_visit_num, num=num, flag=flag)
    # np.save('./feature/' + train_data_name, train_features)
    #
    # test_feature, test_user = visit2array_test(user_place_visit_num, num=num, flag=flag)
    # np.save('./feature/' + test_data_name, test_feature)
    #
    # print('train_features & test_feature shape:', train_features.shape, test_feature.shape)
    #
    # print('(train_users, test_users) = (%d, %d)' % (len(train_user), len(test_user)))
    # train_user, test_user = set(train_user), set(test_user)
    # print('unique -- (train_users, test_users) = (%d, %d)' % (len(train_user), len(test_user)))
    #
    # common_users = train_user & test_user
    # different_users = test_user - common_users
    # print('common users:', len(common_users))
    # print('different users:', len(different_users))
    #
    # print('train users:', len(train_user))
    # print('test user:', len(test_user))
    # print('common user:', len(train_user & test_user))
    #
    # print('all done!')


if __name__ == '__main__':
    main()