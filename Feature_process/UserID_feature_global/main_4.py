# -*- coding: utf-8 -*-
import numpy as np
import pickle
import sys
import time

def get_9(value, y_train):
    tmp = np.zeros(9)
    for index in value:
        label = y_train[index] - 1
        tmp[label] += 1
    return tmp

print('start read pkl: ')
user_place_visit_num = pickle.load(open('./data/tmp/user_place_visit_num.pkl', 'rb'))
print('user_place_visit_num read done!')

y_train = np.load('/home/For_U_44w/' + 'y_train_44w.npy')
print('y_train loaded -- shape = ', y_train.shape)

Max_CNT_label = np.zeros(9)
ind = 0
for key, value in user_place_visit_num.items():
    tmp = get_9(value, y_train)
    Max_CNT_label = [max(Max_CNT_label[_], tmp[_]) for _ in range(9)]
    ind += 1
    # user_place_visit_num.pop(key)
    if not (ind % 10000 == 0):
        continue
    sys.stdout.write(
        '\r>> Processing train visit data %d/%d' % (ind + 1, 95987868))
    sys.stdout.flush()

sys.stdout.write('\n')

print('Max_CNT_label:', Max_CNT_label)





