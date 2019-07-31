from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import lightgbm as lgbm
# from check_feature_is_good import ceshi
import time

data_file_path = '/home/For_U_44w/'
y_name = 'y_train_44w.npy'
y = np.load(data_file_path + y_name)

clf = RandomForestClassifier(random_state=42)

# XX = []
# XX_T = []
# for i in [1, 3]:
#     X_name = 'train_basic_feature_%d.npy' % i
#     X_T_name = 'test_basic_feature_%d.npy' % i
#     X = np.load(data_file_path + X_name)
#     X_T = np.load(data_file_path + X_T_name)
#     XX.append(X)
#     XX_T.append(X_T)
#     print('X, X_T:', X.shape, X_T.shape)
#
# X = np.concatenate(XX, axis=1)
# X_T = np.concatenate(XX_T, axis=1)

# X_name_1 = 'train_basic_feature_%d.npy' % 1
# X_T_name_1 = 'test_basic_feature_%d.npy' % 1
# X_name_3 = 'train_basic_feature_%d.npy' % 3
# X_T_name_3 = 'test_basic_feature_%d.npy' % 3

# X = np.concatenate([np.load(data_file_path + X_name_1), np.load(data_file_path + X_name_3)], axis=1)
# X_T = np.concatenate([np.load(data_file_path + X_T_name_1), np.load(data_file_path + X_T_name_3)], axis=1)

X = np.load(data_file_path + 'train_basic_13_RF_1581.npy')
X_T = np.load(data_file_path + 'test_basic_13_RF_1581.npy')

st = time.time()
print('X shape:', X.shape)
clf.fit(X, y)
print('fitting time: %.2f s' % (time.time() - st))

print('start:')
Acc = []
# thrd = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
thrd = [20, 10, 5, 1.5, 1.0]
reduce_f_len = []
for th in thrd:
# for th in [1.5]:
    model = SelectFromModel(clf, prefit=True, threshold='{} * mean'.format(th))
    X_new = model.transform(X)
    X_T_new = model.transform(X_T)

    # del X, X_T, model, clf

    print('#' * 60)
    print('threshold={} * mean'.format(th))
    print('X_new shape:', X_new.shape)
    print('X_T_new shape:', X_T_new.shape)
    reduce_f_len.append(X_new.shape[-1])

    np.save(data_file_path + 'train_basic_13_RF_%d.npy' % X_new.shape[-1], X_new)
    np.save(data_file_path + 'test_basic_13_RF_%d.npy' % X_T_new.shape[-1], X_T_new)

    # acc = ceshi([lgbm.LGBMClassifier(random_state=42, n_jobs=-1)], X_new, y)
    # acc = ceshi([lgbm.LGBMClassifier(random_state=42, n_jobs=30, n_estimators=500)], X_new, y)

    # print('acc:', acc)
    # Acc.append(acc)

    print('*' * 20)
    print('thresold:', thrd)
    print('reduce_f_len:', reduce_f_len)
    # print('Acc: ', Acc)


print('result:')
print('reduce_f_len:', reduce_f_len)
print('thresold:', thrd)
# print('Acc: ', Acc)



'''
f_len =     [4076,                  2906,               2009,               1363                932
thresold =  [0.5,                   0.75,               1.0,                1.25,               1.5]')
Accc =      [ 0.6628744022149509,   0.661213189025925,  0.6605336018122325, 0.6612635288195319, 0.6605084319154292]

combine 1 & 2
All_data (6903,  0.6618172665492071)

RF_1381   0.6565064183236848

train_basic_feature_3.npy       combine 1 & 3
0.6493581676315127,             0.6621193053108482

f_len 2450
acc: 0.6620689655172414

reduce_f_len: [12601, 9807, 7331, 5352, 3916, 2843]
thresold: [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
Acc:	     [-,	-,	-,        	   -,	-]


f_len =         [1577       ]
thresold =      [1.5        ]
fitting time:   [1522.39 s  ]
acc =           [0.6573335666608334]

44w shuju
thrd = 1.5
reduce_f_len: [1581]
Acc:  [0.6678143287313221]

reduce_f_len: [3738]
Acc:  [0.6708596102494176]


'''