from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import lightgbm as lgbm
from check_feature_is_good import ceshi
import time

data_input_name = 'feature'
data_file_path = './%s/' % data_input_name
y_name = 'y_train_44w.npy'
y = np.load(data_file_path + y_name)

clf = RandomForestClassifier(random_state=42)

X_name_1 = 'train_basic_feature_%d.npy' % 1
X_T_name_1 = 'test_basic_feature_%d.npy' % 1
X_name_3 = 'train_basic_feature_%d.npy' % 3
X_T_name_3 = 'test_basic_feature_%d.npy' % 3

X = np.concatenate([np.load(data_file_path + X_name_1), np.load(data_file_path + X_name_3)], axis=1)
X_T = np.concatenate([np.load(data_file_path + X_T_name_1), np.load(data_file_path + X_T_name_3)], axis=1)


st = time.time()
print('X shape:', X.shape)
clf.fit(X, y)
print('fitting time: %.2f s' % (time.time() - st))

print('start:')
Acc = []
# thrd = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
thrd = [1.0, 0.75, 0.5, 0.25]
reduce_f_len = []
for th in thrd:
# for th in [1.5]:
    model = SelectFromModel(clf, prefit=True, threshold='{} * mean'.format(th))
    X_new = model.transform(X)
    X_T_new = model.transform(X_T)

    del X, X_T, model, clf

    print('#' * 60)
    print('threshold={} * mean'.format(th))
    print('X_new shape:', X_new.shape)
    print('X_T_new shape:', X_T_new.shape)
    reduce_f_len.append(X_new.shape[-1])

    np.save(data_file_path + 'train_basic_13_RF_%d.npy' % X_new.shape[-1], X_new)
    np.save(data_file_path + 'test_basic_13_RF_%d.npy' % X_T_new.shape[-1], X_T_new)

	# 如果不想测试准确率，直接注释掉
    acc = ceshi([lgbm.LGBMClassifier(random_state=42, n_jobs=30, n_estimators=500)], X_new, y)

    print('acc:', acc)
    Acc.append(acc)

    print('*' * 20)
    print('thresold:', thrd)
    print('reduce_f_len:', reduce_f_len)
    # print('Acc: ', Acc)


print('result:')
print('reduce_f_len:', reduce_f_len)
print('thresold:', thrd)
print('Acc: ', Acc)

