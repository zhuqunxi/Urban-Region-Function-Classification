import pickle  # pickle

import pandas as pd
import time
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgbm
import warnings
from scipy.special import softmax

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

data_file_path = '/home/For_U_44w/'

def ceshi(classifiers, X, y, X_T):
    log_cols = ["Classifier", "Accuracy"]
    log = pd.DataFrame(columns=log_cols)

    # sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    # sss = KFold(n_splits=5, shuffle=True, random_state=0)
    sss = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    acc_dict = {}

    print('start:')
    ST = time.time()
    cnt_val = 0
    XG_LGBM_prob_Test = {}
    prob_val = np.zeros((len(y), 9)) + 0.0
    tmp = np.zeros((len(y),))
    for train_index, test_index in sss.split(X, y):
        cnt_val += 1
        # if cnt_val <= 2:
        #     continue
        print('validation %d' % cnt_val)
        print('valiadation len:', len(test_index))
        tmp[test_index] = 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for clf in classifiers:
            name = clf.__class__.__name__
            print('name:', name)
            # print('weight:', Weight[name])
            st = time.time()
            clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
                    verbose=True, early_stopping_rounds=30)
            print('fitting time: %.2f s' % (time.time() - st))

            # save Model
            # with open('./ENSEMBLE_model/%s_fold_%d_num_n_f_%d.pickle' % (name, cnt_val, n_f), 'wb') as f:
            #     pickle.dump(clf, f)

            if name == "XGBClassifier":
                best_iteration = clf.get_booster().best_iteration
            else:
                best_iteration = clf.best_iteration_

            test_pred_prob = clf.predict(X_test, best_iteration)
            test_pred_prob = softmax(test_pred_prob, axis=1)
            test_pred = np.argmax(test_pred_prob, axis=-1)
            prob_val[test_index] = test_pred_prob
            acc = accuracy_score(y_test, test_pred)
            Res_prob = clf.predict(X_T, best_iteration)
            Res_prob = softmax(Res_prob, axis=1)
            if name in acc_dict:
                acc_dict[name] += acc
                XG_LGBM_prob_Test[name] += Res_prob
            else:
                acc_dict[name] = acc
                XG_LGBM_prob_Test[name] = Res_prob
            print('name:', name, 'acc:', acc)
            # print('check acc:', accuracy_score(y_test, np.argmax(test_pred_prob, axis = -1) + 1))
            print('*' * 60)

    print('result:')
    for clf in acc_dict:
        acc_dict[clf] = acc_dict[clf] / 5.0
        log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
        log = log.append(log_entry)
        print(clf, ' 5-fold avg acc :', acc_dict[clf])

    print('tmp sum:', np.sum(tmp))
    print('time used %.2f s' % (time.time() - ST))
    return XG_LGBM_prob_Test['XGBClassifier'], prob_val

def get_train_test_data(files_list, n_dim):
    X = None
    for X_name in files_list:
        tmp = np.load(data_file_path + X_name)[:n_dim]
        print(tmp.shape, end='\t')
        if X is None:
            X = tmp
        else:
            X = np.concatenate([X, tmp], axis=1)
    print(' #')
    return X

learning_rate = 0.08
n_train = 440000
y_name = 'y_train_44w.npy'
train_files_list = ['train_basic_13_RF_1581.npy',
                    'train_X_UserID_normal_local_day.npy',
                    'train_X_UserID_normal_local_hour.npy',
                    'train_X_UserID_normal_local_hour_std.npy',
                    'train_X_UserID_normal_local_work_rest_fangjia_day.npy',
                    'train_X_UserID_normal_local_work_rest_fangjia_hour.npy',
                    'train_X_UserID_normal_local_work_rest_fangjia_hour_std.npy',
                    'train_X_UserID_normal_global_feature.npy',
                    'train_X_UserID_normal_global_feature_more.npy'] + ['train_feature_user_hour_visit_44w.npy',
                                                                        'train_feature_user_hour_user_44w.npy',
                                                                        'train_feature_user_holiday_visit_44w.npy',
                                                                        'train_feature_user_holiday_user_44w.npy'
                                                                        ]
test_files_list = ['test_basic_13_RF_1581.npy',
                   'test_X_UserID_normal_local_day.npy',
                   'test_X_UserID_normal_local_hour.npy',
                   'test_X_UserID_normal_local_hour_std.npy',
                   'test_X_UserID_normal_local_work_rest_fangjia_day.npy',
                   'test_X_UserID_normal_local_work_rest_fangjia_hour.npy',
                   'test_X_UserID_normal_local_work_rest_fangjia_hour_std.npy',
                   'test_X_UserID_normal_global_feature.npy',
                   'test_X_UserID_normal_global_feature_more.npy'] + ['test_feature_user_hour_visit_44w.npy',
                                                                      'test_feature_user_hour_user_44w.npy',
                                                                      'test_feature_user_holiday_visit_44w.npy',
                                                                      'test_feature_user_holiday_user_44w.npy'
                                                                      ]


print('learning_rate =', learning_rate)
print('train_files_list:', train_files_list)
print('test_files_list:', test_files_list)
print('done!')

X = get_train_test_data(train_files_list, n_dim=n_train)
X_T = get_train_test_data(test_files_list, n_dim=100000)
y = np.load(data_file_path + y_name)[:n_train] - 1
print('X = (N_sample, N_feature) =', X.shape)
print('X_T = (N_sample, N_feature) =', X_T.shape)


from sklearn.feature_selection import SelectFromModel
clf = RandomForestClassifier(random_state=42)

st = time.time()
print('X shape:', X.shape)
clf.fit(X, y)
print('fitting time: %.2f s' % (time.time() - st))


print('start:')
Acc = []
# thrd = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
thrd = [1.5, 1.0, 0.75, 0.5, 0.25]
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

    np.save(data_file_path + 'train_all_RF_%d.npy' % X_new.shape[-1], X_new)
    np.save(data_file_path + 'test_all_RF_%d.npy' % X_T_new.shape[-1], X_T_new)

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

