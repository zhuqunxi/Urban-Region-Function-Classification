import pickle  # pickle

import pandas as pd
import time
import numpy as np

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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

warnings.filterwarnings(action='ignore', category=DeprecationWarning)


def ceshi(classifiers, X, y, data_out_name=None, n_f=None):
    log_cols = ["Classifier", "Accuracy"]
    log = pd.DataFrame(columns=log_cols)

    sss = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    seed = 2019

    acc_dict = {}

    print('start:')
    ST = time.time()
    cnt_val = 0

    En_prob = np.zeros((len(y), 9)) + 0.0

    tmp = np.zeros((len(y),))

    for train_index, test_index in sss.split(X, y):
        cnt_val += 1
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
            # clf.fit(X_train, y_train)
            clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
                    verbose=True, early_stopping_rounds=25)
            print('fitting time: %.2f s' % (time.time() - st))

            # save Model
            # with open('./ENSEMBLE_model/%s_fold_%d_num_n_f_%d.pickle' % (name, cnt_val, n_f), 'wb') as f:
            #     pickle.dump(clf, f)

            # train_predictions = clf.predict(X_test)
            # acc = accuracy_score(y_test, train_predictions)
            # test_pred_prob = clf.predict_proba(X_test)
            # En_prob[test_index] = test_pred_prob

            if name == "XGBClassifier":
                best_iteration = clf.get_booster().best_iteration
            else:
                best_iteration = clf.best_iteration_
            train_predictions = clf.predict_proba(X_test, best_iteration)
            train_predictions = np.argmax(train_predictions, axis=-1) + 1
            acc = accuracy_score(y_test, train_predictions)
            test_pred_prob = clf.predict_proba(X_test, best_iteration)
            En_prob[test_index] += test_pred_prob



            if name in acc_dict:
                acc_dict[name] += acc
            else:
                acc_dict[name] = acc
            print('name:', name, 'acc:', acc)
            # print('check acc:', accuracy_score(y_test, np.argmax(test_pred_prob, axis = -1) + 1))
            print('*' * 60)

        break

    # print('result:')
    # for clf in acc_dict:
    #     acc_dict[clf] = acc_dict[clf] / 5.0
    #     log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    #     log = log.append(log_entry)
    #     print(clf, ' 5-fold avg acc :', acc_dict[clf])
    #
    # print('train ensemble acc:', accuracy_score(y, np.argmax(En_prob, axis=-1) + 1))
    # print('train sample 10 predicted label:', np.argmax(En_prob[:20], axis=-1) + 1)
    # print('train sample 10 label :', y[:20])
    # print('tmp sum:', np.sum(tmp))
    # print('time used %.2f s' % (time.time() - ST))

    # return accuracy_score(y, np.argmax(En_prob, axis=-1) + 1)

    return acc

if __name__ == '__main__':

    data_input_name = 'feature'
    data_file_path = './%s/' % data_input_name


    print('done!')

    n_t = 10000
    #############################################################################################
    classifiers = [
        lgbm.LGBMClassifier(random_state=42, n_jobs=-1),
    ]


    y_name = 'y_vis_train_feature_4003_X_1468_304_2231.npy'
    y = np.load(data_file_path + y_name)

    # X_name = 'train_basic_feature_2.npy'
    # X_name_test = 'test_basic_feature_2.npy'


    Acc = []
    XX = []
    for i in [1, 3]:
        X_name = 'train_basic_feature_%d.npy' % i
        X_name_test = 'test_basic_feature_%d.npy' % i
        X = np.load(data_file_path + X_name)
        X_T = np.load(data_file_path + X_name_test)
        XX.append(X)
        n_f = X.shape[-1]
        print('X = (N_sample, N_feature) =', X.shape)
        print('X_T = (N_sample, N_feature) =', X_T.shape)
        if i==1:
            continue
        acc = ceshi(classifiers, X, y, data_out_name=None, n_f=None)
        Acc.append(acc)


    X = np.concatenate(XX, axis=1)
    print('X = (N_sample, N_feature) =', X.shape)
    acc = ceshi(classifiers, X, y, data_out_name=None, n_f=None)
    Acc.append(acc)
    print('#' * 50)
    print(Acc)

    # X = np.load(data_file_path + 'RF_X_train.npy')
    # print('X = (N_sample, N_feature) =', X.shape)
    # acc = ceshi(classifiers, X, y, data_out_name=None, n_f=None)
    # print('acc: ', acc)

    print('done!')


'''
#########################
train_basic_feature_1.npy   train_basic_feature_2.npy       combine 1 & 2
[0.6566826076013088,        0.6473949156808457,             0.6618172665492071]

train_basic_feature_3.npy       combine 1 & 3
0.6493581676315127, 0.6621193053108482
RF_1381
0.6565064183236848

'''