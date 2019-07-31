# coding=utf-8
import numpy as np
import pandas as pd
base_path = "/home/download-20190701/"
def generate(prd_name='./submit/Label_Ensemble_XG_LGBM.npy', filenameto='./submit/submit.txt'):
    table = pd.read_csv(base_path + 'test.txt', header=None)
    filenames = [a[0].split("/")[-1].split('.')[0] for a in table.values]
    # prd_name = './submit/LGBMClassifier.npy'
    # prd_name = './submit/Label_Ensemble_XG_LGBM.npy'  # 预测的label名字
    # prd_name = './submit/Label_LGBM.npy'  # 预测的label名字
    print(prd_name)
    # pred_label = np.load('./submit/pred_label.npy')
    pred_label = np.load(prd_name)
    length = len(filenames)
    print(filenames[:5])
    print(len(filenames))

    dict = {}
    for index, Id in enumerate(filenames):
        dict[int(Id)] = (Id, pred_label[index])

    f = open(filenameto, "w+")
    for index in range(length):
        # f.write( Id + '\t%d' % 1+"\n")
        Id, Class = dict[int(index)]
        f.write(Id + '\t00%d' % Class + "\n")
        if index < 10:
            print(Id + '\t00%d' % Class)
    f.close()

if __name__ == '__main__':
    generate(prd_name='./submit/Label_LGBM_1.npy', filenameto='./submit/submit_lgbm.txt')

"""
000000	002
000001	003
000002	007
000003	006
000004	006
000005	007
000006	007
000007	003
000008	007
000009	006
000010	007
000011	007
000012	003
000013	003
000014	003
000015	006
000016	006
"""
