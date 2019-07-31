import pandas as pd
import time
import sys
import numpy as np
main_data_path = '/home/download-20190701/'

train_feature_out_path = './feature/train/'
train_table_path = main_data_path + 'train_44w.txt'
train_main_visit_path = main_data_path + "train_visit/"

def visit2array_train():
    table = pd.read_csv(train_table_path, header=None)
    filenames = [a[0].split("/")[-1].split('.')[0] for a in table.values]
    length = len(filenames)
    start_time = time.time()
    NUM_USER = []
    for index, filename in enumerate(filenames):
        table = pd.read_table(train_main_visit_path + filename + ".txt", header=None)
        users = table[0]
        num_user = len(users)
        NUM_USER.append(num_user)
        sys.stdout.write(
            '\r>> Processing train visit data %d/%d --- num of users = %d' % (index + 1, length, num_user))


    sys.stdout.write('\n')
    print("using time:%.2fs" % (time.time() - start_time))
    NUM_USER = np.array(NUM_USER)
    np.save('Num_User.npy', NUM_USER)

visit2array_train()