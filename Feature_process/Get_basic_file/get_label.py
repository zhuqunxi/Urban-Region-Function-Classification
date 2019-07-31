import os
import sys
import numpy as np
import pandas as pd
main_data_path = '/home/download-20190701/'

train_visit_path = main_data_path + 'train_visit/'
# test_visit_path = main_data_path + 'test_image/'


ID = []
LABEL = []
Files = os.listdir(train_visit_path)
file_num = len(Files)
print('file_num:', file_num)
for index, file in enumerate(Files):
    Id = file[:file.find('.')]
    label = int(Id[-1])
    ID.append(Id)
    LABEL.append(label)

    # print('#' * 40)
    # print('index:', index)
    # print('file:', file)
    # print('Id:', Id)
    # print('label:', label)

    sys.stdout.write(
        '\r>> Processing visit data %d/%d --- file name: %s -- Id: %s , label: %d' % (index + 1, file_num, file, Id, label))
    sys.stdout.flush()
    # if index > 5:
    #     break
sys.stdout.write("\n")
np.save('y_train_44w.npy', np.array(LABEL))