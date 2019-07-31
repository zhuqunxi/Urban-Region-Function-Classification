import os
import sys

main_data_path = '/home/download-20190701/'

train_visit_path = main_data_path + 'train_visit/'
test_visit_path = main_data_path + 'test_image/'

files = []
for file in os.listdir(train_visit_path):
    files.append(train_visit_path + file)
sys.stdout.write("\n")
print('done!')
file_num = len(files)
print(file_num)

f = open(main_data_path + "train_44w.txt", "w+")
for index, item in enumerate(files):
    # print(item)
    sys.stdout.write(
        '\r>> Processing visit data %d/%d --- file name: %s' % (index + 1, file_num, item))
    sys.stdout.flush()
    f.write(item+"\n")
    # if index > 5:
    #     break
f.close()
sys.stdout.write("\n")

####################################################################################################

files = []
for file in os.listdir(test_visit_path):
    files.append(test_visit_path + file)
sys.stdout.write("\n")
print('done!')
file_num = len(files)
print(file_num)

f = open(main_data_path + "test.txt", "w+")
for index, item in enumerate(files):
    # print(item)
    sys.stdout.write(
        '\r>> Processing visit data %d/%d --- file name: %s' % (index + 1, file_num, item))
    sys.stdout.flush()
    f.write(item+"\n")
    # if index > 5:
    #     break
f.close()
sys.stdout.write("\n")

