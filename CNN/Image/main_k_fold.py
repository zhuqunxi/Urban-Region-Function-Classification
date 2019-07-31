from __future__ import print_function
import os
import time
import json
import torch
import random
import warnings
import torchvision
import numpy as np
import pandas as pd

from utils import *
from multimodal import MultiModalDataset, MultiModalNet, \
    VisitModalNet, ImageModalNet, CosineAnnealingLR

from tqdm import tqdm
from config import config
from datetime import datetime
from torch import nn, optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.model_selection import StratifiedKFold
from timeit import default_timer as timer
from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as F
import time


log = Logger()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.cuda.set_device(1)
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

if not os.path.exists("./logs/"):
    os.mkdir("./logs/")

def train(train_loader, model, criterion, optimizer, epoch, valid_metrics, best_results, start):
    losses = AverageMeter()
    f1 = AverageMeter()
    acc = AverageMeter()
    
    model.train()

    Train_ST = time.time()
    for i, (images, visit, target) in enumerate(train_loader):
        ST = time.time()
        # visit=visit.to(device)
        del visit
        images = images.to(device)
        if images.size(0) == 1:
            continue
        indx_target = target.clone()
        target = torch.from_numpy(np.array(target)).long().to(device)
        # compute output
        # output = model(images,visit)

        st = time.time()
        output = model(images)
        time_used = 'sample_num_now = %d. batchsize = %d, batch_id = %d, single batch time used %.2f s' \
                    % (config.batch_size * (i + 1), config.batch_size, i, time.time() - st)
        
        # output = metric_fc(output, target)
        # output = metric_fc(output)
        
        # print(output.cpu().data.numpy()[0])
        # print(F.softmax(output).cpu().data.numpy()[0])
        # print('predict:\t', np.argmax(F.softmax(output).cpu().data.numpy(),axis=1))
        # print('   true:\t', target.cpu().data.numpy())
        
        loss = criterion(output, target)
        losses.update(loss.item(), images.size(0))
        f1_batch = f1_score(target.cpu().data.numpy(), np.argmax(F.softmax(output).cpu().data.numpy(), axis=1),
                            average='macro')
        acc_score = accuracy_score(target.cpu().data.numpy(), np.argmax(F.softmax(output).cpu().data.numpy(), axis=1))
        f1.update(f1_batch, images.size(0))
        acc.update(acc_score, images.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print('\r', end='', flush=True)
        message = '%s %5.1f %6.1f      |   %0.3f  %0.3f  %0.3f  | %0.3f  %0.3f  %0.4f   | %s  %s  %s |   %s' % ( \
            "train", i / len(train_loader) + epoch, epoch,
            acc.avg, losses.avg, f1.avg,
            valid_metrics[0], valid_metrics[1], valid_metrics[2],
            str(best_results[0])[:8], str(best_results[1])[:8], str(best_results[2])[:8],
            time_to_str((timer() - start), 'min'))
        print(message, end='', flush=True)

        singe_for_time = 'singe for time used: %.2f s' % (time.time() - ST)

        # log.write(message)
        # log.write("\n")
        # print('#' * 60)
        # print(time_used)
        # print(singe_for_time)
        # print('#' * 60)
        #
        # if config.batch_size * (i + 1) > 1000:
        #     print('#' * 60)
        #     print('sample_num_now = %d. batchsize = %d, batch_id = %d' \
        #               % (config.batch_size * (i + 1), config.batch_size, i))
        #     print('total time used: %.2f s' % (time.time() - Train_ST))
        #     print('#' * 60 + '\n')
        #     break

    log.write("\n")
    # log.write(message)
    # log.write("\n")
    return [acc.avg, losses.avg, f1.avg]

# 2. evaluate function
def evaluate(val_loader, model, criterion, epoch, train_metrics, best_results, start):
    # only meter loss and f1 score
    losses = AverageMeter()
    f1 = AverageMeter()
    acc = AverageMeter()
    # switch mode for evaluation
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (images, visit, target) in enumerate(val_loader):
            del visit
            images = images.to(device)
            # visit=visit.to(device)
            indx_target = target.clone()
            target = torch.from_numpy(np.array(target)).long().to(device)
            
            # output = model(images_var,visit)
            output = model(images)
            # output = metric_fc(output, target)
            # output = metric_fc(output)
            
            # print('output:', output.cpu().data.numpy())
            loss = criterion(output, target)
            
            losses.update(loss.item(), images.size(0))
            f1_batch = f1_score(target.cpu().data.numpy(), np.argmax(F.softmax(output).cpu().data.numpy(), axis=1),
                                average='macro')
            acc_score = accuracy_score(target.cpu().data.numpy(),
                                       np.argmax(F.softmax(output).cpu().data.numpy(), axis=1))
            f1.update(f1_batch, images.size(0))
            acc.update(acc_score, images.size(0))
            print('\r', end='', flush=True)
            message = '%s   %5.1f %6.1f     |     %0.3f  %0.3f   %0.3f    | %0.3f  %0.3f  %0.4f  | %s  %s  %s  |  %s' % ( \
                "val", i / len(val_loader) + epoch, epoch,
                acc.avg, losses.avg, f1.avg,
                train_metrics[0], train_metrics[1], train_metrics[2],
                str(best_results[0])[:8], str(best_results[1])[:8], str(best_results[2])[:8],
                time_to_str((timer() - start), 'min'))
            
            print(message, end='', flush=True)
        log.write("\n")
        # log.write(message)
        # log.write("\n")
    
    return [acc.avg, losses.avg, f1.avg]


# 3. test model on public dataset and save the probability matrix
def ttest(test_loader, model, folds):
    sample_submission_df = pd.read_csv("./test.csv")
    # 3.1 confirm the model converted to cuda
    filenames, labels, submissions = [], [], []
    model.to(device)
    model.eval()
    submit_results = []
    for i, (images, visit, filepath) in tqdm(enumerate(test_loader)):
        del visit
        # 3.2 change everything to cuda and get only basename
        filepath = [os.path.basename(x) for x in filepath]
        with torch.no_grad():
            images = images.to(device)
            # visit=visit.to(device)
            # y_pred = model(image_var,visit)
            y_pred = model(images)
            
            label = F.softmax(y_pred).cpu().data.numpy()
            labels.append(label == np.max(label))
            filenames.append(filepath)
    
    for row in np.concatenate(labels):
        subrow = np.argmax(row)
        submissions.append(subrow)
    sample_submission_df['Predicted'] = submissions
    sample_submission_df.to_csv('./submit/%s_bestloss_submission.csv' % config.model_name, index=None)

def get_prob_val_and_test(loader, model):
    model.to(device)
    model.eval()
    Prob = None
    for i, (images, visit, _) in tqdm(enumerate(loader)):
        del visit
        with torch.no_grad():
            images = images.to(device)
            # visit=visit.to(device)
            # y_pred = model(image_var,visit)
            y_pred = model(images)

            label_prob = F.softmax(y_pred).cpu().data.numpy()

            if Prob is None:
                Prob = label_prob
            else:
                Prob = np.concatenate([Prob, label_prob])

    return np.array(Prob)


# 4. main function
def main(fold, train_data_list, val_data_list):
    log.open("logs/%s_log_train_fold_%d.txt" % (config.model_name, fold), mode="a")
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
    log.write(
        '                           |------------ Train -------|----------- Valid ---------|----------Best Results---|------------|\n')
    log.write(
        'mode     iter     epoch    |    acc  loss  f1_macro   |    acc  loss  f1_macro    |    loss  f1_macro       | time       |\n')
    log.write(
        '-------------------------------------------------------------------------------------------------------------------------|\n')
    
    # 4.1 mkdirs
    if not os.path.exists(config.submit):
        os.makedirs(config.submit)
    if not os.path.exists(config.weights + config.model_name + os.sep + str(fold)):
        os.makedirs(config.weights + config.model_name + os.sep + str(fold))
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")
    
    # 4.2 get model
    model = ImageModalNet(config.pre_train_name, "dpn26", 0.25)
    
    # print(model)
    
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('pre_train_name:', config.pre_train_name)
    print('parameters: ', para)
    print('Model  params: {:4f}M'.format(para * 4 / 1000 / 1000))
    
    # 4.3 optim & criterion
    # optimizer = optim.SGD([[{'params': model.parameters()},
    #                         {'params': metric_fc.parameters()}]],
    #                       lr = config.lr,
    #                       momentum=0.9,
    #                       weight_decay=1e-4)
    optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                 lr=config.lr,
                                 weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=config.lr,
    #                              weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = ArcFaceLoss().to(device)
    # criterion = My_loss().to(device)
    # criterion = FocalLoss().to(device)
    
    start_epoch = 0
    best_acc = 0
    best_loss = np.inf
    best_f1 = 0
    best_results = [0, np.inf, 0]
    val_metrics = [0, np.inf, 0]
    
    Reused = True
    if Reused:
        filename = config.weights + config.model_name + os.sep + \
                   str(fold) + os.sep + "checkpoint.pth.tar"
        current_model = torch.load(filename)
        # print(current_model["state_dict"])
        model.load_state_dict(current_model["state_dict"])

        print('current_model_epoch:', current_model["epoch"])
        print('current_model_model_name:', current_model["model_name"])
        print('current_model_best_loss:', current_model["best_loss"])
        print('current_model_best_acc:', current_model["best_acc"])
        print('current_model_best_f1:', current_model["best_f1"])
        del current_model
    
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    model.to(device)
    # metric_fc.to(device)
    
    
    test_files = pd.read_csv("./test.csv")
    
    # train_data_list, val_data_list = train_test_split(all_files, test_size=0.1, random_state=2019)
    
    # load dataset
    train_gen = MultiModalDataset(train_data_list, config.train_data, config.train_vis, mode="train")
    train_loader = DataLoader(train_gen, batch_size=config.batch_size, shuffle=True, pin_memory=True,
                              num_workers=1)  # num_worker is limited by shared memory in Docker!
    # train_loader = DataLoader(train_gen, batch_size=config.batch_size, shuffle=False, pin_memory=True,
    #                           num_workers=1)  # num_worker is limited by shared memory in Docker!
    
    val_gen = MultiModalDataset(val_data_list, config.train_data, config.train_vis, augument=False, mode="train")
    val_loader = DataLoader(val_gen, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=1)
    
    test_gen = MultiModalDataset(test_files, config.test_data, config.test_vis, augument=False, mode="test")
    test_loader = DataLoader(test_gen, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=1)
    
    # scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    # n_batches = int(len(train_loader.dataset) // train_loader.batch_size)
    # scheduler = CosineAnnealingLR(optimizer, T_max=n_batches*2)
    start = timer()
    
    # train
    for epoch in range(0, config.epochs):
        print('lr:', config.lr / (2 ** np.sqrt(epoch + 0.1)))
        optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                     lr=config.lr / (2 ** np.sqrt(epoch + 0.1)),
                                     weight_decay=1e-4)
        
        scheduler.step(epoch)
        # train
        train_metrics = train(train_loader, model, criterion, optimizer, epoch, val_metrics, best_results, start)
        # val
        val_metrics = evaluate(val_loader, model, criterion, epoch, train_metrics, best_results, start)
        # check results
        is_best_acc = val_metrics[0] > best_results[0]
        best_results[0] = max(val_metrics[0], best_results[0])
        is_best_loss = val_metrics[1] < best_results[1]
        best_results[1] = min(val_metrics[1], best_results[1])
        is_best_f1 = val_metrics[2] > best_results[2]
        best_results[2] = max(val_metrics[2], best_results[2])
        # save model
        save_checkpoint({
            "epoch": epoch + 1,
            "model_name": config.model_name,
            "state_dict": model.state_dict(),
            "best_acc": best_results[0],
            "best_loss": best_results[1],
            "optimizer": optimizer.state_dict(),
            "fold": fold,
            "best_f1": best_results[2],
        }, is_best_acc, is_best_loss, is_best_f1, fold)
        # print logs
        print('\r', end='', flush=True)
        log.write(
            '%s  %5.1f %6.1f      |   %0.3f   %0.3f   %0.3f     |  %0.3f   %0.3f    %0.3f    |   %s  %s  %s | %s' % ( \
                "best", epoch, epoch,
                train_metrics[0], train_metrics[1], train_metrics[2],
                val_metrics[0], val_metrics[1], val_metrics[2],
                str(best_results[0])[:8], str(best_results[1])[:8], str(best_results[2])[:8],
                time_to_str((timer() - start), 'min'))
        )
        log.write("\n")
        time.sleep(0.01)


    
    print('load best loss model!')
    best_model = torch.load(
        "%s/%s_fold_%s_model_best_loss.pth.tar" % (config.best_models, config.model_name, str(fold)))
    # best_model = torch.load("%s/%s_fold_%s_model_best_acc.pth.tar"%(config.best_models,config.model_name,str(fold)))
    print('model_best_loss:', best_model["best_loss"])
    print('model_best_acc:', best_model["best_acc"])
    print('model_best_f1:', best_model["best_f1"])

    
    model.load_state_dict(best_model["state_dict"])

    Prob_Val = get_prob_val_and_test(val_loader, model)
    Prob_Test = get_prob_val_and_test(test_loader, model)


    return Prob_Val, Prob_Test


def rearrange_data(PROB_Train=None, PROB_Test=None):
    all_files = pd.read_csv("./train_44w.csv")
    y = all_files['Target'].values
    main_data_path = config.main_path
    train_map_Id2num, test_map_Id2num = {}, {}
    Id_train = pd.read_csv("./train_44w.csv")['Id'].values
    Id_test = pd.read_csv("./test.csv")['Id'].apply(lambda x: str(x).zfill(6)).values

    for i in range(len(Id_train)):
        train_map_Id2num[Id_train[i]] = i
    for i in range(len(Id_test)):
        test_map_Id2num[Id_test[i]] = i

    print(Id_train[:5])
    print(Id_test[:5])

    train_table = pd.read_csv(main_data_path + "train_44w.txt", header=None).values
    train_filenames = [a[0].split("/")[-1].split('.')[0] for a in train_table]
    test_table = pd.read_csv(main_data_path + "test.txt", header=None).values
    test_filenames = [a[0].split("/")[-1].split('.')[0] for a in test_table]

    print(train_filenames[:5])
    print(test_filenames[:5])

    train_index, test_index = [], []
    for index in range(len(train_table)):
        filename = train_filenames[index]
        train_index.append(train_map_Id2num[filename])

    for index in range(len(test_table)):
        filename = test_filenames[index]
        train_index.append(test_map_Id2num[filename])

    train_index, test_index = np.array(train_index).astype(np.int64), np.array(test_index).astype(np.int64)
    y = y[train_index]

    # yy = np.load('y_vis_train_feature_4003_X_1468_304_2231.npy')

    print(y[:10])
    # print(yy[:10])

    return PROB_Train[train_index], y[train_index], PROB_Test[test_index]

if __name__ == "__main__":
    # rearrange_data()

    All_seed = [2019, 2050, 1314, 520, 748]
    all_files = pd.read_csv("./train_44w.csv")
    N_train = len(all_files)
    y = all_files['Target'].values
    print(y[:5])
    sss = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # sss = StratifiedKFold(n_splits=5, shuffle=False, random_state=0)

    PROB_Train = np.zeros((N_train, 9))
    PROB_Test = None
    st = time.time()
    for fold, (train_index, val_index) in enumerate(sss.split(all_files, y)):
        #if fold != 2:
        #   continue
        print('*' * 60)
        print('fold:', fold)
        print('train_index[:10]:', train_index[:10])
        print('train data = %d, valid data = %d' % (len(train_index), len(val_index)))
        train_data_list, val_data_list = all_files.iloc[train_index], all_files.iloc[val_index]

        Prob_Val, Prob_Test = main(fold, train_data_list, val_data_list)
        print('Prob_Val shape:', Prob_Val.shape)
        print('Prob_Test shape:', Prob_Test.shape)
        print('time used: %.2f s' % (time.time() - st))
        np.save('./feature/PROB_Train_fold_%d.npy' % fold, Prob_Val)
        np.save('./feature/PROB_Test_fold_%d.npy' % fold, Prob_Test)

        PROB_Train[val_index] = Prob_Val
        if PROB_Test is None:
            PROB_Test = Prob_Test
        else:
            PROB_Test += Prob_Test

    PROB_Test /= 5
    print('y shape, max, min:', y.shape, y.max(), y.min())
    print('PROB_Train shape, max, min:', PROB_Train.shape, PROB_Train.max(), PROB_Train.min())
    print('PROB_Test shape, max, min:', PROB_Test.shape, PROB_Test.max(), PROB_Test.min())

    #PROB_Train, y, PROB_Test = rearrange_data(PROB_Train=PROB_Train, PROB_Test=PROB_Test)

    if not os.path.exists('./feature/'):
        os.makedirs('./feature/')
    np.save('./feature/PROB_Train.npy', PROB_Train)
    np.save('./feature/PROB_Test.npy', PROB_Test)
    #np.save('./feature/y.npy', y)


'''
rob_Val shape: (7942, 9)
Prob_Test shape: (10000, 9)
time used: 3500.88 s
y shape, max, min: (39730,) 8 0
PROB_Train shape, max, min: (39730, 9) 0.9999995231628418 2.1071732469951825e-32
PROB_Test shape, max, min: (10000, 9) 0.99993116 2.0780656e-13

y = [7 5 5 2 7 6 1 8 7 0]
yy = [8 6 6 3 8 7 2 9 8 1]

'''
