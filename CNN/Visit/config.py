#coding=utf-8
import warnings
import os

class DefaultConfigs(object):
    
    metric =  'linear' # 'arc_margin'
    # metric = 'arc_margin'
    
    loss_name ='focal_loss'
    pre_train_name = 'densenet121'  # se_resnext101_32x4d
    # pre_train_name = 'se_resnext101_32x4d'
    env='default'
    # model_name = "multimodal"
    model_name = "multimodal_%s_%s_%s" % (pre_train_name, loss_name, metric)
    
    main_path = '/mnt/ssd/zhoujie/download-20190701/'
    train_data = main_path + "train_image/" # where is your train images data
    test_data = main_path +  "test_image/"   # your test data
    train_vis = main_path + "npy/train_visit"  # where is your train visits data
    test_vis = main_path + "npy/test_visit"
    load_model_path = None
    
    weights = "./checkpoints/"
    best_models = "./checkpoints/best_models/"
    debug_file='./tmp/debug'
    submit = "./submit/"
    
    num_classes = 9
    img_weight = 100
    img_height = 100
  
    # channels = 3
    # vis_channels=7
    # vis_weight=24
    # vis_height=26

    lr = 0.0005
    lr_decay = 0.5
    weight_decay =1e-5
    batch_size = 128
    epochs = 0
    
def parse(self, kwargs):
    """
    update config by kwargs
    """
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfigs.parse = parse
config = DefaultConfigs()

# print(os.listdir(config.train_data))
