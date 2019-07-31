#coding=utf-8
import warnings


class DefaultConfigs(object):

    main_data_path = '/home/download-20190701/'
    
    train_feature_out_path = './feature/train/'
    train_table_path = main_data_path + 'train_44w.txt'
    train_main_visit_path = main_data_path + "train_visit/"
    
    test_feature_out_path = './feature/test/'
    test_table_path = main_data_path + 'test.txt'
    test_main_visit_path = main_data_path + "test_visit/"
    
    True_Small = False # True 只测试小部分数据
    
    part_size = 2000

    
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
# print(config.main_data_path)
