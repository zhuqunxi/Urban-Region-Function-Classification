## **[Urban Region Function Classification](https://dianshi.baidu.com/competition/30/rank) Top 1 方案**

### 团队介绍

队名：海疯习习

成员：朱群喜，复旦大学数学系博士，目前在哈佛医学院交流；周杰，华东师范大学计算机系博士，曾获得[KDD CUP 2017](https://github.com/12190143/Black-Swan)和[KDD CUP 2018](https://github.com/12190143/KDD_CUP_2018) Top3.

名次：初赛第一，复赛？

### 任务描述
给定用户访问数据和卫星图片，判断城市用地功能，包括Residential area, School, Industrial park, Railway station, Airport, Park, Shopping area, Administrative district和Hospital9个类别，具体任务描述见[官网](https://dianshi.baidu.com/competition/30/question) 

### Environmental Requirement
- Python 3.6
- Pytorch 0.4.0
- sklearn
- numpy
- XGboost
- Lightgbm

### 思路
#### 特征

- 用户Basic特征
  - 1）提取一个地区不同时间段（节假日，工作日，休息日）的统计特征，包括sum，mean，max等8个统计量，由于特征维度过大，使用RF进行特征选择

- 用户id特征挖掘

  - 1）统计一个用户出现在不同的地方次数，这里以一天表示1次，然后特征中对8个统计量进行统计（注意：样本统计中需要将去当前样本的信息，从而防止透露label信息而过拟合）（主要特征，直接到线上86+）
  - 2）统计一个用户全局的Global特征，及一个用户在不同地方次数的统计信息
  - 3）用户有规律出入到特征统计，如统计用户节假日情况，将一天分成四个时间段统计用户在这些时间段特征。

- 初赛和复赛数据合并，从40w增加到44w样本（复赛大概提高1个点）

#### 模型
- 深度学习模型
   - 1）图像特征抽取使Densenet121，获得样本概率，用于stacking（线下0.566375）
   - 2）7*26*24特征用dpn26网络进行训练，得到样本概率，用于stacking（线下0.643914）
- 机器学习模型
   - 1）Lightgbm模型和Xgboost模型，除了学习率和迭代次数，其他使用默认参数，结合前面抽取特征 （线下0.905048）
- stacking
   - 使用Lightgbm进行Stacking （线上90.02）

### Contacts
- qxzhu16@fudan.edu.cn
- jzhou@ica.stc.sh.cn
