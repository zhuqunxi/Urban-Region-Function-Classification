## **[Urban Region Function Classification](https://dianshi.baidu.com/competition/30/rank) Top 2 方案**

### 团队介绍

队名：海疯习习

成员：
- 朱群喜，复旦大学数学系博士，目前在哈佛医学院交流；
- 周杰，华东师范大学计算机系博士，目前在加拿大约克大学交流，曾获得[KDD CUP 2017](https://github.com/12190143/Black-Swan)和[KDD CUP 2018](https://github.com/12190143/KDD_CUP_2018) Top3.

名次：初赛第一，复赛第二

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
  - 2）统计一个用户全局的Global特征，及一个用户在不同地方次数的统计信息（ps：online的提高也有将近1个百分点）
  - 3）用户有规律出入到特征统计，如统计用户节假日情况，将一天分成四个时间段统计用户在这些时间段特征。

- 初赛和复赛数据合并，从40w增加到44w样本（复赛大概提高1个点）

#### 模型
- 深度学习模型
   - 1）图像特征抽取使Densenet121，获得样本概率，用于stacking（线下0.566375）
   - 2）7×26×24特征用dpn26网络进行训练，得到样本概率，用于stacking（线下0.643914）
- 机器学习模型
   - 1）Lightgbm模型和Xgboost模型，除了学习率和迭代次数，其他使用默认参数，结合前面抽取特征 （线下0.907）
- stacking
   - 使用Lightgbm进行Stacking （线下0.910786，线上0.90203）
   
### 代码使用说明
- [preprocessing](https://github.com/zhuqunxi/Urban-Region-Function-Classification-/tree/master/data_processing) 数据预处理
	- 1）train.txt, test.txt, train.csv, test.csv生成
	- 2）样本npy文件生成，用于深度学习模型
- [Feature_extracting](https://github.com/zhuqunxi/Urban-Region-Function-Classification-/tree/master/Feature_extracting) 特征抽取 （特征的详细说明，见[blog](https://www.cnblogs.com/skykill/p/11273640.html)）
	- 1）[Basic_feature](https://github.com/zhuqunxi/Urban-Region-Function-Classification/tree/master/Feature_extracting/Basic_feature) 基础特征
	- 2）[UserID_feature_local](https://github.com/zhuqunxi/Urban-Region-Function-Classification/tree/master/Feature_extracting/UserID_feature_local) 用户id的local特征抽取
	- 3）[UserID_feature_global](https://github.com/zhuqunxi/Urban-Region-Function-Classification/tree/master/Feature_extracting/UserID_feature_global) 用户id的global特征抽取
	
- [NN](https://github.com/zhuqunxi/Urban-Region-Function-Classification-/tree/master/CNN) 神经网络模型
	- [Image和Visit概率特征](https://github.com/zhuqunxi/Urban-Region-Function-Classification/tree/master/CNN) 5-fold，得到训练的概率特征，测试的概率特征（取平均）
- [ML](https://github.com/zhuqunxi/Urban-Region-Function-Classification-/tree/master/ML) 模型
	- [LightGBM概率特征](https://github.com/zhuqunxi/Urban-Region-Function-Classification/tree/master/ML) 5-fold，得到训练的概率特征，测试的概率特征（取平均）
- [ML_stack_model](https://github.com/zhuqunxi/Urban-Region-Function-Classification-/tree/master/ML_stack_model) Stacking
	- [LightGBM跑概率特征](https://github.com/zhuqunxi/Urban-Region-Function-Classification-/tree/master/ML_stack_model) 5-fold


### 感想
朱群喜：第一次参加这么大型还有奖金的比赛，有点感想哈。作为一名应用数学专业（学的贼烂T﹏T）的3年级直博生（马上4年级了，老了），最近陷入了学术的针扎中，心中的滋味也就自己能体会。偶然间，不知是从哪里，发现了这个比赛，看着标题和奖金感觉挺有吸引力的。仔细想了想，要不去玩一玩，算是排解下压力，转移下注意力吧。这次比赛，当然要感谢队友华师大NLP**周杰**（一个热爱在[知乎](https://www.zhihu.com/people/zhou-jie-77-75/activities)回答问题和[github](https://github.com/12190143)上开源的的少年），以及开源[baseline_1](https://github.com/czczup/UrbanRegionFunctionClassification), [baseline_2](https://github.com/ABadCandy/BaiDuBigData19-URFC)的大佬们，还有写这篇[blog](https://blog.csdn.net/qq_34919792/article/details/93976813)的博主，以及赛事主办方联合国教科文组织国际工程科技知识中心（IKCEST）、中国工程科技知识中心（CKCEST）、百度公司及西安交通大学的大力支持。最后也祝贺各个进决赛的队伍，希望能从你们那学到点东西。

周杰：楼上大佬有些谦虚，是我本科校友，一直特别认真，这次比赛也是废寝忘食。本人小白有幸参与。也算是一直做学术觉得乏味来参与比赛，提高实践能力同时提高对数据对敏锐度。坚信对于任何比赛，付出时间就不会太差，其他事情也是一样，共勉。

### Contacts
- qxzhu16@fudan.edu.cn, QZHU6@mgh.harvard.edu
- jzhou@ica.stc.sh.cn
