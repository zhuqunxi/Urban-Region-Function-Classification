## 步骤

- step 1: 跑Code_Basic_feature_1文件中的main.py, 得到第1个feature
- step 2: 跑Code_Basic_feature_3文件中的main.py, 得到第2个feature
- step 3: 
	- 1） 跑Combine_feature文件中的RF_selection.py, 将上面的两个特征合并，并且用RF删选出部分特征(不然特征太多了，很多特征的importance非常小); 
	- 2） 跑Combine_feature文件中的RF_RF_selection.py, 将删选出的特征，同样再用RF删选出更少的特征，以便后面用


