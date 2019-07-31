#coding=utf-8
import numpy as np
import pandas as pd
from config import config
# submit1_path=r"./submit/multimodal_bestloss_submission.csv"
# submit1_path=r"./submit/multimodal_se_resnext101_32x4d_bestloss_submission.csv"
# submit1_path=r"./submit/multimodal_se_resnext101_32x4d_focal_loss_bestloss_submission.csv"
submit1_path = './submit/%s_bestloss_submission.csv'% config.model_name
submit1=pd.read_csv(submit1_path)
submit1.drop('Target',axis=1,inplace=True)
submit1.Predicted=submit1.Predicted.apply(lambda x: "00"+str(int(x)+1))
submit1.Id=submit1.Id.apply(lambda x: str(x).zfill(6))
submit1=submit1.sort_values('Id',ascending=True)
submit1.to_csv("./submit/submit.txt",sep='\t',index=None,header=None)


"""
000000	006
000001	008
000002	002
000003	006
000004	006
000005	001
000006	007
000007	003
000008	007
000009	006
000010	007
000011	005
000012	002
000013	007
000014	003
000015	006
000016	001
000017	007
000018	006
000019	005
000020	001
"""