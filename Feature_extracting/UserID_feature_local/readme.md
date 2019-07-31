## 步骤（由于内存限制，我们只能把特征分成下面几个程序分开跑，并且做了点小技巧，将几个数用k进制转化为一个数，节省内存）
- step 1: 跑normal_local.py, 得到第1个feature
- step 2: 跑normal_hour_local.py, 得到第2个feature
- step 3: 跑normal_hour_local_std.py, 得到第3个feature
- step 4: 跑normal_work_rest_fangjia_hour_local.py, 得到第4个feature
- step 5: 跑normal_work_rest_fangjia_hour_local_std.py, 得到第5个feature
- step 6: 跑normal_work_rest_fangjia_local.py, 得到第6个feature
- step 7: 跑data_precessing_user_id_number_holiday.py, 得到第7个feature
- step 8: 跑data_precessing_user_id_number_hour.py, 得到第8个feature


