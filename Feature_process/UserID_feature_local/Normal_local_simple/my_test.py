# jinzhi = 185

# def map_250_to_num(x):
#     res = 0
#     for i in range(len(x)):
#         res += x[i] * (jinzhi ** i)
#     return res
#
#
# def map_num_to_250(res, k=None):
#     assert not (k == None)
#     x = []
#     for i in range(k):
#         x.append(res % jinzhi)
#         res = res // jinzhi
#     return x


def map_250_to_num(x, jinzhi=250):
    res = 0
    for i in range(len(x)):
        res += x[i] * (jinzhi ** i)
    return res


def map_num_to_250(res, k=None, jinzhi=250):
    assert not (k==None)
    x = []
    for i in range(k):
        x.append(res % jinzhi)
        res = res // jinzhi
    return x


k = 5
# x = np.array([240.0 + _  for _ in range(k)])
x = [240.0 + _  for _ in range(k)]
res = map_250_to_num(x)
print(x)

x = map_num_to_250(res, k=k)
print(res)
print(x)

x = map_num_to_250(res+0.0, k=k)
print(res)
print(x)



import numpy as np

def fuyuan_feature(user_place_visit_num_, k_wei=None, len_feature=None, num=None):
    res = []
    for label in range(9):
        for j in range(len_feature):
            if num==2:
                res += map_num_to_250(user_place_visit_num_[j, label], k=k_wei, jinzhi=250)
            elif num ==3:
                res += map_num_to_250(user_place_visit_num_[j, label], k=k_wei, jinzhi=185)
            elif num ==4:
                res += map_num_to_250(user_place_visit_num_[j, label], k=k_wei, jinzhi=150)
    return res


user_place_visit_num_ = np.zeros((2, 9)) + 5.0

f_n_user = []
tmp = fuyuan_feature(user_place_visit_num_, k_wei=5, len_feature=2, num=2)
f_n_user.append(tmp)
f_n_user = np.array(f_n_user)

print(f_n_user.shape[-1] == 5 * 2 * 9)