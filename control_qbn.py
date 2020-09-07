import CNN_qMNIST

import numpy as np

acc_list = np.zeros((20))
for i in range(20):
    acc_list[i], *_ = CNN_qMNIST.CNN_KF18bn_RBF_AUG()
    print(acc_list)

print(acc_list.mean())