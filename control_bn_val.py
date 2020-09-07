import CNN_MNIST_Val
import numpy as np
n_tr = 5
acc_list = np.zeros((n_tr))
acct_list = np.zeros((n_tr))
rate = 0.001
dr = 1000
for i in range(n_tr):
    acc_list[i], acct_list[i], *_ = CNN_MNIST_Val.CNN_KF18bn_RBF_AUG(lrate=rate, drate=dr)
    print(acc_list)
    print(acct_list)

print(acc_list.mean(), acct_list.mean(), rate, dr)