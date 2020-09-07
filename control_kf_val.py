import CNN_MNIST_Val
import numpy as np
n_tr = 5
acc_list = np.zeros((n_tr))
acct_list = np.zeros((n_tr))
l3 = 0.0
l6 = 1.0
g_fac = 1.0
for i in range(n_tr):
    acc_list[i], acct_list[i], *_ = CNN_MNIST_Val.CNN_KF18kf2_RBF_AUG(lam3=l3, lam6=l6, gamma_fac=g_fac)
    print(acc_list)
    print(acct_list)

print(acc_list.mean(), acct_list.mean(), l3, l6, g_fac)