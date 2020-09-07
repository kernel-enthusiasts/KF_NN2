import CNN_MNIST
import numpy as np

acc_list = np.zeros((20))
acct_list = np.zeros((20))
for i in range(20):
    acc_list[i], acct_list[i], *_ = CNN_MNIST.CNN_KF180bn_RBF_AUG()
    print(acc_list)
    print(acct_list)

print(acc_list.mean(), acct_list.mean())